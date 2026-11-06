#!/usr/bin/env python
"""Per-checkpoint diagnosis of the qwen35 "unlearning wall" (2026-09-29).

Question: GD/SatImp/WGA/CEU push the forget-corpus objective at least as far
on qwen35 as on Intern-S2 (GD NLL 213 vs 172, CEU entropy 11.6 vs 9.9), yet
WMDP-bio MCQ barely moves on qwen35. Hypothesis: an output-layer shortcut --
the objective is met by flattening / rescaling the output distribution while
the 4-way A/B/C/D ranking at "Answer:" (all lm_eval scores) survives.

For ONE model dir on ONE GPU this script records
  mcq   : WMDP-bio (1273) + a 14-subject MMLU slice, lm_eval prompt format.
          Per item: log-probs of " A".." D", full-vocab entropy / logsumexp /
          max logit / logit std at the answer position, argmax token.
  lens  : logit lens -- final norm + lm_head applied to every decoder layer's
          output at the answer position -> 4-way acc per layer (is the bio
          answer still linearly decodable in the middle of the network?).
  cloze : WMDP-bio scored as "Question: q\\nAnswer: <choice text>" (knowledge
          in raw-text form, the form the forget objective acts on).
  nll   : NLL / entropy / logsumexp on bio-forget (trained-on), bio-retain
          and fineweb documents + per-layer routed-expert counts (router shift).
  wdiff : (optional, --base) streaming per-tensor weight diff vs the base
          checkpoint (see weight_diff.py).
Writes <out_dir>/<name>.json (+ <name>.wdiff.json).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

HF_DS = "/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/jincheng/.cache/huggingface/datasets"
WMDP_BIO = f"{HF_DS}/cais___wmdp/wmdp-bio/0.0.0/7125571f22f032c56415e7980f48d877dd830ff8/wmdp-test.arrow"
MMLU = f"{HF_DS}/cais___mmlu/%s/0.0.0/c30699e8356da336a370243923dbaf21066bb9fe/mmlu-test.arrow"
MMLU_SUBJECTS = (
    "college_biology high_school_biology virology anatomy college_medicine medical_genetics "
    "clinical_knowledge high_school_us_history philosophy high_school_mathematics "
    "computer_security econometrics marketing world_religions"
).split()
DATA = "/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/jincheng/data"
TEXT_SETS = {
    "bio_forget": f"{DATA}/wmdp/bio-forget-corpus/*.parquet",
    "bio_retain": f"{DATA}/wmdp/bio-retain-corpus/*.parquet",
    "fineweb": f"{DATA}/fineweb-edu-5b-shuffle/shard_00013.parquet",
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def read_arrow(path):
    with open(path, "rb") as f:
        return pa.ipc.open_stream(f).read_all().to_pylist()


def mcq_items(tok, limit=None):
    items = []
    rows = read_arrow(WMDP_BIO)
    for r in rows:
        items.append(("wmdp_bio", "biology", r))
    for s in MMLU_SUBJECTS:
        for r in read_arrow(MMLU % s):
            items.append(("mmlu", s.replace("_", " "), r))
    out = []
    for task, subj, r in items:
        ctx = (
            f"The following are multiple choice questions (with answers) about {subj}.\n\n"
            + r["question"].strip()
            + "\n"
            + "".join(f"{c}. {x}\n" for c, x in zip("ABCD", r["choices"]))
            + "Answer:"
        )
        out.append({"task": task, "subject": subj, "ids": tok(ctx).input_ids, "answer": int(r["answer"])})
    if limit:
        by = {}
        for it in out:
            by.setdefault(it["task"], []).append(it)
        out = [it for v in by.values() for it in v[:limit]]
    return out


def cloze_items(tok, limit=None):
    rows = read_arrow(WMDP_BIO)[: limit or None]
    out = []
    for r in rows:
        ctx = tok(f"Question: {r['question'].strip()}\nAnswer:").input_ids
        conts = [tok(" " + str(c).strip(), add_special_tokens=False).input_ids or [tok.eos_token_id] for c in r["choices"]]
        out.append({"ctx": ctx, "conts": conts, "chars": [len(str(c).strip()) + 1 for c in r["choices"]], "answer": int(r["answer"])})
    return out


def text_docs(tok, pattern, n, seq, seed=0):
    files = sorted(glob.glob(pattern))
    rng = random.Random(seed)
    texts = []
    for f in files:
        pf = pq.ParquetFile(f)
        for g in range(pf.num_row_groups):
            texts.extend(pf.read_row_group(g, columns=["text"]).column("text").to_pylist())
            if len(texts) >= 40 * n:
                break
        if len(texts) >= 40 * n:
            break
    rng.shuffle(texts)
    docs = []
    for t in texts:
        ids = tok(t or "", add_special_tokens=False).input_ids
        if len(ids) >= 256:
            docs.append(ids[:seq])
        if len(docs) >= n:
            break
    return docs


class Probe:
    """Finds decoder layers / final norm / lm_head and records hooks."""

    def __init__(self, model):
        self.model = model
        self.lm_head = model.get_output_embeddings()
        base = model.model
        if hasattr(base, "language_model"):
            base = base.language_model
        self.base = base
        self.layers = base.layers
        self.norm = base.norm
        self.pos = None  # (rows, cols) LongTensors for gathering
        self.layer_out = []
        self.route_counts = None
        self.handles = []
        for i, layer in enumerate(self.layers):
            self.handles.append(layer.register_forward_hook(self._layer_hook(i)))
            gate = getattr(getattr(layer, "mlp", None), "gate", None)
            if gate is not None:
                self.handles.append(gate.register_forward_hook(self._router_hook(i)))

    def _layer_hook(self, i):
        def hook(mod, args, out):
            if self.pos is None:
                return
            h = out[0] if isinstance(out, tuple) else out
            self.layer_out.append(h[self.pos[0], self.pos[1]].float())
        return hook

    def _router_hook(self, i):
        def hook(mod, args, out):
            if self.route_counts is None:
                return
            idx = out[2]
            self.route_counts[i] += torch.bincount(idx.reshape(-1), minlength=self.route_counts.shape[1]).to(self.route_counts)
        return hook

    def logits(self, h_normed):
        return self.lm_head(h_normed.to(self.lm_head.weight.dtype)).float()


def pad_batch(seqs, pad_id, device):
    T = max(len(s) for s in seqs)
    x = torch.full((len(seqs), T), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, : len(s)] = torch.tensor(s)
    return x.to(device)


def batches_by_tokens(order_lens, max_tokens, max_bs):
    idx = sorted(range(len(order_lens)), key=lambda i: order_lens[i])
    cur, cur_max = [], 0
    for i in idx:
        m = max(cur_max, order_lens[i])
        if cur and (m * (len(cur) + 1) > max_tokens or len(cur) >= max_bs):
            yield cur
            cur, m = [], order_lens[i]
        cur.append(i)
        cur_max = m
    if cur:
        yield cur


def vocab_stats(z):
    """z: [N, V] float32 logits -> dict of [N] tensors."""
    lse = torch.logsumexp(z, -1)
    p = torch.softmax(z, -1)
    ent = -(p * (z - lse[:, None])).sum(-1)
    return {"lse": lse, "entropy": ent, "max": z.max(-1).values, "std": z.std(-1), "argmax": z.argmax(-1)}


@torch.no_grad()
def run_mcq(probe, items, letter_ids, pad_id, device, max_tokens):
    letters = torch.tensor(letter_ids, device=device)
    n_layers = len(probe.layers)
    rec = []
    lens = [len(it["ids"]) for it in items]
    done = 0
    for b in batches_by_tokens(lens, max_tokens, 64):
        seqs = [items[i]["ids"] for i in b]
        x = pad_batch(seqs, pad_id, device)
        rows = torch.arange(len(b), device=device)
        cols = torch.tensor([len(s) - 1 for s in seqs], device=device)
        probe.pos, probe.layer_out = (rows, cols), []
        out = probe.base(input_ids=x, use_cache=False)
        probe.pos = None
        h_final = out.last_hidden_state[rows, cols]
        z = probe.logits(h_final)
        st = vocab_stats(z)
        lp = torch.log_softmax(z, -1)[:, letters]
        # logit lens: every layer output -> final norm -> lm_head, 4-way only
        lens_pred = []
        lens_mass = []
        for h in probe.layer_out[:n_layers]:
            zl = probe.logits(probe.norm(h.to(out.last_hidden_state.dtype)))
            lpl = torch.log_softmax(zl, -1)[:, letters]
            lens_pred.append(lpl.argmax(-1))
            lens_mass.append(lpl.exp().sum(-1))
        lens_pred = torch.stack(lens_pred, 1).cpu()
        lens_mass = torch.stack(lens_mass, 1).cpu()
        hn = h_final.norm(dim=-1)
        pre = probe.layer_out[n_layers - 1].norm(dim=-1) if len(probe.layer_out) >= n_layers else hn
        for j, i in enumerate(b):
            rec.append((i, {
                "lp4": [round(v, 5) for v in lp[j].tolist()],
                "lse": st["lse"][j].item(), "entropy": st["entropy"][j].item(),
                "max": st["max"][j].item(), "std": st["std"][j].item(), "argmax": int(st["argmax"][j]),
                "hnorm_final": hn[j].item(), "hnorm_pre": pre[j].item(),
                "lens_pred": lens_pred[j].tolist(), "lens_mass": [round(v, 4) for v in lens_mass[j].tolist()],
            }))
        done += len(b)
        if done % 1000 < len(b):
            log(f"  mcq {done}/{len(items)}")
    rec.sort(key=lambda t: t[0])
    return [r for _, r in rec]


@torch.no_grad()
def run_cloze(probe, items, pad_id, device, max_tokens):
    flat = []
    for qi, it in enumerate(items):
        for ci, c in enumerate(it["conts"]):
            flat.append((qi, ci, it["ctx"], c))
    lens = [len(f[2]) + len(f[3]) for f in flat]
    scores = {}
    for b in batches_by_tokens(lens, max_tokens, 64):
        seqs = [flat[i][2] + flat[i][3] for i in b]
        x = pad_batch(seqs, pad_id, device)
        hs = probe.base(input_ids=x, use_cache=False).last_hidden_state
        for j, i in enumerate(b):
            qi, ci, ctx, cont = flat[i]
            pos = torch.arange(len(ctx) - 1, len(ctx) - 1 + len(cont), device=device)
            z = probe.logits(hs[j, pos])
            lp = torch.log_softmax(z, -1).gather(1, torch.tensor(cont, device=device)[:, None]).sum().item()
            scores[(qi, ci)] = lp
    out = []
    for qi, it in enumerate(items):
        s = [scores[(qi, c)] for c in range(len(it["conts"]))]
        out.append({"lp": [round(v, 4) for v in s], "answer": it["answer"],
                    "acc": int(int(np.argmax(s)) == it["answer"]),
                    "acc_norm": int(int(np.argmax([v / n for v, n in zip(s, it["chars"])])) == it["answer"])})
    return out


@torch.no_grad()
def run_nll(probe, docs, device, n_experts):
    n_layers = len(probe.layers)
    probe.route_counts = torch.zeros(n_layers, n_experts, dtype=torch.float64, device=device)
    tot = {"nll": 0.0, "entropy": 0.0, "lse": 0.0, "max": 0.0, "std": 0.0, "top1": 0.0, "n": 0}
    per_doc = []
    for d in docs:
        x = torch.tensor([d], device=device)
        hs = probe.base(input_ids=x, use_cache=False).last_hidden_state[0, :-1]
        tgt = x[0, 1:]
        dn = 0.0
        for s in range(0, hs.shape[0], 256):
            z = probe.logits(hs[s : s + 256])
            st = vocab_stats(z)
            nll = st["lse"] - z.gather(1, tgt[s : s + 256, None])[:, 0]
            tot["nll"] += nll.sum().item()
            dn += nll.sum().item()
            tot["entropy"] += st["entropy"].sum().item()
            tot["lse"] += st["lse"].sum().item()
            tot["max"] += st["max"].sum().item()
            tot["std"] += st["std"].sum().item()
            tot["top1"] += (st["argmax"] == tgt[s : s + 256]).sum().item()
        tot["n"] += hs.shape[0]
        per_doc.append(dn / hs.shape[0])
    counts = probe.route_counts.cpu().numpy()
    probe.route_counts = None
    n = tot.pop("n")
    res = {k: v / n for k, v in tot.items()}
    res["n_tokens"] = n
    res["per_doc_nll"] = [round(v, 4) for v in per_doc]
    res["route_counts"] = counts.astype(np.int64).tolist()
    return res


def summarize_mcq(items, recs):
    out = {}
    for task in ("wmdp_bio", "mmlu"):
        idx = [i for i, it in enumerate(items) if it["task"] == task]
        if not idx:
            continue
        lp = np.array([recs[i]["lp4"] for i in idx])
        ans = np.array([items[i]["answer"] for i in idx])
        n4 = lp - np.logaddexp.reduce(lp, 1, keepdims=True)
        corr = lp[np.arange(len(ans)), ans]
        other = np.where(np.eye(4, dtype=bool)[ans], -np.inf, lp).max(1)
        lens = np.array([recs[i]["lens_pred"] for i in idx])
        lmass = np.array([recs[i]["lens_mass"] for i in idx])
        out[task] = {
            "n": len(idx),
            "acc": float((lp.argmax(1) == ans).mean()),
            "choice_mass": float(np.exp(lp).sum(1).mean()),
            "p4_correct": float(np.exp(n4[np.arange(len(ans)), ans]).mean()),
            "entropy4": float((-(np.exp(n4) * n4).sum(1)).mean()),
            "margin": float((corr - other).mean()),
            "choice_logit_spread": float(lp.std(1).mean()),
            "vocab_entropy": float(np.mean([recs[i]["entropy"] for i in idx])),
            "vocab_lse": float(np.mean([recs[i]["lse"] for i in idx])),
            "vocab_max": float(np.mean([recs[i]["max"] for i in idx])),
            "vocab_std": float(np.mean([recs[i]["std"] for i in idx])),
            "hnorm_final": float(np.mean([recs[i]["hnorm_final"] for i in idx])),
            "hnorm_pre": float(np.mean([recs[i]["hnorm_pre"] for i in idx])),
            "top1_is_letter": float(np.mean([recs[i]["argmax"] in LETTER_IDS for i in idx])),
            "lens_acc": (lens == ans[:, None]).mean(0).round(4).tolist(),
            "lens_mass": lmass.mean(0).round(4).tolist(),
        }
    return out


LETTER_IDS = []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--path", required=True)
    ap.add_argument("--base", default="", help="base ckpt dir for weight diff (empty = skip)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-docs", type=int, default=48)
    ap.add_argument("--doc-len", type=int, default=1024)
    ap.add_argument("--max-tokens", type=int, default=24576)
    ap.add_argument("--limit", type=int, default=0, help="debug: items per task")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip", default="", help="comma list of phases to skip: mcq,cloze,nll,wdiff")
    args = ap.parse_args()
    skip = set(filter(None, args.skip.split(",")))

    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{args.name}.json")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.path, trust_remote_code=True)
    letter_ids = []
    for c in "ABCD":
        ids = tok(" " + c, add_special_tokens=False).input_ids
        assert len(ids) == 1, (c, ids)
        letter_ids.append(ids[0])
    LETTER_IDS[:] = letter_ids
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    result = {"name": args.name, "path": args.path, "letter_ids": letter_ids}
    need_model = not {"mcq", "cloze", "nll"} <= skip
    if need_model:
        log(f"loading {args.path}")
        dtype = torch.float32 if args.device == "cpu" else torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(
            args.path, dtype=dtype, trust_remote_code=True, device_map=args.device
        ).eval()
        log(f"loaded {type(model).__name__} in {time.time() - t0:.0f}s")
        probe = Probe(model)
        n_experts = getattr(getattr(probe.layers[0].mlp, "gate", None), "num_experts", 1)
        dev = probe.lm_head.weight.device
        result["lm_head_row_norm_mean"] = probe.lm_head.weight.float().norm(dim=1).mean().item()
        result["lm_head_letter_row_norm"] = probe.lm_head.weight[letter_ids].float().norm(dim=1).tolist()
        result["final_norm_gain_mean"] = (1 + probe.norm.weight.float()).mean().item() if "Qwen3_5" in type(probe.norm).__name__ else probe.norm.weight.float().mean().item()

        if "mcq" not in skip:
            items = mcq_items(tok, args.limit)
            log(f"mcq: {len(items)} items")
            recs = run_mcq(probe, items, letter_ids, pad_id, dev, args.max_tokens)
            result["mcq"] = summarize_mcq(items, recs)
            result["mcq_items"] = [{"task": it["task"], "answer": it["answer"], **r} for it, r in zip(items, recs)]
            for k, v in result["mcq"].items():
                log(f"  {k}: acc={v['acc']:.4f} mass={v['choice_mass']:.3f} H={v['vocab_entropy']:.2f} lse={v['vocab_lse']:.2f}")
        if "cloze" not in skip:
            citems = cloze_items(tok, args.limit)
            log(f"cloze: {len(citems)} items")
            crec = run_cloze(probe, citems, pad_id, dev, args.max_tokens)
            result["cloze"] = {"acc": float(np.mean([r["acc"] for r in crec])),
                               "acc_norm": float(np.mean([r["acc_norm"] for r in crec])), "items": crec}
            log(f"  cloze acc={result['cloze']['acc']:.4f} acc_norm={result['cloze']['acc_norm']:.4f}")
        if "nll" not in skip:
            result["nll"] = {}
            for k, pat in TEXT_SETS.items():
                docs = text_docs(tok, pat, args.n_docs, args.doc_len)
                result["nll"][k] = run_nll(probe, docs, dev, n_experts)
                r = result["nll"][k]
                log(f"  nll[{k}] n={r['n_tokens']} nll={r['nll']:.3f} H={r['entropy']:.3f} lse={r['lse']:.2f} top1={r['top1']:.3f}")
        for h in probe.handles:
            h.remove()
        del probe, model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        result["elapsed_s"] = time.time() - t0
        with open(out_path + ".tmp", "w") as f:
            json.dump(result, f)
        os.replace(out_path + ".tmp", out_path)
        log(f"wrote {out_path}")

    if args.base and "wdiff" not in skip:
        from weight_diff import weight_diff

        wd = weight_diff(args.base, args.path, device=args.device, tok=tok, letter_ids=letter_ids)
        wpath = os.path.join(args.out_dir, f"{args.name}.wdiff.json")
        with open(wpath + ".tmp", "w") as f:
            json.dump(wd, f)
        os.replace(wpath + ".tmp", wpath)
        log(f"wrote {wpath} ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
