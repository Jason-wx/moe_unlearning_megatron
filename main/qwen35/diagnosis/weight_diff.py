"""Streaming per-tensor weight diff: unlearned ckpt vs base (HF safetensors).

Only the language-model tensors present in the ckpt are compared (the base
also carries mtp.* and the ckpt exports visual.* unchanged). Per tensor:
  w_norm, d_norm, rel = |dW|/|W|, frac_changed (bf16 elements that differ),
  cos(W, dW) (negative = shrinking, positive = growing).
Extras:
  experts  (3D [E, ...])  -> per-expert rel list
  lm_head / embed         -> per-row |dW| stats, top-changed tokens, the
                             " A".." D" rows, and the share of |dW|^2 that is
                             a vocab-common shift (softmax-invariant for lm_head)
  norms                   -> mean change of the effective gain
Plus a grouped summary by (layer, module kind).
"""
from __future__ import annotations

import json
import os
import re
import time
from collections import defaultdict

import torch
from safetensors import safe_open

KIND_RULES = [
    (r"mlp\.experts\.", "experts"),
    (r"mlp\.shared_expert_gate", "shared_gate"),
    (r"mlp\.shared_expert\.", "shared_expert"),
    (r"mlp\.gate\.", "router"),
    (r"linear_attn\.", "linear_attn"),
    (r"self_attn\.", "self_attn"),
    (r"layernorm", "layer_norm"),
    (r"embed_tokens", "embed"),
    (r"^lm_head", "lm_head"),
    (r"^model\.(language_model\.)?norm\.", "final_norm"),
]


def kind_of(key):
    for pat, k in KIND_RULES:
        if re.search(pat, key):
            return k
    return "other"


def layer_of(key):
    m = re.search(r"\.layers\.(\d+)\.", key)
    return int(m.group(1)) if m else -1


class ShardReader:
    def __init__(self, root):
        self.root = root
        with open(os.path.join(root, "model.safetensors.index.json")) as f:
            self.map = json.load(f)["weight_map"]
        self.handles = {}

    def get(self, key, device):
        fn = self.map[key]
        if fn not in self.handles:
            self.handles[fn] = safe_open(os.path.join(self.root, fn), framework="pt", device="cpu")
        return self.handles[fn].get_tensor(key).to(device)


def _rowstats(W, D, tok, letter_ids, topk=20):
    Wf, Df = W.float(), D.float()
    rn = Df.norm(dim=1)
    wn = Wf.norm(dim=1)
    mean_row = Df.mean(0)
    common = (mean_row.norm() ** 2 * Df.shape[0]).item()
    tot = (Df.norm() ** 2).item()
    top = torch.topk(rn, topk)
    out = {
        "row_d_mean": rn.mean().item(),
        "row_d_median": rn.median().item(),
        "row_rel_mean": (rn / wn.clamp_min(1e-8)).mean().item(),
        "common_shift_share": common / tot if tot > 0 else 0.0,
        "top_rows": [
            {"id": int(i), "tok": tok.convert_ids_to_tokens(int(i)) if tok is not None else "", "d": float(v), "rel": float(v / wn[i].clamp_min(1e-8))}
            for v, i in zip(top.values.tolist(), top.indices.tolist())
        ],
        "letter_rows": [
            {"id": int(i), "d": rn[i].item(), "rel": (rn[i] / wn[i]).item(), "w_norm": wn[i].item(),
             "w_norm_rank": int((wn > wn[i]).sum().item())}
            for i in letter_ids
        ],
    }
    return out


@torch.no_grad()
def weight_diff(base_dir, ckpt_dir, device="cuda", tok=None, letter_ids=(), key_regex=None):
    t0 = time.time()
    base, ck = ShardReader(base_dir), ShardReader(ckpt_dir)
    keys = [k for k in ck.map if not k.startswith(("model.visual", "mtp"))]
    missing = [k for k in keys if k not in base.map]
    keys = [k for k in keys if k in base.map]
    if key_regex:
        keys = [k for k in keys if re.search(key_regex, k)]
    per = {}
    groups = defaultdict(lambda: [0.0, 0.0, 0])  # (layer, kind) -> [d2, w2, n_tensors]
    for n, key in enumerate(sorted(keys)):
        W = base.get(key, device)
        C = ck.get(key, device)
        if W.shape != C.shape:
            per[key] = {"shape_mismatch": [list(W.shape), list(C.shape)]}
            continue
        kind = kind_of(key)
        Wf = W.float()
        D = C.float() - Wf
        w2 = (Wf.norm() ** 2).item()
        d2 = (D.norm() ** 2).item()
        rec = {
            "shape": list(W.shape),
            "kind": kind,
            "layer": layer_of(key),
            "w_norm": w2 ** 0.5,
            "d_norm": d2 ** 0.5,
            "rel": (d2 / w2) ** 0.5 if w2 > 0 else float("nan"),
            "frac_changed": (C != W).float().mean().item(),
            "cos_w_d": (torch.sum(Wf * D) / (w2 ** 0.5 * d2 ** 0.5)).item() if d2 > 0 and w2 > 0 else 0.0,
        }
        if kind == "experts" and W.dim() == 3:
            ed = D.flatten(1).norm(dim=1)
            ew = Wf.flatten(1).norm(dim=1)
            rec["expert_rel"] = (ed / ew.clamp_min(1e-8)).tolist()
        if kind in ("lm_head", "embed") and W.dim() == 2:
            rec.update(_rowstats(W, D, tok, letter_ids))
        if kind in ("final_norm", "layer_norm"):
            # Qwen3.5 RMSNorm gain = 1 + w
            rec["gain_base_mean"] = (1 + Wf).mean().item()
            rec["gain_delta_mean"] = D.mean().item()
            rec["gain_delta_abs_mean"] = D.abs().mean().item()
        per[key] = rec
        g = groups[(rec["layer"], kind)]
        g[0] += d2
        g[1] += w2
        g[2] += 1
        del W, C, Wf, D
        if n % 100 == 0:
            print(f"[wdiff] {n}/{len(keys)} {time.time() - t0:.0f}s", flush=True)
    summary = [
        {"layer": l, "kind": k, "d_norm": v[0] ** 0.5, "w_norm": v[1] ** 0.5, "rel": (v[0] / v[1]) ** 0.5 if v[1] else float("nan"), "n": v[2]}
        for (l, k), v in sorted(groups.items())
    ]
    tot_d2 = sum(v[0] for v in groups.values())
    by_kind = defaultdict(float)
    for (l, k), v in groups.items():
        by_kind[k] += v[0]
    return {
        "base": base_dir,
        "ckpt": ckpt_dir,
        "missing_in_base": missing,
        "total_d_norm": tot_d2 ** 0.5,
        "d2_share_by_kind": {k: v / tot_d2 for k, v in by_kind.items()} if tot_d2 else {},
        "groups": summary,
        "tensors": per,
        "elapsed_s": time.time() - t0,
    }
