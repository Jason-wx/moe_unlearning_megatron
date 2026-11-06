"""Summarize diag results: python summarize.py <results_dir> -> summary.md"""
import glob
import json
import os
import sys

R = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
L = []
models = {}
for f in sorted(glob.glob(f"{R}/*.json")):
    if f.endswith(".wdiff.json"):
        continue
    d = json.load(open(f))
    models[d["name"]] = d

L.append("## MCQ (Answer: position)\n")
L.append("| model | task | acc | A-D mass | p4(correct) | margin | vocab H | lse | max | hnorm pre/final | top1∈ABCD | cloze acc/norm |")
L.append("|" + "---|" * 12)
for n, d in models.items():
    for t, v in d.get("mcq", {}).items():
        c = d.get("cloze", {}) if t == "wmdp_bio" else {}
        L.append(f"| {n} | {t} | {v['acc']:.4f} | {v['choice_mass']:.3f} | {v['p4_correct']:.3f} | {v['margin']:.2f} | "
                 f"{v['vocab_entropy']:.2f} | {v['vocab_lse']:.2f} | {v['vocab_max']:.2f} | {v['hnorm_pre']:.1f}/{v['hnorm_final']:.1f} | "
                 f"{v['top1_is_letter']:.2f} | {c.get('acc', float('nan')):.4f}/{c.get('acc_norm', float('nan')):.4f} |")

L.append("\n## logit lens: WMDP-bio 4-way acc by layer (every 4th)\n")
for n, d in models.items():
    v = d.get("mcq", {}).get("wmdp_bio")
    if v:
        L.append(f"- {n}: " + " ".join(f"L{i}:{a:.2f}" for i, a in enumerate(v["lens_acc"]) if i % 4 == 3 or i == len(v["lens_acc"]) - 1))

L.append("\n## text NLL / entropy / lse / top1\n")
L.append("| model | set | nll | H | lse | max | top1 |")
L.append("|---|---|---|---|---|---|---|")
for n, d in models.items():
    for k, v in d.get("nll", {}).items():
        L.append(f"| {n} | {k} | {v['nll']:.3f} | {v['entropy']:.3f} | {v['lse']:.2f} | {v['max']:.2f} | {v['top1']:.3f} |")

L.append("\n## weight diff\n")
for f in sorted(glob.glob(f"{R}/*.wdiff.json")):
    w = json.load(open(f))
    n = os.path.basename(f)[: -len(".wdiff.json")]
    share = ", ".join(f"{k}:{v:.3f}" for k, v in sorted(w["d2_share_by_kind"].items(), key=lambda x: -x[1]))
    L.append(f"### {n}  |dW|={w['total_d_norm']:.3f}\n- d² share: {share}")
    t = w["tensors"]
    for key in ("lm_head.weight", "model.language_model.norm.weight", "model.language_model.embed_tokens.weight"):
        if key in t:
            v = t[key]
            ex = f" common_shift={v['common_shift_share']:.3f} letters_rel={[round(r['rel'], 5) for r in v['letter_rows']]}" if "letter_rows" in v else ""
            ex += f" gainΔ={v['gain_delta_mean']:.5f}" if "gain_delta_mean" in v else ""
            L.append(f"- {key}: rel={v['rel']:.5f} changed={v['frac_changed']:.3f} cos={v['cos_w_d']:.3f}{ex}")
    by_layer = {}
    for g in w["groups"]:
        by_layer.setdefault(g["layer"], {})[g["kind"]] = g["rel"]
    kinds = ["experts", "shared_expert", "router", "linear_attn", "self_attn", "layer_norm"]
    L.append("\n| layer | " + " | ".join(kinds) + " |")
    L.append("|---|" + "---|" * len(kinds))
    for l in sorted(k for k in by_layer if k >= 0):
        L.append(f"| {l} | " + " | ".join(f"{by_layer[l].get(k, float('nan')):.2e}" for k in kinds) + " |")
    L.append("")

open(f"{R}/summary.md", "w").write("\n".join(L) + "\n")
print(f"wrote {R}/summary.md")
