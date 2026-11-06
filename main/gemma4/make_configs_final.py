#!/usr/bin/env python
"""Generate the gemma4 FINAL-line configs (main/gemma4/configs-final/).

Nine algorithms on the NOCAP shadow model + BOS-anchored caches, each saving
the final checkpoint. Deterministic string transforms over per-algorithm
base configs; run from the repo root:

    envs/open-unlearning/bin/python main/gemma4/make_configs_final.py

Bases (most-recently-valid predecessor per algorithm):
  GA family  -> main/gemma4/bos-ablation/*_nocap.yaml   (already nocap+BOS)
  RMU        -> main/gemma4/rmu.yaml                    (best arm, saved ckpt)
  PDU        -> main/gemma4/pdu_eps_sweep/pdu_eps2p50.yaml (eps -> 2.60)
  PureNPO/SimNPO -> main/gemma4/{pure_npo,simnpo}.yaml  (lr 5e-5, the only
                effective bracket on gemma4; 1e-5/1e-6 arms were dead)
"""

from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
LINE = Path(__file__).resolve().parent / "configs-final"
NOCAP = "/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/wangxiao1/checkpoints/gemma/gemma-26b-a4b-nocap"
CAPPED = "/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/jincheng/sgtm/checkpoints/gemma/gemma-26b-a4b"
FORGET_OLD = "gemma4_wmdp_bio_softpack4096/forget"
FORGET_BOS = "gemma4_wmdp_bio_softpack4096_bos/forget"
RETAIN_OLD = "fineweb_edu_5b_gemma4_softpack4096_seed42"
RETAIN_BOS = "fineweb_edu_5b_gemma4_softpack4096_bos_seed42"
EVAL_DIR = "/mnt/shared-storage-user/safethm-share/wangxiao1/Safe_MoE/open-unlearning/main/gemma4/configs-final/online-evaluation"
OUTPUT_ROOT = "/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/wangxiao1/checkpoints/unlearn/megatron/main"

PREAMBLE = """# Gemma4 FINAL line — nocap shadow model + BOS-anchored caches, final ckpt
# saved. Regime proven by the 10-03 BOS ablation (GD 0.79->0.273/0.725 with
# MMLU -4.2pt while every no-BOS run froze at 0.78-0.80); all pre-BOS
# gemma4-bio results are superseded. Online-eval records ->
# main/gemma4/configs-final/online-evaluation/; weights ->
# {OUTPUT_ROOT}/<output_name>/checkpoint-step-2133 (~66G each).
"""

SPECS = {
    "grad_ascent.yaml": {
        "base": "main/gemma4/bos-ablation/grad_ascent_nocap.yaml",
        "rename": ("gemma4-bos-grad-ascent-nocap", "gemma4-final-grad-ascent-nocap"),
        "eval_old": "main/gemma4/bos-ablation/online-evaluation",
        "note": "# Base: bos-ablation/grad_ascent_nocap.yaml (identical plan; only the\n"
                "# rename + final-ckpt saving differ). bos-ablation outcome: pure ascent\n"
                "# collapses fully (0.247/0.248 at step 300) — kept for the record.\n",
        "no_save_block": True,
    },
    "grad_diff.yaml": {
        "base": "main/gemma4/bos-ablation/grad_diff_nocap.yaml",
        "rename": ("gemma4-bos-grad-diff-nocap", "gemma4-final-grad-diff-nocap"),
        "eval_old": "main/gemma4/bos-ablation/online-evaluation",
        "note": "# Base: bos-ablation/grad_diff_nocap.yaml (identical plan; only the\n"
                "# rename + final-ckpt saving differ). bos-ablation outcome: healthy —\n"
                "# monotone 0.79 -> 0.273/0.725, margin 0.452 at the final step.\n",
        "no_save_block": True,
    },
    "ceu.yaml": {
        "base": "main/gemma4/bos-ablation/ceu_nocap.yaml",
        "rename": ("gemma4-bos-ceu-nocap", "gemma4-final-ceu-nocap"),
        "eval_old": "main/gemma4/bos-ablation/online-evaluation",
        "note": "# Base: bos-ablation/ceu_nocap.yaml (identical plan; only the rename +\n"
                "# final-ckpt saving differ). FIRST run on the fixed fp32 leave-gold-out\n"
                "# normalization (objectives.py, 10-03): every earlier CEU run sat on the\n"
                "# (1-p_gold).clamp_min(1e-12) bug and froze at ~0.78 even with BOS.\n",
        "no_save_block": True,
    },
    "satimp_forgetonly.yaml": {
        "base": "main/gemma4/bos-ablation/satimp_forgetonly_nocap.yaml",
        "rename": ("gemma4-bos-satimp-forgetonly-nocap", "gemma4-final-satimp-forgetonly-nocap"),
        "eval_old": "main/gemma4/bos-ablation/online-evaluation",
        "note": "# Base: bos-ablation/satimp_forgetonly_nocap.yaml (identical plan; only the\n"
                "# rename + final-ckpt saving differ). bos-ablation outcome: overshoot at\n"
                "# step 300 then rebound with MMLU erosion — expect the same shape.\n",
        "no_save_block": True,
    },
    "WGA_forgetonly.yaml": {
        "base": "main/gemma4/bos-ablation/WGA_forgetonly_nocap.yaml",
        "rename": ("gemma4-bos-wga-forgetonly-nocap", "gemma4-final-wga-forgetonly-nocap"),
        "eval_old": "main/gemma4/bos-ablation/online-evaluation",
        "note": "# Base: bos-ablation/WGA_forgetonly_nocap.yaml (identical plan; only the\n"
                "# rename + final-ckpt saving differ). bos-ablation outcome: overshoot at\n"
                "# step 300 then rebound with MMLU erosion — expect the same shape.\n",
        "no_save_block": True,
    },
    "rmu.yaml": {
        "base": "main/gemma4/rmu.yaml",
        "rename": ("gemma4-main-rmu", "gemma4-final-rmu-nocap"),
        "eval_old": "main/gemma4/online-evaluation",
        "note": "# Base: main/gemma4/rmu.yaml — coeff-alpha sweep best arm (steer 1200 /\n"
                "# alpha 4.45 / target 9 / layers 7-9, lr 1e-6, activation_recompute\n"
                "# null is REQUIRED for RMU). RMU transferred even without BOS\n"
                "# (0.346/0.726 online); the BOS cache unifies the regime. The old ckpt\n"
                "# was deleted in the 10-03 disk cleanup — this re-run is the artifact.\n",
    },
    "pdu.yaml": {
        "base": "main/gemma4/pdu_eps_sweep/pdu_eps2p50.yaml",
        "rename": ("gemma4-pdu-eps2p50", "gemma4-final-pdu-nocap"),
        "eval_old": "main/gemma4/pdu_eps_sweep/online-evaluation",
        "note": "# Base: pdu_eps_sweep/pdu_eps2p50.yaml with eps 2.50 -> 2.60 (user-set\n"
                "# absolute value, no sweep; 2.60 was the gemma4-cyber R1 best arm).\n"
                "# Anchor caveat: retain NLL 2.0945 was measured on the NO-BOS cache;\n"
                "# the BOS-cache anchor is unmeasured — read this arm as absolute-eps,\n"
                "# not as anchor+offset. mbs 1; the int32 split-forward guard trips\n"
                "# automatically from config.json (vocab 262144).\n",
        "eps": ("retain_loss_eps: 2.50", "retain_loss_eps: 2.60"),
        "output_root_old": "/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/wangxiao1/checkpoints/unlearn/megatron/gemma4/pdu_eps_sweep",
    },
    "pure_npo.yaml": {
        "base": "main/gemma4/pure_npo.yaml",
        "rename": ("gemma4-main-pure-npo", "gemma4-final-pure-npo-nocap"),
        "eval_old": "main/gemma4/online-evaluation",
        "note": "# Base: main/gemma4/pure_npo.yaml — lr 5e-5 is the ONLY effective\n"
                "# bracket on gemma4 (1e-5/1e-6 arms dead, 09-30). Caveat: the\n"
                "# qwen35/gemma4 pure_npo carries an alpha=1.0 retain-CE anchor\n"
                "# (intern main is forget-only) — cross-model comparisons discounted.\n",
    },
    "simnpo.yaml": {
        "base": "main/gemma4/simnpo.yaml",
        "rename": ("gemma4-main-simnpo", "gemma4-final-simnpo-nocap"),
        "eval_old": "main/gemma4/online-evaluation",
        "note": "# Base: main/gemma4/simnpo.yaml — lr 5e-5 (only effective bracket on\n"
                "# gemma4, 09-30); SimNPO semantics identical across the three models.\n",
    },
}


def transform(spec: dict) -> str:
    text = (REPO / spec["base"]).read_text()
    # Drop the base's leading comment block; a fresh header is prepended.
    text = text[text.index("name:") :]
    old_name, new_name = spec["rename"]
    assert text.count(old_name) == 2, (spec["base"], text.count(old_name))
    text = text.replace(old_name, new_name)
    text = text.replace(CAPPED, NOCAP)
    text = text.replace(FORGET_OLD, FORGET_BOS)
    text = text.replace(RETAIN_OLD, RETAIN_BOS)
    # CEU (and other forget-only algorithms) carry retain_cache: null.
    has_retain = "retain_cache: null" not in text
    assert text.count(FORGET_BOS) == 1, spec["base"]
    assert text.count(RETAIN_BOS) == (1 if has_retain else 0), spec["base"]
    assert spec["eval_old"] in text
    text = text.replace(spec["eval_old"], "main/gemma4/configs-final/online-evaluation")
    if spec.get("no_save_block"):
        block = (
            "  # Diagnostic A/B: the online-eval curve is the readout; no checkpoint\n"
            "  # artifacts (saves ~66G per arm).\n"
            "  save_final_checkpoint: false"
        )
        assert block in text, spec["base"]
        text = text.replace(block, "  save_final_checkpoint: true")
    if "eps" in spec:
        old, new = spec["eps"]
        assert text.count(old) == 1
        text = text.replace(old, new)
    if "output_root_old" in spec:
        assert text.count(spec["output_root_old"]) == 1
        text = text.replace(spec["output_root_old"], OUTPUT_ROOT)
    header = PREAMBLE.format(OUTPUT_ROOT=OUTPUT_ROOT) + spec["note"]
    return header + text


def flatten(payload, prefix=""):
    out = {}
    for key, value in payload.items():
        tag = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            out.update(flatten(value, tag))
        else:
            out[tag] = value
    return out


def main() -> None:
    LINE.mkdir(exist_ok=True)
    for filename, spec in SPECS.items():
        text = transform(spec)
        target = LINE / filename
        assert not target.exists() or True
        target.write_text(text)
        payload = yaml.safe_load(text)
        flat = flatten(payload)
        expected_name = spec["rename"][1]
        checks = {
            "name": flat["name"] == expected_name,
            "output_name": flat["training.output_name"] == expected_name,
            "nocap": flat["model.path"] == NOCAP,
            "forget_bos": flat["data.forget_cache"].endswith(FORGET_BOS),
            "retain_bos": flat["data.retain_cache"] is None
            or flat["data.retain_cache"].endswith(RETAIN_BOS),
            "eval_dir": flat["backends.megatron.evaluation.output_dir"] == EVAL_DIR,
            "save_final": flat["training.save_final_checkpoint"] is True,
            "save_mid_off": flat["training.save_steps"] == []
            and flat["training.save_interval_steps"] == 0
            and flat["training.save_every_epoch"] is False,
            "steps_2133": flat["backends.megatron.evaluation.steps"][-1] == 2133,
        }
        base_flat = flatten(yaml.safe_load((REPO / spec["base"]).read_text()))
        diffs = sorted(
            [k for k in flat if flat[k] != base_flat.get(k)]
            + [k for k in base_flat if k not in flat]
        )
        status = "OK" if all(checks.values()) else f"FAIL {checks}"
        print(f"{filename:28s} {status:6s} vs-base diffs={len(diffs)}: {diffs}")
    print(f"\nwrote {len(SPECS)} configs -> {LINE}")


if __name__ == "__main__":
    main()
