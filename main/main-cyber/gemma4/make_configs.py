#!/usr/bin/env python3
"""Generate the fourteen gemma4 main-cyber G1 configs (nine algorithms;
PDU split into six absolute-eps arms) from the qwen35-cyber R2 yamls.
Output goes to configs-r1/ next to this script (the G1/R1 pure-curve
configs live in their own subdir; later save-point rounds get their own
configs-r2/ etc.).

One-shot migration tool (run once; regenerate only if the qwen35-cyber
sources change). Per-file transforms, with occurrence-count assertions so
a drifted source fails loudly instead of silently copying wrong fields:

  - model block      : qwen35 -> gemma4 (path = gemma-26b-a4b-nocap,
                       num_experts 256 -> 128, pad_token_id 248044 -> 0;
                       the nocap + BOS-caches combo is the one whose
                       transfer the BOS ablation confirmed, see
                       main/gemma4/GA_family_nocap_rootcause_20261002.md)
  - data caches      : qwen35 cyber forget -> gemma4_wmdp_cyber_softpack4096_bos,
                       retain -> fineweb_edu_5b_gemma4_softpack4096_bos_seed42
  - names            : qwen35-main-cyber-* -> gemma4-main-cyber-*
  - online-eval dir  : main-cyber/qwen35/online-evaluation ->
                       main-cyber/gemma4/online-evaluation
  - RMU              : algorithm section asserted BYTE-IDENTICAL to the
                       gemma4 coeff-alpha sweep best arm
                       (configs/training/megatron/gemma4/coeff-alpha_sweep/
                       rmu_s1200_rf001_a4.45_softpack4096_l7-9_lr1e-6_3ep.yaml):
                       steer 1200 / alpha 4.45 / target 9 / layers 7-9 -
                       R:F is not portable across models, so the section
                       is aligned, not just the two numbers
  - PDU              : six absolute-eps arms 2.0 / 2.1 / 2.2 / 2.4 /
                       2.6 / 3.0 (2026-10-03 grid decision; mirrors the
                       softcap bio sweep grid with 2.25->2.4 / 2.5->2.6).
                       The nocap x BOS anchor is NOT measured - measuring
                       it in parallel is recommended for reading the
                       curves (arm-to-anchor offsets), not a blocker
  - save plan        : G1 = qwen R1 protocol: save_steps [] and
                       save_final_checkpoint false everywhere (the
                       qwen35 sources carry their R2 single save points;
                       those are dropped, NOT copied)

Everything else is copied byte-identical from the qwen35-cyber source:
algorithm sections, lr schedule (5e-6 unified / 5e-5 NPO family, min_lr
1e-6, 10 epochs), mbs per algorithm, online-eval batch sizes and the
[23..230] per-epoch eval steps (the gemma4 cyber BOS forget cache packs
to 748 packs vs qwen's 743; drop_last gives floor(748/32) = 23 steps per
epoch either way, so the 230-step schedule carries over exactly).
"""

from __future__ import annotations

import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE.parent / "qwen35"
RMU_REF = (
    HERE.parents[2]
    / "configs/training/megatron/gemma4/coeff-alpha_sweep"
    / "rmu_s1200_rf001_a4.45_softpack4096_l7-9_lr1e-6_3ep.yaml"
)

QWEN_PATH = "/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/jincheng/sgtm/checkpoints/qwen/qwen3p5-35b-a3b"
GEMMA_PATH = "/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/wangxiao1/checkpoints/gemma/gemma-26b-a4b-nocap"

FORGET_OLD = "qwen35_wmdp_cyber_softpack4096/forget"
FORGET_NEW = "gemma4_wmdp_cyber_softpack4096_bos/forget"
RETAIN_OLD = "fineweb_edu_5b_qwen35_softpack4096_seed42"
RETAIN_NEW = "fineweb_edu_5b_gemma4_softpack4096_bos_seed42"
EVALDIR_OLD = "main/main-cyber/qwen35/online-evaluation"
EVALDIR_NEW = "main/main-cyber/gemma4/online-evaluation"

COMMON_NOTE = """\
# Schedule / mbs / eval batch_size copied unchanged from the qwen35-cyber
# config: the gemma4 cyber BOS forget cache packs to 748 packs (vs
# intern/qwen 743), and with drop_last floor(748/32) = 23 steps per
# epoch, so 10 epochs = 230 steps and the per-epoch eval [23..230]
# carries over 1:1. Eval bs keeps the qwen35-cyber tiering (32; 16 for
# the full-logit trio gd/pure_npo/simnpo): gemma4's A4B static footprint
# on cyber-length prompts is unmeasured - on first-eval CUDA OOM halve
# it (32 -> 16, 16 -> 8). Output names / online-eval dir under
# main-cyber/gemma4; checkpoint output_root is shared with the intern/
# qwen cyber lines (runs are name-distinct; G1 writes nothing there).\
"""

G1_NOTE = """\
# G1 round: NO checkpoints anywhere (save_steps [], save_final_checkpoint
# false) - the online-eval curve is the only readout; save points, if
# any, get picked from the curves in a later round (the qwen35-cyber
# R1 -> R2 protocol).\
"""

GA_FAMILY_NOTE = """\
# This family is the BOS ablation's positive control: gemma4 GD x BOS
# x lr 1e-6 moved bio 0.79 -> 0.578 by step 300 where every no-BOS arm
# stayed pinned at ~0.79 (main/gemma4/GA_family_nocap_rootcause_20261002.md
# + 10-03 confirmation) - this run is that combo's first test on cyber.
# lr 5e-6 is the cyber-line policy, 5x the ablation arm: watch the first
# three eval points for MMLU collapse.\
"""

PDU_ARM_NOTE = """\
# One of SIX PDU eps arms on this line (absolute grid 2.0 / 2.1 / 2.2 /
# 2.4 / 2.6 / 3.0, decided 2026-10-03; mirrors the softcap bio sweep
# grid with 2.25 -> 2.4 and 2.5 -> 2.6). The nocap x BOS anchor is NOT
# measured: run tools.measure_retain_nll on gemma-26b-a4b-nocap over
# fineweb_edu_5b_gemma4_softpack4096_bos_seed42 alongside this round
# (rjob pattern: main/gemma4/pdu_eps_sweep/rjob_anchor_safethm.sh,
# 4 GPU / ~20-40min) - not a submission blocker, but the curves get read
# against arm-to-anchor offsets; if the anchor lands well above 3.0 the
# whole grid sits on the tight side and a second round re-centers on it.
# Fixed knobs gamma/alpha/dual_step_size = 1.0/1.0/0.1 (V4-validated)
# unchanged. mbs 1 is mandatory on gemma4 (paired full-logits hit the
# 2^31 int32 cap; the split-forward guard auto-engages from config.json
# vocab - GPU-verified, no runner env needed)."""

# (file suffix, eps value, nominal offset vs the VOID softcap-no-BOS
# anchor 2.0945 - kept as a grid-spacing reference only, NOT a live
# anchor: every prior anchor is invalid for nocap x BOS).
PDU_EPS_ARMS: list[tuple[str, str, str]] = [
    ("2p00", "2.0", "-0.09"),
    ("2p10", "2.1", "+0.01"),
    ("2p20", "2.2", "+0.11"),
    ("2p40", "2.4", "+0.31"),
    ("2p60", "2.6", "+0.51"),
    ("3p00", "3.0", "+0.91"),
]

ALGOS: dict[str, dict] = {
    "rmu": {
        "file": "rmu.yaml",
        "src": "rmu.yaml",
        "tag_old": "rmu",
        "tag_new": "rmu",
        "note": """\
# The algorithm section is BYTE-IDENTICAL to the gemma4 coeff-alpha
# sweep best arm (steer 1200 / alpha 4.45 / R:F = 0.001 / target 9 /
# layers 7-9; enforced by an assertion in make_configs.py) - the
# intern/qwen alphas are NOT portable across models. Caveat: that arm
# was calibrated on the SOFTCAP model over no-BOS bio caches; nocap
# shifts the layer-9 activation scale somewhat. mbs 4 /
# activation_recompute null are the gemma4-verified RMU profile (the
# hook runs under no-grad; config validation rejects full recompute
# for RMU).""",
    },
    "grad_diff": {
        "file": "grad_diff.yaml",
        "src": "grad_diff.yaml",
        "tag_old": "grad-diff",
        "tag_new": "grad-diff",
        "note": """\
# Algorithm section copied unchanged (identical across the intern/qwen
# bio main lines; mbs 1 / eval bs 16 tier).
""" + GA_FAMILY_NOTE,
    },
    "grad_ascent": {
        "file": "grad_ascent.yaml",
        "src": "grad_ascent.yaml",
        "tag_old": "grad-ascent",
        "tag_new": "grad-ascent",
        "note": """\
# Algorithm section copied unchanged (pure ascent, no knobs; mbs 4).
# GA collapsed MMLU on both the intern and qwen cyber lines; expect the
# fastest blow-up of the family at lr 5e-6.
""" + GA_FAMILY_NOTE,
    },
    "pure_npo": {
        "file": "pure_npo.yaml",
        "src": "pure_npo.yaml",
        "tag_old": "pure-npo",
        "tag_new": "pure-npo",
        "note": """\
# Algorithm section and lr 5e-5 copied unchanged (mbs 1 / eval bs 16
# tier). The one gemma4 recipe that transferred forgetting even on
# NO-BOS caches (online min 0.521 with MMLU 0.694, rootcause section 2);
# with BOS it should move at least as well - watch MMLU.""",
    },
    "simnpo": {
        "file": "simnpo.yaml",
        "src": "simnpo.yaml",
        "tag_old": "simnpo",
        "tag_new": "simnpo",
        "note": """\
# Algorithm section and lr 5e-5 copied unchanged (mbs 1 / eval bs 16
# tier). The lr 1e-6 arms stayed frozen on gemma4 no-BOS (0.786);
# 5e-5 is the only regime that ever moved this algorithm on gemma4.""",
    },
    "satimp_forgetonly": {
        "file": "satimp_forgetonly.yaml",
        "src": "satimp_forgetonly.yaml",
        "tag_old": "satimp-forgetonly",
        "tag_new": "satimp-forgetonly",
        "note": """\
# Algorithm section copied unchanged; forget-only (retain null, mbs 4).
# SatImp saturated on gemma4 no-BOS (token weights ~1e-17) - this run
# tests the BOS regime.
""" + GA_FAMILY_NOTE,
    },
    "WGA_forgetonly": {
        "file": "WGA_forgetonly.yaml",
        "src": "WGA_forgetonly.yaml",
        "tag_old": "wga-forgetonly",
        "tag_new": "wga-forgetonly",
        "note": """\
# Algorithm section copied unchanged; forget-only (retain null, mbs 4).
# Same family context as satimp above.
""" + GA_FAMILY_NOTE,
    },
    "ceu": {
        "file": "ceu.yaml",
        "src": "ceu.yaml",
        "tag_old": "ceu",
        "tag_new": "ceu",
        "note": """\
# Algorithm section copied unchanged; forget-only (retain null, mbs 1,
# full-logit objective at 2^30 elements - under the int32 cap).
# KNOWN BUG, UNFIXED: CEU's fp32 target normalization amplifies early
# gradients up to ~6e3x on high-confidence tokens (objectives.py:278,
# rootcause section 4.1) - read this run's curve with that in mind; all
# three models' CEU results carry the same caveat.""",
    },
}

for _suffix, _eps, _offset in PDU_EPS_ARMS:
    ALGOS[f"pdu_eps{_suffix}"] = {
        "file": f"pdu_eps{_suffix}.yaml",
        "src": "pdu_eps2p51.yaml",
        "tag_old": "pdu-eps2p51",
        "tag_new": f"pdu-eps{_suffix}",
        "eps": _eps,
        "note": (
            f"# This arm: eps = {_eps} (nominal offset {_offset} vs the VOID\n"
            "# softcap-no-BOS anchor 2.0945 - grid-spacing reference only).\n"
            + PDU_ARM_NOTE
        ),
    }


def replace_counted(text: str, old: str, new: str, expected: int, where: str) -> str:
    found = text.count(old)
    if found != expected:
        raise SystemExit(f"ASSERT {where}: {found!r} occurrences of {old!r}, expected {expected}")
    return text.replace(old, new)


def _algorithm_block(text: str) -> str:
    start = text.index("algorithm:\n")
    end = text.index("\ntraining:\n")
    return text[start:end]


def migrate(algo: str, spec: dict) -> str:
    text = (SRC / spec["src"]).read_text()

    body = text[text.index("name: "):]

    name_old = f"qwen35-main-cyber-{spec['tag_old']}"
    name_new = f"gemma4-main-cyber-{spec['tag_new']}"
    body = replace_counted(body, name_old, name_new, 2, f"{algo}/names")
    body = replace_counted(body, "  name: qwen35\n", "  name: gemma4\n", 1, f"{algo}/model.name")
    body = replace_counted(body, QWEN_PATH, GEMMA_PATH, 1, f"{algo}/model.path")
    body = replace_counted(body, "num_experts: 256", "num_experts: 128", 1, f"{algo}/num_experts")
    body = replace_counted(body, "pad_token_id: 248044", "pad_token_id: 0", 1, f"{algo}/pad_token_id")
    body = replace_counted(body, FORGET_OLD, FORGET_NEW, 1, f"{algo}/forget_cache")
    # satimp / WGA / ceu wire no retain stream (retain_cache null);
    # only replace the retain line where it exists.
    if RETAIN_OLD in body:
        body = replace_counted(body, RETAIN_OLD, RETAIN_NEW, 1, f"{algo}/retain_cache")
    body = replace_counted(body, EVALDIR_OLD, EVALDIR_NEW, 1, f"{algo}/eval_dir")

    if algo == "rmu":
        body = replace_counted(body, "alpha: 63.4", "alpha: 4.45", 1, f"{algo}/alpha")
        body = replace_counted(
            body, "steering_coeff: 300.0", "steering_coeff: 1200.0", 1, f"{algo}/steering_coeff"
        )
        # Structural alignment: the algorithm section must equal the
        # gemma4-calibrated sweep best arm byte for byte.
        ref_block = _algorithm_block(RMU_REF.read_text())
        if _algorithm_block(body) != ref_block:
            raise SystemExit(
                f"ASSERT {algo}: algorithm section drifted from the coeff-alpha sweep "
                f"best arm ({RMU_REF.name}):\n--- generated ---\n{_algorithm_block(body)}"
                f"--- reference ---\n{ref_block}"
            )
    if "eps" in spec:
        body = replace_counted(
            body,
            "retain_loss_eps: 2.51",
            f"retain_loss_eps: {spec['eps']}",
            1,
            f"{algo}/eps",
        )

    # G1 save plan: drop the qwen R2 single save point; save nothing.
    body, n = re.subn(r"save_steps: \[\d+\]", "save_steps: []", body, count=1)
    if n != 1 or "save_steps: []" not in body or "save_final_checkpoint: false" not in body:
        raise SystemExit(f"ASSERT {algo}: G1 save plan (save_steps [] / save_final false) not set")

    # No qwen-specific value may survive outside comments.
    for leftover in ("qwen35", "qwen3p5", "Qwen"):
        for line in body.splitlines():
            if leftover in line and not line.lstrip().startswith("#"):
                raise SystemExit(f"ASSERT {algo}: leftover {leftover!r} in non-comment line: {line}")

    header = (
        f"# Gemma4 main-cyber {algo} configuration (G1 curve round).\n"
        f"# Auto-migrated from main/main-cyber/qwen35/{spec['src']} (R2 edition) by\n"
        f"# main/main-cyber/gemma4/make_configs.py - training plan byte-copied\n"
        f"# (cyber-line policy), model switched to gemma-26b-a4b-nocap and caches\n"
        f"# to the BOS variants (the combo whose transfer the BOS ablation\n"
        f"# confirmed on bio GD: 0.79 -> 0.578 by step 300 at lr 1e-6).\n"
        + spec["note"] + "\n"
        + G1_NOTE + "\n"
        + COMMON_NOTE + "\n"
    )
    return header + body


def main() -> None:
    out_dir = HERE / "configs-r1"
    out_dir.mkdir(exist_ok=True)
    written = []
    for algo, spec in ALGOS.items():
        out = out_dir / spec["file"]
        out.write_text(migrate(algo, spec))
        written.append(out.name)
    print(f"Wrote {len(written)} configs to {out_dir}:")
    for name in written:
        print(f"  {name}")


if __name__ == "__main__":
    sys.exit(main())
