#!/usr/bin/env python3
"""Generate the eleven qwen35 main-cyber configs (9 algorithms; PDU split
into three eps arms) from the intern-cyber R3 yamls.

One-shot migration tool (run once; regenerate only if the intern-cyber
sources change). Per-file transforms, with occurrence-count assertions so a
drifted source fails loudly instead of silently copying wrong fields:

  - model block      : interns2 -> qwen35 (path/name; fields from the
                       qwen35 bio main line, main/qwen35/*.yaml)
  - data caches      : cyber forget -> qwen35_wmdp_cyber_softpack4096/forget,
                       retain -> fineweb_edu_5b_qwen35_softpack4096_seed42
  - names            : interns2-main-cyber-* -> qwen35-main-cyber-*
  - online-eval dir  : main-cyber/interns2/online-evaluation ->
                       main-cyber/qwen35/online-evaluation
  - RMU alpha        : 350.0 -> 63.4 (qwen-calibrated R:F sweep best; the
                       intern value is not portable across models)
  - PDU eps          : 2.05 -> three arms 2.16 / 2.30 / 2.51 (2026-10-01
                       decision; anchor = retain NLL 2.0082 over the same
                       fineweb retain cache, so offsets are comparable to
                       the qwen35 bio eps sweep)
  - save plan        : R1 (first submission, 2026-10-01) saved nothing;
                       after its curves landed, this R2 edit (same day)
                       points save_steps at each run's margin-best step
                       (margin = mmlu - wmdp_cyber, argmax over the R1
                       online-eval curve, earliest step wins ties); the
                       final checkpoint stays OFF

Everything else is copied byte-identical from the intern-cyber source:
algorithm sections, lr schedule (5e-6 unified / 5e-5 NPO family, min_lr 1e-6,
10 epochs), mbs per algorithm, online-eval batch sizes and the [23..230]
per-epoch eval steps (the qwen35 cyber forget cache has the same 743 packs
as intern's, so the 230-step schedule carries over exactly).
"""

from __future__ import annotations

import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE.parent / "interns2"

INTERN_PATH = "/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/jincheng/sgtm/checkpoints/intern/Intern-S2-base05"
QWEN_PATH = "/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/jincheng/sgtm/checkpoints/qwen/qwen3p5-35b-a3b"

FORGET_OLD = "interns2_wmdp_cyber_softpack4096/forget"
FORGET_NEW = "qwen35_wmdp_cyber_softpack4096/forget"
RETAIN_OLD = "fineweb_edu_5b_interns2_softpack4096_seed42"
RETAIN_NEW = "fineweb_edu_5b_qwen35_softpack4096_seed42"
EVALDIR_OLD = "main/main-cyber/interns2/online-evaluation"
EVALDIR_NEW = "main/main-cyber/qwen35/online-evaluation"

COMMON_NOTE = """\
# Schedule / mbs / eval batch_size copied unchanged from the intern-cyber
# R3 config: the qwen35 cyber forget cache packs to the same 743 packs,
# so 10 epochs = 230 steps and the per-epoch eval [23..230] carries over
# 1:1. Save-plan history: R1 (2026-10-01 first submission) saved NO
# checkpoints - a pure curve round; its curves then picked each run's
# margin-best step, and this R2 edit (same day) sets save_steps to exactly
# that one point. Final checkpoint stays OFF. Output names / online-eval
# dir under main-cyber/qwen35; checkpoint output_root is shared with the
# intern-cyber line (runs are name-distinct).\
"""

# R2 save points (2026-10-01): ONE checkpoint per run at its margin-best
# step - margin = mmlu - wmdp_cyber, argmax over the R1 online-eval curve
# (earliest step wins ties), read off online-evaluation/*_online_eval.jsonl:
#   rmu 92/0.559   pdu-2p16 115/0.550  pdu-2p30 92/0.549  pdu-2p51 115/0.559
#   grad_diff 69/0.401   grad_ascent 46/0.331   pure_npo 69/0.392
#   simnpo 46/0.395   satimp 46/0.419   wga 184/0.414   ceu 184/0.397
# (base margin 0.259 for reference). Verified against the jsonl after
# regeneration - see README R2 table.
SAVE_STEPS: dict[str, int] = {
    "rmu.yaml": 92,
    "pdu_eps2p16.yaml": 115,
    "pdu_eps2p30.yaml": 92,
    "pdu_eps2p51.yaml": 115,
    "grad_diff.yaml": 69,
    "grad_ascent.yaml": 46,
    "pure_npo.yaml": 69,
    "simnpo.yaml": 46,
    "satimp_forgetonly.yaml": 46,
    "WGA_forgetonly.yaml": 184,
    "ceu.yaml": 184,
}

ALGOS: dict[str, dict] = {
    "rmu": {
        "file": "rmu.yaml",
        "tag": "rmu",
        "note": """\
# The algorithm section is byte-identical to main/qwen35/rmu.yaml (the
# qwen35 bio main line): alpha 350.0 -> 63.4 (steer 300 unchanged) - the
# intern alpha is NOT portable across models (gemma4 R:F-mismatch
# verdict); 63.4 is the qwen35-calibrated R:F-sweep best arm (bio online
# 0.323/0.817, main/qwen35/... RMU_rf_sweep_report_20260924.md). Only the
# algorithm section is aligned - optimizer/schedule keep the cyber-line
# policy below. Eval bs stays 32: cyber stems are longer than bio's (qwen
# bio ran 96 on bio-length prompts); lower to 16 only on first-eval CUDA
# OOM.""",
    },
    "pdu_eps2p16": {
        "file": "pdu_eps2p16.yaml",
        "src": "pdu.yaml",
        "src_tag": "pdu",
        "tag": "pdu-eps2p16",
        "eps": "2.16",
        "note": """\
# One of THREE PDU eps arms on this line (2.16 / 2.30 / 2.51, decided
# 2026-10-01; replaces the single bio-best 2.04 arm). eps anchors to the
# retain NLL of the init model over the retain cache (2.0082 for qwen35,
# tools.measure_retain_nll); the retain cache is the SAME fineweb pack set
# as the qwen35 bio eps sweep, so anchor offsets are comparable across
# lines: 2.16 = anchor+0.15 (bio reference point: 0.636/0.810). lr keeps
# the cyber-line unified 5e-6 (qwen eps was tuned at 4e-6 - same regime;
# see the intern-cyber lr-eps coupling note). mbs 1 is mandatory on qwen35
# (full-logit int32 cap at mbs 4; eps-sweep verdict) and matches the qwen
# calibration.""",
    },
    "pdu_eps2p30": {
        "file": "pdu_eps2p30.yaml",
        "src": "pdu.yaml",
        "src_tag": "pdu",
        "tag": "pdu-eps2p30",
        "eps": "2.3",
        "note": """\
# One of THREE PDU eps arms on this line (2.16 / 2.30 / 2.51, decided
# 2026-10-01; replaces the single bio-best 2.04 arm). eps anchors to the
# retain NLL of the init model over the retain cache (2.0082 for qwen35,
# tools.measure_retain_nll); the retain cache is the SAME fineweb pack set
# as the qwen35 bio eps sweep, so anchor offsets are comparable across
# lines: 2.30 = anchor+0.29 (new midpoint between the bio 2.16 and 2.51
# arms). lr keeps the cyber-line unified 5e-6 (qwen eps was tuned at 4e-6 -
# same regime; see the intern-cyber lr-eps coupling note). mbs 1 is
# mandatory on qwen35 (full-logit int32 cap at mbs 4; eps-sweep verdict)
# and matches the qwen calibration.""",
    },
    "pdu_eps2p51": {
        "file": "pdu_eps2p51.yaml",
        "src": "pdu.yaml",
        "src_tag": "pdu",
        "tag": "pdu-eps2p51",
        "eps": "2.51",
        "note": """\
# One of THREE PDU eps arms on this line (2.16 / 2.30 / 2.51, decided
# 2026-10-01; replaces the single bio-best 2.04 arm). eps anchors to the
# retain NLL of the init model over the retain cache (2.0082 for qwen35,
# tools.measure_retain_nll); the retain cache is the SAME fineweb pack set
# as the qwen35 bio eps sweep, so anchor offsets are comparable across
# lines: 2.51 = anchor+0.50 (bio reference point: 0.482/0.793 - deepest
# bio forgetting, MMLU eroding). lr keeps the cyber-line unified 5e-6
# (qwen eps was tuned at 4e-6 - same regime; see the intern-cyber lr-eps
# coupling note). mbs 1 is mandatory on qwen35 (full-logit int32 cap at
# mbs 4; eps-sweep verdict) and matches the qwen calibration.""",
    },
    "grad_diff": {
        "file": "grad_diff.yaml",
        "tag": "grad-diff",
        "note": """\
# Algorithm section copied unchanged (identical between the intern and
# qwen35 bio main lines; mbs 1 verified on qwen35 bio).""",
    },
    "grad_ascent": {
        "file": "grad_ascent.yaml",
        "tag": "grad-ascent",
        "note": """\
# Algorithm section copied unchanged (identical between the intern and
# qwen35 bio main lines; mbs 4 verified on qwen35 bio).""",
    },
    "pure_npo": {
        "file": "pure_npo.yaml",
        "tag": "pure-npo",
        "note": """\
# Algorithm section and lr 5e-5 copied unchanged (identical between the
# intern and qwen35 bio main lines; mbs 1, eval bs 16 verified on qwen35
# bio).""",
    },
    "simnpo": {
        "file": "simnpo.yaml",
        "tag": "simnpo",
        "note": """\
# Algorithm section and lr 5e-5 copied unchanged (identical between the
# intern and qwen35 bio main lines; mbs 1, eval bs 16 verified on qwen35
# bio).""",
    },
    "satimp_forgetonly": {
        "file": "satimp_forgetonly.yaml",
        "tag": "satimp-forgetonly",
        "note": """\
# Algorithm section copied unchanged (identical between the intern and
# qwen35 bio main lines; mbs 4 verified on qwen35 bio).""",
    },
    "WGA_forgetonly": {
        "file": "WGA_forgetonly.yaml",
        "tag": "wga-forgetonly",
        "note": """\
# Algorithm section copied unchanged (identical between the intern and
# qwen35 bio main lines; mbs 4 verified on qwen35 bio).""",
    },
    "ceu": {
        "file": "ceu.yaml",
        "tag": "ceu",
        "note": """\
# Algorithm section copied unchanged (identical between the intern and
# qwen35 bio main lines; mbs 1 verified on qwen35 bio).""",
    },
}


def replace_counted(text: str, old: str, new: str, expected: int, where: str) -> str:
    found = text.count(old)
    if found != expected:
        raise SystemExit(f"ASSERT {where}: {found!r} occurrences of {old!r}, expected {expected}")
    return text.replace(old, new)


def migrate(algo: str, spec: dict) -> str:
    src_file = SRC / spec.get("src", spec["file"])
    text = src_file.read_text()

    # Swap the hand-written intern header for the qwen migration header.
    body_at = text.index("name: ")
    header = (
        f"# Qwen3.5 main-cyber {algo} configuration.\n"
        f"# Auto-migrated from {src_file.relative_to(HERE.parents[2])} (R3 plan)\n"
        f"# by main/main-cyber/qwen35/make_configs.py - model switched to\n"
        f"# qwen3p5-35b-a3b (fields from the qwen35 bio main line), caches\n"
        f"# switched to the qwen35 cyber forget pack (plus the qwen35 fineweb\n"
        f"# retain pack where the source wires a retain stream).\n"
        + spec["note"] + "\n"
        + (
            f"# R2 save point: ONE checkpoint at step {SAVE_STEPS[spec['file']]}"
            f" (epoch {SAVE_STEPS[spec['file']] // 23}) - the margin-best\n"
            f"# point of this run's R1 curve (margin = mmlu - wmdp_cyber, argmax,\n"
            f"# earliest step wins ties). save_final_checkpoint stays false.\n"
        )
        + COMMON_NOTE + "\n"
    )
    body = text[body_at:]

    name_old = f"interns2-main-cyber-{spec.get('src_tag', spec['tag'])}"
    name_new = f"qwen35-main-cyber-{spec['tag']}"
    body = replace_counted(body, name_old, name_new, 2, f"{algo}/names")
    body = replace_counted(body, f"  name: interns2\n", "  name: qwen35\n", 1, f"{algo}/model.name")
    body = replace_counted(body, INTERN_PATH, QWEN_PATH, 1, f"{algo}/model.path")
    body = replace_counted(body, FORGET_OLD, FORGET_NEW, 1, f"{algo}/forget_cache")
    # satimp / WGA / ceu wire no retain stream in their sources
    # (retain_cache null); only replace the retain line where it exists.
    has_retain = RETAIN_OLD in body
    if has_retain:
        body = replace_counted(body, RETAIN_OLD, RETAIN_NEW, 1, f"{algo}/retain_cache")
    body = replace_counted(body, EVALDIR_OLD, EVALDIR_NEW, 1, f"{algo}/eval_dir")

    if algo == "rmu":
        body = replace_counted(body, "alpha: 350.0", "alpha: 63.4", 1, f"{algo}/alpha")
        # Inline-list formatting so the algorithm section is byte-identical to
        # main/qwen35/rmu.yaml (the qwen35 bio reference).
        body = replace_counted(
            body,
            "train_layers:\n    - 7\n    - 8\n    - 9",
            "train_layers: [7, 8, 9]",
            1,
            f"{algo}/train_layers",
        )
    if "eps" in spec:
        body = replace_counted(
            body, "retain_loss_eps: 2.05", f"retain_loss_eps: {spec['eps']}", 1, f"{algo}/eps"
        )

    # No intern-specific value may survive outside comments.
    for leftover in ("interns2", "Intern-S2"):
        for line in body.splitlines():
            if leftover in line and not line.lstrip().startswith("#"):
                raise SystemExit(f"ASSERT {algo}: leftover {leftover!r} in non-comment line: {line}")

    # R2 save plan (2026-10-01): exactly ONE checkpoint, at this run's
    # margin-best step (R1-curve argmax of mmlu - wmdp_cyber). Replaces the
    # intern-R3 point save in the source; the final checkpoint stays OFF.
    save_step = SAVE_STEPS[spec["file"]]
    body = re.sub(r"save_steps: \[\d+\]", f"save_steps: [{save_step}]", body, count=1)
    if f"save_steps: [{save_step}]\n" not in body or "save_final_checkpoint: false" not in body:
        raise SystemExit(
            f"ASSERT {algo}: save plan must be save_steps [{save_step}] / save_final false"
        )
    return header + body


def main() -> None:
    written = []
    for algo, spec in ALGOS.items():
        out = HERE / spec["file"]
        out.write_text(migrate(algo, spec))
        written.append(out.name)
    print(f"Wrote {len(written)} configs to {HERE}:")
    for name in written:
        print(f"  {name}")


if __name__ == "__main__":
    sys.exit(main())
