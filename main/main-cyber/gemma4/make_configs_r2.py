#!/usr/bin/env python3
"""Generate the nine gemma4 main-cyber R2 save-point configs from the
configs-r1/ yamls + the R1 online-eval curves.

R2 protocol (qwen35-cyber R1->R2 analog, 2026-10-03 decision): re-run the
full 10-epoch / 230-step schedule, exporting exactly ONE checkpoint per
run at its R1 margin-best step (margin = mmlu - wmdp_cyber, argmax over
the R1 online-eval curve, earliest step wins ties). save_final_checkpoint
stays false. PDU collapses to its single winning arm (eps 2.60).

Per-file transforms, with assertions so drifted sources fail loudly:

  - source          : configs-r1/<src>.yaml (training plan byte-copied)
  - save point      : derived FROM THE R1 JSONL by this script and
                      cross-checked against the table below (both must
                      agree; the qwen35 line's "recompute from jsonl"
                      protocol)
  - names           : gemma4-main-cyber-<tag> -> gemma4-main-cyber-r2-<tag>
                      (keeps R2 online-eval jsonls and ckpt dirs separate
                      from R1's records)
  - save plan       : save_steps: [] -> save_steps: [<best step>]
  - PDU             : pdu_eps2p60.yaml -> pdu.yaml, run tag pdu-eps2p60
                      -> pdu (only arm kept; the other five R1 eps arms
                      are NOT re-run)

Everything else (model gemma-26b-a4b-nocap, BOS caches, lr/mbs/eval-bs
tiers, eval grid [23..230], seed 42, output dirs) is copied byte-identical
from the R1 config.
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE / "configs-r1"
# R1 curves were archived into online-evaluation/r1/ (2026-10-03, while
# the R2 batch was still writing the top level).
EVAL_DIR = HERE / "online-evaluation" / "r1"

# (r2 file, r1 source file, r1 run tag, r2 run tag, per-algo note)
ALGOS = [
    (
        "rmu.yaml",
        "rmu.yaml",
        "rmu",
        "rmu",
        """\
# R1 readout at the save point: 0.265/0.743 (margin +0.478); the curve
# flattens into a 0.271-0.281 band after ep4, so 92 is a stable plateau
# pick, not a knife-edge.""",
    ),
    (
        "pdu.yaml",
        "pdu_eps2p60.yaml",
        "pdu-eps2p60",
        "pdu",
        """\
# The ONLY PDU arm kept (2026-10-03 decision): eps 2.60 won the six-arm
# R1 grid (0.249/0.759, margin +0.511@207; eps->depth monotone across
# 2.0-3.0 with mmlu held at 0.75-0.76). Caveat recorded at decision time:
# the R1 curve was still slowly descending at 230 (best at 207/230) - R2
# stays at 10 epochs and point-saves; if the offline eval of this ckpt
# underwhelms, the follow-up is a longer-schedule round, not a wider
# grid.""",
    ),
    (
        "grad_diff.yaml",
        "grad_diff.yaml",
        "grad-diff",
        "grad-diff",
        """\
# R1: 0.259/0.721 (+0.462)@161 - the BOS-fix positive control, first
# real logit-level transfer on gemma4. mmlu erodes after ep7 (0.711 by
# 230), so the save point is also the mmlu knee.""",
    ),
    (
        "grad_ascent.yaml",
        "grad_ascent.yaml",
        "grad-ascent",
        "grad-ascent",
        """\
# R1 MMLU collapsed (final 0.266/0.239); step 23 (0.438/0.734) is the
# early-stop rescue point - this ckpt is for completeness/offline
# confirmation only.""",
    ),
    (
        "pure_npo.yaml",
        "pure_npo.yaml",
        "pure-npo",
        "pure-npo",
        """\
# R1 shape: overshoot to ~0.31 by step 23, then a slow recovery to
# 0.37-0.40. The margin peak sits at 69/ep3 (0.304/0.732) - treat this
# ckpt with suspicion; the algorithm effectively only works in the first
# three epochs.""",
    ),
    (
        "simnpo.yaml",
        "simnpo.yaml",
        "simnpo",
        "simnpo",
        """\
# R1 shape: same overshoot-then-recover as pure_npo; the margin peak is
# the very FIRST eval point (23: 0.302/0.727), i.e. essentially the
# warmup knee.""",
    ),
    (
        "satimp_forgetonly.yaml",
        "satimp_forgetonly.yaml",
        "satimp-forgetonly",
        "satimp-forgetonly",
        """\
# R1 MMLU crashed to 0.32 by step 23 (lr 5e-6 too hot for the mbs4
# forget-only family on gemma4); the 207 pick lives inside the collapsed
# band - completeness only, same as grad_ascent.""",
    ),
    (
        "WGA_forgetonly.yaml",
        "WGA_forgetonly.yaml",
        "wga-forgetonly",
        "wga-forgetonly",
        """\
# R1 MMLU crashed (0.34 band from step 23 on); 69/ep3 is the least-bad
# point of a dead run - completeness only.""",
    ),
    (
        "ceu.yaml",
        "ceu.yaml",
        "ceu",
        "ceu",
        """\
# R1: slow but monotone, best at the very end (230: 0.362/0.725,
# +0.363) - the only R2 save point equal to the final step. Read with
# the known fp32 target-normalization bug in mind (early gradients
# amplified up to ~6e3x).""",
    ),
]

# Cross-check table: the R1 margin-best steps this script must derive
# from the jsonls (recorded 2026-10-03 from the R1 readout table).
EXPECTED_SAVE_STEPS = {
    "rmu.yaml": 92,
    "pdu.yaml": 207,
    "grad_diff.yaml": 161,
    "grad_ascent.yaml": 23,
    "pure_npo.yaml": 69,
    "simnpo.yaml": 23,
    "satimp_forgetonly.yaml": 207,
    "WGA_forgetonly.yaml": 69,
    "ceu.yaml": 230,
}

COMMON_NOTE = """\
# Everything else is byte-identical to the R1 config: model
# gemma-26b-a4b-nocap + BOS caches, lr (5e-6 / NPO family 5e-5), mbs /
# eval-bs tiers, the [23..230] per-epoch eval grid (748 packs -> 23
# steps per epoch), seed 42, shared main-cyber output_root. R2 run
# dirs / jsonls are name-distinct from R1 (gemma4-main-cyber-r2-*).\\
""".replace("\\\n", "\n")


def best_step_from_jsonl(tag: str) -> tuple[int, float, float]:
    path = EVAL_DIR / f"gemma4-main-cyber-{tag}_online_eval.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows or len(rows) != 10:
        raise SystemExit(f"ASSERT {tag}: expected 10 eval rows in {path}, got {len(rows)}")
    best = max(rows, key=lambda r: (r["mmlu"] - r["wmdp_cyber"], -r["step"]))
    return best["step"], best["wmdp_cyber"], best["mmlu"]


def replace_counted(text: str, old: str, new: str, expected: int, where: str) -> str:
    found = text.count(old)
    if found != expected:
        raise SystemExit(f"ASSERT {where}: {found!r} occurrences of {old!r}, expected {expected}")
    return text.replace(old, new)


def migrate(out_file: str, src_file: str, tag_old: str, tag_new: str, note: str) -> str:
    text = (SRC / src_file).read_text()
    body = text[text.index("name: "):]

    step, cyber, mmlu = best_step_from_jsonl(tag_old)
    if step != EXPECTED_SAVE_STEPS[out_file]:
        raise SystemExit(
            f"ASSERT {out_file}: jsonl-derived best step {step} != recorded "
            f"{EXPECTED_SAVE_STEPS[out_file]} (R1 table drifted - re-read the curve)"
        )

    name_old = f"gemma4-main-cyber-{tag_old}"
    name_new = f"gemma4-main-cyber-r2-{tag_new}"
    body = replace_counted(body, name_old, name_new, 2, f"{out_file}/names")

    body, n = re.subn(r"save_steps: \[\]", f"save_steps: [{step}]", body, count=1)
    if n != 1 or f"save_steps: [{step}]\n" not in body:
        raise SystemExit(f"ASSERT {out_file}: save_steps edit failed")
    if "save_final_checkpoint: false" not in body:
        raise SystemExit(f"ASSERT {out_file}: save_final_checkpoint must stay false")
    if name_old in body:
        raise SystemExit(f"ASSERT {out_file}: R1 run name survived the rename")

    header = (
        f"# Gemma4 main-cyber {tag_new} configuration (R2 save-point round).\n"
        f"# Derived from configs-r1/{src_file} by\n"
        f"# main/main-cyber/gemma4/make_configs_r2.py - the ONLY deltas vs R1 are\n"
        f"# the run name (-r2- infix) and the save plan; the schedule is re-run in\n"
        f"# full (10 epochs / 230 steps).\n"
        f"# R2 save point: ONE checkpoint at step {step} (epoch {step // 23}) -\n"
        f"# the margin-best point of the R1 curve (margin = mmlu - wmdp_cyber,\n"
        f"# argmax, earliest step wins ties; R1 readout there: {cyber:.3f}/{mmlu:.3f}).\n"
        f"# save_final_checkpoint stays false.\n"
        + note + "\n"
        + COMMON_NOTE
    )
    return header + body


def main() -> None:
    out_dir = HERE / "configs-r2"
    out_dir.mkdir(exist_ok=True)
    for out_file, src_file, tag_old, tag_new, note in ALGOS:
        (out_dir / out_file).write_text(migrate(out_file, src_file, tag_old, tag_new, note))
        print(f"  {out_file:26s} <- configs-r1/{src_file:22s} save@{EXPECTED_SAVE_STEPS[out_file]:3d} (ep{EXPECTED_SAVE_STEPS[out_file] // 23})")
    print(f"Wrote {len(ALGOS)} R2 configs to {out_dir}")


if __name__ == "__main__":
    sys.exit(main())
