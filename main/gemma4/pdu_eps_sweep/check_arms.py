#!/usr/bin/env python3
"""Invariant check for the generated gemma4 PDU eps-sweep arms.

Verifies every pdu_eps*.yaml against the committed sweep protocol:
anchor-derived eps grid, fixed PDU knobs (gamma/alpha/eta/lr), gemma4
model/data specifics, 2133-step schedule, final-ckpt-only saving.
"""
import glob
import sys

import yaml

EXPECTED_EPS = {
    "pdu_eps2p00.yaml": 2.00,
    "pdu_eps2p10.yaml": 2.10,
    "pdu_eps2p20.yaml": 2.20,
    "pdu_eps2p25.yaml": 2.25,
    "pdu_eps2p50.yaml": 2.50,
    "pdu_eps3p00.yaml": 3.00,
}
EXPECTED_EVAL_STEPS = [1, 300, 700, 1000, 1400, 1700, 1800, 1900, 2000, 2100, 2133]

failures = 0
files = sorted(glob.glob("pdu_eps*.yaml"))
for path in files:
    cfg = yaml.safe_load(open(path))
    t, a = cfg["training"], cfg["algorithm"]
    m, d, mb = cfg["model"], cfg["data"], cfg["backends"]["megatron"]
    par, ev = mb["parallel"], mb["evaluation"]
    exp_name = "gemma4-pdu-eps" + f"{EXPECTED_EPS[path]:.2f}".replace(".", "p")
    checks = [
        ("eps", a["retain_loss_eps"] == EXPECTED_EPS[path], a["retain_loss_eps"]),
        ("name", cfg["name"] == exp_name and t["output_name"] == exp_name, (cfg["name"], t["output_name"])),
        ("primal_dual", a.get("primal_dual") is True, a.get("primal_dual")),
        ("gamma 1.0", a.get("gamma") == 1.0, a.get("gamma")),
        ("alpha 1.0", a.get("alpha") == 1.0, a.get("alpha")),
        ("eta 0.1", a.get("dual_step_size") == 0.1, a.get("dual_step_size")),
        ("lr 4e-6", t["learning_rate"] == 4e-6, t["learning_rate"]),
        ("gbs 32", t["global_batch_size"] == 32, t["global_batch_size"]),
        ("epochs 3", t["num_train_epochs"] == 3, t["num_train_epochs"]),
        ("mbs 1", mb["micro_batch_size"] == 1, mb["micro_batch_size"]),
        ("save_final only", t.get("save_final_checkpoint") is True
         and t.get("save_steps") == [] and t.get("save_every_epoch") is False,
         (t.get("save_steps"), t.get("save_every_epoch"))),
        ("eval steps", ev.get("steps") == EXPECTED_EVAL_STEPS, ev.get("steps")),
        ("eval tasks", ev["tasks"] == ["wmdp_bio", "mmlu"], ev["tasks"]),
        ("mmlu limit 35", ev.get("task_limits", {}).get("mmlu") == 35, ev.get("task_limits")),
        ("eval bs 64", ev.get("batch_size") == 64, ev.get("batch_size")),
        ("eval output dir", ev.get("output_dir", "").endswith("main/gemma4/pdu_eps_sweep/online-evaluation"),
         ev.get("output_dir")),
        ("model path", m["path"].endswith("/sgtm/checkpoints/gemma/gemma-26b-a4b"), m["path"]),
        ("num_experts 128", m.get("num_experts") == 128, m.get("num_experts")),
        ("pad 0", d.get("pad_token_id") == 0, d.get("pad_token_id")),
        ("seq 4096", d.get("sequence_length") == 4096, d.get("sequence_length")),
        ("forget cache", d["forget_cache"].endswith("Xtuner/gemma4_wmdp_bio_softpack4096/forget"),
         d["forget_cache"]),
        ("retain cache", d["retain_cache"].endswith("Xtuner/fineweb_edu_5b_gemma4_softpack4096_seed42"),
         d["retain_cache"]),
        ("ckpt output root", mb.get("output_root", "").endswith("megatron/gemma4/pdu_eps_sweep"),
         mb.get("output_root")),
        ("world 8 / TP1 / PP1 / EP8", (par["world_size"], par["tensor_parallel"],
         par["pipeline_parallel"], par["expert_parallel"]) == (8, 1, 1, 8), par),
        ("dist-opt on", par.get("use_distributed_optimizer") is True, par.get("use_distributed_optimizer")),
    ]
    bad = [(n, v) for n, ok, v in checks if not ok]
    if bad:
        failures += 1
        print(f"FAIL {path}:")
        for n, v in bad:
            print(f"  {n}: got {v!r}")
    else:
        print(f"PASS {path}")

print(f"\n{len(files) - failures}/{len(files)} arms pass all checks")
sys.exit(1 if failures else 0)
