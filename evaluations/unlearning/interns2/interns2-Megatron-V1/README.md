# Intern-S2 All-11 Organized Results

This directory presents the WMDP/MMLU evaluation artifacts from the current
Intern-S2 unlearning round in an algorithm-first layout. Each completed
algorithm has one verbatim copy of the lm-eval `result.json` for step 300 and
one for epoch 1.

The unmodified base-model result is stored separately as `base/result.json`.
It was copied from
`../../../results/baselines/lm_eval_results_interns2_base_wmdp_mmlu/result.json`.

## Layout

```text
base/
└── result.json

<algorithm>/
├── step300.json
└── epoch1.json
```

## Included results

The following results were copied without modification from
`../interns2-all11-20260825-v1/lm_eval_results_<algorithm>_<checkpoint>_wmdp_mmlu/result.json`:

- `pure_npo`
- `token_npo`
- `ceu`
- `rmu`
- `grad_ascent`
- `undial`
- `grad_diff`
- `wga`
- `token_simnpo`
- `satimp`
- `pdu`

## Completion status

All 11 algorithms in this sweep now include both the step-300 and epoch-1
evaluation results.

The original sweep directories remain the source of record and are not changed
by this organized view.
