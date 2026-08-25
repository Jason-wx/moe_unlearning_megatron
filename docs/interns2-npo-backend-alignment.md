# Intern-S2 PureNPO: DeepSpeed / Megatron alignment

This document records the *effective* training semantics used for the
DeepSpeed-HF and Megatron-Core comparison. A matching field name is not enough:
the value after Accelerate, DeepSpeed and Megatron apply their defaults is what
must match.

## Aligned settings

| Category | Effective value on both backends | Where it is fixed |
|---|---|---|
| Initial student/reference | Same `Intern-S2-base05` checkpoint; reference is frozen | model and reference configs |
| Forget data | Same pre-tokenized `interns2_wmdp_bio_max1024_stride1024/forget` cache | data configs |
| Sequence construction | length 1024, right padding, next-token labels, padding excluded by `-100`/loss mask | cache and collators |
| Objective | sequence-level PureNPO, mean NLL per valid sequence, `beta=0.1`, `gamma=1.0`, no retain term | `npo.py` / `objectives.py` |
| Trainable weights | routed expert projections plus shared-expert projections; router/gate and all non-expert weights frozen | backend parameter selectors |
| Precision | BF16 model computation, FP32 optimizer/gradient reduction state | backend runtime configs |
| Optimizer | AdamW-equivalent Adam, LR `1e-6`, betas `(0.9, 0.999)`, epsilon `1e-8`, weight decay `0` | both optimizer configs |
| LR schedule | constant, zero warmup | both scheduler configs |
| Gradient clipping | disabled (`0.0`) | HF arguments, DeepSpeed JSON and Megatron optimizer config |
| Global batch | 128 examples per optimizer step | DS: `4 x DP8 x GA4`; MCore: `16 x DP1 x GA8` |
| Gradient averaging | mean inside each microbatch, then mean across data-parallel ranks and accumulation microbatches | framework runtimes |
| Sampling seed | model seed and data seed `42` | both launch/config files |
| Examples/steps | 1344 complete global batches = 172032 examples; the final 105 of 172137 are not trained | explicit `max_steps`/`train_steps` plus `drop_last` |
| Epochs/checkpoint | one effective epoch over the complete batches; final model weights exported | launch/config files |
| MoE auxiliary loss | disabled for this manually computed NPO loss | effective backend model configs |

## Mismatches found and corrected

1. Megatron inherited Adam `beta2=0.95`; HF `adamw_torch` uses `0.999`.
   Megatron now explicitly uses `0.999`, and the HF values are also explicit.
2. Megatron inherited gradient clipping `1.0`. The successful DeepSpeed run was
   effectively unclipped because its runtime value was `gradient_clipping=0.0`.
   Both are now explicitly `0.0`.
3. The 172137-example cache previously produced 1345 DeepSpeed updates but only
   1344 complete Megatron updates. Both now stop at 1344 and consume 172032
   examples.
4. DeepSpeed used seed `0` while the requested Megatron default was `42`. Model
   seed and sampler/data seed are now explicitly `42` on both sides.

## Backend differences that remain by design

These cannot be made byte-identical without turning the two runs into the same
backend. They should not change the intended objective, but can produce small
floating-point trajectory differences.

- DeepSpeed uses DP8 + ZeRO-3; Megatron uses TP4 + PP2 + EP4 + DP1.
- DeepSpeed dynamically pads each microbatch; Megatron pads to the configured
  1024 length. Padding positions are masked on both sides.
- DeepSpeed uses HF model kernels and activation checkpointing; Megatron uses
  MCore/Transformer-Engine kernels and its current recomputation setting.
- Parameter and optimizer states are sharded differently, and checkpoint export
  follows different code paths.
- With seed 42, the two samplers target the same shuffled example stream, but
  different rank/microbatch grouping means the order of floating-point
  additions is not bitwise identical.

## Validation gates before attributing an accuracy gap to NPO

1. Compare the exact trainable parameter name categories and logical parameter
   count printed by each backend; the known HF run has 200 trainable tensors and
   32,338,083,840 trainable parameters.
2. On optimizer step 1, log the same global sample IDs, student/reference
   sequence NLL, NPO loss and gradient norm. Compare before any optimizer update.
3. Run a zero-step TP4/PP2/EP4 MCore load-and-export, then evaluate WMDP-Bio and
   MMLU. A single-GPU round trip does not test distributed aggregation.
4. Only after the zero-step export matches the base model, rerun the aligned
   1344-step PureNPO training and evaluation.

`token_npo` is an ablation, not a backend-equivalent reproduction of the
DeepSpeed `TokenSimNPO` result: Megatron `token_npo` is reference-based token
NPO, whereas the repository's `TokenSimNPO` is reference-free and includes a
retain objective.
