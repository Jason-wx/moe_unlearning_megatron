# Intern-S2 Megatron V2 结果

更新时间：2026-08-31

V2 延续 V1 的 1024-token chunk、global batch 128 和 expert-only 训练方式，
重点调试 WGA、SatImp，并收录单独运行的 6-GPU RMU。评测口径为 WMDP-Bio
越低越好、MMLU 越高越好；Base 为 83.66% / 85.15%。

## 完整结果

| 算法/配置 | Checkpoint | WMDP-Bio ↓ | MMLU ↑ |
| --- | --- | ---: | ---: |
| WGA 原始配置 | step 300 | 23.96% ± 1.20% | 27.50% ± 0.38% |
| WGA 原始配置 | epoch 1 | 28.12% ± 1.26% | 26.39% ± 0.37% |
| WGA，`beta=0.5, lr=1e-6` | step 300 | 76.20% ± 1.19% | 82.32% ± 0.31% |
| SatImp，`beta1=2, beta2=0.1, lr=4e-6` | epoch 1 | 66.85% ± 1.32% | 76.13% ± 0.33% |
| RMU，6 GPU | step 300 | 39.59% ± 1.37% | 84.44% ± 0.29% |
| RMU，6 GPU | epoch 1 | 37.55% ± 1.36% | 84.67% ± 0.29% |

## 配置与结论

- WGA 调参版本为 forget-only：`alpha=0`、`beta=0.5`、`gamma=1`、
  `lr=1e-6`，训练 300 steps。原始 V2 WGA 虽然 WMDP-Bio 很低，但 MMLU
  同步崩溃，属于破坏性结果。
- SatImp 为 forget-only：`alpha=0`、`beta1=2`、`beta2=0.1`、`gamma=1`、
  `lr=4e-6`，训练 1344 steps。遗忘增强，但 MMLU 下降 9.02 个百分点。
- RMU 使用 6 GPU、`lr=1e-5`、`train_layers=[7,8,9]`、`target_layer=9`。
  epoch 1 相对 Base 将 WMDP-Bio 降低 46.11 个百分点，MMLU 仅降低
  0.48 个百分点，是 V2 中最平衡的结果。

## 原始数据

- `lm_eval_results_wga_step300_wmdp_mmlu/result.json`
- `lm_eval_results_wga_epoch1_wmdp_mmlu/result.json`
- `lm_eval_results_wga-beta05-lr1e6-step300_wmdp_bio_mmlu/result.json`
- `lm_eval_results_satimp-b2-b01-lr4e6_epoch1_wmdp_bio_mmlu/result.json`
- `lm_eval_results_rmu_step300_wmdp_mmlu/result.json`
- `lm_eval_results_rmu_epoch1_wmdp_mmlu/result.json`
