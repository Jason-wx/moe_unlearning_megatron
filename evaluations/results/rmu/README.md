# RMU 测评记录

本目录记录 RMU 在 Intern-S2、Qwen3.5 MoE 和 Gemma-4 MoE 上的测评结果。
这里只保留 WMDP-Bio 和 MMLU 两项指标：WMDP-Bio 越低表示遗忘越强，
MMLU 越高表示通用能力保留越好。

## Base 参考

| 模型 | WMDP-Bio | MMLU |
|---|---:|---:|
| Gemma-4 | 79.26% | 74.24% |
| Intern-S2 | 83.66% | 85.15% |
| Qwen3.5 MoE | 83.27% | 82.47% |

## RMU 结果

| 模型 / 实验 | WMDP-Bio | MMLU |
|---|---:|---:|
| Gemma-4 · All experts · FineWeb retain, L10 fixed | 64.81% | 74.13% |
| Intern-S2 · All experts · FineWeb retain | 37.31% | 83.46% |
| Intern-S2 · Sparse experts · FineWeb retain | 38.10% | 84.46% |
| Intern-S2 · All experts · WMDP retain | 39.83% | 75.81% |
| Qwen3.5 MoE · All experts · FineWeb retain | 52.47% | 82.06% |
| Qwen3.5 MoE · Sparse experts · FineWeb retain | 54.12% | 82.09% |
| Qwen3.5 MoE · All experts · WMDP retain | 59.54% | 81.10% |

## 原始结果

- [Gemma-4 FineWeb retain](lm_eval_results_gemma4_moe_rmu_bio_l10_fixed_wmdp_mmlu/result.json)
- [Intern-S2 FineWeb retain](lm_eval_results_interns2_fineweb_retain_wmdp_mmlu/result.json)
- [Intern-S2 Sparse experts](lm_eval_results_interns2_sparse_experts_wmdp_mmlu/result.json)
- [Intern-S2 WMDP retain](lm_eval_results_interns2_wmdp_retain_wmdp_mmlu/result.json)
- [Qwen3.5 FineWeb retain](lm_eval_results_qwen35_fineweb_retain_wmdp_mmlu/result.json)
- [Qwen3.5 Sparse experts](lm_eval_results_qwen35_sparse_experts_wmdp_mmlu/result.json)
- [Qwen3.5 WMDP retain](lm_eval_results_qwen35_wmdp_retain_wmdp_mmlu/result.json)
