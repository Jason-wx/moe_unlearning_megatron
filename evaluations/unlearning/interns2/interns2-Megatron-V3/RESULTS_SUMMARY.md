# Interns2 Megatron V3 评测结果汇总

本目录包含 12 组 Interns2-Megatron-V3 在 `WMDP + MMLU` 上的 epoch-1 评测结果。
下表取各 `result.json` 中的 `acc,none`，以百分比表示。

## 总体结果

WMDP / WMDP-Bio 越低表示遗忘越充分；MMLU 越高表示通用能力保持越好。

| 算法 | 训练阶段 | WMDP | WMDP-Bio | MMLU | 结果文件 |
|---|---|---:|---:|---:|---|
| CEU | 1 epoch | 25.57% | 24.67% | 22.95% | `lm_eval_results_ceu_epoch1_wmdp_mmlu` |
| Grad-Ascent | 1 epoch | 24.45% | 24.35% | 24.16% | `lm_eval_results_grad_ascent_epoch1_wmdp_mmlu` |
| Grad-Diff | 1 epoch | 61.59% | 68.89% | 83.18% | `lm_eval_results_grad_diff_epoch1_wmdp_mmlu` |
| NPO | 1 epoch | **68.95%** | **84.37%** | 84.87% | `lm_eval_results_npo_epoch1_wmdp_mmlu` |
| PDU | 1 epoch | 68.13% | 83.35% | **84.88%** | `lm_eval_results_pdu_epoch1_wmdp_mmlu` |
| Pure-NPO | 1 epoch | 65.84% | 78.32% | 78.82% | `lm_eval_results_pure_npo_epoch1_wmdp_mmlu` |
| RMU | 1 epoch | 57.77% | 52.00% | 85.02% | `lm_eval_results_rmu_epoch1_wmdp_mmlu` |
| SatImp | 1 epoch | 51.91% | 59.62% | 66.77% | `lm_eval_results_satimp_epoch1_wmdp_mmlu` |
| SimNPO | 1 epoch | 68.35% | 83.90% | 84.71% | `lm_eval_results_simnpo_epoch1_wmdp_mmlu` |
| Token-NPO | 1 epoch | 36.45% | 37.39% | 37.79% | `lm_eval_results_token_npo_epoch1_wmdp_mmlu` |
| Token-SimNPO | 1 epoch | 66.68% | 81.23% | 84.39% | `lm_eval_results_token_simnpo_epoch1_wmdp_mmlu` |
| UNDIAL | 1 epoch | 66.98% | 82.09% | 84.01% | `lm_eval_results_undial_epoch1_wmdp_mmlu` |
| WGA | 1 epoch | 30.70% | 33.46% | 33.91% | `lm_eval_results_wga_epoch1_wmdp_mmlu` |

## MMLU 子集结果

| 算法 | Humanities | STEM | Social Sciences | Other |
|---|---:|---:|---:|---:|
| CEU | 24.21% | 21.25% | 21.71% | 23.98% |
| Grad-Ascent | 24.02% | 23.66% | 23.20% | 25.84% |
| Grad-Diff | 77.41% | 82.75% | 90.32% | 85.29% |
| NPO | 78.81% | 84.59% | **91.81%** | **87.45%** |
| PDU | 79.26% | **84.78%** | 91.39% | 87.06% |
| Pure-NPO | 72.67% | 82.87% | 88.85% | 74.09% |
| RMU | **79.49%** | 84.59% | 91.68% | 87.22% |
| SatImp | 52.99% | 72.76% | 76.15% | 72.29% |
| SimNPO | 78.53% | 85.03% | 91.45% | 87.06% |
| Token-NPO | 41.72% | 35.17% | 40.53% | 31.80% |
| Token-SimNPO | 78.28% | 84.62% | 91.10% | 86.77% |
| UNDIAL | 78.07% | 83.25% | 90.61% | 87.25% |
| WGA | 30.71% | 39.42% | 33.83% | 33.22% |

## Summary

- **遗忘最强**：Grad-Ascent 的 WMDP-Bio 最低（24.35%），CEU 紧随其后（24.67%）；两者的 MMLU 仅 24.16% / 22.95%，说明通用能力严重退化。
- **能力保持最好**：RMU 的 MMLU 最高（85.02%），同时 WMDP-Bio 降至 52.00%，是较好的综合折中。
- **综合遗忘/保持折中**：Grad-Diff 的 WMDP-Bio 为 68.89%、MMLU 为 83.18%；相比 RMU 遗忘较弱、能力保持略低。
- **NPO 与 PDU**：MMLU 保持较高（84.87% / 84.88%），但 WMDP-Bio 仍为 84.37% / 83.35%，遗忘效果很弱。
- **WGA 与 Token-NPO**：WMDP-Bio 较低，但 MMLU 分别降至 33.91% 和 37.79%，存在明显能力损失。
- **SatImp**：WMDP-Bio 为 59.62%，但 MMLU 降至 66.77%，整体折中不如 RMU。
- **SimNPO**：WMDP-Bio 为 83.90%、MMLU 为 84.71%，能力保持较好但遗忘效果较弱。
- **Token-SimNPO**：WMDP-Bio 为 81.23%、MMLU 为 84.39%，在保持通用能力的同时只有有限遗忘。
- **UNDIAL**：WMDP-Bio 为 82.09%、MMLU 为 84.01%，通用能力保持较好但遗忘效果较弱。

## 训练配置

V3 各算法均使用 Megatron 后端、1 epoch、8 GPU，并采用相同的模型结构、数据和
并行配置；主要差异在算法目标和学习率。

### 公共配置

| 配置项 | 取值 |
|---|---|
| Base model | Intern-S2-base05（40 层，256 routed experts/层，top-K=8） |
| Forget / retain 数据 | WMDP Bio / FineWeb Edu 5B（seed 42）；NPO、Pure-NPO 无 retain 数据 |
| 训练序列长度 | 4096 |
| Epoch / batch | 1 / global batch 4 / micro batch 4 |
| Optimizer | Adam，betas `(0.9, 0.999)`，epsilon `1e-8` |
| LR scheduler | cosine，warmup ratio 0.03 |
| Weight decay / clipping | 0.0 / **默认开启，`gradient_clip=1.0`** |
| Seed / dtype | 42 / BF16 |
| Trainable parameters | full model |
| TP / PP / EP | 4 / 2 / 4 |
| ETP / CP / DP | 1 / 1 / 1 |
| Sequence parallel / distributed optimizer | 开启 / 开启 |
| Reference model / Grouped GEMM | 同分配且冻结 / 关闭 |

说明：本次训练没有关闭 gradient clipping，沿用配置默认值 `gradient_clip=1.0`。
UNDIAL 的结果来自显存优化重试配置 `undial_mbs1_ga4.yaml`，其 micro batch size
为 1、global batch size 为 4（4 个 microbatch 累计）；其他算法使用 micro batch
size 为 4。

### 算法特定配置

| 算法 | 学习率 | 关键参数 |
|---|---:|---|
| CEU | `1e-6` | `gamma=1.0`，忽略前 1 token |
| Grad-Ascent | `1e-6` | `gamma=1.0` |
| Grad-Diff | `1e-6` | `gamma=1.0`，`alpha=1.0`，retain loss=NLL |
| NPO | `1e-6` | `beta=0.1`，`gamma=1.0` |
| PDU | `1e-6` | `alpha=100`，`retain_loss_eps=0.3`，dual step=5 |
| Pure-NPO | `1e-6` | `beta=0.1`，`gamma=1.0`，无 retain 数据 |
| RMU | `1e-5` | target layer=9；train layers=7,8,9；`alpha=350`；steering=300 |
| SatImp | `4e-6` | `beta1=2.0`，`beta2=0.1`，`gamma=1.0` |
| SimNPO | `1e-6` | `beta=0.1`，`delta=0`，`gamma=1.0`，`alpha=0.1`，retain loss=NLL |
| Token-NPO | `1e-6` | `beta=0.1`，`gamma=1.0` |
| Token-SimNPO | `1e-6` | `beta=0.1`，`delta=0`，`gamma=1.0`，`alpha=0.1`，retain loss=NLL |
| UNDIAL | `1e-6` | `beta=3.0`，`gamma=1.0`，`alpha=0.0`，reference logit window=1 |
| WGA | `1e-6` | `beta=0.5`，`gamma=1.0`，`alpha=0.0` |

## 评测环境

- 模型来源：Interns2 Megatron V3 各算法 checkpoint-epoch-1
- 评测框架：lm-eval 0.4.11
- Transformers：5.5.4
- 模型来源类型：Hugging Face (`model_source=hf`)
- 最大序列长度：262,144
- 任务：WMDP（含 WMDP-Bio/Chem/Cyber）与 MMLU

原始详细结果位于各算法目录下的 `result.json`。
