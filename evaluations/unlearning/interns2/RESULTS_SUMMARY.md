# Intern-S2 遗忘评测汇总

评测目标是降低 WMDP-Bio，同时尽量保持 MMLU。数值均为准确率：WMDP-Bio
越低越好，MMLU 越高越好。

| 模型 | WMDP-Bio ↓ | MMLU ↑ |
| --- | ---: | ---: |
| Base | 83.66% ± 1.04% | 85.15% ± 0.29% |

## 版本概览

| 版本 | 主要变化 | 已评测内容 |
| --- | --- | --- |
| [V1](interns2-Megatron-V1/RESULTS_SUMMARY.md) | 1024-token chunk、global batch 128、expert-only、constant LR | 11 个算法的 step 300 和 epoch 1 |
| [V2](interns2-Megatron-V2/V2_CHANGES.md) | 延续 V1 数据与训练方式，重点调 WGA、SatImp；另有 6-GPU RMU | WGA、SatImp、RMU |
| [V3](interns2-Megatron-V3/) | 4096-token XTuner soft-pack、global batch 4、full trainable、gradient clip 1.0、warmup + cosine | GradDiff、NPO、PureNPO、RMU 的 epoch 1 |

三个版本同时改变了数据组织、batch、可训练参数范围或调度策略，因此结果适合比较最终效果，
不应视为严格的单变量消融。

## V1：第一轮完整算法对比

V1 用统一配置横向比较 11 个算法。GradAscent 遗忘最强，但 MMLU 严重下降；
在兼顾两项指标时，RMU 和 GradDiff 更有代表性。PDU、SatImp、WGA、UNDIAL
在该配置下几乎没有产生有效遗忘。

| 代表结果 | Checkpoint | WMDP-Bio ↓ | MMLU ↑ |
| --- | --- | ---: | ---: |
| GradAscent | epoch 1 | 24.67% | 23.53% |
| GradDiff | epoch 1 | 69.21% | 84.26% |
| RMU | epoch 1 | 68.26% | 84.85% |
| TokenSimNPO | epoch 1 | 79.26% | 84.16% |

完整的 22 组 checkpoint 指标和逐算法配置见
[V1 结果](interns2-Megatron-V1/RESULTS_SUMMARY.md)。

## V2：定向调参与独立 RMU

V2 不再做全算法扫描，主要验证 WGA、SatImp 和 RMU。WGA 原始配置出现破坏性退化，
调成 forget-only、`beta=0.5`、`lr=1e-6` 后明显恢复；SatImp 增强了遗忘，
但 MMLU 损失仍较大。6-GPU RMU 在保持 MMLU 的同时，将 WMDP-Bio 降至 37.55%。

| 结果 | Checkpoint | WMDP-Bio ↓ | MMLU ↑ |
| --- | --- | ---: | ---: |
| WGA 原始配置 | epoch 1 | 28.12% | 26.39% |
| WGA 调参后 | step 300 | 76.20% | 82.32% |
| SatImp 调参后 | epoch 1 | 66.85% | 76.13% |
| RMU，6 GPU | epoch 1 | 37.55% | 84.67% |

全部 V2 checkpoint 见 [V2 结果](interns2-Megatron-V2/V2_CHANGES.md)。

## V3：当前统一训练配置

V3 切换到 4096-token XTuner soft-pack，并统一使用 micro/global batch 4、
`trainable_param_mode=full`、`gradient_clip=1.0`、3% warmup 和 cosine decay。
当前已完成 4 个算法。

| 算法 | WMDP-Bio ↓ | 相对 Base | MMLU ↑ | 相对 Base |
| --- | ---: | ---: | ---: | ---: |
| GradDiff | 68.89% | -14.77 | 83.18% | -1.97 |
| NPO | 84.37% | +0.71 | 84.87% | -0.28 |
| PureNPO | 78.32% | -5.34 | 78.82% | -6.33 |
| RMU | 52.00% | -31.66 | 85.02% | -0.14 |

当前 V3 中 RMU 的综合效果最好；NPO 基本保持 MMLU，但没有产生遗忘；
PureNPO 的 MMLU 损失大于遗忘收益。原始结果在各算法的 `result.json` 中。

## RMU 跨版本参考

| 来源 | WMDP-Bio ↓ | MMLU ↑ |
| --- | ---: | ---: |
| V1 RMU，epoch 1 | 68.26% | 84.85% |
| V2 RMU，epoch 1 | 37.55% | 84.67% |
| V3 RMU，epoch 1 | 52.00% | 85.02% |
| [独立 HF/DeepSpeed RMU](lm_eval_results_rmu_wmdp_mmlu/result.json) | 37.16% | 83.43% |

V2 RMU 的遗忘最强；V3 RMU 的 MMLU 保持最好。独立 RMU 不属于 Megatron
V1/V2/V3，其对应权重仍保存在
`saves/unlearn/interns2_moe_rmu_bio_l10_fixed_wmdp_forget_fineweb_retain_cached`。
