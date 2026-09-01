# 实验进展

更新时间：2026-09-01

本页只记录正式实验的完成情况。运行方法见 [Runbook](../RUNBOOK.md)，详细指标见
[评测结果](../evaluations/README.md)。

## 当前进展

| 项目 | 状态 | 结果 |
| --- | --- | --- |
| 三个基础模型评测 | 已完成 | [Gemma 4、Intern-S2、Qwen3.5 MoE](../evaluations/results/baselines/) |
| 三个模型 RMU 对比 | 已完成 | [RMU 汇总](../evaluations/results/rmu/README.md) |
| Intern-S2 Megatron V1 | 已完成 | 11 个算法，step 300 和 epoch 1 均已评测 |
| Intern-S2 Megatron V2 | 已完成 | WGA、SatImp 和 6-GPU RMU 调参结果 |
| Intern-S2 Megatron V3 | 进行中 | 已完成 GradDiff、NPO、PureNPO、RMU |

Intern-S2 三个版本的配置变化、完整数据和横向对比见
[Intern-S2 结果汇总](../evaluations/unlearning/interns2/RESULTS_SUMMARY.md)。

## V3 状态

V3 是当前统一配置：4096-token XTuner soft-pack、micro/global batch 4、
`trainable_param_mode=full`、`gradient_clip=1.0`、3% warmup 后 cosine decay，
共训练 1 epoch。

| 状态 | 算法 |
| --- | --- |
| 已有结果 | GradDiff、NPO、PureNPO、RMU |
| 尚未收录结果 | CEU、GradAscent、PDU、SatImp、SimNPO、TokenNPO、TokenSimNPO、UNDIAL、WGA |

当前 V3 中 RMU 综合表现最好：WMDP-Bio 为 52.00%，MMLU 为 85.02%。不同版本
同时改变了数据组织、batch、可训练参数范围和学习率调度，不能作为严格的单变量消融。

## 维护方式

完成新实验后：

1. 将原始 `result.json` 放入 `evaluations/` 对应目录。
2. 更新该版本的结果汇总。
3. 更新本页的完成状态，不在 `docs/` 新建临时实验日志。
