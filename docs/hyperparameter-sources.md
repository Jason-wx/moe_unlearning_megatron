# 早期 Intern-S2 V3 配置的超参数来源

> 本页记录早期配置 `configs/training/megatron/interns2/*.yaml`（V3 统一基线）的参数来源，
> 仅作历史参考；这些早期配置只保存在集群存储上，不包含在 Git 仓库中。
> **最终实验的配置**在 `configs/training/megatron/final-training-configs/`，
> 其 batch（global batch 32）、训练长度（bio 3 epochs、cyber 10 epochs）、warmup（5%）和部分算法的学习率
> 均与本页不同，以各配置文件及其注释为准。

`configs/training/megatron/interns2/` 固定一版经验超参数。方法参数和 peak LR
优先采用已完成的 Intern-S2 实验；数据、batch、训练长度、可训练参数范围和 LR
调度采用 V3 统一基线。

## 已迁移的经验参数

| 方法 | 当前方法参数 | Peak LR | 依据 |
| --- | --- | ---: | --- |
| GradAscent | `gamma=1` | `1e-6` | Megatron V1 |
| GradDiff | `gamma=1, alpha=1` | `1e-6` | V1：epoch 1 WMDP-Bio `69.21%`，MMLU `84.26%` |
| NPO | `beta=0.1, gamma=1` | `1e-6` | 与已测 PureNPO 的 NPO 家族参数对齐 |
| PureNPO | `beta=0.1, gamma=1` | `1e-6` | Megatron V1 |
| TokenNPO | `beta=0.1, gamma=1` | `1e-6` | Megatron V1 |
| SimNPO | `beta=0.1, delta=0, gamma=1, alpha=0.1` | `1e-6` | 与已测 TokenSimNPO 的 SimNPO 家族参数对齐 |
| TokenSimNPO | `beta=0.1, delta=0, gamma=1, alpha=0.1` | `1e-6` | Megatron V1 |
| WGA | `beta=0.5, gamma=1, alpha=0` | `1e-6` | V2 最佳：WMDP-Bio `76.20%`，MMLU `82.32%` |
| SatImp | `beta1=2, beta2=0.1, gamma=1, alpha=0` | `4e-6` | V2 最佳：WMDP-Bio `66.85%`，MMLU `76.13%` |
| CEU | `gamma=1, ignore_first_n_tokens=1` | `1e-6` | Megatron V1 |
| PDU | `gamma=1, alpha=100, retain_loss_eps=0.3, dual_step_size=5` | `1e-6` | Megatron V1 |
| UNDIAL | `beta=3, gamma=1, alpha=0` | `1e-6` | Megatron V1 |
| RMU | `gamma=1, alpha=350, target_layer=9, train_layers=[7,8,9], steering_coeff=300` | `1e-5` | 独立 RMU：epoch 1 WMDP-Bio `37.55%`，MMLU `84.67%` |

Base 模型为 WMDP-Bio `83.66%`、MMLU `85.15%`。完整原始结果保存在：

- `evaluations/unlearning/interns2/interns2-Megatron-V1/`
- `evaluations/unlearning/interns2/interns2-Megatron-V2/`
- `evaluations/unlearning/interns2/interns2-Megatron-V2/lm_eval_results_rmu_epoch1_wmdp_mmlu/`

## v3 统一覆盖项

上述历史实验使用 1024-token cache、global batch 128、expert-only、constant LR，部分
实验还使用 6 卡布局。当前 v3 改为 soft-pack 4096、micro/global batch 4、8 卡、
`trainable_param_mode=full`、1 epoch、梯度裁剪 1，以及 3% warmup 后 cosine 衰减。
因此这里只迁移方法参数和 peak LR；历史 step 数、micro-batch、并行布局与 scheduler
不直接迁移，v3 结果不能视为历史分数的严格复现。
