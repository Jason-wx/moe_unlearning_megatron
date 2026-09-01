# Intern-S2 本轮遗忘结果总结

更新时间：2026-08-28

- 遗忘目标：WMDP-Bio；准确率越低，表示遗忘越强。
- 通用能力：MMLU 总体准确率；准确率越高，表示能力保持越好。
- 数值均为 `accuracy ± standard error`，并换算为百分比。

## 本轮训练超参数

以下参数对应本目录中的 Intern-S2 Megatron V1（all-11）结果。它们来自实际成功运行的 RJob 日志、共享实验配置和 Megatron objective 实现，不包含后续单独运行的 6-GPU、学习率 `1e-5` RMU 实验。

### 共同配置

| 参数 | 实际值 |
| --- | --- |
| 基础模型 | `Intern-S2-base05`，256 experts |
| 训练后端 | Megatron-Core / MCore-Bridge，8 × H200 |
| 并行布局 | TP=4，PP=2，EP=4，CP=1，ETP=1，DP=1 |
| Forget 数据 | WMDP-Bio token cache |
| Retain 数据 | FineWeb-Edu token cache；仅需要 retain 的算法加载 |
| 序列长度 | 1024 |
| Global batch size | 128 |
| 训练长度 | 1344 optimizer steps，1 epoch；消费 172,032 个完整 forget chunks |
| 优化器 | Adam，`lr=1e-6`，`betas=(0.9, 0.999)`，`eps=1e-8` |
| Weight decay | 0 |
| LR schedule | constant，zero warmup |
| Gradient clipping | 0，即关闭 |
| 数值精度 | BF16 模型计算，FP32 optimizer / gradient reduction |
| Seed | 42 |
| Checkpoint | step 300、epoch 1 |
| 可训练参数 | 全层 routed experts 和 shared experts；router/gate 与其他非 expert 参数冻结 |
| Reference model | 需要 reference 的算法使用与 student 同源、同 TP/PP/EP 布局的冻结模型 |

### 逐算法配置

| 算法 | Objective 与算法超参数 | Retain | 配置 micro batch | 每步 microbatches |
| --- | --- | --- | ---: | ---: |
| GradAscent | token mean gradient ascent；`gamma=1` | 否 | 16 | 8 |
| GradDiff | `gamma=1, alpha=1`；retain=NLL | 是 | 8 | 16 |
| PureNPO | sequence-level、length-normalized；`beta=0.1, gamma=1` | 否 | 16 | 8 |
| TokenNPO | token-level reference NPO；`beta=0.1, gamma=1` | 否 | 16 | 8 |
| TokenSimNPO | token-level；`beta=0.1, delta=0, gamma=1, alpha=0.1`；retain=NLL | 是 | 8 | 16 |
| WGA | `beta=1, gamma=1, alpha=1`；retain=NLL | 是 | 8 | 16 |
| SatImp | `beta1=5, beta2=1, gamma=1, alpha=0.1`；retain=NLL | 是 | 8 | 16 |
| CEU | `gamma=1, ignore_first_n_tokens=1` | 否 | 16 | 8 |
| PDU | `gamma=1`；初始 dual multiplier `alpha=100`；`retain_loss_eps=0.3`；`dual_step_size=5`；primal-dual 开启 | 是 | 8 | 16 |
| UNDIAL | `beta=3, gamma=1, alpha=0`；reference logit window=1 | 否 | 8 | 16 |
| RMU | `gamma=1, alpha=350`；`target_layer=9`；`steering_coeff=300`；`control_norm_mode=fixed` | 是 | 16 | 8 |

对于配置 micro batch 为 8 的 retain 算法，每个 microbatch 会拼接 8 条 forget 和 8 条 retain，因此模型实际接收 16 条序列；配置层的 global batch size 仍为 128。UNDIAL 的 `alpha=0`，不加载 retain 数据，其 mbs=8 是为了限制 full-vocabulary reference logits 的峰值显存。

### RMU 实际生效范围说明

RMU 算法配置包含 `train_layers=[7, 8, 9]`（零基层号，第 8–10 层），但本轮 all-11 继承共享 profile 的 `trainable_param_mode=moe_experts`。当前实现只有在 `rmu_experts` 模式下才使用 `train_layers` 限制 `requires_grad`，所以本轮运行实际将全层 routed/shared expert 参数标记为可训练；RMU loss 仍在第 10 层（零基 `target_layer=9`）计算。

后续单独运行的 6-GPU RMU 使用 `trainable_param_mode=rmu_experts`、`lr=1e-5` 和不同并行布局，其结果与超参数不属于本目录这一轮对比。

## 已完成结果

| 算法 | Checkpoint | WMDP-Bio ↓ | MMLU ↑ |
| --- | --- | ---: | ---: |
| Base | 未训练 | 83.66% ± 1.04% | 85.15% ± 0.29% |
| CEU | step 300 | 84.29% ± 1.02% | 85.14% ± 0.29% |
| CEU | epoch 1 | 77.06% ± 1.18% | 79.97% ± 0.32% |
| GradAscent | step 300 | 37.08% ± 1.35% | 47.42% ± 0.39% |
| GradAscent | epoch 1 | 24.67% ± 1.21% | 23.53% ± 0.36% |
| GradDiff | step 300 | 77.14% ± 1.18% | 84.72% ± 0.29% |
| GradDiff | epoch 1 | 69.21% ± 1.29% | 84.26% ± 0.29% |
| PDU | step 300 | 83.66% ± 1.04% | 84.95% ± 0.29% |
| PDU | epoch 1 | 83.35% ± 1.04% | 84.95% ± 0.29% |
| PureNPO | step 300 | 78.79% ± 1.15% | 79.47% ± 0.31% |
| PureNPO | epoch 1 | 39.28% ± 1.37% | 61.22% ± 0.37% |
| RMU | step 300 | 71.17% ± 1.27% | 85.04% ± 0.29% |
| RMU | epoch 1 | 68.26% ± 1.31% | 84.85% ± 0.29% |
| SatImp | step 300 | 83.50% ± 1.04% | 84.95% ± 0.29% |
| SatImp | epoch 1 | 83.74% ± 1.03% | 84.72% ± 0.29% |
| TokenNPO | step 300 | 72.58% ± 1.25% | 69.79% ± 0.35% |
| TokenNPO | epoch 1 | 76.20% ± 1.19% | 69.12% ± 0.38% |
| TokenSimNPO | step 300 | 82.25% ± 1.07% | 84.70% ± 0.29% |
| TokenSimNPO | epoch 1 | 79.26% ± 1.14% | 84.16% ± 0.30% |
| UNDIAL | step 300 | 83.11% ± 1.05% | 84.52% ± 0.29% |
| UNDIAL | epoch 1 | 82.48% ± 1.07% | 84.28% ± 0.29% |
| WGA | step 300 | 83.27% ± 1.05% | 84.92% ± 0.29% |
| WGA | epoch 1 | 83.50% ± 1.04% | 84.57% ± 0.29% |

## 当前观察

- 未训练 Base 模型的 WMDP-Bio 为 83.66%，MMLU 为 85.15%；各算法应相对这一行判断遗忘强度和能力损失。
- GradAscent 的遗忘最强，但 MMLU 同时严重下降，属于破坏性遗忘。
- GradDiff 在 epoch 1 将 WMDP-Bio 降至 69.21%，同时将 MMLU 保持在 84.26%，遗忘强度和能力保持明显优于本轮 CEU。
- PDU 两个 checkpoint 的 MMLU 都保持在 84.95%，但 WMDP-Bio 与 Base 基本相同，当前配置没有产生有效遗忘。
- PureNPO 在 epoch 1 获得较强遗忘，但 MMLU 下降明显。
- RMU 的 MMLU 基本保持在 85%，但当前 WMDP-Bio 降幅相对有限。
- SatImp 两个 checkpoint 的 WMDP-Bio 都与 Base 接近，MMLU 基本保持，当前配置没有产生明显遗忘。
- CEU 在 step 300 基本保持 MMLU，训练到 epoch 1 后遗忘增强，同时通用能力开始下降。
- TokenNPO 从 step 300 训练到 epoch 1 后，WMDP-Bio 反而升高，MMLU 略微下降；当前结果不支持继续延长训练。
- TokenSimNPO 训练到 epoch 1 后 WMDP-Bio 比 Base 低约 4.40 个百分点，MMLU 下降约 0.99 个百分点，属于温和遗忘。
- UNDIAL 在 step 300 和 epoch 1 都基本保持 MMLU，但 WMDP-Bio 仅比 Base 低约 0.55–1.18 个百分点，当前遗忘效果很弱。
- WGA 两个 checkpoint 的 WMDP-Bio 都与 Base 接近，当前配置几乎没有产生有效遗忘。

## 完成状态

本轮 11 个算法均已补齐 step 300 和 epoch 1 评测结果。对应原始数据为各算法目录中的
`step300.json` 和 `epoch1.json`。
