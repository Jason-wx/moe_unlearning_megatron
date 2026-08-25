# 超参数来源

如果原方法仓库公开了 WMDP 配置，启动器就优先采用 WMDP-Bio 设置。数据路径和
特定模型的分布式设置不属于论文超参数对齐范围。

| 方法 | 启动器采用的默认值 | 主要来源 |
|---|---|---|
| GradAscent | 学习率 `5e-6`，`500` steps | SimNPO 官方 WMDP 基线配置：https://github.com/OPTML-Group/Unlearn-Simple/tree/main/WMDP/configs/unlearn/wmdp |
| GradDiff | 学习率 `5e-6`，`500` steps，gamma `1`，随机种子 `3` | https://github.com/OPTML-Group/Unlearn-Simple/blob/main/WMDP/configs/unlearn/wmdp/GradDiff.json |
| NPO | 当前纯 NPO：学习率 `1e-6`，显式 `max_grad_norm=1.0`，beta `0.1`，gamma `1`，随机种子 `0`，序列 NLL 按有效预测 token 数取平均；WMDP 原配置为未归一化 NLL、beta `0.0225`，学习率 `5e-6` | https://github.com/OPTML-Group/Unlearn-Simple/blob/main/WMDP/configs/unlearn/wmdp/NPO.json |
| SimNPO | 当前长文本适配：学习率 `9.5e-6`，token-level 目标，beta `0.1`，forget 系数 `1`，retain 系数 `0.1`，随机种子 `0`；官方 WMDP 基线为序列平均目标和 retain 系数 `1` | https://github.com/OPTML-Group/Unlearn-Simple/blob/main/WMDP/configs/unlearn/wmdp/SimNPO.json |
| UNDIAL | 全参数微调学习率 `1e-6`，强度 `3`，10 epochs | https://github.com/dong-river/LLM_unlearning |
| SatImp | 学习率 `1e-5`，beta1 `5`，beta2 `1`，forget:retain 系数比 `10:1`，5 epochs | 论文：https://arxiv.org/abs/2505.11953；官方实现：https://github.com/Puning97/SatImp-for-LLM-Unlearning |
| WGA | 学习率 `1e-5`，beta `1`，5 epochs | 论文：https://arxiv.org/abs/2502.19301；官方实现：https://github.com/tmlr-group/G-effect |
| CEU | 学习率 `1e-5`，10 epochs，忽略第一个回答 token | 论文：https://arxiv.org/abs/2503.01224；作者的 OpenUnlearning 复现脚本 |
| PDU | 学习率 `1e-5`，10 epochs，gamma `1`，初始 retain 偏好 `100`，epsilon `0.3`，对偶步长 `5`，对偶 warmup `5` epochs | 论文：https://arxiv.org/abs/2506.05314；`community/methods/PDU/run.sh` |

batch size 有意与缓存数据版 RMU 启动器对齐，而不是与论文对齐：每卡训练 batch
为 `8`，评估 batch 为 `1`，梯度累积为 `2`。8 张训练 GPU 上的有效 batch 为
`128`；GPU 7 留给参考模型时为 `112`。表中的优化器相关参数和方法专属参数仍与
论文或官方配置对齐。

所有启动器使用 `max_steps=-1`、`num_train_epochs=1` 和
`data.anchor=forget`，因此训练长度严格等于完整遍历一次 forget 缓存，retain
样本数量不决定 epoch 长度。该设置明确覆盖论文中的 `500` steps 和多 epoch
计划。PDU 论文中的 5 epoch 对偶 warmup 也被缩短为 `0`，否则在单 epoch 训练中
永远不会启用对偶更新。

这些值用于复现论文配置，不保证是 35B/26B MoE 的最优值。每个启动器都保留环境
变量覆盖入口，便于后续针对具体模型调参。
