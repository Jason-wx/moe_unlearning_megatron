# Unlearning 算法调试与改进记录

本文档用于持续记录各 unlearning 算法在本项目中的实现调整、训练配置、评测结果和结论。
当前只有 NPO 和 SimNPO 完成了两轮可直接对比的 Intern-S2 正式训练与评测；其他算法
暂不根据单轮结果推断改进效果。

## 统一实验背景

- 模型：`Intern-S2-base05`
- 遗忘目标：WMDP-Bio
- 训练数据：预分词 WMDP-Bio forget cache，共 172,137 个 chunks
- 历史正式训练后端：Accelerate + DeepSpeed ZeRO-3
- 历史正式训练长度：1 个 forget epoch，1,345 optimizer steps
- 历史有效 global batch：128
- 可训练范围：所有层的 routed/shared experts，共 32.338B / 35.120B 参数，约 92.08%
- 评测：WMDP-Bio accuracy 越低越好，MMLU accuracy 越高越好

当前目录中没有未执行 unlearning 的基础模型评测结果。因此下文只比较同一算法两轮
实验之间的变化，不将结果表述为相对基础模型的绝对下降。

## 结果总览

| 算法演进 | 第一轮 WMDP-Bio ↓ | 第二轮 WMDP-Bio ↓ | 变化 | 第一轮 MMLU ↑ | 第二轮 MMLU ↑ | 变化 |
|---|---:|---:|---:|---:|---:|---:|
| NPO → PureNPO | 82.56% | 38.88% | -43.68 pp | 84.89% | 62.04% | -22.85 pp |
| SimNPO → TokenSimNPO | 81.30% | 74.39% | -6.91 pp | 83.88% | 84.16% | +0.28 pp |

## NPO：NPO+retain → PureNPO

### 第一轮：旧 NPO

旧实验权重名为 `Intern-S2-base05_NPO_wmdp_bio`。虽然命名为 NPO，实际目标是
NPO forget loss 与 retain NLL 的组合：

```text
L_old = gamma * L_NPO_sum(forget) + alpha * L_NLL(retain)
```

关键配置：

- `beta=0.1`
- `gamma=1.0`
- `alpha=1.0`
- `retain_loss_type=NLL`
- 学习率 `5e-6`
- forget/reference 序列 NLL 对有效 token 直接求和，不做长度归一化
- 每个 forget 样本配对一个 FineWeb-Edu retain 样本

正式训练结果：

- WMDP-Bio：82.56% ± 1.06%
- MMLU：84.89% ± 0.29%
- 训练时间：35,909 秒，约 9 小时 58 分钟
- 现象：通用能力保持良好，但目标遗忘很弱

### 第二轮：PureNPO

新实验权重名为 `Intern-S2-base05_PureNPO_wmdp_bio`。本轮做了以下改变：

1. 完全删除 retain 数据和 retain NLL，强制 `alpha=0`。
2. NPO 仍是 sequence-level objective，但序列 NLL 从 token 求和改为对有效预测 token
   取平均，即 length-normalized sequence NLL。
3. 学习率从 `5e-6` 降到 `1e-6`。
4. 显式设置 `max_grad_norm=1.0`，缓解前期梯度尖峰。
5. reference 从旧的独立推理服务改为冻结模型与 student 一起在 8 卡上做 ZeRO-3
   分片；这是工程调整，不改变目标函数。

PureNPO 目标为：

```text
mean_nll = sum(valid_token_nll) / valid_token_count
delta_nll = mean_nll_student - mean_nll_reference
L_pure_npo = -2 / beta * log_sigmoid(beta * delta_nll)
```

正式训练结果：

- WMDP-Bio：38.88% ± 1.37%
- MMLU：62.04% ± 0.37%
- 训练时间：17,682 秒，约 4 小时 55 分钟
- 训练 loss 从初期约 13.81 下降到后期接近 0，表明大量 forget 样本进入 NPO
  饱和区

### NPO 阶段结论

- 删除 retain 约束显著增强了遗忘，但造成严重的通用能力损失。
- length normalization 解决了 1024-token 长序列之间尺度不一致的问题，但不能提供
  能力保持约束。
- 第二轮学习率更低且带梯度裁剪，因此遗忘增强不能归因于更激进的优化器配置；主要
  原因是删除 retain NLL。
- PureNPO 适合作为强遗忘基线，但当前配置不是理想的 WMDP/MMLU 平衡点。
- 后续应优先测试有限 retain 权重、减少训练 steps 或分阶段评测，而不是直接增大学习率。

### 第三版候选：Megatron TokenPureNPO

在保留现有 sequence-level PureNPO 的前提下，Megatron 后端新增独立算法名
`token_npo`。本轮只改变 sigmoid 与 token 聚合的先后顺序：

```text
token_delta[t] = NLL_student[t] - NLL_reference[t]
token_loss[t] = -2 / beta * log_sigmoid(beta * token_delta[t])
loss = mean(token_loss[t] over valid tokens)
```

与 sequence-level PureNPO 相比，以下条件保持不变：

- 仍然是 forget-only，不增加 retain NLL。
- student/reference 使用相同 token mask 和 next-token 对齐。
- reference 仍冻结，并与 student 使用 TP=4、PP=2、CP=1、EP=4、ETP=1。
- `beta=0.1`、`gamma=1.0`，并继承当前 Megatron NPO 的学习率 `5e-6` 和
  `global_batch_size=32`。
- reference pipeline 仍只缓存 `[microbatch, sequence]` 的逐 token NLL，不缓存完整
  vocabulary logits。

新增监控指标：

- `current_token_nll`
- `reference_token_nll`
- `token_npo_logit`
- `token_gradient_weight`

当前状态：38 个 Megatron 离线数学/配置测试全部通过；尚未进行真实 GPU 训练、保存
checkpoint 或 WMDP/MMLU 评测，因此不能将它记为第三轮性能结果。

配置入口：

- `configs/algorithms/token_npo.yaml`
- `configs/experiments/interns2_pure_npo.yaml --algorithm token_npo`
- Qwen3.5/Gemma 4 使用各自 experiment 并追加同一 `--algorithm token_npo`

相关实现与说明：

- `src/trainer/unlearn/npo.py`
- `src/trainer/utils.py::compute_dpo_loss`
- `configs/algorithms/pure_npo.yaml`
- `docs/hyperparameter-sources.md`

## SimNPO：sequence-level → TokenSimNPO

### 第一轮：旧 SimNPO

旧实验权重名为 `Intern-S2-base05_SimNPO_wmdp_bio`。它先对一条序列的有效 token
NLL 取平均，再对整个 sequence 计算一次有界 SimNPO objective：

```text
sequence_mean_nll = sum(valid_token_nll) / valid_token_count
L_old_simnpo = -2 / beta * log_sigmoid(beta * (sequence_mean_nll - delta))
```

关键配置：

- sequence-level objective
- `beta=4.5`
- `gamma=0.125`，即 forget loss 系数
- `alpha=1.0`，即 retain NLL 系数
- `delta=0`
- 学习率 `9.5e-6`
- 使用 FineWeb-Edu retain NLL

正式训练结果：

- WMDP-Bio：81.30% ± 1.09%
- MMLU：83.88% ± 0.30%
- 训练时间：31,650 秒，约 8 小时 48 分钟
- 现象：能力保持良好，但遗忘很弱

长序列下，一条序列中部分 token 的 NLL 被推高后，整个 sequence 的 sigmoid 权重会
一起进入饱和区，使其余仍未遗忘 token 也失去有效梯度。

### 第二轮：TokenSimNPO

新实验权重名为 `Intern-S2-base05_TokenSimNPO_wmdp_bio`。本轮同时做了以下调整：

1. 从 sequence-level 改为真正的 token-level objective：每个有效 next-token NLL
   独立经过 sigmoid，再对 token loss 求平均。
2. `beta` 从 `4.5` 降为 `0.1`。
3. forget 系数 `gamma` 从 `0.125` 提高为 `1.0`。
4. retain 系数 `alpha` 从 `1.0` 降为 `0.1`，但没有删除 retain NLL。
5. 增加 `forget_mean_nll`、`sigmoid_argument` 和 `forget_gradient_weight` 指标，用于
   观察 token 饱和过程。

TokenSimNPO 目标为：

```text
for each valid token i:
    token_loss_i = -2 / beta * log_sigmoid(beta * (token_nll_i - delta))

L_token_simnpo = gamma * mean(token_loss_i) + alpha * L_NLL(retain)
```

正式训练结果：

- WMDP-Bio：74.39% ± 1.22%
- MMLU：84.16% ± 0.30%
- 训练时间：31,245 秒，约 8 小时 41 分钟
- step 10 平均 forget gradient weight：约 0.877
- step 20 平均 forget gradient weight：约 0.217
- step 30 平均 forget gradient weight：约 0.043
- step 1340 平均 forget gradient weight：约 0.0027

### SimNPO 阶段结论

- 第二轮在 MMLU 基本不变的情况下，将 WMDP-Bio 降低 6.91 个百分点，是当前确认
  有效的温和改进。
- token-level 设计避免少数高 NLL token 让整条 1024-token sequence 一起饱和。
- 由于第二轮还同时调整了 `beta/gamma/alpha`，现有两轮实验不能把全部收益严格归因
  于 token-level 改动；若要做消融，需要固定三个系数后只切换聚合方式。
- retain NLL 仍然存在且权重为 0.1，这很可能是 MMLU 得以保持的重要原因。
- 后期 gradient weight 已接近 0，完整一 epoch 可能偏长；后续可以评测更早的中间
  checkpoint，寻找更好的遗忘/保持折中。

相关实现与说明：

- `src/trainer/unlearn/simnpo.py`
- `src/trainer/utils.py::compute_token_nll`
- `configs/algorithms/token_simnpo.yaml`
- `docs/hyperparameter-sources.md`

## Megatron 迁移状态说明

- 当前 Megatron 配置名 `npo` 对应 PureNPO：forget-only、length-normalized
  sequence-level objective，不对应旧的 NPO+retain。
- 当前 Megatron 配置名 `token_npo` 对应 TokenPureNPO：forget-only、逐 token
  student/reference NLL difference objective。
- 当前 Megatron 配置名 `simnpo` 对应 TokenSimNPO：token-level forget objective，
  同时保留权重为 0.1 的 retain NLL。
- Megatron 版本目前完成的是 Intern-S2 真实 8 卡 50-step 稳定性测试，还没有保存权重
  或执行 WMDP/MMLU 正式评测，不能把 50-step 运行结果加入上述性能对比。

### SequenceNPO 空序列边界修复（2026-08-22）

- 正式训练在第 160 个 optimizer step 遇到单条 `valid_token_count=0` 的缓存序列，
  旧版 `masked_sequence_mean` 因无法进行长度归一化而终止。
- 修复后，只过滤没有任何监督 next-token 的空序列；其余序列仍分别除以自己的
  有效 token 数，再进入 sequence-level NPO sigmoid。
- 如果整个 micro-batch 都没有有效序列，仍然抛出异常，避免静默产生无效更新。
- 新增 `valid_sequences` 和 `skipped_empty_sequences` 日志与离线 W&B 指标。

## 后续算法记录模板

### 算法名称：第一轮 → 第二轮

- 模型与数据：
- 后端与并行布局：
- 第一轮 checkpoint：
- 第二轮 checkpoint：
- 固定不变的训练条件：
- 代码改动：
- 超参数改动：
- 第一轮 WMDP/MMLU：
- 第二轮 WMDP/MMLU：
- 训练稳定性与显存：
- 能确认的结论：
- 尚不能确认的因果关系：
- 下一轮实验建议：

## 数据来源

- `evaluations/unlearning/interns2/RESULTS_SUMMARY.md`
- `evaluations/unlearning/interns2/lm_eval_results_*_wmdp_mmlu/result.json`
- 各 checkpoint 中的 `trainer_state.json`

本页曾参考 NPO、PureNPO、SimNPO 和 TokenSimNPO 的历史训练日志；这些运行日志已于
2026-08-25 按要求清理，结论以本页摘要和仍保留的原始评测 JSON 为准。
