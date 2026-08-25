# WMDP 论文中的 RMU 设置

本文整理 **The WMDP Benchmark: Measuring and Reducing Malicious Use With Unlearning** 论文中的 RMU 设置，并说明这些设置和本项目实现之间的对应关系。

## RMU 设置表

| 模型 | 总层数 | RMU hook 层 `l` | hook 位置 | 实际微调层 | 微调范围 | retain 权重 `alpha` | unlearning 系数 `c` | 梯度更新步数 |
|---|---:|---:|---:|---|---|---:|---:|---|
| `Zephyr-7B-Beta` | 32 | 7 | 约 22% 深度 | 5, 6, 7 | MLP 参数；官方代码和本项目都偏向 `mlp.down_proj.weight` 这类参数 | 1200 | 6.5 | 在 `[150, 300, 500]` 中搜索 |
| `Yi-34B-Chat` | 60 | 15 | 约 25% 深度 | 13, 14, 15 | MLP 参数 | 350 | 300 | 在 `[150, 300, 500]` 中搜索 |
| `Mixtral-8x7B-Instruct-v0.1` | 32 | 7 | 约 22% 深度 | 5, 6, 7 | MLP 参数 | 1600 | 300 | 在 `[150, 300, 500]` 中搜索 |

这里的层号通常按 **0-indexed** 的 transformer block 编号理解。比如 `layer 7` 是第 8 个 transformer block；对于 32 层模型，合法 block 编号是 `0...31`。

## 核心思想

RMU 是 **Representation Misdirection for Unlearning** 的缩写。它不是直接在最终 logits 上优化，让模型故意答错，而是在选定的 transformer 中间层 `l` 上对 activation 施加遗忘目标。

对于 forget 数据，RMU 会把选定层的 activation 推向一个随机方向：

```text
c * u
```

其中 `u` 是随机单位向量，`c` 是 unlearning coefficient，也可以理解为 steering coefficient。

对于 retain 数据，RMU 会让更新后的模型在同一层的 activation 尽量接近原始 frozen model。完整 loss 是：

```text
L = L_forget + alpha * L_retain
```

其中：

- `L_forget`：让危险知识或目标领域数据的 activation 远离原本表示，转向随机控制方向。
- `L_retain`：在 benign/general retain 数据上保持 activation 接近原始模型。
- `alpha`：控制 retain 约束的强度。
- `c`：控制 forget activation 被推得多远。

## 为什么只更新少数层

WMDP 论文报告说，只在第 `l` 层计算 loss，并只更新下面三层就足够：

```text
l - 2, l - 1, l
```

所以对于 Zephyr 和 Mixtral，`l = 7`，RMU 更新第 `5, 6, 7` 层。对于 Yi-34B，`l = 15`，RMU 更新第 `13, 14, 15` 层。

这说明 RMU 不是全参数微调。它只更新 hook 层附近的一小部分 MLP 参数，从而降低显存消耗，也减少对模型其它能力的副作用。

## 和梯度回传的关系

因为 RMU loss 是挂在第 `l` 层 activation 上的，所以这个 loss 只依赖到该层为止的计算图。第 `l` 层之后的层不参与这个 activation loss。

但这里要区分两个问题：

1. 哪些参数理论上可以从 activation loss 收到梯度？
2. 哪些参数实际是 trainable，并且会被传给 optimizer 更新？

hook 层决定 loss 在哪里计算；可训练参数过滤规则决定真正更新哪些参数。在本项目的 WMDP RMU 配置中，optimizer 被限制到：

```yaml
model\.layers\.(5|6|7)\.mlp\.down_proj\.weight
```

所以即使更前面的参数也在计算图里，最终也只有正则表达式选中的参数会被更新。

## 和本项目实现的对应关系

本项目的 RMU 实现在：

```text
src/trainer/unlearn/rmu.py
```

WMDP 实验配置在：

```text
configs/experiment/unlearn/wmdp/default.yaml
```

该配置使用：

```yaml
module_regex: model\.layers\.7
trainable_params_regex:
  - model\.layers\.(5|6|7)\.mlp\.down_proj\.weight
retain_loss_type: EMBED_DIFF
```

所以本项目遵循了同样的整体设计：

- hook 一个选定的 transformer 层；
- 在 forget corpus 上计算 activation-space forget loss；
- 在 retain corpus 上计算相对于 frozen reference model 的 activation-space retain loss；
- 只更新 hook 层附近被正则表达式选中的少量参数。

## 官方 WMDP/RMU 数据处理逻辑

官方实现的数据处理更接近 post-training/SFT 的逐样本 batch 方式，而不是预训练语料的全 corpus 拼接切块方式。

官方 `rmu/utils.py` 的 `get_data()` 逻辑可以概括为：

```text
读取 corpus jsonl
-> 每一行取 line["text"]
-> 按 min_len 过滤太短样本
-> 每 batch_size 条 text 组成一个 batch
-> 返回 List[List[str]]
```

官方默认配置里常见值：

| 参数 | 含义 | 典型值 |
|---|---|---:|
| `batch_size` | 每个 RMU batch 的原始文本条数 | `4` |
| `min_len` | 过滤太短文本 | `0` |
| `max_len` | 函数签名和配置中存在，但官方代码没有用它做截断 | `2000` |

训练时，官方 `rmu/unlearn.py` 对 batch 内文本直接调用 tokenizer：

```text
tokenizer(batch_texts, padding=True, truncation=True, max_length=...)
```

长度设置是：

| 数据 | tokenizer `max_length` |
|---|---:|
| bio forget | `512` |
| cyber forget | `768` |
| retain | `512` |

因此官方 WMDP/RMU 的核心语义是：

```text
每条原始 text 独立作为一个样本；
超过 max_length 的样本在 tokenizer 阶段截断；
不足 max_length 的样本在 batch 内 padding。
```

官方实现没有把整个 forget/retain corpus 先拼接成一个巨大字符串，也没有把连续 token stream 按 `max_length` 切成 chunk。

## 当前项目的长文处理方案

本地 WMDP bio parquet 数据的单条文本平均约 7k-8k tokens。如果直接照搬官方 `max_length=512` 的逐条截断，大部分文档内容不会进入训练。

当前 Qwen MoE RMU 配置改为 **单文档内 chunking**：

```text
每条原始文档独立 tokenizer
-> 在该文档内部按 max_length 切成多个 chunk
-> 每个 chunk 作为一条训练样本
-> 不跨文档拼接
```

这个方案和官方逐条样本处理相比，保留了长文档的更多内容；和旧 `PretrainingDataset` 的全 corpus 拼接相比，避免了不同文档跨界混在同一个 chunk 里，也避免了全量 tokenizer 导致的内存峰值。

当前实现：

```text
src/data/pretraining.py::DocumentChunkDataset
```

主要参数：

| 参数 | 含义 | 当前建议 |
|---|---|---:|
| `max_length` | 每个 chunk 的最终 token 长度上限 | `1024` 起步，可扫 `2048` |
| `stride` | 文档内部滑窗步长；空值表示不重叠 | 默认等于 `max_length` |
| `min_length` | 过滤过短 chunk | `1` |
| `drop_last` | 是否丢弃最后不足 `max_length` 的尾部 chunk | `false` |

chunk 后样本数会显著增加。当前 `bio_l10_fixed` 采用按 epoch 训练：

```text
num_train_epochs = 1
max_steps = -1
```

## 来源

- WMDP 论文：<https://arxiv.org/pdf/2403.03218>
- WMDP 官方仓库：<https://github.com/centerforaisafety/wmdp>
- 官方 RMU 数据加载：<https://github.com/centerforaisafety/wmdp/blob/main/rmu/utils.py>
- 官方 RMU 训练循环：<https://github.com/centerforaisafety/wmdp/blob/main/rmu/unlearn.py>
- Mistral 7B 架构，用于推断 Zephyr 的层数：<https://arxiv.org/pdf/2310.06825>
- Mixtral 8x7B 论文：<https://arxiv.org/pdf/2401.04088>
- Yi-34B-Chat config：<https://huggingface.co/01-ai/Yi-34B-Chat/raw/main/config.json>
