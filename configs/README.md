# 配置目录

~~~text
configs/
├── data_processing/                     离线 token cache 的构建配置
│   ├── xtuner/<model>/<dataset>.yaml
│   └── document_chunk/<model>/<dataset>.yaml
├── training/
│   ├── megatron/final-training-configs/<model>/<domain>/<算法>.yaml   最终实验
│   └── deepspeed/interns2/<算法>.yaml   DeepSpeed 后端示例（Intern-S2）
└── internal/deepspeed/                  DeepSpeed 的 Hydra 桥接层
~~~

数据处理配置见 [data_processing/README.md](data_processing/README.md)。

## 训练配置

每份训练 YAML 自包含一次训练的全部设置，按顺序包含：

1. `model`：模型路径、专家数量。
2. `data`：forget / retain cache、序列长度、是否 shuffle。
3. `algorithm`：算法名称及全部算法专属参数。
4. `training`：batch、优化器与学习率调度、训练长度、seed、保存策略、输出名称与 W&B。
5. `backends`：只包含当前文件所属后端的分布式设置；Megatron 还包含在线评测
   （`backends.megatron.evaluation`）。

创建新实验时复制最接近的完整文件，然后只修改新文件。

~~~bash
scripts/inspect.sh --config configs/training/megatron/final-training-configs/qwen/bio/GradDiff.yaml
scripts/train_megatron.sh configs/training/megatron/final-training-configs/qwen/bio/GradDiff.yaml --dry-run
scripts/train_deepspeed.sh configs/training/deepspeed/interns2/pure_npo.yaml --dry-run
~~~

### 最终实验配置

`training/megatron/final-training-configs/` 按 `{intern,qwen,gemma}/{bio,cyber}/` 分目录，
每个目录 9 个算法：GradAscent、GradDiff、PureNPO、SimNPO、SatImp、WGA、CEU、PDU、RMU
（文件名与 `algorithm.name` 的对应见 [RUNBOOK](../RUNBOOK.md)）。目录中的 `Trace.yaml`
不属于本框架，仅作记录保留，不能用本仓库脚本运行。

54 份配置的共同设置：

- forget / retain 均为第 1 步构建的 4096-token XTuner soft-pack cache，`shuffle: false`、`seed: 42`；
- `global_batch_size: 32`，TP1 / PP1 / EP8 共 8 卡，`micro_batch_size` 为 4 或 1
  （物化全词表 logits 的算法用 1 以避免 OOM，不改变 global batch）；
- cosine 学习率调度、5% warmup、`gradient_clip: 1.0`、`weight_decay: 0.0`、bf16；
- `trainable_param_mode: full`；
- 训练中在线评测 WMDP 与 MMLU。

| 数据集 | 训练长度 | 最低学习率 | 保存的 checkpoint |
| --- | --- | --- | --- |
| bio | 3 epochs | 0 | 最后一步（`save_final_checkpoint: true`） |
| cyber | 10 epochs | 1e-6 | `save_steps` 指定的一步（按在线评测曲线选定） |

算法超参和学习率见各配置文件；文件开头的注释说明了取值来源。

### 保存策略

- 权重输出到 `backends.megatron.output_root/training.output_name/checkpoint-step-<N>/`。
- `training.save_steps` 指定额外导出的 step；`training.save_interval_steps` 按固定间隔导出
  （`0` 表示不启用）；`save_final_checkpoint: true` 导出最后一步；`save_every_epoch` 在每个
  epoch 结束时导出 `checkpoint-epoch-<E>/`。

### 可训练参数模式

`training.trainable_param_mode` 只接受以下四个值：

- `attention`：只训练 attention 模块内部参数，包括 Q/K/V norm；block 间 norm 冻结。
- `expert`：训练 routed/shared experts。Gemma4 的普通 MLP 视为 shared expert。
- `router`：训练所有直接决定路由权重的参数，包括 gate、router scale 和
  per-expert scale。
- `full`：严格等于 `attention + expert + router`，不是整个模型或整个 Decoder Block
  的全参数训练。

四种模式都冻结 embedding、最终 norm、LM head、视觉/音频编码器、projector、MTP
以及其他文本 Transformer Block 外参数。Qwen3.5 和 Intern-S2 共用 Qwen3.5 MoE
适配器；Gemma4 使用独立适配器。Megatron 下融合进 projection 的 block norm 仍按
语义识别并保持冻结。RMU 还受 `train_layers` 限制，只在指定层内训练。

## DeepSpeed 桥接层

`internal/deepspeed/` 是上游 Hydra 入口（`src/train.py`）的机械兼容层，不保存模型路径、
cache 路径、算法超参、优化器超参、并行布局、输出目录或 W&B 实验设置，不应作为训练入口
修改。DeepSpeed adapter 把 `training/deepspeed/` 下的配置翻译成 Hydra override 套在这一层上。

## 后端能力

| 算法 | DeepSpeed | Megatron |
| --- | --- | --- |
| GradAscent / GradDiff | 是 | 是 |
| NPO / PureNPO / TokenNPO | 是 | 是 |
| SimNPO / TokenSimNPO / WGA / SatImp | 是 | 是 |
| CEU / PDU / UNDIAL | 是 | 是 |
| RMU | 否 | 是 |
