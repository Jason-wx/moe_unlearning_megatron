# 配置目录

公开配置分为训练和数据处理两类：

- `training/deepspeed/interns2/`：仅维护 Intern-S2 的 DeepSpeed 框架示例。
- `training/megatron/<model>/`：正式实验配置，按模型分目录扩展。
- `data_processing/`：离线 token cache 和 XTuner soft-pack 构建配置。

`internal/deepspeed/` 是 Hydra 机械兼容层，不保存模型路径、cache 路径、算法超参、
优化器超参、并行布局、输出目录或 W&B 实验设置，不应作为训练入口修改。

## 训练配置

每个 `training/*.yaml` 按顺序完整包含：

1. `model`：模型、tokenizer、handler、专家数量。
2. `data`：预处理 cache、长度、顺序和 dataset handler。
3. `algorithm`：算法名称及全部算法专属参数。
4. `training`：batch、优化器、训练长度、seed、保存、输出与 W&B。
5. `backends`：只包含当前文件所属后端的完整分布式设置。

每个“后端/模型”目录对每种算法只保留一个 YAML，公开配置统一以 8 卡为默认布局；
临时消融通过 CLI 覆盖参数，不再复制为 `v2`、`6gpu`、`mbs8` 或 `step300` 配置。

```bash
scripts/inspect.sh --config configs/training/megatron/interns2/pure_npo.yaml
scripts/train_deepspeed.sh configs/training/deepspeed/interns2/pure_npo.yaml --dry-run
scripts/train_megatron.sh configs/training/megatron/interns2/pure_npo.yaml --dry-run
```

创建新实验时复制最接近的完整文件，然后只修改新文件。CLI 仅保留临时调试覆盖；正式
批量训练不覆盖 YAML 中的 model、dataset、algorithm、训练长度、保存策略或输出名称。

### Intern-S2 Megatron 固定训练基线

`training/megatron/interns2/` 中所有配置统一使用 XTuner soft-pack 4096 cache：

- forget：`${OU_CACHE_ROOT}/Xtuner/interns2_wmdp_bio_softpack4096/forget`，22,864 packs；
- retain：`${OU_CACHE_ROOT}/Xtuner/fineweb_edu_5b_interns2_softpack4096_seed42`，只由需要
  retain loss 的算法加载；
- `micro_batch_size=4`、`global_batch_size=4`、`sequence_length=4096`；
- `shuffle=false`、`seed=42`、`num_train_epochs=1`、`train_steps=-1`。
- 所有训练配置统一使用全局 L2 `gradient_clip=1.0`。
- 学习率在前 3% optimizer steps 从 0 线性 warmup，随后 cosine 衰减到 0；衰减终点
  自动对齐实际训练总步数。
- MCore 配置名为 `optimizer=adam`，但默认启用 decoupled weight decay，底层语义为
  AdamW；当前 `weight_decay=0.0` 时二者数值等价。
- 权重输出到 `backends.megatron.output_root/training.output_name`，每份 YAML 使用独立
  子目录。
- 所有算法使用 `trainable_param_mode=full`。RMU 仍受 `train_layers` 限制，只在指定
  层内训练 attention、experts 和 router；不会扩展到其他层或 block 外参数。

当前 TP4/PP2 的 8 卡布局中 data parallel size 为 1，因此每个 optimizer step 消耗
4 个 forget packs，一轮自动计算为 `22864 / 4 = 5716` steps。需要 retain 的算法会在
每个 step 额外配对 4 个 retain packs；这不改变 forget-anchor 的 epoch 长度。

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
语义识别并保持冻结。

## 后端能力

| 算法 | DeepSpeed | Megatron |
| --- | --- | --- |
| GradAscent / GradDiff | 是 | 是 |
| NPO / PureNPO / TokenNPO | 是 | 是 |
| SimNPO / TokenSimNPO / WGA / SatImp | 是 | 是 |
| CEU / PDU / UNDIAL | 是 | 是 |
| RMU | 否 | 是 |
