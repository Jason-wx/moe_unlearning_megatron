# Qwen3.5-35B-A3B-Base 上的 RMU 注意点

> 本文保留早期调试过程。文中的专用 shell/RJob 文件已删除；当前 Megatron RMU
> 使用统一 experiment 加 `--algorithm rmu`，DeepSpeed 专用实现仍可通过
> `python -m qwen_moe_rmu.train` 或 `python -m gemma_moe_rmu.train` 调用。

本文记录在 `Qwen/Qwen3.5-35B-A3B-Base` 这类 MoE 模型上做 RMU 时需要注意的实现差异。

## 模型结构要点

| 项目 | 配置 |
|---|---|
| 模型 | `Qwen/Qwen3.5-35B-A3B-Base` |
| 类型 | MoE causal language model |
| 总参数量 | 约 35B |
| 激活参数量 | 约 3B |
| 语言模型层数 | 40 |
| MoE experts | 256 |
| 每 token 激活 expert | 8 routed experts + 1 shared expert |

本地 checkpoint 路径：

```text
/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/zskj-hub/models-Qwen-Qwen3.5-35B-A3B-Base
```

该模型不是普通 dense Transformer。做 RMU 时，不能直接沿用 dense 模型里的：

```yaml
model\.layers\.(5|6|7)\.mlp\.down_proj\.weight
```

已从本地 `model.safetensors.index.json` 确认：语言模型参数前缀是：

```text
model.language_model.layers.{layer_id}.*
```

所以 hook 正则应该类似：

```yaml
module_regex: model\.language_model\.layers\.9
```

而不是：

```yaml
module_regex: model\.layers\.9
```

如需重新确认，可打印实际模块名或参数名：

```python
for name, _ in model.named_parameters():
    print(name)
```

## Hook 层选择

WMDP/RMU 论文中，Zephyr 和 Mixtral 都选择约 22%-25% 深度的层做 hook。

对于 40 层 Qwen3.5 MoE，可以先尝试：

```text
hook layer l = 9 或 10
```

注意这里的 `l` 是 Hugging Face 模块名中的 **0-indexed** 编号。中文说“第 10 层”时，对应的是：

```text
model.language_model.layers.9
```

对应更新：

```text
l = 9  -> 更新 layers 7, 8, 9
l = 10 -> 更新 layers 8, 9, 10
```

该模型的 attention 类型是 3 层 `linear_attention` + 1 层 `full_attention` 循环。例如：

```text
layers 0, 1, 2   linear_attention
layer  3         full_attention
layers 4, 5, 6   linear_attention
layer  7         full_attention
layers 8, 9, 10  linear_attention
layer  11        full_attention
```

## MoE 参数更新建议

RMU 论文在 dense 模型上主要更新 hook 层附近的 MLP 参数。对应到 MoE，建议先更新这三层里的 expert MLP 参数：

```text
model.language_model.layers.(l-2|l-1|l).mlp.experts.*
model.language_model.layers.(l-2|l-1|l).mlp.shared_expert.*
```

本地 checkpoint 中，routed experts 是打包 tensor，不是逐 expert 展开的 `experts.0.*`、`experts.1.*` 形式。代表性参数名如下：

```text
model.language_model.layers.8.mlp.experts.gate_up_proj
model.language_model.layers.8.mlp.experts.down_proj
```

shared expert 是普通投影权重：

```text
model.language_model.layers.8.mlp.shared_expert.gate_proj.weight
model.language_model.layers.8.mlp.shared_expert.up_proj.weight
model.language_model.layers.8.mlp.shared_expert.down_proj.weight
```

如果选择 `l = 9`，推荐第一版正则：

```yaml
module_regex: model\.language_model\.layers\.9

trainable_params_regex:
  - model\.language_model\.layers\.(7|8|9)\.mlp\.experts\.(gate_up_proj|down_proj)
  - model\.language_model\.layers\.(7|8|9)\.mlp\.shared_expert\.(gate_proj|up_proj|down_proj)\.weight
```

如果选择 `l = 10`：

```yaml
module_regex: model\.language_model\.layers\.10

trainable_params_regex:
  - model\.language_model\.layers\.(8|9|10)\.mlp\.experts\.(gate_up_proj|down_proj)
  - model\.language_model\.layers\.(8|9|10)\.mlp\.shared_expert\.(gate_proj|up_proj|down_proj)\.weight
```

## Router 是否更新

第一版不建议更新 router。

原因：

- RMU 原论文主要更新 MLP/FFN 参数；
- router 控制 token 到 expert 的分配，更新它可能影响大量普通 token；
- router 的 top-k 路由带有离散选择，训练副作用可能更大；
- retain 能力可能更容易下降。

本地 checkpoint 中 router 相关参数名是：

```text
model.language_model.layers.8.mlp.gate.weight
model.language_model.layers.8.mlp.shared_expert_gate.weight
```

第一版建议不要加入这些正则。

建议实验顺序：

```text
1. 只更新 selected layers 的 routed expert MLP 参数
2. routed expert MLP + shared expert
3. 如果遗忘不够，再单独 ablation：expert MLP + router
```

如果加入 router，建议单独设置更小 learning rate，并重点观察 retain 和通用能力指标。

## Forget target 的强度对齐

原始 RMU 常见写法是把 forget activation 推向一个固定强度的随机向量：

```text
target = unit_random_vector * steering_coeff
```

这个做法的问题是：不同模型、不同层的 hidden state L2 norm 可能差很多，`steering_coeff=6.5` 在 Zephyr 上合适，不一定适合 Qwen MoE。

新增 Qwen MoE RMU 代码默认使用 reference model 的 forget activation 做强度对齐：

```text
target = unit_random_vector * ||h_ref||_2 * control_scale_coeff
```

其中：

- `h_ref` 是 frozen reference model 在 hook 层的 forget activation；
- `||h_ref||_2` 是每个 token 的 hidden state L2 norm；
- `control_scale_coeff=1.0` 表示随机 target 和原始 hidden state 同 L2 强度；
- `control_scale_coeff>1.0` 表示更强的遗忘推动；
- `control_scale_coeff<1.0` 表示更温和。

支持的模式：

| `control_norm_mode` | 含义 |
|---|---|
| `ref_activation_l2` | 默认；按 frozen reference model 的 forget activation L2 norm 对齐 |
| `batch_activation_l2` | 按当前模型本 batch activation 的 L2 norm 对齐 |
| `fixed` | 使用旧式固定强度 `steering_coeff` |

建议第一版使用：

```yaml
control_norm_mode: ref_activation_l2
control_scale_coeff: 1.0
```

后续可扫：

```text
control_scale_coeff = 0.5, 1.0, 1.5, 2.0
```

## 简短结论

在 `Qwen/Qwen3.5-35B-A3B-Base` 上做 RMU 时：

- hook 层可先试 `l = 9` 或 `l = 10`；
- 更新层使用 `l-2, l-1, l`；
- 参数正则需要使用 `model.language_model.layers.*` 前缀；
- 初始实验建议只更新 expert MLP 和 shared expert；
- forget target 默认用 reference activation 的 L2 norm 对齐；
- router 不建议一开始更新，除非 expert-only 效果不足。
- 正则应避免匹配 `model.visual.*` 和 `mtp.layers.*`。

## 新增适配代码

为避免影响原始训练入口和原始 RMU，本项目新增了一套独立适配：

```text
src/qwen_moe_rmu/train.py
src/qwen_moe_rmu/trainer.py
src/qwen_moe_rmu/modeling.py
configs/qwen_moe_rmu/unlearn.yaml
configs/qwen_moe_rmu/bio_l10_fixed.yaml
scripts/qwen35_moe_rmu_bio_l10_fixed.sh
```

旧的 `src/train.py`、`src/trainer/unlearn/rmu.py` 和 trainer registry 不需要修改。

## 脚本可调超参

启动脚本：

```text
scripts/qwen35_moe_rmu_bio_l10_fixed.sh
```

可通过环境变量覆盖主要超参，例如：

```bash
NUM_TRAIN_EPOCHS=1 bash scripts/qwen35_moe_rmu_bio_l10_fixed.sh
```

| 类别 | 环境变量 | 默认值 | 作用 |
|---|---|---:|---|
| 资源/分布式 | `CUDA_VISIBLE_DEVICES` | `0,1,2,3,4` | 使用哪几张 GPU |
| 资源/分布式 | `NUM_PROCESSES` | `5` | Accelerate/DeepSpeed 进程数，通常和 GPU 数一致 |
| 模型 | `MODEL_PATH` | 本地 Qwen3.5 MoE 路径 | 模型 checkpoint 路径 |
| 模型 | `MODEL_HANDLER` | `AutoModelForImageTextToText` | Transformers AutoModel 类；如本地版本映射不同可改 |
| 数据 | `WMDP_DATA_ROOT` | 本地 WMDP parquet 根目录 | forget/retain corpus 根目录 |
| 训练 | `PER_DEVICE_TRAIN_BATCH_SIZE` | `8` | 单卡 micro batch size |
| 训练 | `PER_DEVICE_EVAL_BATCH_SIZE` | `1` | 单卡 eval batch size |
| 训练 | `GRADIENT_ACCUMULATION_STEPS` | `2` | 梯度累积步数 |
| 训练 | `GRADIENT_CHECKPOINTING` | `false` | 默认关闭 gradient checkpointing，避免 MoE 反向重算不一致 |
| 训练 | `GRADIENT_CHECKPOINTING_USE_REENTRANT` | `false` | 仅在手动开启 checkpointing 时生效 |
| 训练 | `GRADIENT_CHECKPOINTING_DETERMINISM_CHECK` | `default` | 仅在手动开启 checkpointing 时生效，保留重算 metadata 校验 |
| 训练 | `NUM_TRAIN_EPOCHS` | `1` | 训练 epoch 数；当前脚本固定 `trainer.args.max_steps=-1`，训练量由 epoch 控制 |
| 训练 | `LEARNING_RATE` | `1e-5` | 学习率 |
| 训练 | `WARMUP_STEPS` | `0` | warmup steps |
| 训练 | `LOGGING_STEPS` | `1` | 日志间隔 |
| 训练 | `SAVE_TOTAL_LIMIT` | `1` | 最多保留 checkpoint 数 |
| 训练 | `SEED` | `42` | 随机种子 |
| RMU | `ALPHA` | `350.0` | retain loss 权重 |
| RMU | `GAMMA` | `1.0` | forget loss 权重 |
| RMU | `STEERING_COEFF` | `300.0` | 固定 control vector 强度 |
| RMU | `CONTROL_NORM_MODE` | `fixed` | forget target 强度模式 |
| RMU | `CONTROL_SCALE_COEFF` | `1.0` | 相对 reference hidden norm 的缩放系数 |
| RMU | `RETAIN_LOSS_TYPE` | `EMBED_DIFF` | retain loss 类型 |
| 日志/任务 | `TASK_NAME` | `qwen35_moe_rmu_bio_l10_fixed` | 输出目录名和 W&B run 名 |
| 日志/任务 | `REPORT_TO` | `wandb` | Trainer 日志后端 |
| 日志/任务 | `WANDB_PROJECT` | `Wangxxx` | W&B project 名 |
| 日志/任务 | `WANDB_ENTITY` | `jesse-jin-southeast-university` | W&B entity/team 名 |
| 日志/任务 | `WANDB_NAME` | `${TASK_NAME}` | W&B run 名 |
| 日志/任务 | `WANDB_MODE` | `offline` | W&B 模式，可改为 `online` |

当前 `bio_l10_fixed` 固定更新第 7, 8, 9 层的 routed expert MLP 和 shared expert MLP：

```yaml
module_regex: model\.language_model\.layers\.9

trainable_params_regex:
  - model\.language_model\.layers\.(7|8|9)\.mlp\.experts\.(gate_up_proj|down_proj)
  - model\.language_model\.layers\.(7|8|9)\.mlp\.shared_expert\.(gate_proj|up_proj|down_proj)\.weight
```

router 仍然不在第一版更新范围内。

### Hook 早停显存优化

RMU 当前使用 `EMBED_DIFF` retain loss 和 fixed control forget loss，二者都只依赖 `module_regex` 指定层的 activation。对于 `bio_l10_fixed`：

```text
hook: model.language_model.layers.9
trainable: layers 7, 8, 9 的 expert/shared expert MLP
```

因此训练时不需要继续计算 hook 层之后的 layers 10-39 和 lm_head。`src/qwen_moe_rmu/trainer.py` 的 `forward_with_cache()` 在 hook 层捕获 activation 后会提前停止 forward：

```text
input -> layers 0-9 -> capture layer 9 activation -> stop
```

这样梯度仍然可以从 layer 9 activation 回传到可训练的 layers 7-9，但不会为后续无关层产生额外 forward 显存峰值。

注意：如果以后把 retain loss 改成 `NLL` 或 `KL`，这些 loss 需要完整 logits，不能使用同样的早停路径。

## rlaunch 执行手册

这里记录当前镜像环境的固定注意事项。rlaunch/rjob 使用的镜像不会因为本仓库代码修改而自动更新，所以每次进入新环境后先做版本检查。

### 1. 进入仓库

```bash
cd /mnt/shared-storage-user/safethm-share/wangxiao1/Safe_MoE/open-unlearning
```

### 2. 检查环境版本

当前 rlaunch/rjob 使用镜像：

```text
registry.h.pjlab.org.cn/ailab-safer2ai-safer2ai_gpu/unlearning:unlearning-cu126-torch271-v1
```

关键依赖已经打进镜像，不需要在 rjob 里再执行 pip install：

| 组件 | 镜像内旧状态 | 现象 | 处理 |
|---|---:|---|---|
| `transformers` | `5.13.1` | 训练侧 API 较新 | 保持 |
| `accelerate` | `1.14.0` | 满足 `Trainer` 要求 | 保持 |

进入环境后可以检查：

```bash
/home/wangxiao1/softmare/miniconda3/envs/unlearning/bin/python -c "import transformers, accelerate, sentencepiece; print('transformers', transformers.__version__); print('accelerate', accelerate.__version__); print('sentencepiece', sentencepiece.__version__)"
```

期望至少满足：

```text
transformers 5.13.1
accelerate 1.14.0
sentencepiece 0.2.1
```

### 3. 当前默认训练设置

当前脚本默认：

```text
GPU: 5 卡
CUDA_VISIBLE_DEVICES: 0,1,2,3,4
NUM_PROCESSES: 5
PER_DEVICE_TRAIN_BATCH_SIZE: 8
GRADIENT_ACCUMULATION_STEPS: 2
NUM_TRAIN_EPOCHS: 1
GRADIENT_CHECKPOINTING: false
WANDB_MODE: offline
```

`GRADIENT_CHECKPOINTING=false` 是有意设置。Qwen MoE 动态路由在 checkpoint 反向重算时出现过 metadata 不一致：

```text
saved shape: [128]
recomputed shape: [0]
```

为了实验结果可信，不默认绕过该检查，而是直接关闭 checkpointing，靠增加 GPU 或降低 batch 解决显存。

### 4. rlaunch 内直接执行

默认 5 卡执行：

```bash
bash scripts/qwen35_moe_rmu_bio_l10_fixed.sh
```

显式指定 5 卡执行：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4 \
NUM_PROCESSES=5 \
bash scripts/qwen35_moe_rmu_bio_l10_fixed.sh
```

如果仍然 OOM，优先降低单卡 batch：

```bash
PER_DEVICE_TRAIN_BATCH_SIZE=8 bash scripts/qwen35_moe_rmu_bio_l10_fixed.sh
```

再不行：

```bash
PER_DEVICE_TRAIN_BATCH_SIZE=4 bash scripts/qwen35_moe_rmu_bio_l10_fixed.sh
```

### 5. rjob 提交执行

脚本已同步为 5 卡：

```bash
./scripts/rjob_rmu.sh
```

该脚本内部关键设置：

```text
--gpu=5
CUDA_VISIBLE_DEVICES=0,1,2,3,4
NUM_PROCESSES=5
```

提交后的容器会先打印版本，然后执行：

```bash
bash scripts/qwen35_moe_rmu_bio_l10_fixed.sh
```

容器内 stdout/stderr 会同时写入 rjob 平台日志和仓库本地日志文件。每次提交会生成一个带时间戳的新文件：

```text
scripts/qwen35_rjob_YYYYmmdd_HHMMSS.log
```

例如：

```bash
tail -f scripts/qwen35_rjob_20260715_210000.log
```

查看任务：

```bash
rjob get <rjob_name>
```

查看日志：

```bash
rjob logs job <rjob_name> --tail-lines 200
```

如果日志命令受代理影响，可临时取消代理：

```bash
env -u http_proxy -u https_proxy -u all_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  rjob logs job <rjob_name> --tail-lines 400
```

### 6. 监控 GPU

一秒刷新并保留颜色：

```bash
watch -n 1 -c gpustat --color
```

如果当前 `gpustat` 版本不支持 `--color`：

```bash
watch -n 1 -c gpustat --force-color
```

### 7. 常见日志说明

| 日志/报错 | 含义 | 处理 |
|---|---|---|
| `ImportError: Trainer with PyTorch requires accelerate>=1.1.0` | 镜像不是当前 v1 或环境不对 | 确认使用 `unlearning-cu126-torch271-v1` |
| `unwrap_model() got an unexpected keyword argument keep_torch_compile` | `transformers` 新、`accelerate` 旧导致 API 不匹配 | 确认使用 `unlearning-cu126-torch271-v1` |
| `torch_dtype is deprecated; use dtype` | Transformers 参数名更新 | 代码已改为 `dtype`，无需处理 |
| `use_cache=True is incompatible with gradient checkpointing` | 训练时 cache 不应开启 | 代码已在模型加载后设为 `use_cache=False` |
| `CheckpointError: Recomputed values ... different metadata` | MoE + gradient checkpointing 重算不一致 | 当前默认已关闭 `GRADIENT_CHECKPOINTING` |
| `CUDA out of memory` | 显存不足 | 先确认 checkpointing 已关闭，再降低 `PER_DEVICE_TRAIN_BATCH_SIZE` |
| `libtinfo.so.6: no version information available` | conda `libtinfo` 与系统 bash 的 warning | 通常不影响训练 |
| `pydantic UnsupportedFieldAttributeWarning` | 依赖包 warning | 通常不影响训练 |

### 8. W&B offline 日志

当前默认：

```text
WANDB_MODE=offline
REPORT_TO=wandb
```

训练结束后，日志会提示类似：

```bash
wandb sync /mnt/shared-storage-user/safethm-share/wangxiao1/Safe_MoE/open-unlearning/wandb/offline-run-xxxx
```

需要上传时按提示执行 `wandb sync`。
