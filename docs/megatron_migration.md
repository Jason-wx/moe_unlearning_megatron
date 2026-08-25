# Megatron 增量迁移计划

> 历史说明：本文记录迁移过程，文中的 `configs/megatron/*`、`src/train_megatron.py`
> 和 `scripts/Megatron/*` 命令已经被统一 `ou train --backend megatron` 入口替代。
> 当前用法以根目录 README、`docs/architecture.md` 和 `configs/README.md` 为准。

## 目标与边界

本迁移在保留现有 Hugging Face Trainer + Accelerate + DeepSpeed ZeRO-3
路径的前提下，新增一套 Megatron 后端。两套后端共享数据语义、算法超参数和
Hugging Face 评测格式，但训练入口、分布式配置与 checkpoint 生命周期彼此独立。

```text
DeepSpeed（保留）:
Hydra -> Transformers Trainer -> Accelerate -> DeepSpeed ZeRO-3

Megatron（新增）:
Megatron unlearning entrypoint -> 本项目维护的 MCore training loop
                                -> MCore-Bridge -> Megatron-Core
```

当前没有 GPU，离线阶段不宣称任何模型已经完成 Megatron 训练。离线验证只覆盖
配置、数据、loss 数学、导入边界和静态检查。显存、吞吐、并行通信、checkpoint
转换一致性必须在后续 8 卡在线阶段确认。

## 技术栈选择

新增后端不复用 ms-swift 的运行时或源码：

- ms-swift 仅作为已经跑通的工程实现参考，不出现在本项目依赖、import 或启动链路中；
- 直接使用 MCore-Bridge 已有的 Qwen3.5 MoE 和 Gemma 4 模型映射；
- Intern-S2 的文本骨干与 Qwen3.5 MoE 同源，可先按 Qwen3.5 bridge 做离线配置
  检查，再在线校验权重映射；
- TP/PP/EP/CP、pipeline schedule 和 distributed optimizer 直接调用 Megatron-Core，
  unlearning 的数据组织、objective、训练循环和 checkpoint glue 由本项目维护。

不会修改相邻 ms-swift 仓库。

## 环境现状

主环境：

```text
envs/open-unlearning
Python 3.11.15
PyTorch 2.7.1+cu126
Transformers 5.5.4
Accelerate 1.14.0
DeepSpeed 0.15.4
```

为避免升级现有 Python、PyTorch 和 CUDA 依赖，新增后端固定在兼容 Python 3.11
的版本线上：Megatron-Core 0.16.1、MCore-Bridge 1.6.1 和 Transformer Engine
2.11.0。前两个包已经加入原环境；Transformer Engine 需要针对当前 CUDA/PyTorch
ABI 安装其 CUDA 核心和 PyTorch 扩展。

Megatron 依赖单独记录在 `requirements-megatron.txt`，不会并入基础 requirements，
这样原 DeepSpeed 安装入口仍可独立复现。除新增可选包外，不升级现有环境的
Python、PyTorch、Transformers、Accelerate、DeepSpeed 或 CUDA wheel。

## 分阶段计划

### 阶段 1：审计和后端无关基础层

- 固化现有 DeepSpeed 行为和结果基线；
- 建立算法能力矩阵；
- 抽离只依赖张量的 unlearning objectives；
- 增加 8 卡并行布局与环境预检；
- 使用 CPU 单元测试检查 loss 方向、mask 和长度归一化。

### 阶段 2：Megatron 最小训练闭环

- 增加独立入口，不改动 `src/train.py`；
- 建立本项目自己的 MCore model provider、pipeline forward step 和 optimizer loop；
- 从现有 `document_chunk_v1` cache 读取样本；
- 完成 HF -> MCore 加载与 MCore -> HF 保存接口；
- 首先实现 GradAscent 单模型、单次 forward 的闭环。

### 阶段 3：成对数据算法

优先将 forget 与 retain 拼成一次模型 forward，并通过样本角色 mask 分开计算
objective，避免在 pipeline schedule 内部任意调用两次模型 forward：

- GradDiff；
- SimNPO；
- WGA；
- SatImp；
- PDU。

该设计需要逐个确认与现有 DeepSpeed 版本的数值语义一致。

### 阶段 4：参考模型算法

- 参考模型与学生模型使用同一个 8 卡通信域，并采用完全相同的
  TP=4、PP=2、CP=1、EP=4、ETP=1 切分；不再采用 7+1 或单卡 reference；
- reference 始终冻结，不创建梯度和 optimizer state；双模型仍会增加每张卡的
  参数显存，因此在线阶段先验证单 batch 峰值；
- NPO：已实现同布局 frozen reference forward。reference pipeline 先计算逐 token
  NLL，学生 pipeline 再计算长度归一化的 pure-NPO objective；缓存规模只有
  `[microbatch, sequence]`，因此每个 optimizer step 保持完整 pipeline schedule；
- TokenPureNPO：以独立算法名 `token_npo` 复用 NPO 的逐 token reference NLL，
  每个有效 token 独立计算有界 NPO loss 后再求平均；不改变 reference 布局和缓存规模；
- UNDIAL：已实现同布局 frozen teacher 和 TP vocab-parallel soft-target CE。由于它
  必须保留 teacher 的词表分布，按默认 2 个 microbatch 分窗执行 reference/student
  pipeline，避免缓存整个 global batch 的教师 logits；
- DPO/AltPO：保留偏好数据路径，晚于纯文本 WMDP 算法迁移。

### 阶段 5：RMU

- 已按 MCore 的全局 `layer_number` 定位目标 Transformer layer；默认目标为
  0-based layer 9，并只训练 layers 7/8/9 的 routed/shared experts；Gemma 的对应
  dense MLP 同样遵循现有实现参与训练；
- 目标 layer 位于 PP 第一个 stage 时，通过内部 early-exit 返回目标 activation，
  后续 PP stage 只传递 pipeline tensor，不再执行目标层之后的 Transformer layers；
- 使用不改变前向 activation 数值的 auxiliary-loss autograd 节点，在目标 stage
  注入 forget steering MSE 和 retain reference-activation MSE；
- sequence-parallel activation 采用 `[local_sequence, batch, hidden]`，token mask
  按 TP rank 切片；MSE numerator/count 跨 TP 归约，与完整 activation 均值等价；
- reference 与学生使用相同 TP4/PP2/CP1/EP4/ETP1 布局，reference activation
  只在拥有目标 layer 的 PP stage 缓存，并始终冻结；
- Qwen3.5、Intern-S2 和 Gemma 4 的 RMU 配置及独立 Megatron 入口已编码，待 8 卡验证。

### 阶段 6：8 卡在线验证

每个模型和算法按以下门禁逐步扩大：

1. checkpoint 转换与单 batch forward；
2. 1 个 optimizer step，不保存权重；
3. 3--10 steps，检查 loss、grad norm、显存和吞吐；
4. 保存 MCore checkpoint 并导出 HF；
5. 比较转换前后固定 token 的 logits/NLL；
6. 完整一个 epoch；
7. 使用现有 lm-eval 路径评测 WMDP/MMLU。

任一步失败都保留 DeepSpeed 启动器作为回滚路径。

## 算法迁移优先级

| 优先级 | 算法 | 原因 |
| --- | --- | --- |
| P0 | GradAscent | 单模型、单数据、单 forward，适合验证最小闭环 |
| P0 | GradDiff | 验证 forget/retain 数据与组合 objective |
| P1 | SimNPO、CEU | 无参考模型，但需要 token 级 loss/logits |
| P1 | WGA、SatImp、PDU | 依赖 token 权重或状态更新 |
| P2 | NPO | 同布局 reference NLL 执行层已编码，待 8 卡单步验证 |
| P2 | UNDIAL | 同布局双流水线与有界 teacher-logit 窗口已编码，待 8 卡验证 |
| P3 | RMU | 入口、目标层 early-exit、SP mask 和 TP 归约已编码，待 8 卡验证 |
| P3 | DPO、AltPO | 使用不同的偏好/QA 数据路径 |

## 8 卡并行策略

项目统一使用下面这一组 8 卡布局：

```text
TP=4, PP=2, CP=1, EP=4, ETP=1, sequence_parallel=true
DP=1, expert_DP=1
```

TP 将 dense 权重和词表切到 4 卡，PP 将 Transformer layers 切为 2 个 stage，
EP 将 experts 切到 4 卡。它不是“只把 expert 分到八卡”：每个 rank 同时属于
TP、PP 和 EP 通信组。当前不使用 gradient checkpointing/activation recomputation；
长序列显存首先由 TP、PP 和 sequence parallel 分担。CP 固定为 1。

所有布局必须满足：

```text
world_size % (TP * PP * CP) == 0
world_size % (ETP * EP * PP) == 0
num_experts % EP == 0
sequence_parallel => TP > 1
```

第一轮在线验证固定该并行布局和算法超参数，只调整确有必要的 batch 参数，避免把
数值差异与并行差异混在一起。

## 兼容性原则

- DeepSpeed 文件和默认入口保持原样；
- Megatron 使用独立目录、配置和启动脚本；
- 共享 `document_chunk_v1` 数据，但启动前验证 tokenizer 元数据；
- Megatron 输出最终转换为标准 Hugging Face checkpoint，再复用当前 lm-eval；
- W&B run name 必须包含后端，例如 `Intern-S2-base05-Megatron-GradDiff-wmdp_bio`；
- 离线测试不得加载 35B 权重或假装验证 CUDA/NCCL 行为。

## 当前落地状态（离线阶段）

已新增且不影响 DeepSpeed 默认入口的内容：

- `requirements-megatron.txt`：Python 3.11 兼容的可选依赖版本；
- `configs/megatron/`：三种模型的 GradAscent、GradDiff、SimNPO、WGA、SatImp、
  CEU、PDU、NPO、UNDIAL、RMU 共 30 份 TP4/PP2/CP1/EP4/ETP1 配置；派生配置通过 `extends`
  只覆盖算法差异；
- `src/train_megatron.py`：独立 torchrun 入口；
- `src/megatron_unlearning/`：配置、并行约束、HF/MCore 配置桥、缓存 batch、
  objective、pipeline forward step 和基础 optimizer loop；
- `scripts/Megatron/run.sh`：单机 8 卡启动器；
- `scripts/Megatron/install_env.sh`：在原挂载环境中可重复安装可选依赖；
- `scripts/check_megatron_readiness.py` 和 `scripts/inspect_megatron_config.py`：
  环境、布局、缓存及模型元数据的离线检查；
- `tests/megatron/`：不需要 GPU 的 objective、mask、标签移位和并行布局测试。

默认且统一的布局为 TP=4、PP=2、CP=1、EP=4、ETP=1，并开启 sequence parallel。
训练配置中不提供 recompute/gradient-checkpointing 开关，避免 MoE 重计算带来的
执行不一致和额外开销。所有需要 reference 的算法都必须复用同一布局。

离线命令：

```bash
envs/open-unlearning/bin/python scripts/check_megatron_readiness.py \
  --world-size 8 --tp 4 --pp 2 --cp 1 --ep 4 --etp 1 \
  --sequence-parallel --num-experts 256
envs/open-unlearning/bin/python scripts/inspect_megatron_config.py \
  configs/megatron/qwen35_grad_ascent.yaml
PYTHONPATH=src envs/open-unlearning/bin/python -m unittest discover \
  -s tests/megatron -p 'test_*.py' -v
```

在线单机 8 卡入口：

```bash
CONFIG=configs/megatron/qwen35_grad_ascent.yaml scripts/Megatron/run.sh
CONFIG=configs/megatron/interns2_grad_diff.yaml scripts/Megatron/run.sh
CONFIG=configs/megatron/qwen35_npo.yaml scripts/Megatron/run.sh
CONFIG=configs/megatron/interns2_undial.yaml scripts/Megatron/run.sh
CONFIG=configs/megatron/gemma4_rmu.yaml scripts/Megatron/run.sh
```

当前无 GPU 节点上 Transformer Engine 会在 import 时因没有活动 Triton CUDA
driver 而停止，因此离线检查只读取 HF 模型元数据，不执行 MCore-Bridge import。
这项检查必须在首个 GPU rjob 中用 `inspect_megatron_config.py --require-bridge`
补齐，随后才允许加载 35B 权重。当前训练循环和最终 HF 导出也都属于“已编码、
待 GPU 单步验证”，不能视为已经跑通。

GradDiff、SimNPO、WGA、SatImp 和 PDU 都把 forget/retain 样本合并到一个物理
micro-batch，只调用一次模型 forward，再按 sequence role 分开计算目标。CEU 和
PDU 使用 TP vocab-sharded logits：CEU 通过跨 TP 的 softmax 归约，PDU 通过跨 TP
的 max/sum 归约，retain NLL 使用 vocab-parallel cross entropy。CPU 已覆盖 TP=1
数学基线，真正的 TP=4 collective、显存与吞吐仍需后续 8 卡单步验证。

NPO 和 UNDIAL 在每个 rank 上同时持有学生分片与对应的冻结 reference 分片；两者
共享完全相同的 TP/PP/CP/EP/ETP 通信布局，但参数对象彼此独立。reference 不创建
梯度或 optimizer state。NPO 的 reference 输出是 token NLL；UNDIAL 的 reference
输出是当前 TP rank 的局部词表 logits，并在同一 rank 上与学生 logits 配对。
UNDIAL 窗口内 loss 乘以 `window_microbatches / total_microbatches`，再由 Megatron
schedule 除以窗口大小，因此多个窗口累计后的梯度仍等价于整个 global batch 平均。

RMU 默认参数与当前三种 MoE 的 fixed-control 实验对齐：目标 layer 9、训练 layers
7/8/9、`gamma=1`、`alpha=350`、`steering_coeff=300`、学习率 `1e-5`。它使用
forget 和 retain 的合并 physical micro-batch，但在目标 layer 分别按 role mask 计算
MSE。辅助 loss 显式按 `1 / num_microbatches` 缩放，所以仍对应 global batch 平均。
