# 架构与目录

## 设计原则

项目只维护一套实验语义、两个后端适配器。每个训练 YAML 自包含模型、数据、算法、
优化器、输出以及当前后端的并行参数，不再拼装公开组件。

```text
configs/data_processing/{xtuner,document_chunk}/   -> src/data/build_*_cache.py -> data_cache/
                                                                                    |
configs/training/                                                     forget/retain cache
        |
        +-- deepspeed/interns2/*.yaml                      -> Accelerate -> src/train.py
        |
        `-- megatron/final-training-configs/<model>/<domain>/*.yaml -> src/megatron_unlearning
```

## 稳定入口

统一 CLI 为 `python -m open_unlearning.cli`（需要 `PYTHONPATH=src`），通常经由脚本调用：

- `scripts/inspect.sh --config ...`：展开环境变量并校验完整配置（`cli inspect`）。
- `scripts/train_megatron.sh CONFIG`：用 torchrun 进入 MCore runtime（`cli train --backend megatron`）。
- `scripts/train_deepspeed.sh CONFIG`：生成或执行 Accelerate/ZeRO-3 命令（`cli train --backend deepspeed`）。
- `scripts/submit_rjob_example.sh`：唯一的 RJob 提交模板，`CLUSTER` 选择资源组，可在一个任务中依次训练多份配置。
- `scripts/eval.sh`：用 lm-eval 离线评测一个 checkpoint。
- `python -m open_unlearning.cli paths`：查看解析后的运行目录。

`configs/internal/deepspeed` 只保留 Hydra trainer 的机械适配，不承载实验参数。

## 数据

训练只读离线 token cache，两种格式都由 `src/data/cached_pretraining.py` 读取：

- XTuner soft-pack（`xtuner_softpack_v1`）：按 seed 固定顺序把文档拼成定长 pack，正式训练使用。
- document chunk（`document_chunk_v1`）：每篇文档内部切 chunk，不跨文档拼接。

格式与参数见 [data-processing.md](data-processing.md)。

## 算法能力边界

DeepSpeed 的底层仍保留上游全部 trainer；统一 CLI 当前覆盖标准 WMDP cache 算法，
模型专用 RMU 与偏好数据 DPO/AltPO 不在统一 DeepSpeed 入口中。Megatron 当前
支持 13 个规范算法名：GradAscent、GradDiff、NPO、PureNPO、TokenNPO、SimNPO、
TokenSimNPO、WGA、SatImp、CEU、PDU、UNDIAL 和 RMU。除 RMU 目前仅支持 Megatron
外，其余 12 个均支持 DeepSpeed 与 Megatron；后端适配器会在启动前拒绝不支持的组合。
最终实验使用其中 9 个（GradAscent、GradDiff、PureNPO、SimNPO、SatImp、WGA、CEU、PDU、RMU）。
Trace 不在本框架内。

三个 NPO 名称不再互为 alias：`npo` 使用 sequence NLL 求和，`pure_npo` 先按有效
token 数归一化，`token_npo` 在逐 token 的 student/reference NLL 差上分别应用 sigmoid。
两个 SimNPO 名称也独立：`simnpo` 先得到 sequence mean NLL，`token_simnpo` 逐 token
应用 sigmoid。

retain 数据：GradDiff、PDU、RMU 始终需要 `data.retain_cache`；NPO、PureNPO、SimNPO、
TokenSimNPO、WGA、SatImp、UNDIAL 仅在 `algorithm.alpha != 0` 时加载 retain 并加入
retain NLL 项，`alpha: 0` 即 forget-only。

## 在线评测

Megatron 训练可在指定步数直接用显存中的参数评测 WMDP-Bio、WMDP-Cyber 和 MMLU，
无需导出 checkpoint（`src/megatron_unlearning/online_eval.py`）。prompt 与打分复用
lm-eval 的 task 定义，与离线评测口径一致。任务、步数、batch size 和输出目录由训练
YAML 的 `backends.megatron.evaluation` 配置。

## 代码边界

- `src/open_unlearning`：稳定外观层（CLI、配置校验、运行目录）；用户脚本只依赖这里。
- `src/megatron_unlearning`：MCore 数据、objective、并行调度、在线评测、checkpoint 与 runtime。
- `src/data`：两种 cache 的构建与读取。
- `src/trainer`、`src/evals`：原 HF/DeepSpeed 实现和上游评测。
- `configs`：配置是真实来源，脚本中不得再次维护一套超参数。

后续内部移动模块时，应保留稳定 CLI 和配置 schema；这比一次性改写所有上游 import
风险更低，也便于对照既有训练结果。
