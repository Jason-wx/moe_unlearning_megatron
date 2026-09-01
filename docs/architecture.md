# 架构与目录

## 设计原则

项目只维护一套实验语义、两个后端适配器。每个训练 YAML 自包含模型、数据、算法、
优化器、输出以及当前后端的并行参数，不再拼装公开组件。

```text
configs/training/
        |
        +-- deepspeed/interns2/*.yaml -> Accelerate -> src/train.py
        |
        `-- megatron/<model>/*.yaml   -> src/megatron_unlearning
```

## 稳定入口

- `ou inspect --config ...`：展开环境变量并校验完整配置。
- `ou train --backend deepspeed --config ...`：生成或执行 Accelerate/ZeRO-3 命令。
- `ou train --backend megatron --config ...`：生成映射或进入 MCore runtime。
- `scripts/train_deepspeed.sh` 与 `scripts/train_megatron.sh`：RJob 友好的薄包装。
- `scripts/submit_rjob_example.sh`：唯一保留的提交模板。

`configs/internal/deepspeed` 只保留 Hydra trainer 的机械适配，不承载实验参数。

## 算法能力边界

DeepSpeed 的底层仍保留上游全部 trainer；统一 CLI 当前覆盖标准 WMDP cache 算法，
模型专用 RMU 与偏好数据 DPO/AltPO 不在统一 DeepSpeed 入口中。Megatron 当前
支持 13 个规范算法名：GradAscent、GradDiff、NPO、PureNPO、TokenNPO、SimNPO、
TokenSimNPO、WGA、SatImp、CEU、PDU、UNDIAL 和 RMU。除 RMU 目前仅支持 Megatron
外，其余 12 个均支持 DeepSpeed 与 Megatron；后端适配器会在启动前拒绝不支持的组合。

三个 NPO 名称不再互为 alias：`npo` 使用 sequence NLL 求和，`pure_npo` 先按有效
token 数归一化，`token_npo` 在逐 token 的 student/reference NLL 差上分别应用 sigmoid。
两个 SimNPO 名称也独立：`simnpo` 先得到 sequence mean NLL，`token_simnpo` 逐 token
应用 sigmoid。NPO 三种版本均为 forget-only；SimNPO 两种版本保留 retain NLL 项。

## 代码边界

- `src/open_unlearning`：新的稳定外观层；用户脚本只依赖这里。
- `src/megatron_unlearning`：MCore 数据、objective、并行调度、checkpoint 与 runtime。
- `src/trainer`、`src/data`、`src/evals`：原 HF/DeepSpeed 实现和上游评测。
- `configs`：配置是真实来源，脚本中不得再次维护一套超参数。

后续内部移动模块时，应保留稳定 CLI 和配置 schema；这比一次性改写所有上游 import
风险更低，也便于对照既有训练结果。
