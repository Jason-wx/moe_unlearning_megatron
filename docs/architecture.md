# 架构与目录

## 设计原则

项目只维护一套实验语义、两个后端适配器。模型、数据、算法、训练 profile 在共享
配置中定义；DeepSpeed 和 Megatron 配置只表达各自的并行与运行时差异。训练产物
强制写入不同后端子目录，避免同名实验互相覆盖。

```text
configs/experiments/*.yaml
        |
        +-- models + datasets + algorithms + profiles
        |
        +-- configs/backends/deepspeed  -> Accelerate -> src/train.py
        |
        `-- configs/backends/megatron   -> src/megatron_unlearning
```

## 稳定入口

- `ou inspect --config ...`：合并环境变量和组件配置并校验。
- `ou train --backend deepspeed --config ...`：生成或执行 Accelerate/ZeRO-3 命令。
- `ou train --backend megatron --config ...`：生成映射或进入 MCore runtime。
- `scripts/train_deepspeed.sh` 与 `scripts/train_megatron.sh`：RJob 友好的薄包装。
- `scripts/submit_rjob_example.sh`：唯一保留的提交模板。

旧的 Hydra、trainer、eval 模块仍是 DeepSpeed 后端实现，不重复复制。旧的具体算法
启动脚本不再作为公开接口；它们的参数来源已归档到共享配置和结果文档中。

## 算法能力边界

DeepSpeed 的底层仍保留上游全部 trainer；统一 CLI 当前覆盖标准 WMDP cache 算法，
模型专用 RMU 与偏好数据 DPO/AltPO 在完成 loader 合并前继续走兼容入口。Megatron 当前
支持 WMDP 文本缓存上的 GradAscent、GradDiff、NPO、TokenNPO、SimNPO、WGA、
SatImp、CEU、PDU、UNDIAL 和 RMU。后端适配器会在启动前拒绝不支持的组合。

`pure_npo` 是配置层的显式名称，运行时规范化为 `npo`，并且不会装载 retain 数据。
这样可以避免过去 NPO 与 NPO+retain 语义混用。

## 代码边界

- `src/open_unlearning`：新的稳定外观层；用户脚本只依赖这里。
- `src/megatron_unlearning`：MCore 数据、objective、并行调度、checkpoint 与 runtime。
- `src/trainer`、`src/data`、`src/evals`：原 HF/DeepSpeed 实现和上游评测。
- `configs`：配置是真实来源，脚本中不得再次维护一套超参数。

后续内部移动模块时，应保留稳定 CLI 和配置 schema；这比一次性改写所有上游 import
风险更低，也便于对照既有训练结果。
