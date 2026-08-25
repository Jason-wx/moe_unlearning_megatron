# 重构状态

更新时间：2026-08-25。

## 已完成

- 新增标准 `pyproject.toml`，并将 DeepSpeed、Megatron、eval 依赖拆为 extras。
- 新增 `open_unlearning` 稳定包、`ou` CLI 和两个后端适配器。
- 建立共享 model/dataset/algorithm/profile schema；Qwen3.5、Intern-S2、Gemma 4
  均有双后端 PureNPO 示例。
- 后端分别拥有 micro batch/并行配置，同时共享 global batch、优化器和算法参数。
- 训练输出自动隔离到 `<experiment>/deepspeed` 与 `<experiment>/megatron`。
- CLI 支持替换组件，不再需要为每个算法复制一套 shell/RJob 文件。
- 新增统一 cache、evaluate、inspect、train 和 RJob 包装。
- 已删除全部旧 Megatron run/submit/roundtrip 文件，并将 35 份逐模型算法配置压缩为
  共享组件和一份历史参数归档。
- 脚本源码由 61 个压缩为 10 个维护文件；历史运行日志已按要求清理。
- Megatron 测试已迁入 `tests/megatron/`；`evaluations/` 只保存原始评测结果。
- `doc/`、`test/docs/` 和脚本说明已合并进 `docs/`。
- 增加 runtime 路径抽象、Git 保护规则、双后端契约测试和配置目录测试。
- 原 `setup.py` 已由 `pyproject.toml` 替代；原 `setup_data.py` 的当前缓存工作流由
  `scripts/build_cache.sh` 承接。

## 暂时保留的兼容层

- `src/train.py`、`src/trainer` 和旧 Hydra 配置仍是 DeepSpeed 后端内部实现。
- `src/qwen_moe_rmu`、`src/gemma_moe_rmu` 仍承载 DeepSpeed 模型专用 RMU loader。
- 原 Megatron run/submit/roundtrip、旧独立入口和逐模型算法配置已经移除。

兼容层存在不等于继续扩展。新实验应只从 `ou`、共享配置和根目录标准脚本进入。

## 待完成

1. 按 `runtime-layout.md` 创建 manifest 后迁移大目录；旧环境在新位置重建。
2. 将 DeepSpeed 专用 Qwen/Gemma RMU loader 并入统一后端适配器。

## 当前验证基线

- 统一 `tests/` 测试：58 项通过，其中共享配置/CLI 12 项、Megatron 46 项。
- 三个模型的两后端路径校验通过。
- DeepSpeed 生成命令已通过旧 Hydra 完整 composition。
- 所有 Python 文件通过 bytecode compile；所有 shell 文件通过 `bash -n`。
- `git diff --check` 通过。

Intern-S2 Megatron aligned NPO 任务在 492/1344 step 被关闭，没有生成最终 checkpoint
或进入 lm-eval，因此不能登记为性能结果；相关运行日志和 GPU 采样已于 2026-08-25
按要求清理。
