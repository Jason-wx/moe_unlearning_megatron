# 脚本

仓库的全部入口都在这里，按使用顺序排列。所有脚本从仓库根目录运行，默认使用
`envs/open-unlearning` 中的 Python（可用 `OU_PYTHON` 覆盖）。完整流程见 [RUNBOOK](../RUNBOOK.md)。

| 步骤 | 脚本 | 用法 |
| --- | --- | --- |
| 1. 构造 data cache | `build_xtuner_cache_example.sh` | 示例：按一份 XTuner 配置构建 cache |
| | `build_document_chunk_cache_example.sh` | 示例：按一份 document chunk 参数构建 cache |
| 2. 检查配置 | `inspect.sh` | `scripts/inspect.sh --config CONFIG [--backend megatron --check-paths]` |
| 3. 训练 | `train_megatron.sh` | `scripts/train_megatron.sh CONFIG [--dry-run]` |
| | `train_deepspeed.sh` | `scripts/train_deepspeed.sh CONFIG [--dry-run]` |
| 4. 提交 RJob | `submit_rjob_example.sh` | `CLUSTER=safethm scripts/submit_rjob_example.sh CONFIG [CONFIG...]` |
| 5. 离线评测 | `eval.sh` | `scripts/eval.sh MODEL_DIR TASKS OUTPUT_DIR` |

## 说明

- **构造 data cache**：两个示例直接调用 builder 模块（`src/data/build_xtuner_packed_cache.py`、
  `src/data/build_document_chunk_cache.py`）。构建其他 cache 时修改脚本中的配置路径或参数，
  参数来源见 `configs/data_processing/`。
- **训练**：GPU 数取自配置的 `world_size`，在当前节点用 torchrun 启动；`--dry-run` 只打印最终参数。
  训练中的 WMDP / MMLU 在线评测由配置控制，不需要额外脚本。
- **提交 RJob**：后端、GPU 数和任务名都从配置读取，`CLUSTER` 选择资源组（safer2ai、safethm、
  safewt）。传入多份配置时在同一个 RJob 中依次训练，某一份失败不影响后续；日志在
  `outputs/rjob_logs/<任务名>_<时间>/`。`DRY_RUN=1` 只打印 rjob 命令。
- **离线评测**：用 lm-eval 评测一个 HF checkpoint，结果写入 `OUTPUT_DIR/result.json`。
  `OUTPUT_DIR` 每次会被清空重建，因此必须位于 `evaluations/` 下且以 `lm_eval_results_` 开头。
