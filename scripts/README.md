# 维护中的脚本

公开入口只有以下几类：

- `train_deepspeed.sh`：DeepSpeed/Accelerate 后端。
- `train_megatron.sh`：Megatron/MCore 后端。
- `submit_rjob_example.sh`：唯一 RJob 提交模板，通过 `BACKEND` 选择后端。
- `run_megatron_train_eval_group.sh`：在一个 RJob 内串行执行多算法 train→eval。
- `inspect.sh`：解析配置和校验路径。
- `build_cache.sh`：构建 document-chunk cache。
- `build_xtuner_pack_cache.sh`：从单个纯文本 CPT 数据集构建固定顺序的
  `xtuner_softpack_v1` cache。
- `lm_eval_accelerate.sh`：统一的 lm-eval 评测入口。

cache builder 的实现位于 `src/data/build_document_chunk_cache.py`，不作为独立脚本
暴露。旧算法、RJob 和 Megatron roundtrip 源文件已经删除；对应历史日志也已按要求清理。

XTuner packed-cache builder 的实现位于
`src/data/build_xtuner_packed_cache.py`。先复制
`configs/data_processing/xtuner_softpack_template.yaml`，再运行：

```bash
scripts/build_xtuner_pack_cache.sh --config /path/to/cache-build.yaml
```

`submit_rjob_example.sh` 会把位置参数继续传给统一 CLI。正式训练直接传完整 YAML，
训练周期、LR 调度、保存策略和输出子目录都由 YAML 定义：

```bash
BACKEND=megatron \
CONFIG=configs/training/megatron/interns2/pure_npo.yaml \
JOB_NAME=interns2-pure-npo \
scripts/submit_rjob_example.sh
```

Megatron 输出目录由 `backends.megatron.output_root/training.output_name` 组成。

多配置流水线不会覆盖训练参数；每个训练配置完成后评测它的
`checkpoint-epoch-1/`，然后才训练下一个算法。
参数直接传完整训练 YAML，不再传算法组件名：

```bash
SWEEP_NAME=interns2-all-algorithms-v1 WORKFLOW=train-eval-group \
BACKEND=megatron JOB_NAME=ou-interns2-group1 \
scripts/submit_rjob_example.sh \
  configs/training/megatron/interns2/pure_npo.yaml \
  configs/training/megatron/interns2/grad_diff.yaml \
  configs/training/megatron/interns2/wga.yaml

SWEEP_NAME=interns2-all-algorithms-v1 WORKFLOW=train-eval-group \
BACKEND=megatron JOB_NAME=ou-interns2-group2 \
scripts/submit_rjob_example.sh \
  configs/training/megatron/interns2/satimp.yaml \
  configs/training/megatron/interns2/rmu.yaml \
  configs/training/megatron/interns2/undial.yaml
```

默认评测任务是 `wmdp,mmlu`，可通过 `EVAL_TASKS` 覆盖。单个算法训练失败时，
runner 会跳过该算法的评测并继续下一个；单个 checkpoint 评测失败也不会阻止后续任务，
但整个 RJob 最终会以非零状态汇总失败。
