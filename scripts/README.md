# 维护中的脚本

公开入口只有以下几类：

- `train_deepspeed.sh`：DeepSpeed/Accelerate 后端。
- `train_megatron.sh`：Megatron/MCore 后端。
- `submit_rjob_example.sh`：唯一 RJob 提交模板，通过 `BACKEND` 选择后端。
- `run_megatron_train_eval_group.sh`：在一个 RJob 内串行执行多算法 train→eval。
- `inspect.sh`：解析配置和校验路径。
- `build_cache.sh`：构建 document-chunk cache。
- `evaluate.sh`：统一 lm-eval 包装。
- `lm_eval_accelerate.sh`：评测实现，由 `evaluate.sh` 调用。

cache builder 的实现位于 `src/data/build_document_chunk_cache.py`，不作为独立脚本
暴露。旧算法、RJob 和 Megatron roundtrip 源文件已经删除；对应历史日志也已按要求清理。

`submit_rjob_example.sh` 会把位置参数继续传给统一 CLI。例如，下面的 Megatron
任务只训练 300 个 optimizer steps，并只在终点导出一次权重：

```bash
BACKEND=megatron \
CONFIG=configs/experiments/interns2_pure_npo.yaml \
JOB_NAME=interns2-npo-300steps \
scripts/submit_rjob_example.sh \
  --train-steps 300 \
  --save-only-at-end \
  --output-dir saves/interns2-pure-npo-300steps
```

Megatron 输出目录由 `--output-dir` 的末级目录名和
`OU_MEGATRON_CHECKPOINT_ROOT` 组合；上述任务最终只导出到
`interns2-pure-npo-300steps/checkpoint-step-300/`。

完整 epoch 同时保留 step 300 和 epoch 1：

```bash
scripts/train_megatron.sh configs/experiments/interns2_pure_npo.yaml \
  --algorithm pure_npo \
  --train-epochs 1 \
  --save-at-step 300 \
  --save-at-epoch-end
```

多算法流水线会在每个算法训练完成后，依次评测它的
`checkpoint-step-300/` 和 `checkpoint-epoch-1/`，然后才训练下一个算法。
三个 RJob 可以共用同一个 `SWEEP_NAME`，同时提交不同的算法组：

```bash
SWEEP_NAME=interns2-all-algorithms-v1 WORKFLOW=train-eval-group \
BACKEND=megatron JOB_NAME=ou-interns2-group1 \
scripts/submit_rjob_example.sh pure_npo grad_diff wga ceu

SWEEP_NAME=interns2-all-algorithms-v1 WORKFLOW=train-eval-group \
BACKEND=megatron JOB_NAME=ou-interns2-group2 \
scripts/submit_rjob_example.sh token_npo token_simnpo satimp pdu

SWEEP_NAME=interns2-all-algorithms-v1 WORKFLOW=train-eval-group \
BACKEND=megatron JOB_NAME=ou-interns2-group3 \
scripts/submit_rjob_example.sh rmu undial grad_ascent
```

默认评测任务是 `wmdp,mmlu`，可通过 `EVAL_TASKS` 覆盖。单个算法训练失败时，
runner 会跳过该算法的评测并继续下一个；单个 checkpoint 评测失败也不会阻止后续任务，
但整个 RJob 最终会以非零状态汇总失败。
