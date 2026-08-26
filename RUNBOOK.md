# Megatron 完整训练流程

本文只说明 Intern-S2 在 Megatron 后端上的正式流程：准备 token cache、训练、保存两个
checkpoint，并运行 WMDP/MMLU 评测。所有命令均从仓库根目录执行。

## 1. 准备数据缓存

训练默认使用以下两份缓存：

```text
data_cache/interns2_wmdp_bio_max1024_stride1024/forget
data_cache/interns2_fineweb_edu_5b_max1024_stride1024
```

第一份是 WMDP-Bio forget 数据；第二份是需要 retain loss 的算法使用的 FineWeb
retain 数据。每个 cache 目录都必须包含：

```text
meta.json
summary.json
chunks.jsonl
token_ids.uint32.bin
```

确认现有缓存：

```bash
for CACHE_DIR in \
  data_cache/interns2_wmdp_bio_max1024_stride1024/forget \
  data_cache/interns2_fineweb_edu_5b_max1024_stride1024
do
  test -f "${CACHE_DIR}/meta.json"
  test -f "${CACHE_DIR}/chunks.jsonl"
  test -f "${CACHE_DIR}/token_ids.uint32.bin"
  echo "cache ready: ${CACHE_DIR}"
done
```

如果缓存缺失，使用 Intern-S2 tokenizer 重新构建。不要对已有正式 cache 添加
`--overwrite`。

```bash
MODEL_PATH=/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/jincheng/sgtm/checkpoints/Intern-S2-base05

scripts/build_cache.sh \
  --model-path "${MODEL_PATH}" \
  --data-files '/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/jincheng/data/wmdp/bio-forget-corpus/*.parquet' \
  --output-dir data_cache/interns2_wmdp_bio_max1024_stride1024/forget \
  --max-length 1024 --stride 1024 --workers 16

scripts/build_cache.sh \
  --model-path "${MODEL_PATH}" \
  --data-files '/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/jincheng/data/fineweb-edu-5b-shuffle/*.parquet' \
  --output-dir data_cache/interns2_fineweb_edu_5b_max1024_stride1024 \
  --max-length 1024 --stride 1024 --workers 16 --tokenize-batch-size 512
```

## 2. 训练一个算法并评测

推荐直接提交 train→eval 流水线。下面以 Pure NPO 为例：

```bash
SWEEP_NAME=interns2-pure-npo-v1 \
WORKFLOW=train-eval-group \
BACKEND=megatron \
JOB_NAME=ou-interns2-pure-npo \
CONFIG=configs/experiments/interns2_pure_npo.yaml \
scripts/submit_rjob_example.sh pure_npo
```

该 RJob 使用 8 张 GPU、global batch size 128，执行：

```text
训练完整 1 epoch
→ 保存 checkpoint-step-300
→ 保存 checkpoint-epoch-1
→ 评测 step-300 的 wmdp,mmlu
→ 评测 epoch-1 的 wmdp,mmlu
```

可用算法参数为：

```text
grad_ascent  grad_diff  pure_npo  token_npo  token_simnpo  wga
satimp       ceu        pdu       undial    rmu
```

## 3. 用三个 RJob 训练全部算法

三个 RJob 使用相同的 `SWEEP_NAME`。三个作业并行运行，每个作业内按参数顺序串行执行
“训练一个算法 → 评测两个 checkpoint → 下一个算法”。

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

## 4. 查看输出

Megatron 权重默认写入：

```text
/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/wangxiao1/checkpoints/unlearn/megatron/
└── <SWEEP_NAME>-<algorithm>/
    ├── checkpoint-step-300/
    └── checkpoint-epoch-1/
```

评测结果写入：

```text
evaluations/unlearning/interns2/<SWEEP_NAME>/
├── lm_eval_results_<algorithm>_step300_wmdp_mmlu/result.json
└── lm_eval_results_<algorithm>_epoch1_wmdp_mmlu/result.json
```

查看作业状态和日志：

```bash
rjob get <RJOB_NAME>
rjob logs job <RJOB_NAME> --tail-lines 200
```

单个算法训练失败时，流水线会继续下一个算法；单个 checkpoint 评测失败时，也会继续
后续任务。只要发生过失败，该 RJob 最终会返回非零状态并在末尾输出失败数量。
