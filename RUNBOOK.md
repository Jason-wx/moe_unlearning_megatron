# OpenUnlearning Runbook

本文说明如何在共享集群上准备数据、修改配置、启动训练和运行评测。所有命令均从仓库
根目录执行。

## 1. 环境与入口

项目使用下面的 Conda 环境：

~~~bash
cd /mnt/shared-storage-user/safethm-share/wangxiao1/Safe_MoE/open-unlearning
conda activate /mnt/shared-storage-user/safethm-share/wangxiao1/Safe_MoE/open-unlearning/envs/open-unlearning
~~~

常用入口：

| 任务 | 脚本 |
| --- | --- |
| 检查配置 | scripts/inspect.sh |
| 构建 document-chunk cache | scripts/build_cache.sh |
| 构建 XTuner soft-pack cache | scripts/build_xtuner_pack_cache.sh |
| Megatron 训练 | scripts/train_megatron.sh |
| DeepSpeed 训练 | scripts/train_deepspeed.sh |
| checkpoint 评测 | scripts/lm_eval_accelerate.sh |
| 顺序训练并评测多个配置 | scripts/run_megatron_train_eval_group.sh |
| 提交 RJob | scripts/submit_rjob_example.sh |

## 2. 数据准备

项目支持两套数据 cache。两者最终都由训练配置中的 data.forget_cache 和
data.retain_cache 引用。

已有数据缓存位于共享服务器：

~~~text
/mnt/shared-storage-user/safethm-share/wangxiao1/Safe_MoE/open-unlearning/data_cache
~~~

Git 仓库不包含、也不会 pull 这些数据。开始训练前请先登录服务器查看该目录；
已有缓存可直接在训练配置中引用，缺少时再按下面的方法构建。

### 2.1 Document chunk

每篇文档单独 tokenize，再在文档内部按 max length 和 stride 切 chunk，不跨文档拼接。

~~~bash
scripts/build_cache.sh \
  --model-path /path/to/tokenizer-or-model \
  --data-files '/path/to/corpus/*.parquet' \
  --dataset-path parquet \
  --text-key text \
  --output-dir /mnt/shared-storage-user/safethm-share/wangxiao1/Safe_MoE/open-unlearning/data_cache/my_document_chunk_cache \
  --max-length 1024 \
  --stride 1024 \
  --workers 8 \
  --tokenize-batch-size 1024
~~~

JSONL 输入把 dataset-path 改为 json。

### 2.2 XTuner soft-pack

该方案固定数据顺序，并把短文档组合成定长 pack。正式 Intern-S2 Megatron 配置使用
4096-token XTuner soft-pack cache。

构建配置位于：

~~~text
configs/data_processing/
├── xtuner_softpack_template.yaml
├── interns2_wmdp_bio_forget_xtuner_softpack.yaml
├── interns2_wmdp_bio_retain_xtuner_softpack.yaml
└── interns2_fineweb_edu_5b_xtuner_softpack.yaml
~~~

复制模板并修改 tokenizer、数据文件和输出目录，然后执行：

~~~bash
cp configs/data_processing/xtuner_softpack_template.yaml \
  configs/data_processing/my_xtuner_softpack.yaml

scripts/build_xtuner_pack_cache.sh \
  --config configs/data_processing/my_xtuner_softpack.yaml
~~~

更详细的数据格式说明见 docs/data-processing.md。

## 3. 配置与启动

训练配置位于：

~~~text
configs/training/deepspeed/interns2/   DeepSpeed 配置
configs/training/megatron/interns2/    Intern-S2 Megatron 配置
configs/training/megatron/qwen35/      Qwen3.5 Megatron 配置
configs/training/megatron/gemma4/      Gemma 4 Megatron 配置
~~~

选择或复制一份配置进行修改：

~~~bash
cp configs/training/megatron/interns2/pure_npo.yaml \
  configs/training/megatron/interns2/my_experiment.yaml
~~~

启动前检查配置：

~~~bash
scripts/inspect.sh \
  --config configs/training/megatron/interns2/my_experiment.yaml

scripts/inspect.sh \
  --config configs/training/megatron/interns2/my_experiment.yaml \
  --backend megatron \
  --check-paths
~~~

只查看最终启动参数：

~~~bash
scripts/train_megatron.sh \
  configs/training/megatron/interns2/my_experiment.yaml --dry-run

scripts/train_deepspeed.sh \
  configs/training/deepspeed/interns2/pure_npo.yaml --dry-run
~~~

正式启动：

~~~bash
# Megatron
scripts/train_megatron.sh \
  configs/training/megatron/interns2/my_experiment.yaml

# DeepSpeed
scripts/train_deepspeed.sh \
  configs/training/deepspeed/interns2/pure_npo.yaml
~~~

在当前分配的节点上顺序训练多个 Megatron 配置，并在每个配置训练完成后评测：

~~~bash
SWEEP_NAME=interns2-my-sweep \
EVAL_TASKS=wmdp,mmlu \
scripts/run_megatron_train_eval_group.sh \
  configs/training/megatron/interns2/pure_npo.yaml \
  configs/training/megatron/interns2/grad_diff.yaml \
  configs/training/megatron/interns2/wga.yaml
~~~

## 4. RJob

RJob 单独使用 scripts/submit_rjob_example.sh。单个 Megatron 训练：

~~~bash
BACKEND=megatron \
WORKFLOW=single \
CONFIG=configs/training/megatron/interns2/pure_npo.yaml \
JOB_NAME=ou-interns2-pure-npo \
GPUS=8 \
scripts/submit_rjob_example.sh
~~~

单个 DeepSpeed 训练：

~~~bash
BACKEND=deepspeed \
WORKFLOW=single \
CONFIG=configs/training/deepspeed/interns2/pure_npo.yaml \
JOB_NAME=ou-interns2-pure-npo-ds \
GPUS=8 \
scripts/submit_rjob_example.sh
~~~

在一个 RJob 中顺序训练并评测多个 Megatron 配置：

~~~bash
SWEEP_NAME=interns2-all-algorithms-v1 \
WORKFLOW=train-eval-group \
BACKEND=megatron \
JOB_NAME=ou-interns2-group1 \
GPUS=8 \
EVAL_TASKS=wmdp,mmlu \
scripts/submit_rjob_example.sh \
  configs/training/megatron/interns2/pure_npo.yaml \
  configs/training/megatron/interns2/grad_diff.yaml \
  configs/training/megatron/interns2/wga.yaml
~~~

查看任务：

~~~bash
rjob get <RJOB_NAME>
rjob logs job <RJOB_NAME> --tail-lines 200
~~~

## 5. 评测与输出

单独评测一个 checkpoint：

~~~bash
scripts/lm_eval_accelerate.sh \
  /path/to/checkpoint-epoch-1 \
  wmdp,mmlu \
  /mnt/shared-storage-user/safethm-share/wangxiao1/Safe_MoE/open-unlearning/evaluations/unlearning/my_run/lm_eval_results_pure_npo_epoch1_wmdp_mmlu
~~~

主要输出位置：

| 内容 | 位置 |
| --- | --- |
| Megatron checkpoint | backends.megatron.output_root/training.output_name |
| DeepSpeed checkpoint | training.output_dir/deepspeed |
| 评测结果 | evaluations/unlearning/ |
| RJob 日志 | outputs/rjob_logs/ |
| W&B 离线记录 | wandb_offline/ |
