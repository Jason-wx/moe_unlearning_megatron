# OpenUnlearning Runbook

本文分两部分：先构造训练用的 data cache，再用 Megatron 运行遗忘算法。所有命令均从仓库
根目录执行，使用下面的 Conda 环境：

~~~bash
cd /mnt/shared-storage-user/safethm-share/wangxiao1/Safe_MoE/open-unlearning
conda activate /mnt/shared-storage-user/safethm-share/wangxiao1/Safe_MoE/open-unlearning/envs/open-unlearning
~~~

## 1. 构造 data cache

训练不直接读原始语料，而是读离线 tokenize 好的 cache。cache 与 tokenizer 绑定，每个模型
需要自己的一套。

### 1.1 三个模型、两种方法、三个数据集

| 模型 | 配置目录名 | 说明 |
| --- | --- | --- |
| Intern-S2 | `interns2` | |
| Qwen3.5-35B-A3B | `qwen35` | |
| Gemma4-26B-A4B | `gemma4` | 每个样本都以 `<bos>` 开头（见下） |

| 构造方法 | 说明 | 用途 |
| --- | --- | --- |
| XTuner soft-pack | 按 seed 固定文档顺序，把文档拼成 4096-token 的定长 pack | **正式训练使用** |
| document chunk | 每篇文档单独 tokenize，在文档内部切成 1024-token 的 chunk，不跨文档拼接 | 早期实验 |

| 数据集 | 配置文件名 | 角色 |
| --- | --- | --- |
| WMDP-Bio forget corpus | `wmdp_bio_forget.yaml` | forget |
| WMDP-Cyber forget corpus | `wmdp_cyber_forget.yaml` | forget |
| FineWeb-Edu 5B | `fineweb_edu_5b.yaml` | retain |

每个“方法 × 模型 × 数据集”的参数配置都在：

~~~text
configs/data_processing/
├── xtuner/
│   ├── template.yaml                     新数据集的模板
│   └── {interns2,qwen35,gemma4}/{wmdp_bio_forget,wmdp_cyber_forget,fineweb_edu_5b}.yaml
└── document_chunk/
    └── {interns2,qwen35,gemma4}/{wmdp_bio_forget,wmdp_cyber_forget,fineweb_edu_5b}.yaml
~~~

- XTuner 配置是 builder 直接读取的 YAML（tokenizer、dataset、packing、build、output 五部分）。
- document chunk 的 builder 只接受命令行参数，YAML 中每个 key 对应一个同名参数，
  文件开头给出了完整命令；示例脚本为 `scripts/build_document_chunk_cache_example.sh`。
- Gemma4 训练和评测都依赖 `<bos>`，因此 Gemma4 的 cache 默认每个 pack / chunk 都以
  `<bos>` 开头；Intern-S2 与 Qwen3.5 不加 BOS。
- 格式与参数的详细说明见 [configs/data_processing/README.md](configs/data_processing/README.md)。

已经构建好的 cache 在共享存储上，Git 仓库不包含这些数据：

~~~text
/mnt/shared-storage-user/safethm-share/wangxiao1/Safe_MoE/open-unlearning/data_cache/
├── Xtuner/          XTuner soft-pack cache（正式训练使用，9 个均已构建）
└── {interns2,qwen35,gemma4}/   document chunk cache
~~~

训练前先确认需要的 cache 已存在，已有的直接在训练配置中引用，缺少时再构建。

### 1.2 示例：构建一个 XTuner cache

`scripts/build_xtuner_cache_example.sh` 以 Qwen3.5 的 WMDP-Bio forget 为例，参数全部来自
`configs/data_processing/xtuner/qwen35/wmdp_bio_forget.yaml`：

~~~bash
bash scripts/build_xtuner_cache_example.sh
~~~

- 构建其他 cache：把脚本中的 `--config` 换成 `configs/data_processing/xtuner/` 下的目标配置。
- 目标 cache 已存在时 builder 会拒绝覆盖，追加 `--overwrite` 才会重建。
- 新数据集：复制 `configs/data_processing/xtuner/template.yaml`，修改 tokenizer、数据文件和
  输出目录后，再把示例脚本的 `--config` 指向它。

## 2. 运行算法（Megatron）

### 2.1 训练配置

最终实验的训练配置按“模型 × 数据集”分目录，每个目录下每个算法一份 YAML：

~~~text
configs/training/megatron/final-training-configs/
└── {intern,qwen,gemma}/{bio,cyber}/<算法>.yaml
~~~

| 文件名 | `algorithm.name` |
| --- | --- |
| `GradAscent.yaml` | `grad_ascent` |
| `GradDiff.yaml` | `grad_diff` |
| `PureNPO.yaml` | `pure_npo` |
| `SimNPO.yaml` | `simnpo` |
| `SatImp.yaml` | `satimp` |
| `WGA.yaml` | `wga` |
| `CEU.yaml` | `ceu` |
| `PDU.yaml` | `pdu` |
| `RMU.yaml` | `rmu` |

> **Trace 不在本框架内。** 每个目录下的 `Trace.yaml` 属于另一套代码，格式与本仓库的训练
> 配置不同，只作为实验记录保留，不能用下面的脚本运行。

每份 YAML 自包含一次训练的全部设置：

| 部分 | 内容 |
| --- | --- |
| `model` | 模型路径、专家数量 |
| `data` | `forget_cache`、`retain_cache`（第 1 节构建的 XTuner cache）、序列长度 |
| `algorithm` | 算法名称及其全部超参 |
| `training` | batch、学习率与调度、训练轮数、保存策略、输出名称 |
| `backends.megatron` | 并行布局、`output_root`，以及训练中在线评测（`evaluation`）的任务、步数和输出目录 |

做新实验时复制最接近的一份 YAML，只修改新文件。

### 2.2 启动训练

以 Qwen3.5 / WMDP-Bio / GradDiff 为例：

~~~bash
CONFIG=configs/training/megatron/final-training-configs/qwen/bio/GradDiff.yaml

# 1. 展开并校验配置；--check-paths 额外检查模型和 cache 路径是否存在
scripts/inspect.sh --config "${CONFIG}"
scripts/inspect.sh --config "${CONFIG}" --backend megatron --check-paths

# 2. 只打印最终的训练参数，不启动训练
scripts/train_megatron.sh "${CONFIG}" --dry-run

# 3. 在当前节点启动训练，GPU 数取自 backends.megatron.parallel.world_size（8 卡）
scripts/train_megatron.sh "${CONFIG}"
~~~

训练输出：

| 内容 | 位置 |
| --- | --- |
| checkpoint（HF 格式） | `<backends.megatron.output_root>/<training.output_name>/checkpoint-step-<N>/`；bio 配置保存最后一步，cyber 配置保存 `save_steps` 指定的一步 |
| 在线评测结果 | `backends.megatron.evaluation.output_dir` |
| W&B 离线记录 | `wandb_offline/` |

### 2.3 用 RJob 提交

`scripts/submit_rjob_example.sh` 把一份或多份配置提交为一个 RJob。后端、GPU 数和任务名
都从配置读取，`CLUSTER` 选择资源组：

~~~bash
# 单个配置，提交到 safethm
CLUSTER=safethm scripts/submit_rjob_example.sh \
  configs/training/megatron/final-training-configs/qwen/bio/GradDiff.yaml

# 同一个 RJob 中依次训练一组配置；某一份失败不影响后续
CLUSTER=safewt scripts/submit_rjob_example.sh \
  configs/training/megatron/final-training-configs/qwen/bio/GradAscent.yaml \
  configs/training/megatron/final-training-configs/qwen/bio/SatImp.yaml
~~~

| `CLUSTER` | namespace | charged group | 镜像 |
| --- | --- | --- | --- |
| `safer2ai`（默认） | `ailab-safer2ai` | `safer2ai_gpu` | `registry.h.pjlab.org.cn/ailab-safer2ai-safer2ai_gpu/unlearning:unlearning-cu126-torch271-v1` |
| `safethm` | `ailab-safethm` | `sfaethm_gpu` | `registry.h.pjlab.org.cn/ailab-safethm-sfaethm_cpu/safemoe:jc-work-20260506163353` |
| `safewt` | `ailab-safewt` | `safewt_gpu` | `registry.h.pjlab.org.cn/ailab-safethm-sfaethm_cpu/safemoe:jc-work-20260506163353` |

资源默认为配置中的 GPU 数、96 CPU、256000 MiB 内存，可用 `CPUS`、`MEMORY_MB` 覆盖；
`JOB_NAME` 默认取第一份配置的 `name`；`DRY_RUN=1` 只打印 rjob 命令。每份配置的日志在
`outputs/rjob_logs/<任务名>_<时间>/<配置名>.log`。查看任务：

~~~bash
rjob get <RJOB_NAME>
rjob logs job <RJOB_NAME> --tail-lines 200
~~~

### 2.4 离线评测 checkpoint

训练中已有在线评测；需要完整评测某个 checkpoint 时运行 `scripts/eval.sh`（输出目录每次会
被清空重建，因此必须位于 `evaluations/` 下，且以 `lm_eval_results_` 开头）：

~~~bash
scripts/eval.sh \
  <output_root>/<output_name>/checkpoint-step-<N> \
  wmdp_bio,mmlu \
  evaluations/unlearning/<实验名>/lm_eval_results_<实验名>_wmdp_bio_mmlu
~~~

Cyber 实验把任务换成 `wmdp_cyber,mmlu`。结果写入输出目录下的 `result.json`。

### 2.5 脚本一览

全部入口都在 `scripts/`，用法汇总见 [scripts/README.md](scripts/README.md)。

| 步骤 | 脚本 |
| --- | --- |
| 构造 data cache | `scripts/build_xtuner_cache_example.sh`、`scripts/build_document_chunk_cache_example.sh` |
| 检查配置 | `scripts/inspect.sh` |
| 训练 | `scripts/train_megatron.sh`；DeepSpeed 为 `scripts/train_deepspeed.sh`（仅 Intern-S2 示例，配置在 `configs/training/deepspeed/interns2/`） |
| 提交 RJob | `scripts/submit_rjob_example.sh` |
| 离线评测 | `scripts/eval.sh` |
