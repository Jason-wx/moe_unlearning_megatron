# MoE Unlearning

本仓库在 [OpenUnlearning](https://github.com/locuslab/open-unlearning) 的基础上，为三个
MoE 大模型实现了基于 Megatron（MCore）的大规模遗忘训练：在 WMDP-Bio / WMDP-Cyber
语料上遗忘危险知识，并用 WMDP 与 MMLU 评测遗忘效果和通用能力。

> 构造数据、启动训练和评测的完整步骤见 **[Runbook](RUNBOOK.md)**。

## 支持范围

**模型**

| 模型 | 专家数 | 配置目录名 |
| --- | --- | --- |
| Intern-S2 | 256 | `interns2` / `intern` |
| Qwen3.5-35B-A3B | 256 | `qwen35` / `qwen` |
| Gemma4-26B-A4B | 128 | `gemma4` / `gemma` |

**数据**

- forget：WMDP-Bio forget corpus、WMDP-Cyber forget corpus
- retain：FineWeb-Edu 5B
- 两种离线 tokenize 方法：XTuner soft-pack（4096-token 定长 pack，正式训练使用）和
  document chunk（每篇文档内部切 1024-token chunk）

**算法**

Megatron 后端实现了 13 个算法：GradAscent、GradDiff、NPO、PureNPO、TokenNPO、SimNPO、
TokenSimNPO、WGA、SatImp、CEU、PDU、UNDIAL、RMU。最终实验使用其中 9 个（GradAscent、
GradDiff、PureNPO、SimNPO、SatImp、WGA、CEU、PDU、RMU），覆盖 3 个模型 × 2 个数据集。

> **Trace 不在本框架内。** 最终配置目录中的 `Trace.yaml` 属于另一套代码，仅作为实验
> 记录保留，不能用本仓库的脚本运行。

## 在线评测：WMDP 与 MMLU

Megatron 训练支持**在线评测**：在指定的训练步直接用显存中的模型参数评测 WMDP 和 MMLU，
不需要导出 checkpoint，也不需要另起评测任务，可以在一次训练中得到完整的遗忘曲线。

| 任务名 | Benchmark | 用途 |
| --- | --- | --- |
| `wmdp_bio` | WMDP-Bio | 生物危险知识（遗忘目标） |
| `wmdp_cyber` | WMDP-Cyber | 网络安全危险知识（遗忘目标） |
| `mmlu` | MMLU | 通用能力保持 |

- prompt 构造和打分复用 lm-eval 的 task 定义，指标为 lm-eval 的 `acc`，与离线评测口径一致。
- 题目在数据并行的各个 rank 之间切分后汇总，并可把多道题合并为一次 forward（`batch_size`）。
- 可按任务限制题量，例如 `mmlu: 35` 表示每个 MMLU 子任务取 35 题（共约 2000 题），
  未限制的任务（如 WMDP）使用全量。

在训练 YAML 的 `backends.megatron.evaluation` 中开启：

~~~yaml
backends:
  megatron:
    evaluation:
      enabled: true
      tasks: [wmdp_bio, mmlu]          # Cyber 实验用 [wmdp_cyber, mmlu]
      num_fewshot: 0
      task_limits: {mmlu: 35}          # 每个 MMLU 子任务 35 题；WMDP 全量
      batch_size: 16                   # 每次 forward 的题数，只影响速度
      steps: [1, 300, 700, 1000, 1400, 1700, 1800, 1900, 2000, 2100, 2142]
      output_dir: /path/to/online-evaluation
~~~

每个评测步向 `<output_dir>/<run>_online_eval.jsonl` 追加一行，训练过程中即可查看：

~~~json
{"run": "qwen35-main-grad-diff", "step": 300, "unix_time": 1790567257.7, "wmdp_bio": 0.8295, "mmlu": 0.8341}
~~~

最终实验的 54 份配置都已开启在线评测（bio 为 `[wmdp_bio, mmlu]`，cyber 为
`[wmdp_cyber, mmlu]`）。在线评测目前要求 `pipeline_parallel: 1` 和 `context_parallel: 1`。导出的
checkpoint 也可以用 `scripts/eval.sh` 做完整的离线评测，见 [Runbook](RUNBOOK.md)。

## 快速开始

~~~bash
# 1. 构造 data cache（示例：Qwen3.5 / WMDP-Bio forget）
bash scripts/build_xtuner_cache_example.sh

# 2. 检查训练配置并查看最终参数
CONFIG=configs/training/megatron/final-training-configs/qwen/bio/GradDiff.yaml
scripts/inspect.sh --config "${CONFIG}"
scripts/train_megatron.sh "${CONFIG}" --dry-run

# 3. 在 8 卡节点上训练
scripts/train_megatron.sh "${CONFIG}"
~~~

模型权重、data cache 和 checkpoint 都在共享存储上，不包含在 Git 仓库中。

## 目录结构

~~~text
configs/
├── data_processing/
│   ├── xtuner/                         XTuner soft-pack 构建配置（模型 × 数据集）
│   └── document_chunk/                 document chunk 构建参数（模型 × 数据集）
└── training/
    ├── megatron/final-training-configs/  最终实验训练配置（模型 × 数据集 × 算法）
    └── deepspeed/interns2/             DeepSpeed 后端示例
scripts/        构建 cache、训练、评测和 RJob 提交的入口脚本
src/
├── megatron_unlearning/                Megatron 训练 runtime、算法目标函数、在线评测
├── open_unlearning/                    统一 CLI 与配置校验
├── data/                               两种 cache 的构建与读取
└── trainer/ evals/                     上游 HF/DeepSpeed 实现与评测
evaluations/    正式评测结果
main/           最终批量实验的运行记录与评测汇总
docs/           工程文档
tests/          单元测试
~~~

## 文档

| 文档 | 内容 |
| --- | --- |
| [RUNBOOK.md](RUNBOOK.md) | 构造 data cache、运行算法、RJob 提交与评测 |
| [configs/data_processing/README.md](configs/data_processing/README.md) | 两种 cache 的格式与参数 |
| [docs/architecture.md](docs/architecture.md) | 后端结构与代码分层 |
| [docs/README.md](docs/README.md) | 全部工程文档索引 |
| [evaluations/README.md](evaluations/README.md) | 评测结果索引 |

## 上游与许可证

基于 [locuslab/open-unlearning](https://github.com/locuslab/open-unlearning)，采用
[MIT License](LICENSE)。
