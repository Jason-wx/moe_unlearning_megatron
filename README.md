# OpenUnlearning — MoE 双分布式后端

本项目基于 [OpenUnlearning](https://github.com/locuslab/open-unlearning)，增加了
Qwen3.5 MoE、Intern-S2、Gemma 4 适配，并同时维护两个训练后端：

- **DeepSpeed**：Transformers Trainer + Accelerate + ZeRO-3；
- **Megatron**：项目维护的 Megatron-Core / MCore-Bridge 训练循环。

两套后端共享模型、数据、算法和训练配置；只有 micro batch、TP/PP/EP 等后端细节
分别配置。Megatron 是当前主流路径，DeepSpeed 入口继续作为一等公民保留。

正式训练请直接参考：[Megatron 完整训练流程](RUNBOOK.md)。

## 快速开始

```bash
cp .env.example .env
set -a; source .env; set +a

# 合并并检查配置
scripts/inspect.sh --config configs/experiments/interns2_pure_npo.yaml

# 先检查两个后端实际参数
scripts/train_deepspeed.sh configs/experiments/interns2_pure_npo.yaml --dry-run
scripts/train_megatron.sh  configs/experiments/interns2_pure_npo.yaml --dry-run

# RJob：默认 Megatron；BACKEND=deepspeed 可切换
scripts/submit_rjob_example.sh

# Megatron：训练 300 steps，只在终点保存一个 checkpoint
BACKEND=megatron JOB_NAME=interns2-npo-300steps \
scripts/submit_rjob_example.sh \
  --train-steps 300 --save-only-at-end \
  --output-dir saves/interns2-pure-npo-300steps

# Megatron：完整训练 1 epoch，保存 step 300 和 epoch 1
scripts/train_megatron.sh configs/experiments/interns2_pure_npo.yaml \
  --train-epochs 1 --save-at-step 300 --save-at-epoch-end --dry-run
```

不需要为算法复制 shell 文件：

```bash
scripts/train_megatron.sh configs/experiments/interns2_pure_npo.yaml \
  --algorithm grad_diff --profile wmdp_moe_default --dry-run
```

## 目录

```text
src/open_unlearning/       稳定 CLI、共享配置和双后端适配
src/megatron_unlearning/   Megatron/MCore 训练实现
src/trainer/               DeepSpeed/HF trainers
src/data/                  共享数据与 token cache loader
src/evals/                 上游评测实现
configs/                   组件、后端布局和实验
scripts/                   维护入口与唯一 RJob 示例
tests/                     全部源码测试
evaluations/               原始评测结果与汇总
docs/                      工程、迁移和实验文档
```

## 运行资产

以下目录包含环境、数据、权重或结果，不属于源码，也不会被自动删除：

- `envs/`
- `data_cache/`
- `saves/`
- `outputs/`
- `wandb/`、`wandb_offline/`
- `scripts/**/logs/`
- `evaluations/**/lm_eval_results_*/`

当前布局保持兼容。新部署推荐设置：

```bash
export OU_RUNTIME_ROOT=/path/to/open-unlearning-runtime
```

详见 [运行资产规则](docs/runtime-layout.md)。

## 验证

```bash
PYTHONPATH=src envs/open-unlearning/bin/python -m unittest discover -s tests -v
RUFF_CACHE_DIR=/tmp/open-unlearning-ruff \
  envs/open-unlearning/bin/python -m ruff check src/open_unlearning tests
```

## 文档

- [架构](docs/architecture.md)
- [配置与算法能力矩阵](configs/README.md)
- [运行资产布局](docs/runtime-layout.md)
- [重构状态](docs/refactor-status.md)
- [Megatron 迁移记录](docs/megatron_migration.md)
- [正式 Intern-S2 结果](evaluations/unlearning/interns2/RESULTS_SUMMARY.md)

## 上游与许可证

原始 OpenUnlearning 由 Vineeth Dorna、Anmol Mekala 等维护。使用本项目时请同时遵循
上游项目的引用要求。技术报告：

> OpenUnlearning: Accelerating LLM Unlearning via Unified Benchmarking of Methods
> and Metrics, arXiv:2506.12618.

项目采用 [MIT License](LICENSE)。
