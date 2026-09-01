# OpenUnlearning

本仓库在 [OpenUnlearning](https://github.com/locuslab/open-unlearning) 基础上，增加了
Intern-S2、Qwen3.5 MoE 和 Gemma 4 的适配，支持 DeepSpeed 与 Megatron 两套训练后端。
当前正式实验以 Megatron 为主。

> 第一次使用、准备数据、修改配置、启动训练或运行评测，请直接阅读
> **[Runbook](RUNBOOK.md)**。

## 项目状态

- 支持 document chunk 与 XTuner soft-pack 两套数据准备方式。
- Intern-S2 已有 13 种 Megatron 算法配置。
- Intern-S2 V1、V2 已完成汇总，V3 已完成 GradDiff、NPO、PureNPO、RMU。
- Gemma 4、Intern-S2、Qwen3.5 MoE 已保存基础性能和 RMU 对比结果。

详细状态见 [实验进展](docs/experiment-progress.md)。

## 环境

共享集群 Conda 环境：

```text
/mnt/shared-storage-user/safethm-share/wangxiao1/Safe_MoE/open-unlearning/envs/open-unlearning
```

## 常用入口

| 内容 | 位置 |
| --- | --- |
| 完整使用流程 | [RUNBOOK.md](RUNBOOK.md) |
| 训练配置 | [configs/training/](configs/training/) |
| 数据处理配置 | [configs/data_processing/](configs/data_processing/) |
| 一键启动脚本 | [scripts/](scripts/) |
| 评测结果 | [evaluations/](evaluations/) |
| 工程文档 | [docs/](docs/) |

## 文档

| 文档 | 内容 |
| --- | --- |
| [Runbook](RUNBOOK.md) | 数据准备、配置、启动、RJob、训练与评测 |
| [实验进展](docs/experiment-progress.md) | 当前完成情况、版本状态和结果入口 |
| [文档索引](docs/README.md) | 所有长期维护文档 |
| [配置说明](configs/README.md) | 配置结构、算法与后端支持情况 |
| [Intern-S2 结果](evaluations/unlearning/interns2/RESULTS_SUMMARY.md) | V1、V2、V3 数据汇总 |
| [全部评测结果](evaluations/README.md) | Baseline、RMU 和 unlearning 结果索引 |

## 主要目录

```text
configs/       数据处理与训练配置
scripts/       配置检查、训练、评测和 RJob 入口
src/           DeepSpeed、Megatron 与公共实现
evaluations/   正式评测结果
docs/          长期维护文档
```

## 上游与许可证

原始项目：[locuslab/open-unlearning](https://github.com/locuslab/open-unlearning)。
本项目采用 [MIT License](LICENSE)。
