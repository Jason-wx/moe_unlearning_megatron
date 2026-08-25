# 运行资产与迁移规则

## 保护清单

以下目录包含 RJob 环境、预处理数据、权重或正式评测结果，不是垃圾文件。运行日志属于
可再生成的临时资产，已于 2026-08-25 按要求清理；其余资产继续从 Git/source 视图中
隔离并原地保留：

| 现有目录 | 内容 | 当前处理 |
| --- | --- | --- |
| `envs/` | RJob 正在使用的 Python 环境 | 原地保留；禁止直接移动 |
| `data_cache/` | 已分词训练缓存 | 原地保留；由 `OU_CACHE_ROOT` 抽象 |
| `saves/` | 模型权重与 checkpoint | 原地保留；两后端分子目录 |
| `outputs/` | 非日志运行产物 | 原地保留；新日志仍可写入但不提交 Git |
| `wandb/`, `wandb_offline/` | W&B 离线 run 数据 | 原地保留；其中 debug/output 日志已清理 |
| `scripts/**/logs/` | 可再生成的提交日志 | 当前已清空；后续生成内容继续忽略 |
| `evaluations/**/lm_eval_results_*` | 正式 lm-eval 原始结果 | 保留，汇总文件继续链接原始 JSON |

这些路径均在 `.gitignore` 中显式忽略。除运行日志外，忽略表示“不提交”，不表示
“可删除”。

## 当前兼容布局

未设置任何 `OU_*` 变量时，解析器继续使用上述仓库内目录，确保正在运行和已有 RJob
不受影响。可用 `scripts/inspect.sh` 或 `ou paths` 查看实际解析结果。

## 推荐的新部署布局

```text
open-unlearning/             小型源码仓库
open-unlearning-runtime/
  envs/open-unlearning/      重建的环境
  caches/                    数据缓存
  checkpoints/               权重/checkpoint
  runs/                      作业日志和临时输出
  evaluations/               lm-eval 原始结果
  wandb/                     W&B 离线记录
```

设置 `OU_RUNTIME_ROOT` 即可切换；也可分别设置 `OU_ENV_ROOT`、`OU_CACHE_ROOT`、
`OU_CHECKPOINT_ROOT`、`OU_RUN_ROOT`、`OU_EVAL_ROOT`、`OU_WANDB_ROOT`。
Megatron 的 HF 权重导出默认使用
`/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/wangxiao1/checkpoints/unlearn/megatron`，
可通过 `OU_MEGATRON_CHECKPOINT_ROOT` 单独覆盖，不影响 DeepSpeed 输出。

## 安全迁移步骤

1. 等待当前训练及其自动评测完全结束。
2. 记录每个目录的大小、文件数和关键结果 manifest。
3. 在 runtime 根目录复制或原子移动数据；权重和结果先校验数量与校验和。
4. 在新位置重新创建 Python 环境并安装项目，不能直接移动旧 `envs/`。
5. 用 `ou inspect --check-paths` 和两个后端的 `--dry-run` 验证。
6. 完成一次短训练、checkpoint 保存和 lm-eval 后，才移除旧副本。

当前活动任务未完成前，不执行第 3 至第 6 步。
