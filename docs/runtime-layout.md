# 运行目录

## 共享集群位置

| 内容 | 路径 |
| --- | --- |
| 仓库 | /mnt/shared-storage-user/safethm-share/wangxiao1/Safe_MoE/open-unlearning |
| Conda 环境 | /mnt/shared-storage-user/safethm-share/wangxiao1/Safe_MoE/open-unlearning/envs/open-unlearning |
| 数据 cache | 仓库内 data_cache/ |
| DeepSpeed checkpoint | 仓库内 saves/ 或训练 YAML 的 training.output_dir |
| Megatron checkpoint | 训练 YAML 的 backends.megatron.output_root/training.output_name |
| Megatron 在线评测 | 训练 YAML 的 backends.megatron.evaluation.output_dir |
| 运行日志 | 仓库内 outputs/ |
| 评测结果 | 仓库内 evaluations/ |
| W&B 离线记录 | 仓库内 wandb_offline/ |

这些目录包含环境、预处理数据、权重或实验结果。它们不属于源码清理范围，也不应提交到
Git。

## 路径变量

RuntimePaths 支持用环境变量覆盖默认位置：

| 变量 | 内容 |
| --- | --- |
| OU_REPO_ROOT | 仓库根目录 |
| OU_RUNTIME_ROOT | 统一的外部运行根目录 |
| OU_ENV_ROOT | Python 环境根目录 |
| OU_CACHE_ROOT | 数据 cache 根目录 |
| OU_CHECKPOINT_ROOT | 通用 checkpoint 根目录 |
| OU_RUN_ROOT | 日志和运行输出根目录 |
| OU_EVAL_ROOT | 评测结果根目录 |
| OU_WANDB_ROOT | W&B 目录 |

未设置 OU_RUNTIME_ROOT 时使用上表中的仓库内兼容布局。设置后默认使用：

~~~text
<OU_RUNTIME_ROOT>/
├── envs/open-unlearning/
├── caches/
├── checkpoints/
├── runs/
├── evaluations/
└── wandb/
~~~

实际解析结果可通过下面的命令查看：

~~~bash
python -m open_unlearning.cli paths
~~~
