# Contributing

## 开发环境

~~~bash
cd /mnt/shared-storage-user/safethm-share/wangxiao1/Safe_MoE/open-unlearning
conda activate /mnt/shared-storage-user/safethm-share/wangxiao1/Safe_MoE/open-unlearning/envs/open-unlearning
~~~

从独立分支修改，不要提交 envs、data_cache、saves、outputs 或 wandb 离线数据。

## 代码位置

| 修改内容 | 位置 |
| --- | --- |
| 稳定 CLI 和共享配置 | src/open_unlearning/ |
| Megatron runtime 与目标函数 | src/megatron_unlearning/ |
| DeepSpeed/HF trainer | src/trainer/ |
| 数据集和 cache | src/data/ |
| 评测实现 | src/evals/ |
| 最终实验训练配置 | configs/training/megatron/final-training-configs/ |
| 数据构建配置 | configs/data_processing/ |
| 入口脚本 | scripts/ |
| 社区算法和 benchmark | community/ |

configs/internal/deepspeed 只是兼容层，不应保存实验超参。

## 新增或修改算法

1. 在对应后端实现目标函数或 trainer（Megatron 为 `src/megatron_unlearning/objectives.py`）。
2. 在后端能力表中登记算法：`src/open_unlearning/configuration.py` 的 `_SUPPORTED_ALGORITHMS`，
   以及 Megatron 的 `src/megatron_unlearning/config.py` 中的 `SUPPORTED_ALGORITHMS`。
3. 为每个支持的后端添加一份自包含训练 YAML。
4. 添加目标函数、配置映射和 batch 边界测试。
5. 用 inspect 和 dry-run 检查最终配置，再运行实际训练与评测。
6. 将正式结果和原始 JSON 放到 evaluations/，并更新对应汇总。

算法论文、默认超参和与原实现的差异应写进训练配置文件开头的注释，不要新增临时调试日志。

## 提交前检查

~~~bash
PYTHONPATH=src envs/open-unlearning/bin/python -m pytest -q
make quality
git diff --check
~~~

Pull request 应说明修改范围、验证命令、使用的训练配置以及结果路径。不要提交模型权重、
token cache、运行日志或密钥。
