# 配置目录

新的公开配置由少量可组合组件构成：

- `models/`：模型路径、family、专家数以及对应的旧 Hydra model 名。
- `datasets/`：forget/retain cache、序列长度和 pad token。
- `algorithms/`：后端共享的算法名与超参数。
- `profiles/`：global batch、优化器超参数、训练长度和输出位置。
- `backends/`：DeepSpeed 或 Megatron 独有的并行布局。
- `experiments/`：只组合以上组件并覆盖少数实验特例。

三个 `experiments/*_pure_npo.yaml` 示例分别覆盖 Qwen3.5、Intern-S2 和 Gemma 4，
并同时启用 DeepSpeed 和 Megatron。
两边保持相同 global batch、学习率、训练步数、seed 和算法参数；micro batch、梯度
累积和 TP/PP/EP 等后端细节由适配器处理。

环境变量采用 `${NAME:-default}` 语法。建议从根目录 `.env.example` 复制，不要把
真实集群路径或密钥提交到仓库。

不需要为每个算法复制 shell 文件。可以从任一同模型实验替换组件；名称和输出目录
会自动按新组件生成：

```bash
scripts/train_megatron.sh configs/experiments/interns2_pure_npo.yaml \
  --algorithm grad_diff --profile wmdp_moe_default --dry-run
```

也可用 `--model qwen35 --dataset qwen35_wmdp_bio` 成对替换模型和 tokenizer cache，
或用 `--name`、`--output-dir` 显式指定实验标识和权重目录。

旧 `configs/experiment`、`configs/trainer` 等目录仍服务于上游 HF/DeepSpeed 内部，
不再作为新 RJob 脚本的参数来源。完成结果回归前不要删除这些兼容配置。

## 统一 CLI 能力矩阵

| 算法 | DeepSpeed | Megatron | 备注 |
| --- | --- | --- | --- |
| GradAscent / GradDiff | 是 | 是 | 标准 cache |
| PureNPO | 是 | 是 | forget-only，使用冻结 reference |
| TokenNPO | 否 | 是 | Megatron token-level reference objective |
| TokenSimNPO / WGA / SatImp | 是 | 是 | 需要 retain cache |
| CEU / PDU / UNDIAL | 是 | 是 | UNDIAL `alpha=0` 时不加载 retain |
| RMU | 兼容入口 | 是 | DeepSpeed 的 Qwen/Gemma loader 尚未并入统一适配器 |
| DPO / AltPO | 兼容入口 | 否 | 需要偏好/QA 数据而非当前 WMDP 文本 cache |

适配器会在提交任务前拒绝矩阵中不支持的组合，避免作业启动后才失败。
