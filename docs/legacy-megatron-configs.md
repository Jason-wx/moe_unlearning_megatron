# 旧 Megatron 配置归档

早期迁移阶段曾为三个模型和每个算法分别维护 `configs/megatron/*.yaml`，共 35 份。
这些文件已由共享组件配置取代，下面保留其参数来源，便于回放历史实验。

## 共同布局

- 8 GPU，TP=4，PP=2，CP=1，EP=4，ETP=1；
- sequence parallel 和 distributed optimizer 开启；
- 基础实验 micro batch=1、global batch=32、seed=3、bf16；
- Qwen3.5 和 Intern-S2 为 256 experts，Gemma 4 为 128 experts；
- forget/retain 分别使用各 tokenizer 的 WMDP-Bio/FineWeb-Edu cache。

## 算法参数

| 算法 | 学习率 | 关键参数 |
| --- | ---: | --- |
| GradAscent | `5e-6` | `gamma=1` |
| GradDiff | `5e-6` | `gamma=1, alpha=1` |
| NPO / TokenNPO | 基础 `5e-6` | `beta=0.1, gamma=1` |
| SimNPO | `9.5e-6` | `beta=0.1, delta=0, gamma=1, alpha=0.1` |
| WGA | `1e-5` | `beta=1, gamma=1, alpha=1` |
| SatImp | `1e-5` | `beta1=5, beta2=1, gamma=1, alpha=0.1` |
| CEU | `1e-5` | `ignore_first_n_tokens=1` |
| PDU | `1e-5` | `alpha=100, retain_loss_eps=0.3, dual_step_size=5` |
| UNDIAL | 基础 `5e-6` | `beta=10, gamma=1, alpha=0` |
| RMU | `1e-5` | layer 9；训练 7/8/9；`alpha=350, steering=300` |

Intern-S2 aligned NPO 是特例：micro batch=16、global batch=128、1344 steps、
learning rate=`1e-6`、Adam betas=`0.9/0.999`、gradient clip=`0`、seed=42，按 epoch
保存。该配置现在由 `configs/experiments/interns2_pure_npo.yaml` 表达。

旧 single/distributed roundtrip 配置仅用于零步 HF→MCore→HF 转换验证，其结果仍保留
在 `evaluations/unlearning/interns2_megatron/lm_eval_results_mcore_*`。
