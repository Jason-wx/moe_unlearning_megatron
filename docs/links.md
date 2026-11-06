# 参考资料

## 项目

- [OpenUnlearning technical report](https://arxiv.org/abs/2506.12618)
- [OpenUnlearning upstream repository](https://github.com/locuslab/open-unlearning)

## 当前训练算法

| 算法 | 论文或实现 |
| --- | --- |
| GradAscent / GradDiff | 常用遗忘基线，可参考 [TOFU](https://arxiv.org/abs/2401.06121) 和 [MUSE](https://arxiv.org/abs/2407.06460) |
| NPO / PureNPO / TokenNPO | [Negative Preference Optimization](https://arxiv.org/abs/2404.05868)，[官方实现](https://github.com/licong-lin/negative-preference-optimization) |
| SimNPO / TokenSimNPO | [Simple and Efficient Unlearning](https://arxiv.org/abs/2410.07163)，[官方实现](https://github.com/OPTML-Group/Unlearn-Simple) |
| WGA | [G-effect](https://arxiv.org/abs/2502.19301)，[实现](https://github.com/tmlr-group/G-effect) |
| SatImp | [论文](https://arxiv.org/abs/2505.11953)，[实现](https://github.com/Puning97/SatImp-for-LLM-Unlearning) |
| CEU | [Cross-Entropy Unlearning](https://arxiv.org/abs/2503.01224) |
| PDU | [Primal-Dual Unlearning](https://arxiv.org/abs/2506.05314) |
| UNDIAL | [论文](https://arxiv.org/abs/2402.10052)，[实现](https://github.com/dong-river/LLM_unlearning) |
| RMU | [WMDP / RMU](https://arxiv.org/abs/2403.03218)，[实现](https://github.com/centerforaisafety/wmdp/tree/main/rmu) |

PureNPO、TokenNPO 和 TokenSimNPO 是本仓库明确区分的目标函数变体，具体语义见
[architecture.md](architecture.md)。DPO 和 AltPO 只保留在兼容或 community 代码中，
不属于当前统一训练配置。

## Benchmark 与评测

- [WMDP](https://arxiv.org/abs/2403.03218)
- [MMLU](https://arxiv.org/abs/2009.03300)
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [TOFU](https://arxiv.org/abs/2401.06121)
- [MUSE](https://arxiv.org/abs/2407.06460)

最终 MoE 实验在 WMDP-Bio、WMDP-Cyber 上遗忘并用 MMLU 衡量通用能力；TOFU/MUSE 代码属于上游兼容实现。

## 延伸阅读

- [Machine Unlearning in 2024](https://ai.stanford.edu/~kzliu/blog/unlearning)
- [Rethinking Machine Unlearning for Large Language Models](https://arxiv.org/abs/2402.08787)
- [Awesome LLM Unlearning](https://github.com/chrisliu298/awesome-llm-unlearning)
- [Awesome Machine Unlearning](https://github.com/tamlhp/awesome-machine-unlearning)
