# Interns2 Megatron V3 Summary

V3 共包含 13 组算法的 epoch-1 `WMDP + MMLU` 评测结果。完整汇总表（含总体指标、
MMLU 子集及原始结果目录）见 [RESULTS_SUMMARY.md](RESULTS_SUMMARY.md)。

训练配置：Megatron 后端、8 GPU（TP=4、PP=2、EP=4）、1 epoch、global batch
size=4、micro batch size=4、BF16、Adam + cosine scheduler，gradient clipping
沿用默认开启（`gradient_clip=1.0`）；各算法共享模型、数据
和并行配置，学习率及算法特定超参数有所不同，详见
[RESULTS_SUMMARY.md](RESULTS_SUMMARY.md) 的“训练配置”章节。

注：UNDIAL 使用显存优化重试配置，`micro_batch_size=1`、有效梯度累计 4。

## 关键结论

- **RMU 是最佳综合折中**：MMLU 最高，为 85.02%；WMDP-Bio 为 52.00%。
- **Grad-Ascent 遗忘最强**：WMDP-Bio 为 24.35%，但 MMLU 只有 24.16%，通用能力严重下降；CEU 结果相近（24.67% / 22.95%）。
- **NPO/PDU 能力保持较好但遗忘弱**：MMLU 分别为 84.87% 和 84.88%，而 WMDP-Bio
  仍高达 84.37% 和 83.35%。
- **WGA/Token-NPO 的能力损失明显**：WMDP-Bio 较低（33.46% / 37.39%），但 MMLU
  仅 33.91% / 37.79%。
- **SatImp** 在遗忘和保持之间的结果为 WMDP-Bio 59.62%、MMLU 66.77%，不如 RMU。
- **SimNPO** 的 WMDP-Bio 为 83.90%、MMLU 为 84.71%，能力保持较好但遗忘效果较弱。
- **Token-SimNPO** 的 WMDP-Bio 为 81.23%、MMLU 为 84.39%，通用能力保持较好但遗忘有限。
- **UNDIAL** 的 WMDP-Bio 为 82.09%、MMLU 为 84.01%，通用能力保持较好但遗忘效果较弱。

指标解释：WMDP-Bio 越低表示遗忘越充分，MMLU 越高表示通用能力保持越好。
