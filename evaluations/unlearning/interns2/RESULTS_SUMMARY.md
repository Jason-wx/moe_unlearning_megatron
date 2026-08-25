# Intern-S2 Unlearning 实验结果汇总

更新时间：2026-08-19

## 1. 评测说明

- 模型：Intern-S2-base05。
- 遗忘目标：WMDP-Bio。
- `WMDP-Bio Accuracy`：越低表示模型在目标遗忘数据上的正确率越低，即遗忘越强。
- `MMLU Accuracy`：越高表示通用知识与推理能力保持得越好。
- 表中数值为 `accuracy ± standard error`，均换算为百分比。
- 当前目录没有未执行 unlearning 的基础模型结果，因此本表只比较不同 unlearning 方法，不能直接计算相对基础模型的变化量。

## 2. 正式结果

结果按 WMDP-Bio Accuracy 从低到高排列。

| 排名 | 方法 | WMDP-Bio ↓ | MMLU ↑ | 简要观察 |
|---:|---|---:|---:|---|
| 1 | CEU | 24.04% ± 1.20% | 27.44% ± 0.37% | 遗忘很强，但通用能力几乎完全损坏 |
| 2 | GradAscent | 24.12% ± 1.20% | 26.74% ± 0.37% | 遗忘很强，但通用能力几乎完全损坏 |
| 3 | RMU | 37.16% ± 1.35% | 83.43% ± 0.30% | 当前结果中遗忘与能力保持的综合表现最好 |
| 4 | PureNPO | 38.88% ± 1.37% | 62.04% ± 0.37% | 遗忘明显，但通用能力下降严重 |
| 5 | GradDiff | 70.38% ± 1.28% | 83.85% ± 0.30% | 保持能力较好，遗忘程度中等 |
| 6 | TokenSimNPO | 74.39% ± 1.22% | 84.16% ± 0.30% | 比原版 SimNPO 遗忘更强，MMLU 基本保持 |
| 7 | PDU | 80.75% ± 1.11% | 83.30% ± 0.30% | 能力保持尚可，但遗忘效果较弱 |
| 8 | SimNPO | 81.30% ± 1.09% | 83.88% ± 0.30% | 能力保持较好，但遗忘效果较弱 |
| 9 | WGA | 82.09% ± 1.08% | 84.10% ± 0.30% | 能力保持较好，但遗忘效果较弱 |
| 10 | NPO | 82.56% ± 1.06% | 84.89% ± 0.29% | MMLU 很高，但几乎没有明显遗忘 |
| 11 | SatImp | 82.72% ± 1.06% | 83.90% ± 0.30% | 能力保持较好，但几乎没有明显遗忘 |
| 12 | UNDIAL | 83.97% ± 1.03% | 84.89% ± 0.29% | MMLU 很高，但目标遗忘最弱 |

## 3. 关键对比

### NPO 与 PureNPO

| 方法 | WMDP-Bio ↓ | MMLU ↑ |
|---|---:|---:|
| NPO | 82.56% ± 1.06% | 84.89% ± 0.29% |
| PureNPO | 38.88% ± 1.37% | 62.04% ± 0.37% |

PureNPO 相比原版 NPO 将 WMDP-Bio 降低了 43.68 个百分点，但 MMLU 同时降低了 22.85 个百分点。它显著增强了遗忘，同时造成了严重的通用能力损失。

### SimNPO 与 TokenSimNPO

| 方法 | WMDP-Bio ↓ | MMLU ↑ |
|---|---:|---:|
| SimNPO | 81.30% ± 1.09% | 83.88% ± 0.30% |
| TokenSimNPO | 74.39% ± 1.22% | 84.16% ± 0.30% |

TokenSimNPO 相比原版 SimNPO 将 WMDP-Bio 降低了 6.91 个百分点，MMLU 反而提高约 0.28 个百分点。该改动在不明显损害通用能力的情况下增强了遗忘。

## 4. 总体结论

1. **RMU 的综合表现最突出**：WMDP-Bio 为 37.16%，同时保留了 83.43% 的 MMLU，明显优于同等遗忘强度附近的 PureNPO。
2. **CEU 和 GradAscent 属于破坏性遗忘**：WMDP-Bio 已接近四选一随机水平，但 MMLU 也降至约 27%，说明它们不是只移除目标知识，而是大范围损坏了模型能力。
3. **PureNPO 的遗忘很强但代价过高**：相较原版 NPO，它解决了“几乎不遗忘”的问题，但 MMLU 降幅过大。
4. **TokenSimNPO 是有效的温和改进**：它优于原版 SimNPO，在维持 MMLU 的同时获得了更低的 WMDP-Bio，但遗忘强度仍明显弱于 RMU。
5. **NPO、PDU、SatImp、SimNPO、WGA 和 UNDIAL 的正式结果整体偏保守**：它们保留了较高 MMLU，但 WMDP-Bio 仍然很高，说明当前参数下目标遗忘不足。

## 5. 原始结果目录

| 方法 | 结果文件 |
|---|---|
| CEU | `lm_eval_results_ceu_wmdp_mmlu/result.json` |
| GradAscent | `lm_eval_results_grad_ascent_wmdp_mmlu/result.json` |
| GradDiff | `lm_eval_results_grad_diff_wmdp_mmlu/result.json` |
| NPO | `lm_eval_results_npo_wmdp_mmlu/result.json` |
| PDU | `lm_eval_results_pdu_wmdp_mmlu/result.json` |
| PureNPO | `lm_eval_results_pure_npo_wmdp_mmlu/result.json` |
| RMU | `lm_eval_results_rmu_wmdp_mmlu/result.json` |
| SatImp | `lm_eval_results_satimp_wmdp_mmlu/result.json` |
| SimNPO | `lm_eval_results_simnpo_wmdp_mmlu/result.json` |
| TokenSimNPO | `lm_eval_results_token_simnpo_wmdp_mmlu/result.json` |
| UNDIAL | `lm_eval_results_undial_wmdp_mmlu/result.json` |
| WGA | `lm_eval_results_wga_wmdp_mmlu/result.json` |
