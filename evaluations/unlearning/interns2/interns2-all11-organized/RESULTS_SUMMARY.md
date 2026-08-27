# Intern-S2 本轮遗忘结果总结

更新时间：2026-08-27

- 遗忘目标：WMDP-Bio；准确率越低，表示遗忘越强。
- 通用能力：MMLU 总体准确率；准确率越高，表示能力保持越好。
- 数值均为 `accuracy ± standard error`，并换算为百分比。

## 已完成结果

| 算法 | Checkpoint | WMDP-Bio ↓ | MMLU ↑ |
| --- | --- | ---: | ---: |
| Base | 未训练 | 83.66% ± 1.04% | 85.15% ± 0.29% |
| CEU | step 300 | 84.29% ± 1.02% | 85.14% ± 0.29% |
| CEU | epoch 1 | 77.06% ± 1.18% | 79.97% ± 0.32% |
| GradAscent | step 300 | 37.08% ± 1.35% | 47.42% ± 0.39% |
| GradAscent | epoch 1 | 24.67% ± 1.21% | 23.53% ± 0.36% |
| GradDiff | step 300 | 77.14% ± 1.18% | 84.72% ± 0.29% |
| GradDiff | epoch 1 | 69.21% ± 1.29% | 84.26% ± 0.29% |
| PureNPO | step 300 | 78.79% ± 1.15% | 79.47% ± 0.31% |
| PureNPO | epoch 1 | 39.28% ± 1.37% | 61.22% ± 0.37% |
| RMU | step 300 | 71.17% ± 1.27% | 85.04% ± 0.29% |
| RMU | epoch 1 | 68.26% ± 1.31% | 84.85% ± 0.29% |
| SatImp | step 300 | 83.50% ± 1.04% | 84.95% ± 0.29% |
| SatImp | epoch 1 | 83.74% ± 1.03% | 84.72% ± 0.29% |
| TokenNPO | step 300 | 72.58% ± 1.25% | 69.79% ± 0.35% |
| TokenNPO | epoch 1 | 76.20% ± 1.19% | 69.12% ± 0.38% |
| TokenSimNPO | step 300 | 82.25% ± 1.07% | 84.70% ± 0.29% |
| TokenSimNPO | epoch 1 | 79.26% ± 1.14% | 84.16% ± 0.30% |
| UNDIAL | step 300 | 83.11% ± 1.05% | 84.52% ± 0.29% |
| UNDIAL | epoch 1 | 82.48% ± 1.07% | 84.28% ± 0.29% |
| WGA | step 300 | 83.27% ± 1.05% | 84.92% ± 0.29% |
| WGA | epoch 1 | 83.50% ± 1.04% | 84.57% ± 0.29% |

## 当前观察

- 未训练 Base 模型的 WMDP-Bio 为 83.66%，MMLU 为 85.15%；各算法应相对这一行判断遗忘强度和能力损失。
- GradAscent 的遗忘最强，但 MMLU 同时严重下降，属于破坏性遗忘。
- GradDiff 在 epoch 1 将 WMDP-Bio 降至 69.21%，同时将 MMLU 保持在 84.26%，遗忘强度和能力保持明显优于本轮 CEU。
- PureNPO 在 epoch 1 获得较强遗忘，但 MMLU 下降明显。
- RMU 的 MMLU 基本保持在 85%，但当前 WMDP-Bio 降幅相对有限。
- SatImp 两个 checkpoint 的 WMDP-Bio 都与 Base 接近，MMLU 基本保持，当前配置没有产生明显遗忘。
- CEU 在 step 300 基本保持 MMLU，训练到 epoch 1 后遗忘增强，同时通用能力开始下降。
- TokenNPO 从 step 300 训练到 epoch 1 后，WMDP-Bio 反而升高，MMLU 略微下降；当前结果不支持继续延长训练。
- TokenSimNPO 训练到 epoch 1 后 WMDP-Bio 比 Base 低约 4.40 个百分点，MMLU 下降约 0.99 个百分点，属于温和遗忘。
- UNDIAL 在 step 300 和 epoch 1 都基本保持 MMLU，但 WMDP-Bio 仅比 Base 低约 0.55–1.18 个百分点，当前遗忘效果很弱。
- WGA 两个 checkpoint 的 WMDP-Bio 都与 Base 接近，当前配置几乎没有产生有效遗忘。

## 待补齐算法

以下算法仍在补跑，完成后应在本表中增加 step 300 和 epoch 1 两行：

- PDU

对应原始数据为各算法目录中的 `step300.json` 和 `epoch1.json`。
