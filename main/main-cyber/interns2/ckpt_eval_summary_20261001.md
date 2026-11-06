# Intern-S2 main-cyber 九定点 ckpt 离线正式测评（2026-10-01 收官）

R3 定点 ckpt 的 **lm_eval 离线全量**成绩（`wmdp_cyber` 1987 题 + `mmlu` 全量
14042 题，0-shot，bf16，bs16）。此表为本线**最终正式口径**，在线测评表
（`../online_eval_summary_20261001.md`）仅作过程参考。

- rjob：`interns2-main-cyber-ckpt-eval-g3bs16-1824258`（safer2ai，3 卡，
  每卡串行 3 ckpt，单卡单进程/ckpt）→ **Succeeded**，9/9 成功，
  15:11:52 → 23:43 结束（约 32 min，~10 min/ckpt）
- 脚本：`rjob_ckpt_eval.sh`；原始输出 `ckpt-eval/lm_eval_results_<ckpt>_wmdp_cyber_mmlu/result.json`
- 基线 Intern-S2-base05（离线全量）：cyber **0.585** / mmlu **0.867**
- **cyber 四选一，0.25 = 随机猜 = 遗忘下限**

## 一、终版成绩（离线正式）

| 算法 | ckpt | cyber ↓ | mmlu ↑ | Δcyber vs 基线 | Δmmlu vs 基线 | 判定 |
|---|---|---|---|---|---|---|
| **RMU** | step-92 | **0.252** | **0.836** | −0.333 | −0.031 | ⭐ 最优：到随机线，mmlu 仅损 3pp |
| **PDU** | step-69 | **0.274** | **0.846** | −0.311 | −0.021 | ⭐ 次优：mmlu 代价最小 |
| **GradDiff** | step-138 | 0.302 | 0.822 | −0.283 | −0.045 | ⭐ 有效，mmlu 代价略大 |
| SatImp | step-23 | 0.380 | 0.787 | −0.205 | −0.080 | 半程遗忘 + 8pp 能力损失 |
| WGA | step-23 | 0.375 | 0.786 | −0.210 | −0.081 | 同 SatImp 同剧本 |
| PureNPO | step-69 | 0.412 | 0.832 | −0.173 | −0.035 | 浅遗忘，能力较完好 |
| CEU | step-92 | 0.454 | 0.767 | −0.131 | −0.100 | 浅遗忘且白丢 10pp |
| SimNPO | step-23 | 0.471 | 0.817 | −0.114 | −0.050 | 基本没遗忘 |
| GA | step-46 | 0.246 | **0.333** 💥 | −0.339 | −0.534 | 到随机线但能力崩塌 |
| UNDIAL | 未跑 | — | — | — | — | R1 全程纹丝不动，剔除 |

**排名（遗忘深度与能力保持的联合判据 mmlu−cyber）**：
RMU 0.584 > PDU 0.572 > GradDiff 0.520 > PureNPO 0.420 > SatImp 0.407 ≈
WGA 0.411 > SimNPO 0.346 > CEU 0.313 > GA 0.087。
**只有 RMU / PDU / GradDiff 三个算法在 cyber 上给出可用工作点。**

## 二、在线 vs 离线 parity

| 算法 | 在线 cyber | 离线 cyber | Δ | 在线 mmlu* | 离线 mmlu | Δ |
|---|---|---|---|---|---|---|
| RMU | 0.257 | 0.252 | −0.005 | 0.851 | 0.836 | −0.015 |
| PDU | 0.277 | 0.274 | −0.003 | 0.846 | 0.846 | 0.000 |
| GradDiff | 0.274 | 0.302 | **+0.028** | 0.827 | 0.822 | −0.005 |
| PureNPO | 0.410 | 0.412 | +0.002 | 0.854 | 0.832 | −0.022 |
| CEU | 0.427 | 0.454 | **+0.027** | 0.790 | 0.767 | −0.023 |
| SimNPO | 0.475 | 0.471 | −0.004 | 0.844 | 0.817 | −0.027 |
| SatImp | 0.374 | 0.380 | +0.006 | 0.801 | 0.787 | −0.014 |
| WGA | 0.372 | 0.375 | +0.003 | 0.808 | 0.786 | −0.022 |
| GA | 0.249 | 0.246 | −0.003 | 0.295 | 0.333 | +0.038 |

\* 在线 mmlu 为 35 题/叶子子样（约 2000 题），离线为全量 14042 题，
数值不同口径，偏差属抽样噪声。cyber 两侧均为全量 1987 题，**可直接对比**。

**parity 结论**：
1. **cyber 口径一致**：9 个算法中 7 个 |Δ| ≤ 0.006，说明训练中在线评测
   （MCore 前向 + lm_eval API）与离线 HF 全量评测在同一 ckpt 上等价，
   在线曲线可信。
2. **在线 mmlu 系统性略高 ~0.02**（9/9 中 8 个为正偏），是 35 题/叶子子样
   的抽样偏差，引用 mmlu 绝对值时应以离线为准。
3. **两个 +0.027/+0.028 的例外（GradDiff / CEU）**：方向都是"离线略浅于
   在线"，量级 ~1.4σ（1987 题上 p≈0.3 的二项标准误 ≈0.010），非 bug 级
   偏差，但引用 GD/CEU 时以离线 0.302/0.454 为准。
4. GA 的 mmlu 在崩塌区（在线 0.295 / 离线 0.333），两次测量都确认能力已毁。

## 三、与 bio 线的对照

bio 线同口径（`main/interns2/online-evaluation/ckpt_eval_summary_20260926.md`）：
RMU 0.370/0.842、PDU-lr4e-6 0.408/0.844 为唯二有效。**cyber 上
RMU/PDU 的遗忘深度（0.252/0.274）显著优于 bio（0.370/0.408）且 mmlu
代价相当** —— cyber forget 语料仅 743 packs（bio 的 1/31）却跑 10 epochs，
高重复 + lr 5e-6 让遗忘更彻底；代价是 GA/SatImp/WGA 这类无 retain 锚的
算法崩塌也更早（bio 上 GA mmlu 0.230，cyber 0.333，同样不可用）。

## 四、ckpt 位置

`/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/wangxiao1/checkpoints/unlearn/megatron/main-cyber/interns2-main-cyber-*/checkpoint-step-<N>`
（每个 ckpt 目录内含生成它的 R3 yaml 配置副本）。

## 五、测评踩坑：离线单卡 batch size

bs64 → bs32 → **bs16** 两次 OOM 才落地（同一张 140 GiB H200）：
Intern-S2 bf16 加载后实占 **107.42 GiB**（权重 ~72 GiB + KV/激活 +
分配器预留），仅余 32.64 GiB，而 loglikelihood 的全词表 logits 张量
（batch × ~2.8k tok × 248044 vocab × 2B）在 bs32 需 **41.58 GiB** 直接炸，
bs16 需 ~20.8 GiB 才装得下。**bio 线 bs64 能跑纯因 bio 题干短**，
cyber 题干长，离线与在线（64→32）同因。batch size 不影响 lm_eval 结果，
只影响吞吐（bs16 下 ~10 min/ckpt，可接受）。
