# Intern-S2 主线十算法 checkpoint 离线测评汇总（2026-09-26）

## 测评设置

- **对象**：main 线（`main/interns2/`）Intern-S2 十算法 unlearning ckpt，统一训练计划
  gbs 32 / 3 epochs = 2142 步 / seed 42 / final ckpt（step-2142）。
- **脚本**：官方 `scripts/eval.sh`（8×H200 accelerate 数据并行，
  bf16，0-shot，`LM_EVAL_BATCH_SIZE=64`，未加 limit = 全量）。
- **任务**：`wmdp_bio`（全量 1273 题）+ `mmlu`（全量 14042 题，57 子任务）。
- **执行**：safewt 两个 8 卡 rjob（interns2-ckpt-eval-a-2495130 /
  interns2-ckpt-eval-b-3719023，均 Succeeded，零失败）；提交脚本
  `main/interns2/rjob_ckpt_eval.sh`。
- **结果目录**：`main/interns2/online-evaluation/ckpt-eval/`，
  每个 ckpt 一个 `lm_eval_results_<名>_wmdp_bio_mmlu/result.json`。
- **基线**（Intern-S2 base，同口径离线全量）：
  `evaluations/results/baselines/lm_eval_results_interns2_base_wmdp_mmlu/result.json`
  → **wmdp_bio 0.8366 / mmlu 0.8515**。
- **双柱图**（2026-10-04 补，同日加 Trace）：本目录 `offline_eval_wmdp_mmlu_interns2.png/.pdf`
  （十算法 + Trace 参照，按 margin 降序 + Base 虚线；绘图脚本 `plot_offline_eval.py`，
  PDU 取 lr4e-6 代表臂、GD/SimNPO 取 saveckpts 重跑版、UNDIAL=tp4 已作废占位；
  **Trace = 0.2859/0.8246（margin 0.539 全场第一）**，跨 harness 口径，源
  `Safe_MoE/TRACE/evaluation/Intern-S2-base05/intern_s2_base05_forget1_lr1e-6_ep2_ep8_mb4_gb32_forgetonly/checkpoint-1428/`）。

## 十算法选定结果 + TRACE + Ours (SGTM) 参照（本表为主口径）

选择说明：PDU 取 **lr1e-6 版**；GD 与 SimNPO 取 **saveckpts 重跑版**的 2142；
其余算法取各自 final（step-2142）。另列入 **TRACE**（ckpt-1428）与
**Ours (SGTM)** 的 full / ablated 两个点（cp0）作参照（见下节）。

| 算法 | wmdp_bio | mmlu | Δbio vs 基线 | Δmmlu vs 基线 | result.json 路径 |
|---|---|---|---|---|---|
| RMU | **0.3700** | **0.8418** | −46.7pt | −1.0pt | `/mnt/shared-storage-user/safethm-share/wangxiao1/Safe_MoE/open-unlearning/main/interns2/online-evaluation/ckpt-eval/lm_eval_results_interns2-main-rmu-step2142_wmdp_bio_mmlu/result.json` |
| **TRACE (ckpt-1428)** | **0.2859** | **0.8246** | −55.1pt | −2.7pt | `/mnt/shared-storage-user/safethm-share/wangxiao1/Safe_MoE/TRACE/evaluation/Intern-S2-base05/intern_s2_base05_forget1_lr1e-6_ep2_ep8_mb4_gb32_forgetonly/checkpoint-1428/wmdp_bio_wmdp_cyber_mmlu/result.json` |
| **Ours (ablated)** | **0.2859** | 0.8070 | −55.1pt | −4.4pt | `/mnt/shared-storage-user/safethm-share/wangxiao1/Safe_MoE/MoE-Postraining/adv-finetuning/online_eval/sgtm/cp0/result.json` |
| PDU (eps1.98) | 0.5059 | 0.8283 | −33.1pt | −2.3pt | `/mnt/shared-storage-user/safethm-share/wangxiao1/Safe_MoE/open-unlearning/main/interns2/online-evaluation/interns2-main-pdu-eps1p98_online_eval.jsonl`（全量在线 = 离线口径，step 2142） |
| Ours (full) | 0.6198 | 0.8225 | −21.7pt | −2.9pt | `/mnt/shared-storage-user/safethm-share/wangxiao1/Safe_MoE/MoE-Postraining/adv-finetuning/online_eval/sgtm/cp0/result.json` |
| GradAscent | 0.2459 | 0.2295 | −59.1pt | −62.2pt（崩） | `/mnt/shared-storage-user/safethm-share/wangxiao1/Safe_MoE/open-unlearning/main/interns2/online-evaluation/ckpt-eval/lm_eval_results_interns2-main-grad-ascent-step2142_wmdp_bio_mmlu/result.json` |
| GradDiff (GD, saveckpts) | 0.6654 | 0.8238 | −17.1pt | −2.8pt | `/mnt/shared-storage-user/safethm-share/wangxiao1/Safe_MoE/open-unlearning/main/interns2/online-evaluation/ckpt-eval/lm_eval_results_interns2-main-grad-diff-saveckpts-step2142_wmdp_bio_mmlu/result.json` |
| CEU | 0.6976 | 0.7578 | −13.9pt | −9.4pt | `/mnt/shared-storage-user/safethm-share/wangxiao1/Safe_MoE/open-unlearning/main/interns2/online-evaluation/ckpt-eval/lm_eval_results_interns2-main-ceu-step2142_wmdp_bio_mmlu/result.json` |
| WGA | 0.7258 | 0.7653 | −11.1pt | −8.6pt | `/mnt/shared-storage-user/safethm-share/wangxiao1/Safe_MoE/open-unlearning/main/interns2/online-evaluation/ckpt-eval/lm_eval_results_interns2-main-wga-step2142_wmdp_bio_mmlu/result.json` |
| SatImp | 0.7321 | 0.7755 | −10.5pt | −7.6pt | `/mnt/shared-storage-user/safethm-share/wangxiao1/Safe_MoE/open-unlearning/main/interns2/online-evaluation/ckpt-eval/lm_eval_results_interns2-main-satimp-step2142_wmdp_bio_mmlu/result.json` |
| SimNPO (saveckpts) | 0.7423 | 0.7780 | −9.4pt | −7.4pt | `/mnt/shared-storage-user/safethm-share/wangxiao1/Safe_MoE/open-unlearning/main/interns2/online-evaluation/ckpt-eval/lm_eval_results_interns2-main-simnpo-saveckpts-step2142_wmdp_bio_mmlu/result.json` |
| PureNPO | 0.7628 | 0.7876 | −7.4pt | −6.4pt | `/mnt/shared-storage-user/safethm-share/wangxiao1/Safe_MoE/open-unlearning/main/interns2/online-evaluation/ckpt-eval/lm_eval_results_interns2-main-pure-npo-step2142_wmdp_bio_mmlu/result.json` |
| UNDIAL (tp4) | 0.8342 | 0.8421 | −0.2pt（≈无遗忘） | −0.9pt | `/mnt/shared-storage-user/safethm-share/wangxiao1/Safe_MoE/open-unlearning/main/interns2/online-evaluation/ckpt-eval/lm_eval_results_interns2-main-undial-tp4-step2142_wmdp_bio_mmlu/result.json` |

> **2026-10-10 PDU 换版注**：对象存储 `safemoe/final/intern/bio/PDU/` 的权重已
> 替换为 **eps1.98 版**（`interns2-main-pdu-eps1p98`，retain_loss_eps 2.05→1.98
> 收紧 retain NLL 锚点，其余 model/调度/seed/mbs1/lr1e-6/3ep-末步导出全不变；
> 配置源 `main/interns2/pdu_eps1p98.yaml` → `PDU/PDU.yaml`，原 eps2.05 版已删）。
> 本行成绩取该 run 的**全量在线评测**（wmdp_bio 全量 1273 + MMLU 全量 14042，
> step 2142，`..._eps1p98_online_eval.jsonl`）——full-size 口径等同离线。
> 相比原 eps2.05 lr1e-6 版（0.3111/0.7835）：收紧 eps 把 MMLU 损伤从 −6.8pt
> 收回到 −2.3pt，但遗忘也大幅变浅（bio 0.311→0.506，−52.6pt→−33.1pt）。
> 下方结论速览第 1 条对 PDU-lr1e-6 的旧描述为历史版本，未同步改写。

## TRACE 与 Ours (SGTM) 参照详情

- **TRACE**：`/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/wangxiao1/checkpoints/Trace/intern_s2_base05_forget1_lr1e-6_ep2_ep8_mb4_gb32_forgetonly/checkpoint-1428`
  （forget-only，lr 1e-6，mb4/gb32）→ bio 0.2859 / mmlu 0.8246 /
  **wmdp_cyber 0.5954**；结果：`TRACE/evaluation/Intern-S2-base05/intern_s2_base05_forget1_lr1e-6_ep2_ep8_mb4_gb32_forgetonly/checkpoint-1428/wmdp_bio_wmdp_cyber_mmlu/result.json`

### Ours (SGTM)（cp0）

- **full**：`/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/jincheng/sgtm/checkpoints/intern/wmdp-bio/helr40-F2`
  → bio 0.6198 / mmlu 0.8225 / **wmdp_cyber 0.5622**
- **ablated**：`/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/wangxiao1/checkpoints/SGTM-ablated-tmp/cp0`
  （full 的有害专家消融版）→ bio **0.2859** / mmlu 0.8070 / **wmdp_cyber 0.5707**
- 测评：全量 0-shot（bio 1273 / mmlu 14042 / cyber 1987 题，2026-09-27，
  与主线 16 ckpt 同口径）。
- 结果：`/mnt/shared-storage-user/safethm-share/wangxiao1/Safe_MoE/MoE-Postraining/adv-finetuning/online_eval/sgtm/cp0/result.json`
  （含 full/ablated 两份完整 lm_eval 输出与逐任务 drop 汇总）。
- 备注：ablated 的 wmdp_bio 与 TRACE 仓库 checkpoint-1428 的历史测评
  （0.2859387274155538）逐位一致，应为同一权重的两次独立评测；mmlu 两次
  分别为 0.8070 / 0.8246（不同 lm_eval 运行的实现差异）。本表以 cp0 为准。

## 结论速览

1. **达到「遗忘深 + 能力保持」的只有四个点**：RMU（0.370/0.842）、
   PDU-lr4e-6（0.408/0.844，见附录）、**TRACE**（0.286/0.825，MMLU 只掉
   2.7pt）与 **Ours (ablated)**（0.286/0.807，MMLU 掉 4.4pt）。本表选定的
   PDU-lr1e-6 版遗忘深（0.311）但 MMLU 掉 6.8pt（lr×eps 耦合：eps2.05 按
   lr4e-6 调出，lr1e-6 下 λ 约束未接上）。
2. **GradAscent 遗忘最深但能力崩毁**（mmlu 0.230）。
3. **UNDIAL 主线工作点（β3/α0, 3ep）对 bio 无遗忘效果**（0.834 ≈ 基线
   0.837），MMLU 完整；与 V3 单 epoch 结果（0.821）互证。
   ⚠️ **2026-09-29 作废**：undial_tp4 与 V3 均为 TP4，`undial_objective` 在
   TP>1 时梯度错误（log Z 项只按本 shard 的 target 质量回传，label logit 下压
   仅约 23%）。已修复，以 TP1/EP8 重跑 `undial_fixed.yaml`，结论以重跑为准。
4. **其余七算法均为遗忘弱档**（bio 0.66–0.76），遗忘越深者 MMLU 损伤越大
   （CEU/WGA/SatImp/SimNPO/PureNPO 掉 6–9pt；GD 系靠 retain 项护住 mmlu）。

## 附录：本轮全部 16 个 ckpt 结果

| ckpt | wmdp_bio | mmlu | 说明 |
|---|---|---|---|
| interns2-main-grad-ascent-step2142 | 0.2459 | 0.2295 | 崩 |
| interns2-main-pdu-eps1p98-step2142 | 0.5059 | 0.8283 | eps1.98（换版，全量在线=离线） |
| interns2-main-rmu-step2142 | 0.3700 | 0.8418 | 最优 |
| interns2-main-pdu-lr4e-6-step2142 | 0.4077 | 0.8435 | PDU 代表工作点 |
| interns2-main-grad-diff-saveckpts-step2142 | 0.6654 | 0.8238 | 同 seed 重跑 |
| interns2-main-ceu-step2142 | 0.6976 | 0.7578 | |
| interns2-main-grad-diff-saveckpts-step714 | 0.7015 | 0.8313 | 1 epoch |
| interns2-main-wga-step2142 | 0.7258 | 0.7653 | |
| interns2-main-simnpo-saveckpts-step714 | 0.7313 | 0.7864 | 1 epoch |
| interns2-main-satimp-step2142 | 0.7321 | 0.7755 | |
| interns2-main-simnpo-step2142 | 0.7368 | 0.7726 | 原 run final |
| interns2-main-simnpo-saveckpts-step2142 | 0.7423 | 0.7780 | 同 seed 重跑 |
| interns2-main-grad-diff-step2142 | 0.7596 | 0.8333 | 原 run final |
| interns2-main-pure-npo-step2142 | 0.7628 | 0.7876 | |
| interns2-main-simnpo-saveckpts-step300 | 0.7863 | 0.8106 | |
| interns2-main-undial-tp4-step2142 | 0.8342 | 0.8421 | 无遗忘 |

### 备注

- **口径**：本表为离线全量 lm_eval（0-shot）；与训练中在线评测
  （mmlu 每叶子子任务限 35 题）数值有口径差，在线值普遍略偏高/偏低属正常，
  结论方向一致。
- **bf16 非确定性**：GD/SimNPO 的 saveckpts 重跑与原 run 参数、seed 完全
  一致，但 final bio 差异达 ~0.1（GD 0.760→0.665；SimNPO 0.737→0.742）。
  跨 run 的 ±0.05 级轨迹差异（如"谷底回弹"）解读需谨慎，档位判断不受影响。
- ckpt 实体位于
  `/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/wangxiao1/checkpoints/unlearn/megatron/main/`。
