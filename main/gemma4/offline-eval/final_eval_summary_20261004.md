# Gemma4 Bio FINAL 线 — 在线/离线测评汇总（2026-10-04）

9 算法统一 nocap+BOS 重跑（configs-final，10-03 提交）**9 个算法全齐**（PDU 于 10-04 补测
完结，见文末）的最终权重（step-2133）离线全量测评 + 训练全程在线曲线汇总。

**一句话结论：GD 与 RMU 两家独大** —— 离线 GD 0.284/0.706（margin +0.423）、RMU 0.299/0.699
（margin +0.400），对照基线 0.788/0.745 即遗忘 −51/−49pt、MMLU 仅 −3.6/−4.6pt；其余七算法
遗忘不足（PDU 终点回吐后归入此类，另 CEU/PureNPO/SimNPO）或模型受损（GA/SatImp/WGA）。
在线/离线 parity 良好
（WMDP |Δ|≤0.021，MMLU |Δ|≤0.025），无判读翻转。

margin 定义 = MMLU − WMDP-Bio（越大越好）。挑选准则：在线曲线 margin 最大步 + 终步对照。

---

## 1. 训练配置（9 个算法共同）

| 项 | 值 |
|---|---|
| 模型 | gemma-26b-a4b（Gemma4 26B MoE，128 experts）影子副本 `checkpoints/gemma/gemma-26b-a4b-nocap`，**`final_logit_softcapping=null`（nocap）** |
| 遗忘数据 | `Xtuner/gemma4_wmdp_bio_softpack4096_bos/forget`（WMDP-Bio 语料，22,757 packs） |
| 保留数据 | `Xtuner/fineweb_edu_5b_gemma4_softpack4096_bos_seed42`（FineWeb-Edu-5B，1,115,271 packs） |
| 数据格式 | softpack 4096，**每 pack 头插 BOS**（bos_per_pack；2026-10-03 BOS 门控修复，见 `main/gemma4/GA_family_nocap_rootcause_20261002.md`） |
| 训练框架 | Megatron-Core，8×H200，TP=1 / PP=1 / **EP=8**，gbs=32，bf16，全参可训，seed 42 |
| 调度 | 3 epochs = **2133 步**，cosine lr，warmup 0.05，每算法 lr 见 §2 |
| 产物 | 末步自动导出 HF 格式（~66G）。**10-04 已重组**：原 `main/gemma4-final-<算法>-nocap/checkpoint-step-2133` 现为 `main/Gemma4/<算法>/`（权重文件直接在算法目录下 + `<算法>.yaml`，含 Trace 共 10 个） |
| 配置 | `main/gemma4/configs-final/<算法>.yaml`（CEU 为 fp32 归一化 bug 修复后代码首跑） |
| 训练 job | safewt：g2 ceu+rmu 8940949 / g3 pnpo+satimp 40162369 / g4 gd+wga 10120969 / g5 simnpo+ga 42562846（提交脚本 `configs-final/rjob_final_safewt.sh`） |

各算法超参（`training` 段其余字段同上表）：

| 算法 | 关键超参 | lr | micro_bs |
|---|---|---|---|
| grad-diff | γ1.0 α1.0 | 1e-6 | 1 |
| rmu | α4.45, steer 1200, target L9, train L7-9, fixed norm | 1e-6 | 4 |
| pdu | retain_loss_eps 2.60（自 eps2.50 臂上调，split 前向防 2³¹ 溢出） | 4e-6 | 1 |
| ceu | γ1.0, ignore_first_n_tokens 1 | 1e-6 | 1 |
| pure-npo | β0.1 γ1.0 α1.0 | 5e-5 | 1 |
| simnpo | β0.1 δ0.0 γ1.0 α1.0 | 5e-5 | 1 |
| satimp(forgetonly) | β1=2.0 β2=0.1 α0.0 | 1e-6 | 4 |
| wga(forgetonly) | β2.0 α0.0 | 1e-6 | 4 |
| grad-ascent | —（纯上升） | 1e-6 | 4 |

## 2. 在线测评配置

训练中内嵌测评（`evaluation:` 段，MCore 前向 + lm_eval API，不导 ckpt）：

- 任务：wmdp_bio **全量 1273 题** + mmlu **子集 35 题/子任务（57 子任务 = 1995 题）**，0-shot，bs 64
- 触发步：`[1, 300, 700, 1000, 1400, 1700, 1800, 1900, 2000, 2100, 2133]`
- 曲线文件：`main/gemma4/configs-final/online-evaluation/gemma4-final-<算法>-nocap_online_eval.jsonl`
- 在线基线（step-1）：WMDP ≈ 0.789–0.792，MMLU(子集) ≈ 0.766–0.769

### 在线结果（最优步 = margin 最大步；终步 = 2133）

| 算法 | 最优步 | 最优 WMDP/MMLU | margin | 终步 WMDP/MMLU | margin |
|---|---|---|---|---|---|
| **grad-diff** | **2100** | **0.2639 / 0.7278** | **+0.464** | 0.2757 / 0.7278 | +0.452 |
| **rmu** | 2133（=终步） | 0.2969 / 0.7048 | +0.408 | 同左 | +0.408 |
| pdu | 1000 | 0.5004 / 0.7489 | +0.249 | 0.6960 / 0.7524 | +0.056 |
| pure-npo | 700 | 0.5774 / 0.6927 | +0.115 | 0.6465 / 0.7058 | +0.059 |
| simnpo | 300 | 0.6355 / 0.7073 | +0.072 | 0.6426 / 0.6992 | +0.057 |
| satimp | 300 | 0.5137 / 0.5669 | +0.053 | 0.6080 / 0.5759 | −0.032 |
| wga | 300 | 0.5601 / 0.5985 | +0.038 | 0.6512 / 0.6125 | −0.039 |
| ceu | 1900 | 0.7455 / 0.7474 | +0.002 | 0.7447 / 0.7429 | −0.002 |
| grad-ascent | 1900 | 0.2357 / 0.2361 | +0.000 | 0.2561 / 0.2381 | −0.018 |

在线判读：GD 全程单调、终点近优；RMU 终点即最优、无过训；SatImp/WGA 300 步过冲后**反扑**
（WMDP 回升超 MMLU 且 MMLU 塌到 0.57–0.61）；GA 300 步全面崩塌（双指标随机水平 ~0.24）；
CEU 曲线几乎不动（WMDP 始终 ~0.74）= fp32 修复+BOS 双修后仍无效；PDU 1000 步最深 0.500
（margin +0.249）后**单调反扑**回 0.696 = 遗忘回吐（详见 §5）。

## 3. 离线测评配置

- 权重：9 个（HF 格式，无 expert_expansion_manifest，不触发消融路径）；现为 `main/Gemma4/<算法>/`（10-04 重组，权重文件平铺，无 checkpoint-step 子层）
- 工具：`scripts/eval.sh`（lm-eval hf, 0-shot, bf16, trust_remote_code）
- 任务：wmdp_bio **全量 1273** + mmlu **全量 14042**
- 硬件/并行：每 rjob 2×H200，组内 2 ckpt 串行，每条 eval 双卡 accelerate 数据并行
  （`NUM_PROCESSES=2`），`LM_EVAL_BATCH_SIZE=auto`（gemma4 MMLU 长 prompt 在 bs64 连 140G H200 都 OOM，auto 实测选 ~25）
- 提交：safewt，10-04 00:26 提交、00:45 前全部 Succeeded（每 job ~2 ckpt 串行共约 18 分钟）
  - g1 ceu+rmu → 80252346；g2 pnpo+satimp → 81457736；g3 gd+wga → 82595118；g4 simnpo+ga → 83714908
  - 提交脚本 `main/gemma4/configs-final/rjob_ckpt_eval_safewt.sh`（BATCH=g1..g4）
- **PDU 补测（10-04 下午）**：首提 safer2ai 34169318（Inqueue 中撤单），改提 **safethm**
  `gemma4-final-ckpt-eval-pdu-27131156`（2×H200 DP、bs16 固定档，同 harness/任务），Succeeded；
  脚本 `configs-final/rjob_ckpt_eval_pdu.sh`（RESOURCE_GROUP 可切 safethm/safer2ai）
- 结果：本目录 `lm_eval_results_gemma4-final-<算法>-nocap-step2133_wmdp_bio_mmlu/result.json`

### 离线结果 + 在线/离线 parity

| 算法 | WMDP-Bio ↓ | MMLU ↑ | margin | vs 基线遗忘/MMLU代价 | 在线终步 Δ(WMDP/MMLU) |
|---|---|---|---|---|---|
| （基线）base 原版 capped | 0.7926 | 0.7424 | −0.050 | — | — |
| （基线）base-nocap 训练基座 | 0.7879 | 0.7449 | −0.043 | — | — |
| **grad-diff** | **0.2836** | **0.7063** | **+0.423** | −50.4pt / −3.9pt | +0.008 / −0.022 |
| **rmu** | 0.2993 | 0.6989 | +0.400 | −48.9pt / −4.6pt | +0.002 / −0.006 |
| pure-npo | 0.6489 | 0.6924 | +0.044 | −13.9pt / −5.3pt | +0.002 / −0.013 |
| simnpo | 0.6386 | 0.6827 | +0.044 | −14.9pt / −6.2pt | −0.004 / −0.017 |
| pdu | 0.6984 | 0.7334 | +0.035 | −9.0pt / −1.2pt | +0.002 / −0.019 |
| wga | 0.6567 | 0.6123 | −0.044 | −13.1pt / −13.3pt | +0.006 / −0.000 |
| ceu | 0.7518 | 0.7183 | −0.034 | −3.6pt / −2.7pt | +0.007 / −0.025 |
| satimp | 0.6292 | 0.5793 | −0.050 | −15.9pt / −16.6pt | +0.021 / −0.003 |
| grad-ascent | 0.2467 | 0.2292 | −0.017 | −54.1pt / −51.6pt（崩塌） | −0.009 / −0.009 |

- **离线基线**（同 harness）：base 原版 capped=`evaluations/results/baselines/lm_eval_results_gemma4_base_wmdp_mmlu`
  （0.7926/0.7424）；base-nocap=`lm_eval_results_gemma4_base_nocap_wmdp_mmlu`（0.7879/0.7449）。
  两者 |Δ|<0.5pt = softcap null 化对 MCQ 读数无影响；"遗忘/MMLU代价"列按 **base-nocap（训练基座）** 计算。
- 在线→离线的绝对值差主要来自 MMLU 子集（1995）vs 全量（14042）：离线 MMLU 普遍低 0–2.5pt，方向一致。
- **GD 略优于 RMU**（离线两项都更好）；GD 在线最优步 2100 与终步差距 <0.012 margin，若要复现最优可取 step-2100 附近存点（本轮只存了 2133）。

## 4. 产物索引

| 内容 | 位置 |
|---|---|
| 本汇总 | `main/gemma4/offline-eval/final_eval_summary_20261004.md` |
| 离线结果（9 个） | `main/gemma4/offline-eval/lm_eval_results_*_wmdp_bio_mmlu/result.json` |
| 在线曲线（9 个 jsonl） | `main/gemma4/configs-final/online-evaluation/` |
| 训练配置（9 含 PDU） | `main/gemma4/configs-final/*.yaml` |
| 权重（10 个，9×~66G+Trace 49G） | `gpfs2/wenxiaoyu-gpfs02/wangxiao1/checkpoints/unlearn/megatron/main/Gemma4/<算法>/`（CEU/GradAscent/GradDiff/PDU/PureNPO/RMU/SatImp/SimNPO/WGA/Trace，权重平铺+同名 yaml；10-04 重组） |
| 离线基线 | `evaluations/results/baselines/lm_eval_results_gemma4_base_wmdp_mmlu`（capped）与 `lm_eval_results_gemma4_base_nocap_wmdp_mmlu`（nocap） |
| 提交脚本 | 训练 `configs-final/rjob_final_safewt.sh`；离线测评 `configs-final/rjob_ckpt_eval_safewt.sh`；PDU 补测 `configs-final/rjob_ckpt_eval_pdu.sh` |

（注：`configs-final/ckpt-eval/` 下还留有 root 属主的原始结果副本 1.6M，无法以当前账号删除，内容与
offline-eval 相同。）

## 5. PDU 补测结案（10-04）

PDU（eps=2.60，lr 4e-6，mbs 1，训练 job 27915035）此前记录为"曲线止步 1400、末步 ckpt 未
落盘"；10-04 复查确认 **jsonl 已完整至 step-2133 且 checkpoint-step-2133 已落盘**（早前观察
不再成立，异常结案），随后补齐离线测评。

- 在线：1000 步最深 0.5004 / 0.7489（margin +0.249，全程最优）后**单调反扑**，2133 步回吐至
  0.6960 / 0.7524（margin +0.056）
- 离线（safethm 27131156，2 卡 DP bs16）：**0.6984 / 0.7334（margin +0.035）**，vs base-nocap
  遗忘仅 −9.0pt、MMLU −1.2pt；parity 成立（在线终步 Δ +0.002 / −0.019，噪声带内）
- 判读：遗忘**完全回吐**，属"不遗忘"档而非 eval 异常。若要兑现 1000 步的中间最优须定点存盘
  重跑，且 +0.249 仍远逊 GD（+0.464）/RMU（+0.408），价值有限。最终结论不变：**GD 与 RMU
  唯二有效**。
