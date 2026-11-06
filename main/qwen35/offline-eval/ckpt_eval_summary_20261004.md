# Qwen35 main(bio) 线 — 9 ckpt 离线测评 + 在线/离线汇总（2026-10-04）

9 个 step-2142 权重（8 个 `qwen35-main-*` 终步导出 + PDU eps 扫描冠军臂 `qwen35-pdu-eps2p51`
10-04 晋编入线）的离线全量测评（wmdp_bio 1273 + mmlu 14042，lm-eval 0-shot/bf16/bs16）。

**一句话结论：RMU 唯一有效、PDU 半破墙、其余全部撞墙/浅/崩** —— 离线 RMU 0.3229/0.8102
（margin +0.487，遗忘 −51.0pt、MMLU 仅 −1.4pt）；PDU-eps2.51 0.4910/0.7849（+0.294，遗忘
−34.2pt、MMLU −4.0pt）；升向族 GD/SatImp/WGA 与 CEU 全钉死在 0.807–0.815 墙平台（遗忘仅
−1.7~−2.5pt）；NPO 族浅平台（−7pt）；GA 崩塌。在线/离线 parity 9/9 成立
（|ΔWMDP|≤0.020，|ΔMMLU|≤0.019）。

margin = MMLU − WMDP-Bio（越大越好）。Base 离线基线 0.8327/0.8247
（`evaluations/results/baselines/lm_eval_results_qwen35_base_wmdp_mmlu`，全量口径）。

---

## 1. 测评配置

- 权重：9 个（HF 格式）。**10-04 已重组**：原 `main/qwen35-main-<算法>/checkpoint-step-2142` +
  `main/qwen35-pdu-eps2p51/` 现统一为 `main/Qwen35/<算法>/`（权重文件直接在算法目录下 +
  `<算法>.yaml`，含 Trace 共 10 个）
- 工具/任务：`scripts/eval.sh`（lm-eval hf, 0-shot, bf16），wmdp_bio 全量 1273 +
  mmlu 全量 14042，`LM_EVAL_BATCH_SIZE=16`（跨线验证过的稳妥档）
- 硬件/并行：safethm 两 rjob、每卡一 ckpt（`NUM_PROCESSES=1`，round-robin 溢出串行）
  - a3（3 卡）：rmu / pdu-eps2p51 / grad-ascent / pure-npo → `qwen35-main-ckpt-eval-a3-2064309`（Succeeded）
  - a4（4 卡）：gd / simnpo / satimp / wga / ceu → `qwen35-main-ckpt-eval-a4-3272407`（Succeeded）
  - 提交脚本 `main/qwen35/rjob_ckpt_eval_safethm.sh`（BATCH=a3|a4，有 result.json 自动跳过）
- 结果：本目录 `lm_eval_results_<ckpt>_wmdp_bio_mmlu/result.json`（9/9）
- 在线曲线：`main/qwen35/online-evaluation/*_online_eval.jsonl`（11 点至 step-2142；在线 MMLU 为
  35 题/子任务抽样 ~1995 题，离线全量 14042）

## 2. 终表（离线全量 + 在线对照；按离线 margin 降序）

| 算法 | WMDP-Bio ↓ | MMLU ↑ | margin | vs Base 遗忘/MMLU | 在线最优步(读数) | 在线终步 | Δ(离线−在线终步) |
|---|---|---|---|---|---|---|---|
| （基线）Base | 0.8327 | 0.8247 | −0.008 | — | — | — | — |
| **rmu** | **0.3229** | **0.8102** | **+0.487** | −51.0pt / −1.4pt | 2100（0.304/0.817 m+0.513） | 0.312/0.817 | +0.011 / −0.006 |
| **pdu-eps2.51** | 0.4910 | 0.7849 | +0.294 | −34.2pt / −4.0pt | 1400（0.467/0.791 m+0.324） | 0.482/0.793 | +0.009 / −0.008 |
| grad-ascent | 0.2404 | 0.2710 | +0.031 | −59.2pt / −55.4pt（崩塌） | 700（0.327/0.491） | 0.247/0.262 | −0.006 / +0.009 |
| pure-npo | 0.7588 | 0.7731 | +0.014 | −7.4pt / −5.2pt | 700（0.743/0.795） | 0.765/0.791 | −0.006 / −0.018 |
| simnpo | 0.7643 | 0.7736 | +0.009 | −6.8pt / −5.1pt | 700（0.742/0.797） | 0.766/0.787 | −0.002 / −0.013 |
| wga-forgetonly | 0.8091 | 0.8074 | −0.002 | −2.4pt / −1.7pt | 2142（终步即最优） | 0.808/0.826 | +0.002 / −0.019 |
| satimp-forgetonly | 0.8075 | 0.8061 | −0.001 | −2.5pt / −1.9pt | 1400（0.796/0.824） | 0.805/0.819 | +0.002 / −0.013 |
| grad-diff | 0.8154 | 0.8127 | −0.003 | −1.7pt / −1.2pt | 1900（0.782/0.825） | 0.796/0.818 | +0.020 / −0.005 |
| ceu | 0.8123 | 0.8091 | −0.003 | −2.0pt / −1.6pt | 1800（0.797/0.826） | 0.803/0.817 | +0.009 / −0.007 |

## 3. 判读

- **遗忘墙离线坐实**：GD/SatImp/WGA/CEU 四家 WMDP-Bio 全落 0.807–0.815（vs Base 0.8327 仅
  −1.7~−2.5pt），与在线 0.78–0.81 墙平台逐点吻合 → qwen35 bio 遗忘墙**不是在线口径的假象**。
  GD 在 gemma4-final/interns2 都是前二算法，在此线被墙挡死 = 墙是模型×数据特性而非算法缺陷。
- **RMU 绕墙、唯一深遗忘工作点**：−51.0pt 遗忘、MMLU 只 −1.4pt；在线最优步 2100 与终步 2142
  差 <0.008 margin → 手上的终步 ckpt 即近优，无需定点重存。
- **PDU 半破墙**：−34.2pt 遗忘、MMLU −4.0pt；在线 1400 步略深（m+0.324 vs 终步 +0.311），差距小，
  终步 ckpt 可用；想再压深可试更低 ε（bio 方向 eps 越低越深，ε2.04→0.606 先例）。
- **NPO 族浅平台**：PureNPO/SimNPO 遗忘仅 −7pt、MMLU 反蚀 −5pt，双输，判无效（与墙前结论一致）。
- **GA 崩塌**：双指标 0.24–0.27 随机水平；在线 700 步曾到 0.327/0.491 也没有可用工作点。
- **在线/离线 parity 9/9**：WMDP |Δ|≤0.020（最大 GD +0.020，墙平台读数略抬）、MMLU |Δ|≤0.019
  （离线全量系统性低 0–2pt，方向一致）。无判读翻转，在线曲线口径再次获验证。

## 4. 产物索引

| 内容 | 位置 |
|---|---|
| 本汇总 | `main/qwen35/offline-eval/ckpt_eval_summary_20261004.md` |
| 离线结果（9 个） | `main/qwen35/offline-eval/lm_eval_results_*_step2142_wmdp_bio_mmlu/result.json` |
| 在线曲线（9 jsonl） | `main/qwen35/online-evaluation/`（8×`qwen35-main-*` + `qwen35-pdu-eps2p51`） |
| 权重（10 个） | `…/checkpoints/unlearn/megatron/main/Qwen35/<算法>/`（CEU/GradAscent/GradDiff/PDU/PureNPO/RMU/SatImp/SimNPO/WGA/Trace，权重平铺+同名 yaml；10-04 重组） |
| 离线基线 | `evaluations/results/baselines/lm_eval_results_qwen35_base_wmdp_mmlu` |
| 提交脚本 | `main/qwen35/rjob_ckpt_eval_safethm.sh`（BATCH=a3\|a4） |
| 训练配置 | `main/qwen35/*.yaml`（8 算法）+ `main/qwen35/pdu_eps_sweep/qwen35-pdu-eps2p51`（臂） |

（对照其他两线 main(bio) 格局：interns2 = RMU/PDU/GD 三有效；gemma4-final = GD/RMU 唯二；
qwen35 = RMU 唯一深遗忘 + PDU 半破墙，GD 被墙挡 —— 三模型有效算法集互不相同，无跨模型通吃。）
