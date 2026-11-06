# Qwen3.5 main-cyber（wmdp-cyber 十算法迁移 · 11 配置）

Intern-S2 main-cyber 线（同目录 `../interns2/`）向 Qwen3.5-35B-A3B 的批量迁移。
2026-10-01 由 `make_configs.py` 从 intern-cyber R3 配置确定性生成：九个算法
（UNDIAL 除外——undial_fixed 在 intern 上修复 TP bug 后仍无遗忘、超参从未调过；
PDU 拆三个 eps 臂 2.16/2.30/2.51），逐字段复制 intern-cyber 的算法语义与训练
计划，只替换模型/数据/输出/在线测评四类字段（取值参照 qwen35 bio 主线
`main/qwen35/*.yaml` 的已验证值）。

## 迁移口径

**原样复制**（intern-cyber R3 = 10 epochs / 230 steps 全调度 + 每 epoch
在线测评 [23..230]；存点计划见下方 R2 说明）：

- 数据规模巧合但关键：qwen35 cyber forget cache 与 intern 同为 **743 packs**
  （`data_cache/Xtuner/qwen35_wmdp_cyber_softpack4096/forget`，743×4096=
  3,043,328 tokens），故步数/评测步/R3 定点步 1:1 沿用，无需重算
- lr 九份配置统一 5e-6（rmu、PDU 三臂、gd、ga、satimp、WGA、ceu），
  PureNPO/SimNPO 两份保持 5e-5；min_lr 全部 1e-6、gbs 32、cosine + 5%
  warmup、seed 42
- mbs 按算法保留 intern-cyber 值（qwen bio 主线已逐一验证同值可跑）：
  rmu/ga/satimp/WGA = 4，pdu/gd/pnpo/simnpo/ceu = 1
- 在线测评 eval batch_size 保留 cyber 线值（cyber 题干更长）：32；全 logits
  算法（gd/pnpo/simnpo）16
- **存点计划（R1→R2，2026-10-01）**：R1 首轮零 ckpt 纯曲线跑完（11/11 曲线
  齐）；随后按 **margin = mmlu − wmdp_cyber 的 argmax**（同 step 并列取最早）
  从曲线选出各 run 最优点，R2 原地改回 `save_steps: [最优 step]` 单点存 ckpt
  （save_final 仍 false）。存点表见下方逐算法参数表
- checkpoint output_root 与 intern-cyber 共用
  `.../checkpoints/unlearn/megatron/main-cyber`（run 名区分）

**必须改的模型相关字段**：

| 字段 | intern-cyber | qwen35-cyber | 依据 |
| --- | --- | --- | --- |
| model path/name | Intern-S2-base05 | qwen3p5-35b-a3b | qwen35 bio 主线 |
| forget cache | interns2 cyber | qwen35_wmdp_cyber_softpack4096/forget | 本线数据 |
| retain cache | interns2 fineweb | fineweb_edu_5b_qwen35_softpack4096_seed42 | qwen35 bio 同款 |
| RMU alpha | 350.0 | **63.4**（算法段与 `main/qwen35/rmu.yaml` 逐字节一致） | qwen R:F 扫描最优臂（跨模型直搬 R:F 无效，gemma4 定案）；steer 300 不变 |
| PDU retain_loss_eps | 2.05（单臂） | **三臂 2.16 / 2.30 / 2.51** | eps 锚定 retain NLL=2.0082（retain cache 与 bio ε 扫描相同，偏移量跨线可比）：2.16=+0.15（bio 参考点 0.636/0.810）、2.30=+0.29（新中点）、2.51=+0.50（bio 0.482/0.793 最深但 MMLU 侵蚀）；2026-10-01 决策，替换 bio 最优 2.04 单臂 |
| 在线测评输出 | …/main-cyber/interns2/online-evaluation | …/main-cyber/qwen35/online-evaluation | 本目录 |
| name/output_name | interns2-main-cyber-* | qwen35-main-cyber-* | 命名约定 |

算法段其余字段：RMU 算法段与 `main/qwen35/rmu.yaml` 逐字节一致；GD/GA/PureNPO/
SimNPO/SatImp/WGA/CEU 七个算法的算法段在 intern 与 qwen bio 主线本来就逐字节
一致，直接复制；PDU 仅 eps 按上表分三臂。

## 逐算法参数表（迁移后 · R2 存点）

R1 曲线（2026-10-01 跑完）+ R2 单点存 ckpt 计划。margin = mmlu − wmdp_cyber；
Base（lm_eval 离线，evaluations/results/baselines/）= 0.565/0.825，margin 0.259。

| 配置 | lr | mbs | eval bs | R1 最优点 (wmdp/mmlu) | R2 存点 step/ep | margin |
| --- | --- | --- | --- | --- | --- | --- |
| rmu.yaml | 5e-6 | 4 | 32 | 0.253/0.812 | **92/ep4** | **0.559** |
| pdu_eps2p16.yaml | 5e-6 | 1 | 32 | 0.258/0.808 | 115/ep5 | 0.550 |
| pdu_eps2p30.yaml | 5e-6 | 1 | 32 | 0.262/0.811 | 92/ep4 | 0.549 |
| pdu_eps2p51.yaml | 5e-6 | 1 | 32 | 0.252/0.812 | **115/ep5** | **0.559** |
| grad_diff.yaml | 5e-6 | 1 | 16 | 0.322/0.722 | 69/ep3 | 0.401 |
| grad_ascent.yaml | 5e-6 | 4 | 32 | 0.318/0.649 | 46/ep2 | 0.331 |
| pure_npo.yaml | 5e-5 | 1 | 16 | 0.425/0.817 | 69/ep3 | 0.392 |
| simnpo.yaml | 5e-5 | 1 | 16 | 0.430/0.826 | 46/ep2 | 0.395 |
| satimp_forgetonly.yaml | 5e-6 | 4 | 32 | 0.324/0.743 | 46/ep2 | 0.419 |
| WGA_forgetonly.yaml | 5e-6 | 4 | 32 | 0.309/0.723 | 184/ep8 | 0.414 |
| ceu.yaml | 5e-6 | 1 | 32 | 0.297/0.694 | 184/ep8 | 0.397 |

R1 要点：qwen35-cyber **遗忘墙未复现**（bio 线 0.79-0.81 墙全线跌破，印证
语境门控是 bio 语料特性）；最优权衡 RMU@ep4 与 PDU-2.51@ep5 并列 0.559；
GD 深但 MMLU 半崩（终点 0.24/0.52）、GA 全崩（0.27/0.23）——两者的存点
（ep3/ep2）是早停抢救位；PDU 三 eps 臂曲线几乎重合（eps 不敏感区）。

## 已知风险与解读预警（提交前读一遍）

1. **R2 单点存 ckpt**：11 个配置已原地改为 `save_steps: [margin 最优 step]`
   （save_final false，机制同 intern-R3 定点存点）。每个 run 只在最优步导出
   一个 ckpt（full 35B 权重）；R1 曲线与存点由 jsonl 复算核对过（11/11 一致）。
   重跑前确认磁盘余量足够 11 份导出。
2. **PDU lr 口径**：qwen 的 eps 网格在 lr 4e-6 下调出，本线沿用 cyber 统一
   lr 5e-6（与 intern-cyber 当年 4e-6→5e-6 的决策同构，同量级；λ 对偶变量
   会自适应）。若 cyber 上 λ 不咬合，第一候选改动是 lr 回 4e-6。
3. **遗忘墙预期**：qwen bio 上升向族（GD/SatImp/WGA/CEU）集体卡 0.79-0.81
   墙（语境门控抑制，见 `main/qwen35/diagnosis/FINDINGS_20260930.md`）。cyber
   语料是纯预训练格式文本，墙大概率复现——读到"没遗忘"时先想墙，不是算法
   失效；有效算法预期仍是 RMU/PDU 一类。基线：qwen35 bio main 线只有 RMU
   (0.312/0.817) 和 PDU ε2.04 (0.606/0.817) 真正动了。
4. **eval bs 首测盯 OOM**：cyber 题干长于 bio。首步在线测评若 CUDA OOM，按
   各文件头注释的降级路径 32→16（全 logits 算法已是 16→8）。
5. **数据 seed**：qwen cyber forget cache 构建 seed 43（meta.json），retain
   fineweb seed 42；与 bio 线一致，无额外动作。

## 提交

R1（2026-10-01 凌晨，11 配置六组跨三集群）当天全部跑完，job 记录见 git 历史。
**R2（存 ckpt 版）已提交：PDU 只保留 eps2.51 臂（2.16/2.30 曲线与 2.51 几乎
重合，不再重跑），9 配置三组、全部投 safewt**（用户 2026-10-01 分配；ε 扫描
+ R1 实证集群）。三组按 R1 实测时长配平（jsonl 首尾时间戳），长杆在前：

| BATCH | 组成（串行） | 实测时长 | job ID |
| --- | --- | --- | --- |
| r2-pdu-gd-rmu | pdu2p51 → grad_diff → rmu | 1.37+1.08+0.69 = 3.14h | qwen35-main-cyber-r2-pdu-gd-rmu-safewt-89628250 |
| r2-pnpo-simnpo-satimp | pure_npo → simnpo → satimp | 1.23+1.11+0.80 = 3.14h | qwen35-main-cyber-r2-pnpo-simnpo-satimp-safe-19c65 |
| r2-ceu-ga-wga | ceu → grad_ascent → WGA | 1.12+0.85+0.81 = 2.78h | qwen35-main-cyber-r2-ceu-ga-wga-safewt-91993079 |

每个 run 在自己的 margin 最优步导出一份 ckpt（intern 点存实测 66G/份，9 份
~630G；output_root 所在盘余 5.7T 已确认）。**ckpt 已于 2026-10-02 整理为**
`.../checkpoints/unlearn/megatron/main-cyber/qwen35/<算法>/`（一层直下 =
HF 权重文件 + 训练配置 yaml + `ckpt_notes.md`；原 run 名 +
`checkpoint-step-N/` 布局纯 rename 迁移，未动数据）。

```bash
cd main/main-cyber/qwen35
# 复现 R2 提交（三组，均 safewt）：
RESOURCE_GROUP=safewt BATCH=r2-pdu-gd-rmu         bash rjob_main.sh
RESOURCE_GROUP=safewt BATCH=r2-pnpo-simnpo-satimp bash rjob_main.sh
RESOURCE_GROUP=safewt BATCH=r2-ceu-ga-wga         bash rjob_main.sh
# PREVIEW=true 只打印 rjob 命令
```

## ckpt 离线测评（R2 之后，2026-10-02 提交）

> **ckpt 产生方式（完整 10 epochs + 中途定点存盘，及为何不能截断 epochs）
> 与在线/离线成绩单页见 [`ckpt_notes.md`](ckpt_notes.md)**。

9 个 R2 存点 ckpt 的全量离线测评（wmdp_cyber 全量 1987 + MMLU 全量
14042，0-shot，bf16，`scripts/eval.sh`，与 Base 行同口径）。
按 R2 训练组拆 **3 个 job、每组 3 卡、一卡测一个权重**（用户 2026-10-02
分配），`BATCH_SIZE=16`（bs64/32 在 cyber 长题干上 OOM 的实证工作档，
见脚本头注释）。提交脚本 `rjob_ckpt_eval.sh`（模板 = intern 线同名脚本，
组选择与 R2 相同的 BATCH 标签）：

| BATCH | ckpt（step） | job ID |
| --- | --- | --- |
| r2-pdu-gd-rmu | pdu-eps2p51(115) / grad-diff(69) / rmu(92) | qwen35-main-cyber-ckpt-eval-r2-pdu-gd-rmu-1845204 |
| r2-pnpo-simnpo-satimp | pure-npo(69) / simnpo(46) / satimp-forgetonly(46) | qwen35-main-cyber-ckpt-eval-r2-pnpo-simnpo-s-15055 |
| r2-ceu-ga-wga | ceu(184) / grad-ascent(46) / wga-forgetonly(184) | qwen35-main-cyber-ckpt-eval-r2-ceu-ga-wga-4446857 |

结果落 `ckpt-eval/lm_eval_results_<ckpt-name>_wmdp_cyber_mmlu/result.json`
（result.json 已存在则跳过，单卡失败不影响同 job 其他卡）。

```bash
# 复现提交（3 组均 safewt）：
RESOURCE_GROUP=safewt BATCH=r2-pdu-gd-rmu         bash rjob_ckpt_eval.sh
RESOURCE_GROUP=safewt BATCH=r2-pnpo-simnpo-satimp bash rjob_ckpt_eval.sh
RESOURCE_GROUP=safewt BATCH=r2-ceu-ga-wga         bash rjob_ckpt_eval.sh
```

### 离线测评结果（2026-10-02 当晚跑完，3 job 全 Succeeded）

与 Base 同口径（全量 wmdp_cyber 1987 + MMLU 14042，0-shot，bf16）。
on = g2 在线曲线在存点步的值（MMLU 为 ×35/leaf 子采样）；dW/dM = 离线−在线：

| ckpt | 在线 wmdp/mmlu | **离线 wmdp/mmlu** | dW | dM | **margin** |
| --- | --- | --- | --- | --- | --- |
| **Base** | — | **0.5652 / 0.8247** | | | 0.260 |
| **pdu-eps2p51 s115** | 0.245/0.802 | **0.2501 / 0.8037** | +0.006 | +0.002 | **0.554** |
| **rmu s92** | 0.265/0.815 | **0.2597 / 0.8074** | −0.006 | −0.007 | **0.548** |
| satimp s46 | 0.324/0.731 | 0.3296 / 0.7414 | +0.006 | +0.010 | 0.412 |
| wga s184 | 0.315/0.715 | 0.3251 / 0.7263 | +0.011 | +0.012 | 0.401 |
| pure-npo s69 | 0.405/0.810 | 0.4152 / 0.8141 | +0.010 | +0.005 | 0.399 |
| grad-ascent s46 | 0.317/0.642 | 0.3573 / 0.7461 | +0.041 | **+0.104** | 0.389 |
| ceu s184 | 0.358/0.721 | 0.3659 / 0.7561 | +0.008 | +0.035 | 0.390 |
| grad-diff s69 | 0.317/0.740 | 0.4031 / 0.7896 | **+0.087** | +0.050 | 0.387 |
| simnpo s46 | 0.423/0.823 | 0.4248 / 0.8116 | +0.002 | −0.011 | 0.387 |

结论：

- **有效算法 = RMU 与 PDU-2.51，margin 0.548/0.554（Base 0.260 翻倍以上）**：
  wmdp_cyber 砍半（0.565→0.25），MMLU 仅掉 ~2pp；两者离线 parity 极佳
  （|Δ|≤0.007），在线曲线可直接信任。PDU 终点不塌的 R1 特性在存点
  ckpt 上兑现（0.250/0.804）。
- **在线↔离线 parity 7/9 在 |Δ|≤0.012 内**；三个离群（GD dW+0.087、
  GA dM+0.104、CEU dM+0.035）全部是能力受损模型且方向一致——离线
  （HF 前向）读数一律比在线（MCore 前向 + MMLU 子采样）"更健康"。
  与 intern 线同构：**曲线定最优步可信，受损模型的官方数以离线为准**。
- 离线口径下 GD 没有在线显示的那么崩（0.403/0.790，margin 0.387 与
  中游持平）；无算法把 wmdp_cyber 打进 0.25 以下。

在线测评曲线按轮次分目录（2026-10-02 拆分）：
`online-evaluation/g1/` = R1（11 配置首轮，零 ckpt 纯曲线）、
`online-evaluation/g2/` = R2（9 配置存 ckpt 重跑；2p16/2p30 未重跑无 g2 文件）。
原始混装 jsonl（R1/R2 追加在同一文件，root 属主不可改）完整归档在
`online-evaluation-R1R2-mixed/`。注意：若未来再跑，eval 输出仍会写到
`online-evaluation/` 根下（configs 的 output_dir），需要时再拆。
训练日志在 `logs/`。生成器 `make_configs.py` 可在 intern-cyber 源更新后重跑。
