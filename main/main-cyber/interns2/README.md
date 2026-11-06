# Intern-S2 main-cyber：十算法 bio→cyber 迁移

> **正式结论见 [`ckpt_eval_summary_20261001.md`](ckpt_eval_summary_20261001.md)**
> （九定点 ckpt 的离线全量成绩 = 本线最终口径 + 在线/离线 parity）。
> **ckpt 产生方式（完整 10 epochs + 中途定点存盘）与成绩单页见
> [`ckpt_notes.md`](ckpt_notes.md)**。
> 过程记录见 [`../online_eval_summary_20261001.md`](../online_eval_summary_20261001.md)
> （三轮在线测评 + "为何必须完整 10 epochs、只改存盘点"的方法论结论）。

本目录是 Intern-S2 主线（`main/interns2/`，bio）十个选定算法配置的 **wmdp_cyber**
迁移版（2026-09-30 建线，同日按 10-epoch 方案修订）。选点依据 bio 线离线汇总
`main/interns2/online-evaluation/ckpt_eval_summary_20260926.md` 的十算法主表，
逐配置溯源后仅做数据集切换 + 训练计划调整，超参 / 布局 / mbs 档 / 优化器设定 /
seed 全部不动。

## 统一训练计划（十配置相同，2026-09-30 定；**Round 1 口径，Round 2 已逐算法截断**，见下文 Round 2 节）

- **10 epochs**：cyber forget 语料 743 packs / gbs32 = 23 步/epoch（drop_last）
  → **230 optimizer steps**（cosine + 5% warmup ≈ 12 步）
- **lr：八配置统一 5e-6**（bio 均 1e-6；2026-09-30 定）；**PureNPO / SimNPO 保持
  5e-5**（其 bio 工作点即 5e-5，不在统一之列）
- **min_lr 全部 1e-6**（bio 全部 0）：cosine 下限不再到 0——八配置下限为峰值的
  20%，PureNPO/SimNPO 为 2%
- **在线评测每 epoch 一次**：步点 `[23, 46, 69, 92, 115, 138, 161, 184, 207, 230]`
- **只存 final ckpt**（step 230）：`save_final_checkpoint: true`，
  `save_steps: []`（GD/SimNPO 的 bio 中途存点不迁移；谷底/回弹问题由每 epoch
  的在线评测追踪）
- 布局 TP1/PP1/EP8 ws8、distributed optimizer、seed 42、bf16、clip 1.0、
  full trainable、shuffle: false、seq 4096、pad 248044
- mbs 分档沿用 bio 实测安全档：GA/RMU/SatImp/WGA=4，GD/PureNPO/SimNPO/CEU=1，
  UNDIAL=2（2³¹ int32 帽）；RMU `activation_recompute: null`
- 在线评测 [wmdp_cyber 全量 1987 题, mmlu×35/leaf]，eval bs：六配置 32
  （V-cyber 实证 cyber 题干更长，bs64→32）、四配置 16（GD/PureNPO/SimNPO/UNDIAL，
  16 的原因是全词表 eval logits + 静态占用，与数据集无关）
- ckpt 落盘（**2026-10-02 已整理**）：
  `/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/wangxiao1/checkpoints/unlearn/megatron/main-cyber/interns2/<算法>/`
  （一层直下 = HF 权重文件 + 训练配置 yaml + `ckpt_notes.md`；原 run 名
  `interns2-main-cyber-*` + `checkpoint-step-N/` 布局纯 rename 迁移）；
  在线评测输出本目录 `online-evaluation/`

## 选点 → 源配置 → 本目录配置

| 算法 | bio 选定 ckpt（bio/mmlu 离线） | 源配置 | 本目录 |
|---|---|---|---|
| RMU | interns2-main-rmu-step2142（0.370/0.842） | `main/interns2/rmu.yaml` | `rmu.yaml` |
| PDU (lr1e-6) | interns2-main-pdu-step2142（0.311/0.784） | `main/interns2/pdu.yaml` | `pdu.yaml` |
| GradAscent | interns2-main-grad-ascent-step2142（0.246/0.230，崩） | `main/interns2/grad_ascent.yaml` | `grad_ascent.yaml` |
| GradDiff | interns2-main-grad-diff-saveckpts-step2142（0.665/0.824） | `main/interns2/grad_diff_saveckpts.yaml` | `grad_diff.yaml`（改名，见下） |
| CEU | interns2-main-ceu-step2142（0.698/0.758） | `main/interns2/ceu.yaml` | `ceu.yaml` |
| WGA | interns2-main-wga-step2142（0.726/0.765） | `main/interns2/WGA_forgetonly.yaml` | `WGA_forgetonly.yaml` |
| SatImp | interns2-main-satimp-step2142（0.732/0.776） | `main/interns2/satimp_forgetonly.yaml` | `satimp_forgetonly.yaml` |
| SimNPO | interns2-main-simnpo-saveckpts-step2142（0.742/0.778） | `main/interns2/simnpo_saveckpts.yaml` | `simnpo.yaml`（改名，见下） |
| PureNPO | interns2-main-pure-npo-step2142（0.763/0.788） | `main/interns2/pure_npo.yaml` | `pure_npo.yaml` |
| UNDIAL | ~~interns2-main-undial-tp4-step2142~~（**作废**，TP 梯度 bug） | `main/interns2/undial_fixed.yaml`（修复重跑版） | `undial_fixed.yaml` |

> 溯源说明：
> 1. **GD / SimNPO 取 saveckpts 重跑版**（bio 表口径）。两者与本目录内对应的
>    普通 `grad_diff/simnpo` 数学完全相同（saveckpts 版只是多存中途 ckpt），
>    本线改为只存 final 后该差异消失，故文件名去掉 `-saveckpts` 后缀、
>    output_name 同步为 `interns2-main-cyber-grad-diff` / `-simnpo`，溯源关系
>    记录在本表。
> 2. **UNDIAL 不用表里的 undial_tp4**：该结果 2026-09-29 因 TP 梯度 bug 作废；
>    迁移源改为修复后的 `undial_fixed.yaml`（β3/α0 数学一致，TP1/PP1/EP8 主线
>    布局，mbs 2）。目标函数修复在代码里，对本线同样生效。
> 3. **PDU 只迁统一 lr 版**（表口径为 lr1e-6，本线按统一决定跑 5e-6）；bio 表
>    另保留 pdu-lr4e-6（0.408/0.844）为代表工作点，未在十选点内。eps=2.05 是
>    lr4e-6 下调出的，5e-6 与该调参档接近（bio lr1e-6 时 λ 会衰减到 ~0 失效）。

## 统一改动（十个配置相同，相对源配置）

- `data.name`: wmdp_bio → **wmdp_cyber**
- `forget_cache`: `Xtuner/interns2_wmdp_bio_softpack4096/forget`（22864 packs）
  → `Xtuner/interns2_wmdp_cyber_softpack4096/forget`（**743 packs**，语料为
  bio 的 1/31）
- `retain_cache`: **不变**（fineweb_edu_5b_interns2_softpack4096_seed42；forget-only
  三配置仍为 null）。cyber 专用 retain 语料（2455 packs）已建于
  `interns2_wmdp_cyber_softpack4096/retain` 但历史从未用过，本轮沿用 fineweb 锚点
- `num_train_epochs`: 3 → **10**（总步数 2142 → **230**）
- `learning_rate`: 1e-6 → **5e-6**（八配置；PureNPO/SimNPO 维持源配置 5e-5）
- `min_learning_rate`: 0 → **1e-6**（全部十配置）
- 在线评测 `steps` → **每 epoch 末** `[23, 46, ..., 230]`（bio 为 11 点含 step1
  探针，本线按"每 epoch 一测"不再含 step 1）
- `save_steps` → 全部 `[]`，仅 `save_final_checkpoint: true`
- `tasks`: [wmdp_bio, mmlu] → **[wmdp_cyber, mmlu]**
- `name` / `output_name`: `interns2-main-*` → `interns2-main-cyber-*`
- ckpt `output_root`: `.../checkpoints/unlearn/megatron/main` →
  `.../checkpoints/unlearn/megatron/**main-cyber**`
- 在线评测 `output_dir` → `main/main-cyber/interns2/online-evaluation/`
- **eval batch_size 64→32**（六配置：GA/RMU/PDU/SatImp/WGA/CEU）：cyber 题干
  更长，V-cyber 全线实测 32 无碍。**16 档不变**（GD/PureNPO/SimNPO/UNDIAL）

## 对表参照（V-cyber，gbs8/276 步口径，非本线设定）

Intern-S2 基线 cyber **0.585** / mmlu 0.867（离线全量）；V-cyber 各算法
（GA/GD/RMU/PureNPO + PDU ε 扫描）结果见
`configs/training/megatron/interns2/V-cyber/`。注意 V-cyber 为 gbs8/276 步且
RMU 超参未调（本线 RMU 为 bio 调优版 steer300/α350），数值只作量级参照。

## 提交记录（2026-09-30 13:03 提交，14:10 g1 改投 safewt）

分组沿用 bio 线两两配对，每组一个 8 卡 rjob、组内串行
（`main_run_train.sh` + `rjob_main.sh`）。原计划资源组 1:1:3
（safethm : safer2ai : safewt）；g1 起初投 safethm（91542559，始终
未起跑），当日撤单改投 safewt，**最终 0:1:4**：

| 组 | 配置（串行） | 资源组 | rjob |
|---|---|---|---|
| g1 | rmu → grad_diff | safewt | interns2-main-cyber-g1-safewt-57991162（原 safethm-91542559 已撤） |
| g2 | pdu → grad_ascent | safer2ai | interns2-main-cyber-g2-safer2ai-92818186 |
| g3 | pure_npo → simnpo | safewt | interns2-main-cyber-g3-safewt-7431934 |
| g4 | satimp_forgetonly → WGA_forgetonly | safewt | interns2-main-cyber-g4-safewt-8680527 |
| g5 | ceu → undial_fixed | safewt | interns2-main-cyber-g5-safewt-9818497 |

rjob 日志：`outputs/rjob_logs/interns2-main-cyber-<gN>-<组>_<ts>/`；
训练日志（本目录）：`logs/`；在线评测 JSONL：`online-evaluation/`。
五个 job 当日均 Succeeded；Round 1 结果归档于
`online-evaluation/round1-10ep/`（十条完整曲线）与 `logs-round1-10ep/`。

## Round 2：逐算法 epoch 截断（2026-09-30 17:10 定稿，17:14 已提交 9 job）

Round 1 结论（0.25 = 四选一随机线；基线 cyber 0.585 / mmlu 0.867）：
**RMU / PDU / GradDiff 有效**（RMU ep4-5 最优 0.245-0.249/0.843-0.850，三者
后程均出现 cyber 回弹或 mmlu 滑坡）；PureNPO ep3 触底 0.387/0.841 后回弹；
CEU ep4 起平台 0.40/0.77；SimNPO 全程横 0.48-0.50 无效；UNDIAL 纹丝不动；
GA / SatImp / WGA（forget-only 升向族）ep1-2 即"遗忘完成 + MMLU 崩塌"
（0.24-0.45）。

Round 2 仅改 `num_train_epochs`（连带 eval steps / cosine 总长），其余不动；
仍只存 final ckpt。**Round 1 的十个 step-230 ckpt 已删除**（root 属主，
sudo rm）。选点依据 Round 1 曲线：

| 算法 | epochs | 总步数 | eval 点 | Round 1 依据 |
|---|---|---|---|---|
| RMU | 4 | 92 | [23,46,69,92] | ep4 = 0.249/0.850，peak 前截断 |
| PDU | 3 | 69 | [23,46,69] | ep3 = 0.288/0.857，mmlu 滑坡前 |
| GradDiff | 6 | 138 | [23..138] | ep6 = 0.285/0.830，后程 mmlu 滑坡前 |
| PureNPO | 3 | 69 | [23,46,69] | ep3 = 0.387/0.841 触底 |
| CEU | 4 | 92 | [23..92] | ep4 = 0.396/0.769 平台起点 |
| SimNPO | 1 | 23 | [23] | ep1 = 0.485/0.854（本来就无效） |
| SatImp | 1 | 23 | [23] | ep1 = 0.372/0.804，ep2 崩塌前最后 sane 点 |
| WGA | 1 | 23 | [23] | ep1 = 0.375/0.803，同上 |
| GA | 2 | 46 | [23,46] | ep2 = 0.267/0.447（遗忘完成点） |
| UNDIAL | 不动（10） | 230 | — | 全程无效，先不管 |

⚠️ **调度差异提醒**：`lr_decay_steps: auto` 随新总步数自适应，Round 2 的
"step 92"处于 cosine 终点（lr≈floor 1e-6），而 Round 1 的 step 92 还在
中段衰减——同为 92 步训练动力学不同，Round 2 终点不会精确复现 Round 1
同 step 数值（用户拍板"只改 epochs"，接受此差异）。

Round 2 提交记录（2026-09-30 17:14，**每算法一个 8 卡 rjob**，UNDIAL 剔除）：

| 算法 | rjob | 资源组 |
|---|---|---|
| PDU | interns2-main-cyber-pdu-safer2ai-37201633 | safer2ai |
| GA | interns2-main-cyber-ga-safer2ai-38428995 | safer2ai |
| RMU | interns2-main-cyber-rmu-safewt-46619771 | safewt |
| GradDiff | interns2-main-cyber-gd-safewt-47816914 | safewt |
| PureNPO | interns2-main-cyber-pure-npo-safewt-48953026 | safewt |
| SimNPO | interns2-main-cyber-simnpo-safewt-50160563 | safewt |
| SatImp | interns2-main-cyber-satimp-safewt-51303134 | safewt |
| WGA | interns2-main-cyber-wga-safewt-52450412 | safewt |
| CEU | interns2-main-cyber-ceu-safewt-53566230 | safewt |

safer2ai 两个沿用 Round 1 的 PDU+GA 配对；在线评测直接写
`online-evaluation/`（round-2 曲线，round-1 已归档子目录）。

Round 2 终值 vs R1 同步数：PDU 精确复现（0.284/0.854 ≈ 0.288/0.857）、
RMU 接近（0.285/0.856 vs 0.249/0.850）、**GD/PureNPO/GA/SatImp/WGA 全部
"欠遗忘"**——短调度把终点压到 cosine 底（lr 1e-6），累积 lr 质量远小于
R1 同步数，R1 的中间点形态无法靠截断复现（SimNPO 例外，略深 0.435）。
R2 曲线归档 `online-evaluation/round2-truncated/`、日志 `logs-round2-truncated/`。

## Round 3：回 R1 全调度 + 定点存 ckpt（2026-09-30 19:30 定稿，19:29 已提交）

策略修正：**九配置全部回到 10 epochs / 230 步**（cosine 调度与 R1 完全
一致，动力学精确复刻），仅把存盘点换成各算法 R1 选出的最优 step
（`save_steps: [指定步]`，`save_final_checkpoint: false`，即只有定点 ckpt
没有 step-230）；在线评测恢复每 epoch 全量 [23..230]（可顺带验证与 R1
的复现性）。R2 的八个 ckpt 已删；CEU 的 R2 短调度 ckpt 待其 job 结束后
补删。提交布局（1 safer2ai + 4 safewt，组内串行）：

| rjob | 配置（串行） | 资源组 |
|---|---|---|
| interns2-main-cyber-pdu-safer2ai-39926025 | pdu | safer2ai |
| interns2-main-cyber-gd-simnpo-safewt-53365446 | grad_diff → simnpo | safewt |
| interns2-main-cyber-ga-satimp-safewt-54550411 | grad_ascent → satimp | safewt |
| interns2-main-cyber-pnpo-wga-safewt-55676566 | pure_npo → WGA | safewt |
| interns2-main-cyber-ceu-rmu-safewt-3619699 | ceu → rmu | safewt |

（CEU 的 R2 短调度 ckpt 已在其 job 结束后删除，曲线/日志归档
round2-truncated/；ceu-rmu 组 19:38 补交。）

**离线正式测评（10-01 定稿 safer2ai 3 卡 bs16）**：`rjob_ckpt_eval.sh`
→ rjob `interns2-main-cyber-ckpt-eval-g3bs16-1824258`（10-01 23:10 提交，
safer2ai，3 卡，每卡
串行 3 ckpt，单卡/ckpt，任务 `wmdp_cyber,mmlu` 全量 0-shot）。
**bs 64→32→16 的原因**：首投 bs64（g3-43764553）三卡全部第一步 OOM——单卡
Intern-S2 bf16 模型 ~72 GiB，loglikelihood 的 64×~2.8k×248k-vocab logits
张量 ~83 GiB > 剩余 68 GiB（bio 线 bs64 能跑纯因 bio 题干短；与在线测评
64→32 同因）。bs32（g3bs32-38718786）仍 OOM——部分 cyber 题干长于
~2.8k 估计（实测 OOM 行：`Tried to allocate 41.58 GiB`，卡上仅余
32.64 GiB，模型占 107.42 GiB），bs16（最坏 ~21 GiB logits，余量 ~12 GiB）
为工作档。batch 不影响 lm_eval
结果只影响吞吐。GPUS/BATCH_SIZE 均已参数化。前四投 safewt-77474695 /
safethm-28402891 / safethm-g3-45074916 未起跑即撤、g3-43764553 bs64 OOM、
g3bs32-38718786 bs32 OOM。结果写本目录 `ckpt-eval/
lm_eval_results_<ckpt>_wmdp_cyber_mmlu/result.json`，已有 result.json 的
ckpt 重提自动跳过。**已 Succeeded（23:43，9/9 成功，约 32 min，
~10 min/ckpt）**，成绩与 parity 见 `ckpt_eval_summary_20261001.md`：
离线终版 RMU@92 **0.252/0.836** 最优、PDU@69 0.274/0.846、GD@138
0.302/0.822，其余六个无可用工作点；cyber 在线/离线 7/9 算法 |Δ|≤0.006
（在线评测口径获验证），在线 mmlu 因 35 题/叶子子样系统性偏高 ~0.02。

**R3 完结（10-01）**：五 job 全 Succeeded，九定点 ckpt 齐落盘。存盘点
R3 vs R1 偏差：8/9 算法 |Δcyber|≤0.031 且 |Δmmlu|≤0.022（PDU/GD/SatImp/
WGA/SimNPO ≤0.012，RMU +0.008，PureNPO +0.023，CEU +0.031）——复现成立，
定点 ckpt 可作"R1 最优点权重"使用；唯一例外 GA（mmlu 0.295 vs 0.447，
崩塌区混沌放大）。终版定点成绩（R3 实测）：

| 算法 | ckpt | cyber | mmlu |
|---|---|---|---|
| RMU | step-92 | 0.257 | 0.851 |
| PDU | step-69 | 0.277 | 0.846 |
| GradDiff | step-138 | 0.274 | 0.827 |
| PureNPO | step-69 | 0.410 | 0.854 |
| CEU | step-92 | 0.427 | 0.790 |
| SimNPO | step-23 | 0.475 | 0.844 |
| SatImp | step-23 | 0.374 | 0.801 |
| WGA | step-23 | 0.372 | 0.808 |
| GA | step-46 | 0.249 | 0.295（崩） |

| 算法 | save_steps（=R1 最优） | R1 该点 (cyber/mmlu) |
|---|---|---|
| RMU | [92] | 0.249 / 0.850 |
| PDU | [69] | 0.288 / 0.857 |
| GradDiff | [138] | 0.285 / 0.830 |
| PureNPO | [69] | 0.387 / 0.841 |
| CEU | [92] | 0.396 / 0.769 |
| SimNPO | [23] | 0.485 / 0.854 |
| SatImp | [23] | 0.372 / 0.804 |
| WGA | [23] | 0.375 / 0.803 |
| GA | [46] | 0.267 / 0.447 |
| UNDIAL | 不参与（无效，配置维持 R1 原样） | — |

## ⚠️ 解读注意

1. **230 步 + lr 5e-6 vs bio 2142 步 + 1e-6**：八配置 lr 为 bio 的 5×且调度
   更短（warmup≈12 步），GA 这类无 retain 约束的算法崩塌风险比 bio 更早；
   PDU 的对偶动力学与 PureNPO/SimNPO 的 lr5e-5 行为也都与 bio 不同档。
2. SimNPO / CEU / UNDIAL 超参在 bio 上即未调参（见 `main/interns2/README.md`
   警示节），cyber 上若遗忘弱/崩坏同样先怀疑超参。
