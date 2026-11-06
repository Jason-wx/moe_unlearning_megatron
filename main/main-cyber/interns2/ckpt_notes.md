# Intern-S2 main-cyber 定点 ckpt 说明（产生方式 + 测评成绩）

对象：`/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/wangxiao1/checkpoints/unlearn/megatron/main-cyber/interns2/<算法>/`
九个权重文件夹（rmu、pdu、grad-diff、grad-ascent、pure-npo、simnpo、
satimp-forgetonly、wga-forgetonly、ceu；每份 ~66G）——**文件夹一层直下即
HF 格式权重文件（config.json + tokenizer + safetensors）+ 产生该权重的
完整训练配置 yaml**（与仓库 `main/main-cyber/interns2/*.yaml` 逐字节一致，
md5 已核）。原始布局（run 名 `interns2-main-cyber-<算法>/` +
`checkpoint-step-<N>/` 中间层）已于 2026-10-02 整理为现结构，纯 rename
未动数据；各算法的存点 step 见第一节表格（原 checkpoint-step-N）。

## 一、ckpt 产生方式：完整跑 10 epochs，中途定点存 ckpt

**九个权重全部来自 Round 3：完整 10-epoch 训练计划 + 中途 `save_steps`
定点导出，不是提前停止（early stop）的产物。**

训练计划（九配置统一，R1/R3 两轮逐字节相同）：

- **10 epochs = 230 optimizer steps**（cyber forget 语料 743 packs / gbs 32
  = 23 步/epoch），`lr_scheduler_type: cosine` + 5% warmup、
  `min_learning_rate: 1e-6`、seed 42；lr 八配置统一 5e-6（PureNPO/SimNPO
  5e-5）；
- **为什么必须整程跑（本线的实证结论）**：cosine 调度锚定在总步数上，
  截断 epochs 会重算调度形状——同一 step 的 lr 位置与累积 lr 质量
  （∫lr dt）完全改变，step N 处的模型状态不再对应在线曲线上那个点。
  **Round 2 专门做过截断实验证伪**：9 算法截断到各自最优 epoch 重跑，
  7 个"欠遗忘"（GradDiff 终点比 R1 同步数浅 +0.12，SatImp/WGA +0.10，
  GA 没崩也没忘透；仅 PDU/SimNPO 近似复现）。截断对照表见
  `../online_eval_summary_20261001.md` 第 1.2 节；
- 因此 **Round 3（终版）调度不动、只改存盘点**：`save_steps: [N]`（N =
  R1 在线曲线选定的最优点/最后 sane 点，选点依据见下表），
  `save_final_checkpoint: false`（step 230 不存）。调度、数据、seed、超参
  与 R1 曲线轮完全一致，step N 处的动力学被精确复刻——R3 存盘点在线值
  与 R1 同步数偏差 8/9 ≤0.031（PDU/GD/RMU/SatImp/WGA/SimNPO ≤0.012，
  同 seed 非确定性核噪声量级），复现成立。

| run（ckpt 目录名后缀） | 存点 step | epoch | R1 选点依据 |
| --- | --- | --- | --- |
| rmu | 92 | 4 | peak 平台前段，mmlu 尚未滑坡 |
| pdu | 69 | 3 | mmlu 滑坡（0.86→0.77）之前 |
| grad-diff | 138 | 6 | 单调下探中段（后程 mmlu 0.83→0.75） |
| pure-npo | 69 | 3 | 全程触底点（之后回弹到 0.43） |
| ceu | 92 | 4 | 平台起点 |
| simnpo | 23 | 1 | 本来无效，留最小档 |
| satimp-forgetonly | 23 | 1 | ep2 起崩塌（mmlu 0.80→0.35）前最后 sane 点 |
| wga-forgetonly | 23 | 1 | 同上 |
| grad-ascent | 46 | 2 | 遗忘完成点（mmlu 已开始崩） |

（UNDIAL 全程 ~0.585 不动，无效剔除未跑。）

## 二、各权重测评成绩（在线 @ 存点步 + ckpt 离线全量）

两套口径：

- **在线**：训练中每 epoch 一次（MCore 前向），wmdp_cyber 全量 1987 题、
  MMLU ×35/leaf 子采样（~2k 题）；取 R3 存点步的值。曲线按轮次分目录：
  `online-evaluation/`（根，= R3 存点轮）、`online-evaluation/round1-10ep/`
  （R1 曲线轮）、`online-evaluation/round2-truncated/`（R2 截断对照轮）；
- **离线（ckpt 官方口径）**：`scripts/eval.sh` 对导出权重
  跑 lm-eval，wmdp_cyber 全量 1987 + MMLU 全量 14042，0-shot，bf16，
  batch 16。结果在 `ckpt-eval/lm_eval_results_<run>-step<N>_wmdp_cyber_mmlu/result.json`。

| 权重 | R1 曲线@step | 在线 R3@step | **离线 wmdp/mmlu** | margin（离线） |
| --- | --- | --- | --- | --- |
| **Base（Intern-S2-base05）** | — | — | **0.5918 / 0.8515** | 0.260 |
| rmu s92 | 0.249/0.850 | 0.257/0.851 | **0.2516 / 0.8358** | **0.584** |
| pdu s69 | 0.288/0.857 | 0.277/0.846 | **0.2743 / 0.8457** | **0.571** |
| grad-diff s138 | 0.285/0.830 | 0.274/0.827 | 0.3020 / 0.8222 | 0.520 |
| pure-npo s69 | 0.387/0.841 | 0.410/0.854 | 0.4117 / 0.8321 | 0.420 |
| simnpo s23 | 0.485/0.854 | 0.475/0.844 | 0.4706 / 0.8166 | 0.346 |
| satimp s23 | 0.372/0.804 | 0.374/0.801 | 0.3800 / 0.7871 | 0.407 |
| wga s23 | 0.375/0.803 | 0.372/0.808 | 0.3754 / 0.7864 | 0.411 |
| ceu s92 | 0.396/0.769 | 0.427/0.790 | 0.4540 / 0.7671 | 0.313 |
| grad-ascent s46 | 0.267/0.447 | 0.249/0.295 | 0.2461 / 0.3333 | 0.087 |

要点（正式结论见 `ckpt_eval_summary_20261001.md`）：

1. **有效算法 = RMU@92（0.252/0.836，margin 0.584）与 PDU@69
   （0.274/0.846，margin 0.571）**：wmdp_cyber 从 Base 0.592 砍半，
   MMLU 基本不掉。GradDiff 深度第二（0.302/0.822）但代价更大。
2. **GA 崩**（0.246/0.333）；SatImp/WGA/CEU 中庸；SimNPO/PureNPO 浅
   （0.41-0.47）。R3 定点存盘正是为了把 SatImp/WGA 这类"ep2 起崩塌"的
   算法停在崩塌前的 sane 点。
3. **在线↔离线 parity（cyber 口径）7/9 |Δ|≤0.006**：在线评测（MCore
   前向）与离线 HF 全量评测在同一 ckpt 上等价，在线曲线可信。两个例外
   GradDiff +0.028 / CEU +0.027（方向"离线略浅"，~1.4σ 非 bug 级），引用
   时以离线 0.302/0.454 为准；在线 mmlu 因 35 题/叶子子样系统性偏高
   ~0.02，mmlu 绝对值以离线为准；GA 的 mmlu 两次测量（在线 0.295 / 离线
   0.333）都确认能力已毁。

过程与三轮记录见 `../online_eval_summary_20261001.md` 与本目录 `README.md`。
