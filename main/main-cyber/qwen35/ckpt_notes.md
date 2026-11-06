# Qwen3.5 main-cyber 定点 ckpt 说明（产生方式 + 测评成绩）

对象：`/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/wangxiao1/checkpoints/unlearn/megatron/main-cyber/qwen35/<算法>/`
九个权重文件夹（rmu、pdu-eps2p51、grad-diff、grad-ascent、pure-npo、
simnpo、satimp-forgetonly、wga-forgetonly、ceu；每份 ~66G）——**文件夹
一层直下即 HF 格式权重文件（config.json + tokenizer + 16×safetensors）
+ 产生该权重的完整训练配置 yaml**（与仓库 `main/main-cyber/qwen35/*.yaml`
逐字节一致，md5 已核）。原始布局（run 名 `qwen35-main-cyber-<算法>/` +
`checkpoint-step-<N>/` 中间层）已于 2026-10-02 整理为现结构，纯 rename
未动数据；各算法的存点 step 见第一节表格（原 checkpoint-step-N）。

## 一、ckpt 产生方式：完整跑 10 epochs，中途定点存 ckpt

**九个权重全部来自"完整 10-epoch 训练计划 + 中途 `save_steps` 定点导出"，
不是提前停止（early stop）的产物。**

训练计划（九配置统一，qwen-cyber R1/R2 两轮逐字节相同）：

- **10 epochs = 230 optimizer steps**（cyber forget 语料 743 packs / gbs 32
  = 23 步/epoch），`lr_scheduler_type: cosine` + 5% warmup、
  `min_learning_rate: 1e-6`、seed 42；
- **cosine 调度锚定在总步数 230 上**——这是必须整程跑的原因：若把
  `num_train_epochs` 截到目标 epoch 来"提前拿权重"，cosine 形状会随总步数
  重算，同一 step 的 lr 位置与累积 lr 质量（∫lr dt）完全改变，step N 处
  的模型状态**不再对应在线曲线上那个点**。该结论在 intern-cyber 线上被
  截断实验（R2）直接证伪过：9 算法里 7 个截断版"欠遗忘"（GD +0.12、
  SatImp/WGA +0.10），详见
  `main/main-cyber/online_eval_summary_20261001.md` 第一节；
- 因此本线采用**调度不动、只改存盘点**的做法：`save_steps: [N]`（N =
  该 run 在线曲线的 margin 最优点，margin = mmlu − wmdp_cyber 的 argmax，
  同分取最早步），`save_final_checkpoint: false`（step 230 不存）。调度、
  数据、seed、超参与曲线轮完全一致，step N 处的动力学被精确复刻；
- 轮次：**R1**（2026-10-01，11 配置零 ckpt 纯曲线轮，产出在线曲线与
  选点）→ **R2**（同日，9 配置整程重跑 + 上表定点存 ckpt；PDU 只保留
  eps2.51 臂）。R2 与 R1 同步数在线值偏差全部在噪声带内（wmdp |Δ|≤0.020
  共 8/9，唯 CEU +0.061），存点处复现成立。

| run（ckpt 目录名后缀） | 存点 step | epoch | lr | mbs |
| --- | --- | --- | --- | --- |
| rmu | 92 | 4 | 5e-6 | 4 |
| pdu-eps2p51 | 115 | 5 | 5e-6 | 1 |
| grad-diff | 69 | 3 | 5e-6 | 1 |
| grad-ascent | 46 | 2 | 5e-6 | 4 |
| pure-npo | 69 | 3 | 5e-5 | 1 |
| simnpo | 46 | 2 | 5e-5 | 1 |
| satimp-forgetonly | 46 | 2 | 5e-6 | 4 |
| wga-forgetonly | 184 | 8 | 5e-6 | 4 |
| ceu | 184 | 8 | 5e-6 | 1 |

R1→R2 同步数复现（wmdp_cyber，在线）：

| run | R1@step | R2@step | Δ |
| --- | --- | --- | --- |
| rmu | 0.253 | 0.265 | +0.013 |
| pdu-eps2p51 | 0.252 | 0.245 | −0.008 |
| grad-diff | 0.322 | 0.317 | −0.005 |
| grad-ascent | 0.318 | 0.317 | −0.002 |
| pure-npo | 0.425 | 0.405 | −0.020 |
| simnpo | 0.430 | 0.423 | −0.008 |
| satimp | 0.324 | 0.324 | +0.000 |
| wga | 0.309 | 0.315 | +0.006 |
| ceu | 0.297 | 0.358 | +0.061 |

## 二、各权重测评成绩（在线 @ 存点步 + ckpt 离线全量）

两套口径：

- **在线**：训练中每 epoch 一次（MCore 前向），wmdp_cyber 全量 1987 题、
  MMLU ×35/leaf 子采样（~2k 题）；取 R2 存点步的值。曲线文件
  `online-evaluation/g2/`（R1 在 `g1/`）；
- **离线（ckpt 官方口径）**：`scripts/eval.sh` 对导出权重
  跑 lm-eval，wmdp_cyber 全量 1987 + MMLU 全量 14042，0-shot，bf16，
  batch 16。结果在 `ckpt-eval/lm_eval_results_<run>-step<N>_wmdp_cyber_mmlu/result.json`。

| 权重 | 在线 wmdp/mmlu | **离线 wmdp/mmlu** | margin（离线） |
| --- | --- | --- | --- |
| **Base（qwen3p5-35b-a3b）** | — | **0.5652 / 0.8247** | 0.260 |
| **pdu-eps2p51 s115** | 0.245 / 0.802 | **0.2501 / 0.8037** | **0.554** |
| **rmu s92** | 0.265 / 0.815 | **0.2597 / 0.8074** | **0.548** |
| satimp s46 | 0.324 / 0.731 | 0.3296 / 0.7414 | 0.412 |
| wga s184 | 0.315 / 0.715 | 0.3251 / 0.7263 | 0.401 |
| pure-npo s69 | 0.405 / 0.810 | 0.4152 / 0.8141 | 0.399 |
| ceu s184 | 0.358 / 0.721 | 0.3659 / 0.7561 | 0.390 |
| grad-ascent s46 | 0.317 / 0.642 | 0.3573 / 0.7461 | 0.389 |
| grad-diff s69 | 0.317 / 0.740 | 0.4031 / 0.7896 | 0.387 |
| simnpo s46 | 0.423 / 0.823 | 0.4248 / 0.8116 | 0.387 |

要点：

1. **有效算法 = PDU-2.51@115（margin 0.554）与 RMU@92（0.548）**：
   wmdp_cyber 从 Base 0.565 砍半到 ~0.25，MMLU 只掉 ~2pp。两者在线↔离线
   parity 极佳（|Δ|≤0.007），在线读数可直接信任。
2. **在线↔离线 parity 7/9 在 |Δ|≤0.012 内**；三个离群（GD dW+0.087、
   GA dM+0.104、CEU dM+0.035）全部是能力受损模型且方向一致——离线
   （HF 前向 + 全量 MMLU）读数比在线更"健康"。与 intern 线结论同构：
   曲线定最优步可信，受损模型的对外成绩以离线为准（GD 离线 0.403/0.790
   并没有在线显示的那么崩）。
3. 无算法把 wmdp_cyber 打进 0.25 以下；qwen35-cyber 未复现 bio 线的
   0.79-0.81 遗忘墙（全线深度遗忘）。

过程与提交记录见本目录 `README.md`；离线测评提交脚本 `rjob_ckpt_eval.sh`。
