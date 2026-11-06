# Intern-S2 main：十算法规模复跑

本目录以 V-final **archive-gbs8**（2026-09-15 rjob
`interns2-v-final-five-algorithms`，结果归档于
`configs/training/megatron/interns2/V-final/online-evaluation/archive-gbs8/`）
的算法超参为基础，做 main 线规模运行。V-final 的五算法（GA/GD/RMU/PDU/PureNPO）
之外，后续补入 SimNPO / SatImp / WGA / CEU / UNDIAL，共十个算法。
离线全量测评汇总见 `online-evaluation/ckpt_eval_summary_20260926.md`。

## 统一设定（全部配置相同）

- 布局 TP1 / PP1 / EP8，world_size 8，distributed optimizer
- cosine + 5% warmup，seed 42，bf16，clip 1.0，full trainable
- **自动保存 final checkpoint**（save_final_checkpoint: true）
- 在线评测 [wmdp_bio 全量, mmlu×35/leaf]，bs 64（例外见下）
- ckpt 根目录：`/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/wangxiao1/checkpoints/unlearn/megatron/main`
- 在线评测输出：`main/interns2/online-evaluation/`

## 统一训练设定

- gbs 32 / mbs 4，3 epochs = **2142 optimizer steps**（例外见下）
- **mbs 2**：`undial_fixed`（见下方 UNDIAL 说明）
- **mbs 1**：`pdu/ceu/undial/undial_tp4`（full-logit：自算 objective 里物化全
  vocab fp32 logits）以及 `grad_diff/pure_npo/simnpo`（Megatron vocab_parallel CE 的
  `calculate_logits_max` 在 mbs4 物化 ~30.7 GiB 全 logits，叠加带 retain 流/
  参考模型的 114–128 GiB 静态占用，step2 即 OOM，2026-09-23/24 各两轮实测）。
  均已验证 mbs1 稳定（ceu/pdu 全程跑通）。gbs 不变，优化数学等价。
- **eval batch_size 16**：`grad_diff/pure_npo/simnpo/undial*`（含各 saveckpts 版；静态占用高的
  配置 eval bs64 也会顶爆；forget-only 且无参考的 grad_ascent/satimp/WGA 实测
  bs64 无碍，保持 64）
- 在线评测点 [1, 300, 700, 1000, 1400, 1700, 1800, 1900, 2000, 2100, 2142]

## 各算法差异

| 配置 | lr | 调度 | activation_recompute | save 策略 |
|---|---|---|---|---|
| `grad_ascent.yaml` | 1e-6 | cosine→0 | full | 仅 final |
| `grad_diff.yaml` | 1e-6 | cosine→0 | full | 仅 final |
| `rmu.yaml` | 1e-6 | cosine→0 | **null（强制）** | 仅 final |
| `pdu.yaml` | 1e-6 | cosine→0 | full | 仅 final（**mbs 1**，full-logit OOM） |
| `pure_npo.yaml` | **5e-5** | cosine→0（取自 V-final 主对比 run，见文件头注释） | full | 仅 final |
| `simnpo.yaml` | **5e-5** | cosine→0（镜像 pure_npo：beta 0.1/gamma 1/alpha 1 + delta 0） | full | 仅 final |
| `satimp_forgetonly.yaml` | 1e-6 | cosine→0 | full | 仅 final |
| `WGA_forgetonly.yaml` | 1e-6 | cosine→0 | full | 仅 final |
| `ceu.yaml` | 1e-6 | cosine→0 | full | 仅 final（**mbs 1**，full-logit OOM） |
| `undial.yaml` | 1e-6 | cosine→0 | full | 仅 final（**mbs 1**，TP1 下仍 OOM，未跑成） |
| `undial_tp4.yaml` | 1e-6 | cosine→0 | full | 仅 final（TP4/PP2/EP4+SP，mbs 1；**结果作废**，TP 梯度 bug） |
| `undial_fixed.yaml` | 1e-6 | cosine→0 | full | 仅 final（**mbs 2**，TP 梯度修复后的重跑，见下） |

补充重跑（训练数学与原配置完全相同，只改存档/lr）：

| 配置 | 相对原配置的差异 | 目的 |
|---|---|---|
| `grad_diff_saveckpts.yaml` | 额外存 step 714（=1 epoch） | 原 run 在 step 700 触底（在线 0.589/0.828）后回弹，保留谷底 ckpt |
| `simnpo_saveckpts.yaml` | 额外存 step 300、714；在线评测加 714 | 同上（原 run 谷底 step 700） |
| `pdu_lr4e-6.yaml` | lr 1e-6 → 4e-6 | eps 2.05 是在 lr4e-6 下调出的；lr1e-6 时 λ 很快衰减到 ~0、约束失效。**PDU 代表工作点**（离线 0.408/0.844） |

> 离线汇总表里 PDU 取的是统一 lr 的 `pdu.yaml`（0.311/0.784），结论部分以
> `pdu_lr4e-6` 为代表工作点，两者都保留。

> RMU 的 `activation_recompute` 必须为 null：config 校验（`megatron_unlearning/config.py`
> 的 RMU 分支）会直接拒绝 full —— hook 在 checkpoint 的 no-grad 前向里计算激活
> 损失，autograd 边丢失、优化器零梯度。

forget-only 配置（retain_cache: null，alpha=0 或算法本身无 retain 项）：

- `satimp_forgetonly.yaml`：beta1=2.0 / beta2=0.1。V-final 该格 1ep 仅 75.7/83.0
  （遗忘不足），3ep 是其结论里"需要更长训练"的直接验证。
- `WGA_forgetonly.yaml`：beta=2.0 / gamma=1.0，源自 V4/WGA forget-only 对照。
- `ceu.yaml`：gamma=1.0 / ignore_first_n_tokens=1，源自 V4/ceu.yaml（CEU 目标
  无 retain 项）。
- `undial.yaml`：beta=3.0 / gamma=1.0 / alpha=0，源自 V3 根配置；UNDIAL 需要
  冻结参考模型（reference 块真实生效）。
- `undial_fixed.yaml`（2026-09-29）：`undial.yaml` 在 TP1 下 OOM、`undial_tp4.yaml`
  在 TP4 下命中 vocab-parallel 梯度 bug（结果作废）。修复后 objective 改为分块自定义
  autograd（不再物化全 vocab fp32 临时张量），TP1/EP8 布局 mbs 2 可跑（116 GiB）。
  mbs 不上 4：Intern-S2 单行 full-logits 4096×251392≈1.03e9 元素，mbs 4 超 2³¹
  （int32 越界，gemma4 PDU 崩溃同根因）。gbs 仍 32。

## ⚠️ 未调参算法提醒：SimNPO / CEU / UNDIAL

这三个算法在本项目中**没有做过针对性调参**，main 配置的超参是"直搬历史唯一
配置"或"从 PureNPO 镜像"，不是验证过的工作点：

| 配置 | 超参来源 | 历史证据（都不算调参） |
|---|---|---|
| `simnpo.yaml` | β0.1/α1/lr5e-5 **镜像自 pure_npo**，SimNPO 自身没在这个工作点跑过 | V4 的 β(0.5–7.5)/α(1/3/5)/官方对齐轮全在 lr5e-6 量级、全部零遗忘；β0.1 从未测过 |
| `ceu.yaml` | gamma1.0/ignore1 直搬 `V4/ceu.yaml`（该配置 V4 从未运行） | 仅 V1/V3 各有一轮默认超参结果，V1 判"几乎无有效遗忘"，之后未再调 |
| `undial_fixed.yaml` | β3/α0 直搬 V3 根配置 | 仅 V1 一轮默认超参（"几乎无有效遗忘"），β 从未扫过；V3 与 undial_tp4 为 TP4，受梯度 bug 影响，不能作为证据 |

**解读规则**：这三个若在 main 线遗忘弱或崩坏，**优先怀疑超参不适配**
（β 量级、lr、forget-only vs 带 retain），不要直接下"算法无效"的结论；
反之若有效，也说明还有调参空间。需要跟进时，参考 V4 的教训先定
β 量级再细扫。

## 与 V-final 当前 yaml 的差异（复刻依据）

V-final 根目录的五个 yaml 后来被就地改为 gbs32 / lr4e-6 / 714 步（PureNPO 为
lr5e-5），本目录按以下证据还原 gbs8 版本：

- 训练计划：旧日志 `V-final/logs/{grad_ascent,grad_diff,pdu,pure_npo}.log`
  的 `training_schedule=... steps=2858 micro_batch_size=1 global_batch_size=8
  lr_warmup_steps=142`；
- lr：GA/GD/PDU 日志峰值 `lr=1.00e-06`；PureNPO 日志峰值 `lr=1.00e-05` 且
  142 个不同 lr 值后恒定（= warmup 后恒定，min_lr=lr）；RMU 旧日志被 gbs32
  重跑覆盖，lr1e-6 依据 `V-final/RESULTS.md` §3.3 与 V4 源配置
  `RMU_lr_1e-6-full-clip.yaml`；
- 评测点：归档 jsonl 的 step 序列；
- PureNPO save_steps=[2500]：旧日志 `save_steps=[2500]`。

命名从 `interns2-V-final-*` 改为 `interns2-main-*`，checkpoint 根目录与
在线评测输出改指向本目录，避免覆盖 V-final 产物。
