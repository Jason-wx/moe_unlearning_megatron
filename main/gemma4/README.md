# Gemma4 main：wmdp_bio 十二配置批量实验

本目录是 Intern-S2 主线（`main/interns2/`）在 **gemma-26b-a4b** 上的复刻批量：
七个算法训练配置逐字段复制 qwen35 main 线（=intern main 线验证副本，超参与
训练计划不变），外加 RMU 最优臂重跑存 ckpt、PureNPO/SimNPO 各两条 lr 臂
（1e-5 / 1e-6，对应 intern V4「PureNPO 低 lr 无效、下轮移 1e-5」的遗留问题）。
**PDU（待 retain NLL 锚点 + ε 扫描）与 UNDIAL（待调试）本轮不跑。**

## 统一设定（十二个配置相同）

- 模型：`gemma-26b-a4b`（30 层 / hidden 2816 / **128 experts** / vocab 262144 /
  **pad_token_id 0**；路径为重组后真实路径
  `jincheng/sgtm/checkpoints/gemma/gemma-26b-a4b`——历史 yaml 里的
  `checkpoints/gemma-26b-a4b` 是死路径）
- 数据：forget `gemma4_wmdp_bio_softpack4096/forget`（22757 packs），retain
  `fineweb_edu_5b_gemma4_softpack4096_seed42`（111.5 万 packs）
- 布局 TP1/PP1/EP8，world_size 8，distributed optimizer
- gbs 32 / 3 epochs = **2133 步**（22757/32 = 711 步/epoch；注意与
  intern/qwen35 的 2142 不同，gemma 分词 pack 数略少），cosine→0 + 5% warmup，
  seed 42，bf16，clip 1.0，full trainable
- **save_final_checkpoint: true**（只存 final ckpt，save_steps 空）
- 在线评测 [wmdp_bio 全量, mmlu×35/leaf]，评测点
  [1, 300, 700, 1000, 1400, 1700, 1800, 1900, 2000, 2100, 2133]，
  输出 `main/gemma4/online-evaluation/`
- ckpt 落盘：`/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/wangxiao1/checkpoints/unlearn/megatron/main/`
  （与 interns2-main-* / qwen35-main-* 同根，目录名 `gemma4-main-*`）

## 十二个配置

| 配置 | 超参（=intern main） | lr | mbs | eval bs | 来源 |
|---|---|---|---|---|---|
| `rmu.yaml` | **steer1200 / α4.45**（R:F=0.001 校准臂）/ target9 / l7-9 | 1e-6 | 4 | 64 | gemma4 coeff-alpha sweep 最优臂重跑（原臂未存 ckpt，在线 0.341/0.724 待复核） |
| `grad_ascent.yaml` | 无 | 1e-6 | 4 | 64 | interns2 复制 |
| `grad_diff.yaml` | γ1/α1 | 1e-6 | 1 | 16 | interns2 复制 |
| `pure_npo.yaml` | β0.1/γ1/α1 | 5e-5 | 1 | 16 | interns2 复制（intern 有效工作点） |
| `pure_npo_lr1e-5.yaml` | 同上 | 1e-5 | 1 | 16 | lr 臂（V4 遗留：5e-6 全 β 无效后计划移 1e-5） |
| `pure_npo_lr1e-6.yaml` | 同上 | 1e-6 | 1 | 16 | lr 臂（下界探针） |
| `simnpo.yaml` | β0.1/δ0/γ1/α1 | 5e-5 | 1 | 16 | interns2 复制（**从未调参**） |
| `simnpo_lr1e-5.yaml` | 同上 | 1e-5 | 1 | 16 | lr 臂（与 PureNPO 同 brackets） |
| `simnpo_lr1e-6.yaml` | 同上 | 1e-6 | 1 | 16 | lr 臂（下界探针） |
| `satimp_forgetonly.yaml` | β1=2/β2=0.1/α0，retain null | 1e-6 | 4 | 64 | interns2 复制 |
| `WGA_forgetonly.yaml` | β2/γ1/α0，retain null | 1e-6 | 4 | 64 | interns2 复制 |
| `ceu.yaml` | γ1/ignore1，retain null | 1e-6 | 1 | 64 | interns2 复制（**从未调参**） |

mbs/eval 档位决策（2026-09-28 与王潇确认）：**照搬 Intern-S2 实测安全档**
（GD/PureNPO/SimNPO 因 vocab-CE 物化 + 冻结参考驻留 mbs1/eval16；CEU full-logit
目标函数 mbs1；GA/SatImp/WGA mbs4/eval64）。RMU 维持 gemma4 sweep 已验证档
（mbs4 / eval 64 / **activation_recompute 必须 null**——RMU hook 在 no-grad
前向下断边，config 校验拒绝 full；这是 RMU 与其余 11 个配置的唯一布局差异）。

⚠️ 与 intern/qwen35 的已知差异点：gemma4 是 26B 总参 / **A4B 激活**（比 qwen35
A3B 大 1/3），静态显存画像未实测。首跑若 step1-2 或首次 eval forward OOM：
eval bs 64→16 回退；mbs4 的四组（GA/SatImp/WGA/RMU）降 mbs（gbs 不变，优化
数学等价）。

## 提交（2026-09-28 已全部提交）

PureNPO / SimNPO 各三条 lr 臂为两个独立组提 safer2ai；其余六个两两配对提
safewt。`BATCH=g1..g5` 选组，每组一个 8 卡 rjob 串行跑组内配置：

| 批次 | 配置 | 档位 | rjob |
|---|---|---|---|
| g1 | grad_ascent + satimp | mbs4 快组 | `gemma4-main-g1-safewt-44457307` |
| g2 | grad_diff + ceu | mbs1 重组 | `gemma4-main-g2-safewt-46004055` |
| g3 | WGA + rmu | mbs4 快组 | `gemma4-main-g3-safewt-47349885` |
| g4 | pure_npo ×3（5e-5/1e-5/1e-6） | mbs1 重组 ×3 | `gemma4-main-g4-safer2ai-85870847` |
| g5 | simnpo ×3（5e-5/1e-5/1e-6） | mbs1 重组 ×3 | `gemma4-main-g5-safer2ai-87135999` |

```bash
# 补跑/换组提交（脚本本体同 qwen35 模式，三组仅资源组块不同）
BATCH=g1 bash rjob_main_safewt.sh        # safewt / safethm 同镜像
BATCH=g4 bash rjob_main_safer2ai.sh      # safer2ai（镜像 unlearning-cu126）
bash rjob_main_safewt.sh main/gemma4/ceu.yaml   # 显式传配置
BATCH=g1 PREVIEW=true bash rjob_main_safewt.sh  # 只打印 rjob 命令
```

g4/g5 为三条 mbs1 重配置串行，时长预计 ≈ g1/g3 的 4-6 倍。训练日志落
`logs/`（main_run_train.sh 按配置名写），rjob 层日志落 `outputs/rjob_logs/`。

## 后续

- **PDU ε 扫描已提交**（09-28 22:38）：见 [`pdu_eps_sweep/`](pdu_eps_sweep/)——
  锚点 2.0945（200 packs 随机抽样），ε 网格 2.0/2.1/2.2/2.25/2.5/3.0
  （绝对值，王潇 09-28 定），分三组各 2 臂：g1 safethm 14930900、
  g2 safewt 16164748、g3 safer2ai 17402560。
- UNDIAL：待调试。
- 12 个 final ckpt 出来后走 `lm_eval_accelerate.sh` 离线全量（对齐 intern
  ckpt_eval_summary 口径），基线用 gemma4 base 全量。
