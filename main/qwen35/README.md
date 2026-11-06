# Qwen3.5 main：wmdp_bio 八配置批量实验

本目录是 Intern-S2 主线（`main/interns2/`，结果汇总见其
`online-evaluation/ckpt_eval_summary_20260926.md`）在 Qwen3.5-35B-A3B 上的
复刻批量：七个算法训练配置逐字段复制 interns2 main 线（超参与训练计划不变），
另加一个 RMU 调优臂重跑。**PDU（待 retain_loss_eps 初扫定超参）与
UNDIAL（性能一般待调试）本轮不跑。**

## 统一设定（八个配置相同）

- 模型：`qwen3p5-35b-a3b`（256 experts，路径 jincheng/sgtm/checkpoints/qwen/）
- 数据：forget `qwen35_wmdp_bio_softpack4096/forget`，retain
  `fineweb_edu_5b_qwen35_softpack4096_seed42`，seq 4096，pad 248044
- 布局 TP1/PP1/EP8，world_size 8，distributed optimizer
- gbs 32 / 3 epochs = **2142 步**，cosine→0 + 5% warmup，seed 42，bf16，
  clip 1.0，full trainable
- **save_final_checkpoint: true**（只存 final ckpt，save_steps 空）
- 在线评测 [wmdp_bio 全量, mmlu×35/leaf]，评测点
  [1, 300, 700, 1000, 1400, 1700, 1800, 1900, 2000, 2100, 2142]，
  输出 `main/qwen35/online-evaluation/`
- ckpt 落盘：`/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/wangxiao1/checkpoints/unlearn/megatron/main/`
  （与 interns2-main-* 同根，目录名 `qwen35-main-*`）

## 八个配置

| 配置 | 超参（=interns2 main） | lr | mbs | eval bs | 来源 |
|---|---|---|---|---|---|
| `rmu.yaml` | steer300 / **α63.4**（R:F=0.005 校准臂）/ target9 / l7-9 | 1e-6 | 4 | 96 | qwen35 R:F 扫描最优臂重跑（原臂未存 ckpt，在线 0.323/0.817 待复核） |
| `grad_ascent.yaml` | 无 | 1e-6 | 4 | 64 | interns2 复制 |
| `grad_diff.yaml` | γ1/α1 | 1e-6 | 1 | 16 | interns2 复制 |
| `pure_npo.yaml` | β0.1/γ1/α1 | **5e-5** | 1 | 16 | interns2 复制（lr 取 intern 有效工作点） |
| `simnpo.yaml` | β0.1/δ0/γ1/α1 | **5e-5** | 1 | 16 | interns2 复制（**从未调参**，弱遗忘优先怀疑超参） |
| `satimp_forgetonly.yaml` | β1=2/β2=0.1/α0，retain null | 1e-6 | 4 | 64 | interns2 复制 |
| `WGA_forgetonly.yaml` | β2/γ1/α0，retain null | 1e-6 | 4 | 64 | interns2 复制 |
| `ceu.yaml` | γ1/ignore1，retain null | 1e-6 | 1 | 64 | interns2 复制（**从未调参**） |

mbs/eval 档位决策（2026-09-28 与王潇确认）：**照搬 Intern-S2 实测安全档**。
GD/PureNPO/SimNPO（vocab CE 物化 ~30.7 GiB 全 logits）、CEU（full-logit 目标
函数多张全 vocab fp32 张量）必须 mbs1——qwen35 同 vocab（pad 248044）同
seq4096，物化数学一致；GD/PureNPO/SimNPO 因带冻结参考模型驻留，eval bs 降 16。
RMU 维持 qwen35 扫描已验证的 mbs4 + eval bs96，`activation_recompute` 必须
null（RMU hook 在 no-grad 前向下断边，config 校验会拒绝 full）。

⚠️ 与 intern 的已知差异点：qwen35 静态显存画像未实测（模型更大 35B 总参
但激活 3B），首跑若 step1-2 或首次 eval forward OOM：mbs 已是安全档的按
eval bs 64→16 回退；mbs4 的三组（GA/SatImp/WGA + RMU）若 OOM 降 mbs
（gbs 不变，优化数学等价）。

## 提交（三个资源组）

资源组规格见根目录 `RUNBOOK.md`（8 GPU / 96 CPU / 256000 MiB，
哪组有卡用哪组；输出路径全部本地化到共享存储，跨组补跑不混淆产物）。
每组一个独立脚本，`BATCH=g1..g4` 选批次，每组两个配置串行一个 8 卡 rjob：

| 批次 | 配置 | 档位 |
|---|---|---|
| g1 | rmu + grad_ascent | mbs4 快组 |
| g2 | grad_diff + pure_npo | mbs1 重组 |
| g3 | simnpo + ceu | mbs1 重组 |
| g4 | satimp_forgetonly + WGA_forgetonly | mbs4 快组 |

```bash
# safewt / safethm（同镜像 safemoe:jc-work-20260506163353）
BATCH=g1 bash rjob_main_safewt.sh
BATCH=g2 bash rjob_main_safethm.sh
# safer2ai（镜像 unlearning-cu126-torch271-v1，首跑注意环境差异）
BATCH=g3 bash rjob_main_safer2ai.sh
# 显式传配置或预览
bash rjob_main_safewt.sh main/qwen35/rmu.yaml main/qwen35/grad_ascent.yaml
BATCH=g1 PREVIEW=true bash rjob_main_safewt.sh   # 只打印 rjob 命令
```

三个脚本除资源组块（namespace / charged-group / 镜像）外逐字节相同，
均继承 qwen35 sweep 验证过的 pod FQDN /etc/hosts 钉扎块。训练日志落
`logs/`（main_run_train.sh 按配置名写），rjob 层日志落
`outputs/rjob_logs/`。g2/g3 时长预计 ≈ g1/g4 的 3-4 倍。

## 后续

- **PDU ε 扫描已启动**（09-28）：见 [`pdu_eps_sweep/`](pdu_eps_sweep/)——锚点
  测量 rjob（`qwen35-pdu-anchor-safewt-64511264`）跑完后
  `python3 pdu_eps_sweep/make_eps_configs.py` 生成 5 臂（anchor±偏移，
  lr 全臂 4e-6），再提 `rjob_pdu_eps_sweep_{safewt,safethm,safer2ai}.sh`。
- UNDIAL：intern 主线 β3/α0 工作点无遗忘，待调参/调试后再上。
- 8 个 final ckpt 出来后走 `lm_eval_accelerate.sh` 离线全量
  （对齐 intern 的 ckpt_eval_summary 口径），基线用 qwen35 base 全量。
