# Qwen3.5 PDU ε 扫描（retain_loss_eps 定超参）

方法论照搬 `configs/training/megatron/interns2/V4/PDU/`（interns2 × wmdp-bio，
锚点 1.9854，ε 只控制下坠速度、MMLU 伤损只发生在跳崖段、近锚 ε 保护能力侧
最优）。PDU：`L = L_fgt + λ·(L_rtn − ε)`，`λ ← max(0, λ + η·(L_rtn − ε))`；
**ε 是"模型 × retain 语料"组合属性，必须锚定基线 retain NLL 重测**。

## 与 V4 的两处刻意差异（2026-09-28 与王潇确认）

1. **lr 全臂 4e-6**（V4 扫描是 1e-6 但那是 gbs4/1ep=5716 步慢动力学）：
   intern main 线证据——gbs32/3ep 规模下 lr4e-6+ε≈anchor+0.065 有效
   （0.408/0.844），lr1e-6 下 λ 从不咬合、MMLU 侵蚀 6.8pt。锚定的就是
   lr4e-6 耦合工作区。
2. **每臂直接跑满 3ep=2142 步 + 在线评测全曲线**（V4 是 1ep 存 ckpt 再离线
   评测两段式）：曲线即结果，最优臂 final ckpt 直接可入 main 线。

## 工作流

```bash
# 1) 锚点测量（2 卡前向，~30-60min）——已提交 09-28 11:12
#    qwen35-pdu-anchor-safewt-64511264
bash rjob_anchor_safewt.sh
#    工具预设的 qwen35 路径被覆盖：模型=sgtm qwen3p5-35b-a3b（main 线训练
#    初始权重，非 HF 公开 base）、cache=softpack4096 retain（非 1024
#    doc-chunk）、seq 4096。800 packs ≈ 3.1M token（对齐 intern 测量口径）。

# 2) 生成 6 臂配置（锚点落盘后）
python3 make_eps_configs.py
#    锚点实测 2.0082（800 packs/314.7 万 token，σ0.140/SE0.010，与 intern
#    1.9854 仅差 +0.023）。标准 5 臂 ε = anchor + {−0.01, +0.03, +0.08,
#    +0.15, +0.5}（V4 相对偏移语义），即 2.00/2.04/2.09/2.16/2.51；
#    另加 1.98 绝对值臂（anchor−0.028，唯一初始 violation≈+0.03 的下锚
#    探针 + intern V4 最优臂绝对值对齐，--extra-eps 1.98 生成）。

# 3) 提交 sweep —— 2026-09-29 起按臂拆分并行（见下方重启记录）
ARM=pdu_eps2p09 bash rjob_pdu_eps_arm_safewt.sh   # 每臂一个 8 卡 job
ARM=pdu_eps2p16 bash rjob_pdu_eps_arm_safewt.sh
ARM=pdu_eps2p51 bash rjob_pdu_eps_arm_safewt.sh
#    （旧的整包串行脚本 rjob_pdu_eps_sweep_*.sh 会 glob 全部 6 个 yaml
#    重跑已完成臂，仅作模板参考，勿直接提交。）
```

## 2026-09-29 重启记录（mbs 教训，三次实测）

- **mbs1 串行**（job 61155362，09-28 提交）：跑完 eps1p98/2p00/2p04 三臂，
  14:16 在 eps2p09 step 202/2142 被停（太慢：12.6 s/step ≈ 7.5h/臂，剩
  3 臂串行还要 ~22h）。半程产物归档为 `*.stopped61155362.{log,jsonl}`。
- **mbs4**（job 23968175）：3/3 臂在**首个训练步**即 CUDA OOM，崩点在
  FLA 线性注意力 backward 的 Triton kernel autotune（`chunk_bwd_dqkwg`），
  与 grad_diff/ceu 的历史 mbs4 OOM 记录一致；且 mbs4 的 split-forward
  每 role 前向 4×4096×248320=4.07e9 > INT32_MAX，gemma4 同款雷。
- **mbs2**（job 71487638）：能跑（114.1GiB，split 路径每 role 2 行=2.03e9
  压线安全），但只快 ~3%（12.27 vs 12.6 s/step）——int32 上限把每次前向
  钉死在 2 行，launch 数省不了；且 λ 每 step 更新 4→2 次（step3 即见
  λ=0.946 vs mbs1 的 0.882），后 3 臂将与前 3 臂不可比。已停，产物归档
  `*.mbs2_71487638.{log,jsonl}`。
- **结论：mbs 是死路（int32 帽 + FLA 显存 + λ 动力学三重锁），提速只能靠
  臂间并行**。现 3 臂已按 `rjob_pdu_eps_arm_safewt.sh` 拆 3 个并行 mbs1
  job：qwen35-pdu-2p09-safewt-76302187 / -2p16-82533848 / -2p51-88665731，
  均 Running，12.5-12.8 s/step / 99.4GiB，预计 09-29 深夜全部跑完。

固定旋钮（V4 验证）：γ=1.0、α(λ₀)=1.0、η=0.1；main 线配方 gbs32 / mbs 1
（full-logit OOM）/ TP1·PP1·EP8 / cosine+5% warmup / seed 42 / 只存 final
ckpt。在线评测 [wmdp_bio 全量, mmlu×35/leaf] 落本目录
`online-evaluation/`，ckpt 落 `.../megatron/qwen35/pdu_eps_sweep/`。

## 监控检查清单（V4 教训，比结果更能预警）

- `retain_violation`：应在 0 附近**双向震荡**；恒 >0 = λ 发散前兆（V3 死法）
- `retain_preference`（λ）：应有界；单调直线一眼病态
- `forget_loss`（dispersion）：贴地板 = forget 侧没活干（V4 死因二）
- MMLU 下坠发生在跳崖段：在线曲线上先看 300→1000 步的 bio/MMLU 联动形态

## 判读框架（来自 V4 RESULTS）

- ε 只改下坠速度，所有臂终将坠入同一深盆；选臂标准 = **MMLU 满血前提下
  bio 坠得最深/可控**的 ε（intern 上是 ε≈锚点，1ep 未坠到底时最值得延长）
- 若 +0.5 诊断端点也不动 bio → flattening 信号在 qwen35 语料上结构性无效，
  第二轮升 γ 或 lr，或判 PDU 不适合此设定
- intern 参照：eps1p98@1ep 61.2/84.3 ⭐；main 线 lr4e-6+eps2.05@3ep 0.408/0.844
