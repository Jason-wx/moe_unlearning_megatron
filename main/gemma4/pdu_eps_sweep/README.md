# Gemma4 PDU ε 扫描（retain_loss_eps 定超参）

方法论照搬 `main/qwen35/pdu_eps_sweep/`（其本身来自
`configs/training/megatron/interns2/V4/PDU/`）：**ε 是"模型 × retain 语料"
组合属性，必须锚定基线 retain NLL 重测**，不能从 intern（1.9854）或
qwen35（2.0082）搬数值。PDU：`L = L_fgt + λ·(L_rtn − ε)`，
`λ ← max(0, λ + η·(L_rtn − ε))`。

## 与 qwen35 协议的差异（2026-09-28 与王潇确认）

1. **锚点 4 卡 / 200 packs 随机抽样**（qwen35 是 2 卡 / 800 packs）：
   `--num-samples 200 --sample-mode random --seed 42`，≈0.8M valid tokens，
   SE 约 ±0.014（qwen35 800 packs 为 ±0.010）——远小于网格间距 0.03，够用。
   提交到 **safethm**（qwen 锚点用的 safewt）。
2. 其余照抄 qwen35 sweep：lr 全臂 4e-6（intern main 线 lr×ε 耦合证据：
   1e-6 下 λ 从不咬合、MMLU 侵蚀 6.8pt；4e-6+ε≈anchor+0.065 有效
   0.408/0.844）、每臂 3ep 全程在线评测、final ckpt 直接入线。

## 工作流

```bash
# 1) 锚点测量（4 卡前向，~20-40min）——已提交 09-28 21:11
#    gemma4-pdu-anchor-safethm-71799664
bash rjob_anchor_safethm.sh
#    工具预设的 gemma4 路径被覆盖：模型=sgtm gemma-26b-a4b（main 线训练
#    初始权重，重组后 gemma/ 子目录，非 HF 公开模型）、cache=softpack4096
#    retain（非 1024 doc-chunk）、seq 4096。随机抽样 200 packs（带种子）。

# 2) 生成 6 臂配置（锚点落盘后）—— 已执行 09-28
python3 make_eps_configs.py
#    ε 网格为绝对值（王潇 09-28 定）：2.0 / 2.1 / 2.2 / 2.25 / 2.5 / 3.0
#    （锚点 2.0945 → 偏移 −0.09 / +0.01 / +0.11 / +0.16 / +0.41 / +0.91）。
#    生成后跑 python3 check_arms.py 做 25 项不变量校验。

# 3) 提交 sweep：三组各 2 臂，BATCH=g1..g3，每组一个 8 卡 rjob
bash rjob_pdu_eps_sweep_safethm.sh     # g1（默认）：eps2p00 + eps2p10
bash rjob_pdu_eps_sweep_safewt.sh      # g2（默认）：eps2p20 + eps2p25
bash rjob_pdu_eps_sweep_safer2ai.sh    # g3（默认）：eps2p50 + eps3p00
#    任一脚本可 BATCH= 任意组；PREVIEW=true 只打印命令。
```

## 提交记录

**首轮（2026-09-28 22:38）全部作废**：g1 `...-14930900`（手动取消）、g2 `...-16164748`
（rmsnorm 崩溃，见下）、g3 `...-17402560`（排队中取消）。根因修复后**重提**
（2026-09-29 14:28，配置不变，代码含 int32 自动防护）：

| 批次 | ε 臂 | rjob（重提） |
|---|---|---|
| g1（紧臂对） | eps2p00 + eps2p10 | `gemma4-pdu-eps-sweep-g1-safethm-9665831` |
| g2（松臂对） | eps2p20 + eps2p25 | `gemma4-pdu-eps-sweep-g2-safewt-11284431` |
| g3（诊断臂对） | eps2p50 + eps3p00 | `gemma4-pdu-eps-sweep-g3-safer2ai-12821849` |

## ⚠️ g2 崩溃与 debug（2026-09-29）

两个臂崩法**完全一致且确定**：第 1 个 optimizer step 的第一次 backward 即死，
8 rank 同报 TE `rmsnorm_bwd_semi_cuda_kernel` illegal memory access。排除法已
锁定「gemma4 × PDU」组合（同镜像同 venv 下 gemma4 其余 8 算法含 full-logit CEU
全跑满 2133 步，峰值 ≤109GiB；qwen35 PDU 同代码在 safewt 干净跑完 99.4GiB；
崩在 `runtime.py:797` 普通 schedule 分支，非 logit-window 路径）。illegal access
是异步错误，报错 kernel 常只是第一受害者。

debug 作业 `gemma4-pdu-debug-safewt-88688731`（09-29 13:06 提交 safewt——首提
safer2ai `44949101` 排队中被删改提 safewt，仍在出事的 safemoe 镜像上复现）：
`CUDA_LAUNCH_BLOCKING=1` 下串行跑两个 2 步配置（`debug/` 目录，不存 ckpt）：
1. `pdu_debug_blocking.yaml`——原样复现（recompute full），定位真正出错的 kernel
2. `pdu_debug_norecompute.yaml`——`activation_recompute: null` 修复候选
   （RMU 在 gemma4 上的先例：hook/loss 图与 full recompute 冲突必须关）

每组 2 臂 × 2133 步（mbs1 重配置），时长预估参考 qwen35 sweep 单臂时长 × 2。

### debug 轮次记录（全部 safewt 8 卡、CUDA_LAUNCH_BLOCKING=1、2 步不存 ckpt）

| 轮次 | rjob | 内容 | 结果 |
|---|---|---|---|
| 1 | `gemma4-pdu-debug-safewt-88688731` | 原样复现 + `activation_recompute: null` | 两臂同崩 rmsnorm_bwd → recompute 无关 |
| 2 | `gemma4-pdu-debug2-safewt-1152125` | 原样 + `OU_PDU_CONTIG_LOSS=1`（loss 张量 contiguous 化，objectives.py 环境门控） | 两臂同崩 → 张量布局无关 |
| 3 | `gemma4-pdu-debug3-safewt-52326596` | `OU_PDU_BISECT=forget`（仅 dispersion 项）/ `=retain`（仅 λ·violation 项），forward_step.py 环境门控 | **两臂同崩** → loss 组成无关 |
| 4 | `gemma4-pdu-debug4-safewt-53079624` | `OU_PDU_NO_DUAL_UPDATE=1`（update() 置 no-op，λ 钉在 α，完整两项 loss 不动） | **同崩** → PDUState.update 的 forward 中途 all_reduce+.item() 同步无关 |
| 5 | `gemma4-pdu-debug5-safewt-18801241` | 分支级 bisect（彻底跳过另一分支计算；R3 只是改 loss 加权、两分支前向都执行过） | **两臂同崩** → 不是 retain/forget 分支的计算本身 |
| 6b | `gemma4-pdu-debug6b-safewt-16963986` | `OU_PDU_SPLIT_FORWARD=1` 按 role 拆两次前向 | ✅ **跑通**：2 步完成、0 处 rmsnorm、eval 0.7910/0.7644 与基线一致、λ 正常更新 |
| 7 | `gemma4-pdu-debug7-safewt-41553949` | 无环境变量，测自动防护（模型属性内省） | ❌ **内省失败**：日志打出 `vocab size not resolvable` → 防护没触发 → 同崩。解释了 g2 重提首跑失败（无环境变量、依赖内省） |
| 8 | `gemma4-pdu-debug8-safewt-78357523` | 无环境变量，测新的 config 源防护（runtime.py 读 config.json vocab + batch 参数算元素数） | ✅ **跑通**：guard 日志 `elements=2^31 → split=True`，2 步成功、0 rmsnorm、eval 0.7863/0.7699、进度条 loss 正常显示 |

修复至此在三条独立路径 GPU 验证：环境变量强制（6b）、config 判定（8）、真实 sweep
全程（g2 重提 `-45867279` step28+ 持续健康）。

### Round 7 教训与最终防护形态

6b 的修复只对「环境变量强制」做了 GPU 验证；sweep 重提首跑（`-11284431`）没带环境
变量、依赖**模型属性内省**（DDP→bridge→`language_model.output_layer`），内省在真实
包装链上解析不到 vocab → 防护未触发 → 复崩。修复三件套：

1. **config 源防护**（权威路径）：`runtime.py::_pdu_split_forward_from_config` 直接读
   模型目录 `config.json` 的 vocab（支持 gemma4 的 `text_config` 嵌套），用
   packs（mbs×2 paired）× seq × vocab（÷CP）判定，结果经 `force_split` 显式传入
   forward_step。离线已验：gemma4 = 2³¹ → True；qwen35 = 2.03×10⁹ → False。
2. **runner 强制**：`pdu_run_train.sh` 无条件 `export OU_PDU_SPLIT_FORWARD=1`（本
   sweep 的兜底，不依赖任何判定）。
3. **诊断日志**：`[pdu-split-guard]` rank0 打印判定依据（vocab/元素数/决策），任何
   路径失败一眼可见。

优先级：环境变量 > config 判定 > 属性内省（仅诊断）。

### ✅ 根因（round 6 确认）：full-logits 张量 int32 溢出

PDU 的 mixed microbatch = 1 forget + 1 retain pack，gemma4 full-logits 输出 =
**2 × 4096 × 262144 = 2³¹ 个元素 = 恰好 INT32_MAX + 1**（词表 262144 = 2¹⁸ 是 2 的幂，
完美风暴）。任何按 int32 索引的 kernel 在该张量上越界 → 破坏显存 → rmsnorm_bwd
第一个踩雷报 illegal access。全部观测对账：崩的运行都持有 ≥2³¹ 元素 logit 张量；
CEU（forget-only，2³⁰）/ GD（labels≠None，mcore 融合 CE 不物化全量 logits）/
RMU（中间层目标）/ qwen35 PDU（vocab 15.2 万 → 1.25×10⁹）/ intern PDU（vocab ~15 万）
全部在 2³¹ 以下、全部干净。也解释了 R1-R5 全阴：不管怎么改 loss 布局/组成/更新，
模型对混 batch 前向必然物化这个张量。

### 修复（已实现，自动防护）

`forward_step.py`：`_pdu_needs_split_forward` 自动检测（行×seq×vocab > INT32_MAX
即走 split），PDU 按 role 拆两次前向（各 2³⁰，CEU 已验证安全尺寸），loss 合并不变；
低于阈值的老路径一字不变（intern/qwen35 不受影响）。`OU_PDU_SPLIT_FORWARD=1`
可强制开。单测 `tests/megatron/test_pdu_split_forward.py`（5 项，全套 125 过）。
副作用对比：峰值 83.8GiB / 34.8s·步（含 eval），split 后显存反而更低。

固定旋钮（V4/qwen35 验证）：γ=1.0、α(λ₀)=1.0、η=0.1、lr 4e-6；main 线配方
gbs 32 / mbs 1（full-logit OOM）/ TP1·PP1·EP8 / cosine+5% warmup / seed 42 /
只存 final ckpt / **2133 步**（22757 packs，与 intern/qwen 的 2142 不同）。
在线评测 [wmdp_bio 全量, mmlu×35/leaf] 落本目录 `online-evaluation/`，ckpt 落
`.../megatron/gemma4/pdu_eps_sweep/`。

## 监控检查清单（V4 教训，比结果更能预警）

- `retain_violation`：应在 0 附近**双向震荡**；恒 >0 = λ 发散前兆（V3 死法）
- `retain_preference`（λ）：应有界；单调直线一眼病态
- `forget_loss`（dispersion）：贴地板 = forget 侧没活干（V4 死因二）
- MMLU 下坠发生在跳崖段：先看 300→1000 步的 bio/MMLU 联动形态
- ⚠️ gemma4 A4B 显存画像未实测：首次 eval forward 若 OOM，eval bs 64→16

## 判读框架（来自 V4 RESULTS + qwen35 遗忘墙结论）

- ε 只改下坠速度，所有臂终将坠入同一深盆；选臂标准 = **MMLU 满血前提下
  bio 坠得最深/可控**的 ε
- 若 +0.5 诊断端点也不动 bio → flattening 信号在 gemma4 语料上结构性无效，
  第二轮升 γ 或 lr，或判 PDU 不适合此设定
- **额外关注**：gemma4 是"遗忘墙"结论的第三个模型测试点——intern 有效 /
  qwen35 升向族全灭。PDU 的 flattening 机制不属于升向族；gemma4 上它若有效
  而 GD/SatImp/WGA 无效，则"墙只挡升向"的外推得到第三点确认
- 参照：intern eps1p98@1ep 61.2/84.3 ⭐、main 线 lr4e-6+eps2.05@3ep
  0.408/0.844；qwen35 锚点 2.0082、sweep 在跑
