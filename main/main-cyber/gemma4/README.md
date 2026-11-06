# Gemma4 main-cyber（wmdp-cyber 九算法迁移 · G1 纯曲线轮）

Qwen3.5 main-cyber 线（同目录 `../qwen35/`）向 **gemma-26b-a4b-nocap** 的批量迁移，
2026-10-03 由 `make_configs.py` 从 qwen35-cyber R2 配置确定性生成：九个算法（PDU 拆
六臂）逐字段复制 qwen35-cyber 的算法语义与训练计划，只替换模型/数据/输出/在线测评
四类字段，共 **14 份配置，全部在 [`configs-r1/`](configs-r1/)**（本轮 = R1 首轮；
后续存点轮进 configs-r2/ 等）。**R1 = 纯曲线跑：全部配置 `save_steps: []` +
`save_final_checkpoint: false`，不存任何 ckpt**（qwen35 R1→R2 协议：曲线出来后再按
margin 最优点定点存盘）。

## 为什么是 nocap 模型 + BOS 缓存

GA 族在 gemma4 上的零迁移已定案为 **BOS 门控**（10-03 证实）：GD-BOS 消融臂在
lr 1e-6 下 bio 0.79→0.578（step 300，MMLU 仅 −1.2pt），而两个无 BOS 版全程钉死
0.78-0.80；见 `main/gemma4/GA_family_nocap_rootcause_20261002.md`。本线全套采用
`bos_per_pack: true` 缓存 + nocap 权重——即消融验证过迁移成立的组合，首次搬到
cyber 语料。

## 迁移口径

**原样复制**（qwen35-cyber = 10 epochs / 230 steps 全调度 + 每 epoch 在线测评）：

- gemma4 cyber BOS forget cache = **748 packs**（seed 43，3.06M tokens；intern/qwen
  为 743），drop_last 下 floor(748/32) = 23 步/epoch 与 intern/qwen 完全一致，
  230 步总调度与评测网格 [23..230] 1:1 沿用
- lr：七份 5e-6（rmu/pdu/gd/ga/satimp/WGA/ceu）+ NPO 族两份 5e-5（pnpo/simnpo），
  min_lr 全部 1e-6、gbs 32、cosine + 5% warmup、seed 42
- mbs 按算法保留：rmu/ga/satimp/WGA = 4，pdu/gd/pnpo/simnpo/ceu = 1
  （与 gemma4 bio 主线档位一致）；RMU `activation_recompute: null`，其余 full
- 在线测评 eval bs 保留 qwen35-cyber 档：32（gd/pnpo/simnpo 三个全 logits 档 16）；
  gemma4 A4B 在 cyber 长题干上的静态画像未实测，首次 eval OOM 则减半
- WMDP-Cyber 全量（1987 题）+ MMLU 35/leaf，输出 `online-evaluation/`

**必须改的模型相关字段**：

| 字段 | qwen35-cyber | gemma4-cyber | 依据 |
| --- | --- | --- | --- |
| model path/name | qwen3p5-35b-a3b | **gemma-26b-a4b-nocap**（王潇目录） | 用户 10-03 指定；nocap+BOS = 消融验证组合 |
| num_experts / pad | 256 / 248044 | **128 / 0** | gemma4 档案（bio 主线/nocap 消融一致） |
| forget cache | qwen35 cyber（无 BOS） | **gemma4_wmdp_cyber_softpack4096_bos**/forget | BOS 门控修复 |
| retain cache | qwen35 fineweb | **fineweb_edu_5b_gemma4_softpack4096_bos_seed42** | 同上（491 万 packs） |
| RMU alpha/steer | 63.4 / 300 | **4.45 / 1200** | 算法段与 coeff-alpha sweep 最优臂**逐字节一致**（生成器断言强制） |
| PDU eps | 2.51（锚+0.50） | **六臂绝对网格 2.0/2.1/2.2/2.4/2.6/3.0** | 2026-10-03 王潇定（沿 softcap bio 网格，2.25→2.4、2.5→2.6）；见下 |
| 输出/命名 | qwen35-main-cyber-* | gemma4-main-cyber-* | 命名约定 |

## 逐算法参数表（R1 · 14 份配置，文件名相对 `configs-r1/`）

| 配置 | lr | mbs | eval bs | 备注 |
| --- | --- | --- | --- | --- |
| rmu.yaml | 5e-6 | 4 | 32 | 算法段=sweep 最优臂（s1200/α4.45/l7-9/target9），逐字节断言 |
| pdu_eps2p00.yaml | 5e-6 | 1 | 32 | ε=2.0（名义 −0.09 vs 作废锚 2.0945） |
| pdu_eps2p10.yaml | 5e-6 | 1 | 32 | ε=2.1（名义 +0.01） |
| pdu_eps2p20.yaml | 5e-6 | 1 | 32 | ε=2.2（名义 +0.11） |
| pdu_eps2p40.yaml | 5e-6 | 1 | 32 | ε=2.4（名义 +0.31） |
| pdu_eps2p60.yaml | 5e-6 | 1 | 32 | ε=2.6（名义 +0.51） |
| pdu_eps3p00.yaml | 5e-6 | 1 | 32 | ε=3.0（名义 +0.91） |
| grad_diff.yaml | 5e-6 | 1 | 16 | BOS 消融正对照（bio 0.79→0.578） |
| grad_ascent.yaml | 5e-6 | 4 | 32 | 预计族内崩得最快 |
| pure_npo.yaml | 5e-5 | 1 | 16 | 唯一在无 BOS 下也迁移的配方（min 0.521） |
| simnpo.yaml | 5e-5 | 1 | 16 | 1e-6 臂在 gemma4 冻结，仅 5e-5 动过 |
| satimp_forgetonly.yaml | 5e-6 | 4 | 32 | forget-only；无 BOS 下权重饱和 ~1e-17 |
| WGA_forgetonly.yaml | 5e-6 | 4 | 32 | forget-only |
| ceu.yaml | 5e-6 | 1 | 32 | forget-only；fp32 归一化 bug 未修 |

PDU 六臂名义偏移仅作网格间距参照（对作废的 softcap×无BOS 锚 2.0945）。

## ⚠️ 提交前必读

1. **PDU 锚点未测（非阻塞，但强烈建议并行补测）**：六臂绝对网格 2.0-3.0 已定
   （10-03），但 nocap×BOS 的 retain NLL 锚点没人测过——历史锚点全部失效（2.0945
   = softcap×无BOS、4.91 = nocap×无BOS；softcap bio eps 扫描本身跑在无 BOS 上、
   六臂全线 ~0.79 钉死已作废，根因报告 §3.5）。曲线判读靠 臂−锚 偏移：建议随 G1
   并行用 `tools.measure_retain_nll` 测锚（rjob 模板
   `main/gemma4/pdu_eps_sweep/rjob_anchor_safethm.sh`，4 卡 ~20-40min）；若锚点
   明显高于 3.0，整个网格落在紧侧（L_rtn−ε 恒正、λ 一路涨），六臂会集体偏向
   retain 保护，届时二轮以锚点重新对中。mbs 1 强制（paired full-logits 2^31
   int32 帽；split-forward 防护经 config.json 自动生效，已 GPU 验证）。
2. **RMU 校准口径**：α4.45/steer1200 是 softcap 模型 × 无 BOS bio 缓存上调出的；
   nocap 会平移 layer-9 激活尺度。若 step-1 forget loss 量级异常，先查 R:F 再动
   其他旋钮。
3. **lr 5e-6 是 cyber 线策略，比 gemma4 bio 主线（1e-6）高 5×**：消融臂 GD 在
   1e-6 已跑出 0.578，5e-6 下 GA 族前三个评测点（23/46/69）可能直接崩穿——这是
   特性不是 bug（曲线轮就是要看轨迹），但读数时注意。
4. **CEU fp32 归一化 bug 未修**（objectives.py:278，早期梯度最高放大 ~6e3×），
   三模型同污；曲线按"CEU + 随机暖启动"口径解读。
5. **首跑盯 OOM**：eval bs 32→16（trio 16→8）；mbs4 四组可降 mbs 不改 gbs。

## 提交

## 提交（2026-10-03 01:00 已全部提交）

四组跨两资源组（g1/g2 → safethm，g3/g4 → safer2ai），每组一个 8 卡 rjob 串行跑组内
配置（长杆在前）。分组按 gemma4 bio/nocap 线实测折算 230 步配平（PDU ~1h/臂、
mbs1 ~0.6-0.8h、mbs4 ~0.3-0.4h，cyber 题量 ×1.6）：

| BATCH | 组成（串行） | 预估 | rjob |
| --- | --- | --- | --- |
| g1 safethm | pdu2p00 → pdu2p10 → rmu → ga | ~3.3h | `gemma4-cyber-r1-g1-safethm-45736659` |
| g2 safethm | pdu2p20 → pdu2p40 → satimp → WGA | ~3.4h | `gemma4-cyber-r1-g2-safethm-46937662` |
| g3 safer2ai | pdu2p60 → pdu3p00 | ~2.6h | `gemma4-cyber-r1-g3-safer2ai-54662854` |
| g4 safer2ai | pnpo → gd → simnpo → ceu | ~2.8h | `gemma4-cyber-r1-g4-safer2ai-55876243` |

```bash
BATCH=g1 bash rjob_r1_safethm.sh        # safethm（g1|g2）
BATCH=g3 bash rjob_r1_safer2ai.sh       # safer2ai（g3|g4）
BATCH=g1 PREVIEW=true bash rjob_r1_safethm.sh   # 只打印 rjob 命令
```

## R1 结果（2026-10-03 13:06 全部落盘：4/4 job Succeeded，14/14 run × 10 评测点）

曲线 jsonl 已归档至 [`online-evaluation/r1/`](online-evaluation/r1/)（10-03 R2
跑动中整理；**当前轮（r2）的 jsonl 留在 `online-evaluation/` 顶层**，每轮完结后
归入对应 `r<N>/` 子目录；`make_configs_r2.py` 的 EVAL_DIR 已同步指向 r1/）。

margin = mmlu − wmdp_cyber，argmax（并列取最早步）；base 参照 ≈ 0.50/0.762
（各 run step-23 前的未测，GD/RMU/CEU 的 23 步读数最接近 base）。

| run | 最优点 step (epoch) | cyber/mmlu | margin | 终点 | 判读 |
| --- | --- | --- | --- | --- | --- |
| **pdu-eps2p60** | **207 (ep9)** | **0.249/0.759** | **+0.511** | 0.250/0.755 | ⭐ 最优；曲线未饱和 |
| **pdu-eps3p00** | **230 (ep10)** | **0.243/0.752** | **+0.509** | 0.243/0.752 | 最深，仍未饱和 |
| pdu-eps2p20 | 184 (ep8) | 0.261/0.765 | +0.504 | 0.260/0.763 | mmlu 最高 |
| pdu-eps2p40 | 230 (ep10) | 0.265/0.761 | +0.496 | 0.265/0.761 | |
| pdu-eps2p10 | 230 (ep10) | 0.268/0.762 | +0.494 | 0.268/0.762 | |
| rmu | 92 (ep4) | 0.265/0.743 | +0.478 | 0.271/0.740 | softcap 校准臂直接可用 |
| pdu-eps2p00 | 230 (ep10) | 0.285/0.762 | +0.478 | 0.285/0.762 | |
| grad-diff | 161 (ep7) | 0.259/0.721 | +0.462 | 0.262/0.711 | **BOS 修复正对照成立** |
| pure-npo | 69 (ep3) | 0.304/0.732 | +0.428 | 0.373/0.731 | 23 步过冲后回升，判无效 |
| simnpo | 23 (ep1) | 0.302/0.727 | +0.424 | 0.396/0.742 | 同上 |
| ceu | 230 (ep10) | 0.362/0.725 | +0.363 | 0.362/0.725 | 慢但在动（带 fp32 bug） |
| grad-ascent | 23 (ep1) | 0.438/0.734 | +0.295 | 0.266/0.239 | MMLU 全崩 |
| satimp | 207 (ep9) | 0.257/0.383 | +0.127 | 0.275/0.389 | MMLU 崩（step23 已 0.32） |
| wga | 69 (ep3) | 0.236/0.342 | +0.107 | 0.246/0.343 | MMLU 崩 |

R1 要点：

1. **BOS 门控修复在 cyber 上成立**：GD 0.499→0.259（mmlu 0.721 保持）——logit 级
   目标在 gemma4 上的首次真迁移（无 BOS 时代 bio/cyber 全钉死 0.79/0.50 一带），
   rootcause 结论外推到 cyber 语料确认。
2. **PDU 六臂全活且 eps 单调分化**（松→深：0.285→0.243），mmlu 全程保 0.75-0.76。
   无 λ 爆炸/retail 钉死迹象 → nocap×BOS 的 retain 锚点明显低于 2.0（无 BOS 时
   4.91 是假象），2.0-3.0 网格对中良好。最优两臂（2.60/3.00）best 落在 207/230
   步仍未饱和——**R2 若只存点会吃亏，建议 PDU 臂加长（如 20ep）再存**。
3. **GA/SatImp/WGA 在 lr 5e-6 下 MMLU 崩**（0.24-0.39；bio 线 1e-6 时它们只是
   冻结不迁移）：gemma4 上 mbs4 forget-only 族对 5×lr 过猛，且与 qwen-cyber
   （三者 0.72-0.74 健康）不同。考虑到两外线它们 margin 本就不如 PDU/RMU，
   R2 不建议救。
4. NPO 族（lr 5e-5）23 步内过冲到 ~0.30 后回升至 0.37-0.40，best=首点即无效。
5. 时长复盘：单 run ~1.5-3h（eval 占大头，cyber 1987 题），组跨度 4-7.8h，
   比分组预估慢 2-3×；R2 规划按实测。

R2 存点候选（若不加长）：pdu-2p60@207、pdu-3p00@230、pdu-2p20@184、rmu@92、
gd@161、（可选 ceu@230）。

## R2 存点轮配置（2026-10-03 已生成，未提交）

`make_configs_r2.py` 从 configs-r1 + R1 jsonl 确定性生成 **9 份配置**（全部在
[`configs-r2/`](configs-r2/)）：PDU 只保留 eps=2.60 单臂（文件名收敛为 `pdu.yaml`），
九算法完整重跑 10 epochs / 230 步，每份**只在各自 R1 margin 最优步存一个 ckpt**
（`save_steps: [N]`，`save_final_checkpoint: false`，qwen35 R1→R2 协议）。存点步由
脚本从 jsonl 复算并与上表交叉断言（两处必须一致）。**决策记录：PDU 未加长**——
最优臂 2.60 曲线未饱和的既知风险接受（若该 ckpt 离线测评不佳，后手是加长调度轮
而非扩网格）；GA/SatImp/WGA 崩塌臂按"早停抢救位"存点（补全用）；NPO 族存点=
过冲峰，判读需谨慎。

run 命名 `gemma4-main-cyber-r2-<tag>`（R1 记录零冲突：jsonl 新文件名 + ckpt 新
run 目录）；其余字段与 R1 逐语义一致（校验：revert 改名+save_steps 后与 R1 源
deep-equal，9/9 通过）。

| 配置（configs-r2/） | R1 源 | 存点 step (epoch) | R1 读数 |
| --- | --- | --- | --- |
| rmu.yaml | rmu | 92 (ep4) | 0.265/0.743 +0.478 |
| pdu.yaml | pdu_eps2p60 | 207 (ep9) | 0.249/0.759 +0.511 ⭐ |
| grad_diff.yaml | grad_diff | 161 (ep7) | 0.259/0.721 +0.462 |
| grad_ascent.yaml | grad_ascent | 23 (ep1) | 0.438/0.734（抢救位） |
| pure_npo.yaml | pure_npo | 69 (ep3) | 0.304/0.732（过冲峰） |
| simnpo.yaml | simnpo | 23 (ep1) | 0.302/0.727（过冲峰） |
| satimp_forgetonly.yaml | satimp_forgetonly | 207 (ep9) | 0.257/0.383（崩塌带） |
| WGA_forgetonly.yaml | WGA_forgetonly | 69 (ep3) | 0.236/0.342（崩塌带） |
| ceu.yaml | ceu | 230 (ep10) | 0.362/0.725 +0.363 |

### R2 提交（2026-10-03 13:57 已全部提交，全 safewt）

五组五个 8 卡 rjob（PDU 单独一组，其余八算法按 R1 实测时长两两配对，
组内长杆在前；单 run 实测：pdu ~120m、ceu ~79m、pnpo ~72m、gd ~68m、
simnpo ~67m、ga ~58m、wga ~56m、satimp ~55m、rmu ~50m）：

| BATCH | 组成（串行） | 预估 | rjob |
| --- | --- | --- | --- |
| g1 | pdu | ~2.0h | `gemma4-cyber-r2-g1-safewt-56065385` |
| g2 | ceu → rmu | ~2.2h | `gemma4-cyber-r2-g2-safewt-57257831` |
| g3 | pnpo → satimp | ~2.1h | `gemma4-cyber-r2-g3-safewt-58440913` |
| g4 | gd → WGA | ~2.1h | `gemma4-cyber-r2-g4-safewt-59634655` |
| g5 | simnpo → ga | ~2.1h | `gemma4-cyber-r2-g5-safewt-61132058` |

```bash
BATCH=g1 bash rjob_r2_safewt.sh        # g1..g5（safewt，参数照抄 bos-ablation 模板）
BATCH=g1 PREVIEW=true bash rjob_r2_safewt.sh
```
