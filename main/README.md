# main：最终批量实验

本目录保存最终批量实验的运行记录：每条实验线的配置、提交脚本、说明文档以及在线 / 离线
评测结果（训练日志在 `logs/` 下，不提交到 Git）。三个模型 × 两个数据集（WMDP-Bio、
WMDP-Cyber）的最终训练配置已整理到
[configs/training/megatron/final-training-configs/](../configs/training/megatron/final-training-configs/)。

| 目录 | 内容 |
| --- | --- |
| [interns2/](interns2/README.md)、[qwen35/](qwen35/README.md)、[gemma4/](gemma4/README.md) | WMDP-Bio 实验线 |
| [main-cyber/](main-cyber/online_eval_summary_20261001.md) | WMDP-Cyber 实验线（[interns2](main-cyber/interns2/README.md)、[qwen35](main-cyber/qwen35/README.md)、[gemma4](main-cyber/gemma4/README.md)） |
| [Appendix/](Appendix/) | 基线细节附录 |
| `temp_config/` | 临时配置 |

以下为各 bio 实验线的建线记录，内容反映提交当时的状态。

## main 线（Intern-S2 十算法规模复跑）

当前正式规模实验的配置入口在
[`interns2/`](interns2/)：10 个算法统一
gbs 32 / mbs 4 / 3 epochs = 2142 步、在线评测 + 自动保存 final ckpt
（例外：PDU / CEU / UNDIAL 三个 full-logit 算法为 mbs 1，在 mbs 4 下 OOM），
详见该目录的 `README.md`。

**rjob 提交会在三个资源组之间切换**（资源组见 [RUNBOOK](../RUNBOOK.md)）（哪个组有卡就用哪个组）。
所有输出路径都已本地化到共享存储、与提交组无关：

- checkpoint → `/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/wangxiao1/checkpoints/unlearn/megatron/main`
- 在线评测 JSONL → `interns2/online-evaluation/`

因此跨组续跑/补跑不会混淆产物；唯一要注意的是不同组镜像不同，
若在某组首次运行遇到环境问题，先对比上表的镜像差异。

### ⚠️ 未调参算法提醒：SimNPO / CEU / UNDIAL

这三个算法在本项目没有做过针对性调参，main 配置的超参是"直搬历史唯一
配置"或"从 PureNPO 镜像"，不是验证过的工作点：

- `simnpo.yaml`：β0.1/α1/lr5e-5 镜像自 pure_npo；SimNPO 历史只在 lr5e-6
  量级扫过 β(0.5–7.5)/α(1/3/5)+官方对齐轮，全部零遗忘，β0.1 从未测过。
- `ceu.yaml`：gamma1.0/ignore1 直搬 `V4/ceu.yaml`（该配置 V4 从未运行）；
  仅 V1/V3 各一轮默认超参，V1 判"几乎无有效遗忘"。
- `undial.yaml`：β3/α0 直搬 V3 根配置；仅 V1 一轮默认超参，β 从未扫过。

**解读规则**：这三个若在 main 线遗忘弱或崩坏，优先怀疑超参不适配
（β 量级、lr、forget-only vs 带 retain），不要直接下"算法无效"的结论；
若有效则说明还有调参空间。详细出处表见
[`main/interns2/README.md`](interns2/README.md)。

## main 线（Qwen3.5 wmdp-bio 八配置批量）

Qwen3.5 的 main 线入口在 [`qwen35/`](qwen35/)：
七个算法（GA / GD / PureNPO / SimNPO / SatImp / WGA / CEU）训练配置逐字段
复制 Intern-S2 main 线（超参与训练计划不变：gbs 32 / 3 ep = 2142 步 /
lr 1e-6，PureNPO 与 SimNPO 为 5e-5 / 只存 final ckpt / 在线评测同口径），
另含 RMU 调优臂（steer300/α63.4，R:F 扫描最优）重跑以补存 ckpt。
mbs/eval 档位照搬 Intern-S2 实测安全档（GD/PureNPO/SimNPO/CEU mbs 1；
带冻结参考的 eval bs 16）。**PDU（待 retain_loss_eps 初扫）与 UNDIAL
（待调试）本轮不跑。**

rjob 提交按上表三个资源组各有一个独立脚本
（`qwen35/rjob_main_{safewt,safethm,safer2ai}.sh`，`BATCH=g1..g4`
选批次），详见该目录 `README.md`。

## main 线（Gemma4 wmdp-bio 十二配置批量）

Gemma4（gemma-26b-a4b，128 experts / 2133 步）的 main 线入口在
[`gemma4/`](gemma4/)：七算法复制同上 + RMU 最优臂
（steer1200/α4.45，coeff-alpha sweep）重跑存 ckpt + PureNPO/SimNPO 各两条
lr 臂（1e-5/1e-6）。已于 09-28 提交：g1..g3（GA+SatImp / GD+CEU / WGA+RMU）
→ safewt，g4/g5（PureNPO×3 / SimNPO×3 lr 臂）→ safer2ai。
**PDU（待锚点+ε 扫描）与 UNDIAL（待调试）本轮不跑。**
`gemma4/rjob_main_{safewt,safethm,safer2ai}.sh`，`BATCH=g1..g5`，
详见该目录 `README.md`。