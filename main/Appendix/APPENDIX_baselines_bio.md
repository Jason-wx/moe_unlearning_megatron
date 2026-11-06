# 附录草稿：Baseline 训练细节（WMDP-Bio）

依据：`main/train_configs/{gemma,intern,qwen}/bio/*.yaml`（3 个模型 × 9 个 baseline；TRACE 不在此列）、
各权重目录下的 `ckpt_notes.md`，以及 `src/megatron_unlearning/` 中的代码。
各方法的目标函数与 cyber 版（`APPENDIX_baselines_cyber.md` 第 C 节）完全相同，
本文只列出与 cyber 不同的设置，并为保持独立可读而重复必要的内容。

> 符号约定：$\mathcal{D}_f$ 为 forget 集，$\mathcal{D}_r$ 为 retain 集，$\pi_\theta$ 为当前模型，
> $\pi_{\mathrm{ref}}$ 为冻结的初始模型；$\ell_t(\theta)=-\log \pi_\theta(x_t\mid x_{<t})$，$p_t=\exp(-\ell_t)$。

## A. Baseline 实现与训练设置

**A.1 总体设置。** 我们在 Gemma4-26B-A4B、Intern-S2-35B 与 Qwen3.5-35B-A3B 三个 MoE 模型上，针对 WMDP-Bio
实现并训练了九个遗忘 baseline：GradAscent (GA)、GradDiff (GD)、CEU、WGA、SatImp、PureNPO、SimNPO、PDU 与 RMU。
所有 baseline 共用同一套训练框架（Megatron-Core，8 卡，专家并行度 8，张量/流水线/上下文并行度均为 1）、
同一套数据流水线与同一套评测协议，使方法间的差异仅来自遗忘目标本身。

**A.2 数据。** Forget 数据为 WMDP-Bio forget 语料，用各模型自带的 tokenizer 分词，软打包为长度 4096 的序列
（Gemma 在序列开头额外追加 BOS，并使用去除 logit 截断的 `nocap` 版本模型）。按 global batch 32 计，
Intern/Qwen 每个 epoch 为 714 步，Gemma 为 711 步（打包后序列数略少）。需要 retain 数据的方法使用
FineWeb-Edu（约 5B token 子集）作为 retain 语料，同样打包为 4096 token；训练时每个 forget 序列配对一个按固定种子
随机抽取的 retain 序列。数据顺序固定，不做打乱。

**A.3 优化与调度。** 所有方法均使用 AdamW（$\beta_1=0.9$，$\beta_2=0.999$，$\epsilon=10^{-8}$，权重衰减为 0），
梯度裁剪阈值 1.0，bf16 精度，随机种子 42，全局 batch size 为 32 条序列。学习率采用带 5% 线性 warmup（从 0 起）的
cosine 调度，**衰减到 0**。基础学习率为 $1\times10^{-6}$，其中 SimNPO 与 PureNPO 使用 $5\times10^{-5}$，
PDU 在 Gemma 与 Qwen 上使用 $4\times10^{-6}$（在 Intern 上为 $1\times10^{-6}$）。每个 run 训练 **3 个 epoch**：
Intern/Qwen 共 2142 步，Gemma 共 2133 步。除 RMU 仅更新第 7–9 层外，其余方法均更新模型全部参数
（attention、专家与 router）。

由于显存限制，全局 batch 通过不同的 micro-batch 配置实现：GA、RMU、SatImp、WGA 使用 micro-batch 4
（4×8 卡，无梯度累积）；CEU、GD、PDU、PureNPO、SimNPO 使用 micro-batch 1 并累积 4 步（1×8 卡×4），
原因是这些方法需要全词表 logits 或 forget+retain 成对序列，micro-batch 4 会显存溢出。两种配置的每步更新数学上等价。

**A.4 Checkpoint 选择。** 与 cyber 不同，bio 线**不做基于验证曲线的 epoch 挑选**：每个 run 完整训练 3 个 epoch，
只保存训练结束时（最后一步）的权重，并以该权重作为该方法的最终结果。训练过程中在若干步
（1、300、700、1000、1400、1700、1800、1900、2000、2100 与末步）在线评测 WMDP-Bio（全量 1273 题）与
MMLU（每个子任务抽取 35 题），0-shot，仅用于观察曲线。

## B. 是否使用 Retain 数据

与 cyber 完全一致：

- **仅使用 forget 数据（forget-only）**：GA、CEU、WGA、SatImp。WGA 与 SatImp 的 retain 权重取 $\alpha=0$，GA 与 CEU
  的目标函数本身不含 retain 项。GA 的配置文件中虽列有 retain 缓存路径，但其实现不会加载该数据。

**WGA、SatImp 取 forget-only 的原因。** 这两种方法的实现允许配合 retain 项使用，但在我们的实验设置下，加入 retain 项后训练基本不动：WMDP 与 MMLU 的准确率几乎都不发生变化，遗忘无法发生。因此本文按 forget-only 形式运行这两种方法（取 $\alpha=0$）。
- **同时使用 forget 与 retain 数据**：GD、PureNPO、SimNPO（损失中加 $\alpha\cdot\mathcal{L}_r$，$\alpha=1$）、
  PDU（retain 损失作为约束，由对偶变量动态加权）与 RMU（约束 retain 样本的中间层表示接近冻结参考模型）。

## C. 各方法的目标函数

同 `APPENDIX_baselines_cyber.md` 第 C 节（GA、GD、WGA、SatImp、CEU、PureNPO、SimNPO、PDU、RMU 的公式与实现完全一致，
bio 与 cyber 共用同一份代码）。

## D. 关键超参数一览

通用设置：global batch 32；3 个 epoch；$\gamma=1$；cosine 调度衰减到 0；取最后一步的权重。

| 方法 | 使用 retain | 学习率 | 关键超参数 |
|---|---|---|---|
| GA | 否 | $1\times10^{-6}$ | — |
| CEU | 否 | $1\times10^{-6}$ | 忽略首个 token |
| WGA | 否 | $1\times10^{-6}$ | $\beta=2$，$\alpha=0$ |
| SatImp | 否 | $1\times10^{-6}$ | $\beta_1=2$，$\beta_2=0.1$，$\alpha=0$ |
| GD | 是 | $1\times10^{-6}$ | $\alpha=1$ |
| PureNPO | 是 | $5\times10^{-5}$ | $\beta=0.1$，$\alpha=1$，参考模型为初始模型 |
| SimNPO | 是 | $5\times10^{-5}$ | $\beta=0.1$，$\delta=0$，$\alpha=1$ |
| PDU | 是 | Gemma/Qwen $4\times10^{-6}$，Intern $1\times10^{-6}$ | $\lambda_0=1$，$\eta=0.1$；$\varepsilon$：Gemma 2.6 / Intern 1.98 / Qwen 2.51 |
| RMU | 是 | $1\times10^{-6}$ | 目标层 9，训练层 7–9；steering 系数 $c$：Gemma 1200 / Intern 300 / Qwen 300；$\alpha$：Gemma 4.45 / Intern 350 / Qwen 634 |

## E. 与 cyber 设置的差异（便于在正文中统一表述）

| 项 | cyber | bio |
|---|---|---|
| 基础学习率 | $5\times10^{-6}$（NPO 类 $5\times10^{-5}$） | $1\times10^{-6}$（NPO 类 $5\times10^{-5}$；PDU 在 Gemma/Qwen 为 $4\times10^{-6}$） |
| cosine 下限 | $10^{-6}$ | 0 |
| 训练长度 | 10 epoch / 230 步 | 3 epoch / 2142 步（Gemma 2133 步） |
| 最终权重 | 按 MMLU − WMDP-Cyber 最大挑选的 epoch | 训练结束时的最后一步 |

## F. 撰写时需注意 / 待核对

1. **bio 与 cyber 的学习率口径不同**（bio 为 $10^{-6}$，且 PDU 例外），若论文在正文里写"统一学习率"，需要分领域说明。
2. **bio 无 epoch 挑选**，与 cyber 的 oracle 式选点不同；正文需明确两个领域的 checkpoint 选择口径不一致的原因（或统一口径）。
   例如 Intern 的 GD 在线曲线在第 700 步附近最低（WMDP-Bio 0.589 / MMLU 0.828），随后回升到 2142 步的 0.711 / 0.850，
   说明末步并非曲线最优点。
3. **PDU 的超参数不一致**：Intern 的 PDU 配置是 $\varepsilon=1.98$ 的调参副本（cyber 线为 2.05），且学习率与另外两个模型不同；
   其评测配置也不同（仅评末步，MMLU 跑全量）。Qwen 的 RMU 同样是 $\alpha\times10$ 的调参副本（63.4→634）。
   这类"调参副本"需要在论文中交代调参过程，或确认其作为最终 baseline 的合理性。
4. Gemma 使用 `nocap` 模型与 BOS 对齐的缓存；ckpt 说明中记录，BOS 对齐前的 Gemma bio 结果均已作废。
   CEU 的归一化实现在 2026-10-03 修复过，Gemma 之前的 CEU 结果不可用。
5. 评测抽样口径：在线评测中 MMLU 为每个子任务 35 题的子采样；个别 run（Intern PDU、Qwen RMU）在末步用了全量 MMLU，
   汇报最终数字时需统一口径。
6. 代码读自当前工作区（`src/megatron_unlearning/` 有未提交改动），需确认与训练这批权重时的版本一致。
