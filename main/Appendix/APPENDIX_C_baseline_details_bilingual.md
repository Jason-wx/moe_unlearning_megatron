# Appendix C draft — Baseline implementation details / 附录 C 草稿：Baseline 实现细节

> 接在技术报告附录 A、B 之后。依据：`main/train_configs/{gemma,intern,qwen}/{bio,cyber}/*.yaml`、`src/megatron_unlearning/` 与各权重目录的 `ckpt_notes.md`。标 `[TODO]` 处需作者确认。

---

# English

## C. Baseline implementation details

Section 4 compares REX with ten parameter-level unlearning baselines. Nine of them (GA, GD, CEU, WGA, SatImp, PureNPO, SimNPO, PDU and RMU) were trained with one shared Megatron-Core implementation on the three backbones and the two domains, and are described here; TRACE uses its own released trainer and is covered in Appendix A. Every baseline is applied to the original model B in its standard form, with neither expert expansion nor excision.

**Training setup.** All baselines train the full parameter set (attention, experts and routers), except RMU, which updates only layers 7–9. We use AdamW with $\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=10^{-8}$ and no weight decay, gradient clipping at 1.0, bf16 and seed 42. The global batch is 32 sequences of 4096 tokens, and the learning rate follows a cosine schedule with 5% linear warm-up. Training runs on 8 GPUs with expert parallelism 8 and no tensor, pipeline or context parallelism. To fit memory, GA, RMU, SatImp and WGA use a micro-batch of 4 per GPU, while CEU, GD, PDU, PureNPO and SimNPO, which need full-vocabulary logits or paired forget and retain sequences, use a micro-batch of 1 with four gradient-accumulation steps; the two realizations produce the same update. The reference model used by PureNPO and RMU is a frozen copy of the original model.

**Data and use of retain data.** The forget data is the WMDP forget corpus of the target domain, tokenized with each backbone's tokenizer and soft-packed into 4096-token sequences (EOS appended, and BOS prepended for Gemma 4). Methods that use retain data draw one FineWeb-Edu sequence for every forget sequence with a fixed seed, from a 5B-token subset packed in the same way, and the data order is fixed. GD, PureNPO, SimNPO, PDU and RMU use retain data, whereas GA, CEU, WGA and SatImp are forget-only. GA and CEU have no retain term in their objectives. WGA and SatImp admit one, but in our setting adding it left training essentially stationary, with neither WMDP nor MMLU accuracy moving, so we run both without it.

**Objectives and hyperparameters.** Write $\ell_t=-\log\pi_\theta(x_t\mid x_{<t})$ and $p_t=e^{-\ell_t}$ for token-level loss and probability, $\bar\ell(x)$ for the length-normalized sequence loss, and $\mathcal{L}_r$ for the mean retain loss. GA minimizes $-\mathbb{E}_f[\ell_t]$, and GD adds $\mathcal{L}_r$ to it with weight $\alpha=1$. WGA and SatImp weight the ascent term of each token by a detached function of its probability, $p_t^{\beta}$ with $\beta=2$ for WGA and $p_t^{\beta_1}(1-p_t)^{\beta_2}$ with $\beta_1=2$, $\beta_2=0.1$ for SatImp. CEU matches the model distribution at each position to the leave-gold-out distribution $\pi_\theta(v)/(1-\pi_\theta(x_t))$ and ignores the first token. PureNPO and SimNPO use the length-normalized sequence-level form $-\tfrac{2}{\beta}\log\sigma(\beta(\bar\ell_\theta-\bar\ell_{\rm ref}))$ and its reference-free variant $-\tfrac{2}{\beta}\log\sigma(\beta(\bar\ell_\theta-\delta))$, both with $\beta=0.1$ ($\delta=0$ for SimNPO) and an added retain term with $\alpha=1$. PDU minimizes the logit dispersion $(\max_v z_t-\bar z_t)^2$ on forget tokens subject to the constraint $\mathcal{L}_r\le\varepsilon$, which is enforced through a dual variable $\lambda$ that starts at 1 and is updated by $\lambda\leftarrow\max(0,\lambda+\eta(\mathcal{L}_r-\varepsilon))$ with $\eta=0.1$. RMU pulls the layer-9 activations of forget tokens towards a fixed random direction of norm $c$ and keeps those of retain tokens close to the frozen reference with weight $\alpha$. The weight of the forget term is 1 for all methods. The hyperparameters that differ across backbones are listed below.

| | Qwen3.5 | Intern-S2 | Gemma 4 |
|---|---|---|---|
| PDU $\varepsilon$ (Bio / Cyber) | 2.51 / 2.51 | 1.98 / 2.05 | 2.6 / 2.6 |
| RMU steering coefficient $c$ | 300 | 300 | 1200 |
| RMU retain weight $\alpha$ | 634 | 350 | 4.45 |

**Schedule and checkpoint selection.** The two domains use different schedules. On WMDP-Bio, the learning rate is $10^{-6}$ (and $5\times10^{-5}$ for PureNPO and SimNPO, and $4\times10^{-6}$ for PDU on Qwen3.5 and Gemma 4), decayed to zero over 3 epochs, i.e. 2142 steps (2133 on Gemma 4), and the checkpoint at the last step is reported. On WMDP-Cyber, the learning rate is $5\times10^{-6}$ (again $5\times10^{-5}$ for PureNPO and SimNPO), decayed to $10^{-6}$ over 10 epochs of 23 steps. Each Cyber run was evaluated after every epoch on the full WMDP-Cyber and on MMLU with 35 questions per subtask (0-shot), and the checkpoint of the epoch with the largest MMLU minus WMDP-Cyber margin (the earliest on ties) is reported. The selected epochs are given below. This online evaluation serves only to draw training curves and to select the Cyber checkpoint; the numbers in Tables 2 and 3 come from the offline protocol of Section 4.1.

| Cyber, selected epoch | GA | GD | CEU | WGA | SatImp | PureNPO | SimNPO | PDU | RMU |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3.5 | 2 | 3 | 8 | 8 | 2 | 3 | 2 | 2 | 4 |
| Intern-S2 | 2 | 6 | 4 | 1 | 1 | 3 | 1 | 3 | 2 |
| Gemma 4 | 1 | 7 | 10 | 3 | 9 | 3 | 1 | 9 | 4 |

---

**给作者的备注（不属于附录正文）。** 正文 4.1 节写"每个 baseline 与 REX 使用相同的 forget 与 retain 数据"、超参数"遵循原论文的搜索"，与实际有出入：GA、CEU、WGA、SatImp 并不使用 retain 数据，YAML 里也只记录了最终取值，PDU 的 $\varepsilon$ 与 RMU 的 $\alpha$ 和 $c$ 的来源（Bio 的 Intern PDU 与 Qwen RMU 是调参副本）需要补充说明。另外 Cyber 的 checkpoint 依据评测准确率挑选而 Bio 取最后一步，口径不一，建议统一或在正文披露。TRACE 的设置本文未覆盖 `[TODO]`；代码读自带未提交改动的工作区 `[TODO：确认与训练时版本一致]`。

---

# 中文

## C. Baseline 实现细节

第 4 节将 REX 与十个参数级遗忘 baseline 比较。其中九个（GA、GD、CEU、WGA、SatImp、PureNPO、SimNPO、PDU、RMU）在三个骨干和两个领域上使用同一套 Megatron-Core 实现训练，在此说明；TRACE 使用其自带的开源训练器，见附录 A。所有 baseline 都以标准形式直接作用于原始模型 B，既不做专家扩展，也不做切除。

**训练设置。** 除 RMU 仅更新第 7–9 层外，所有 baseline 都对全部参数（attention、专家和 router）做全参数训练。优化器为 AdamW（$\beta_1=0.9$，$\beta_2=0.999$，$\epsilon=10^{-8}$，无权重衰减），梯度裁剪 1.0，bf16，随机种子 42；全局 batch 为 32 条 4096-token 序列，学习率采用带 5% 线性 warm-up 的 cosine 调度。训练使用 8 卡，专家并行度 8，不使用张量、流水线和上下文并行。受显存限制，GA、RMU、SatImp、WGA 每卡 micro-batch 为 4；CEU、GD、PDU、PureNPO、SimNPO 因需要全词表 logits 或成对的 forget/retain 序列，使用 micro-batch 1 并累积 4 步，两种实现的更新等价。PureNPO 与 RMU 使用的参考模型是原始模型的冻结副本。

**数据与 retain 数据的使用。** Forget 数据为目标领域的 WMDP forget 语料，用各骨干的 tokenizer 分词后软打包为 4096-token 序列（末尾加 EOS，Gemma 4 另在开头加 BOS）。使用 retain 数据的方法，为每个 forget 序列按固定种子配一个 FineWeb-Edu 序列（同样打包的 5B-token 子集），数据顺序固定。GD、PureNPO、SimNPO、PDU、RMU 使用 retain 数据，GA、CEU、WGA、SatImp 为 forget-only：GA 与 CEU 的目标函数本身没有 retain 项；WGA 与 SatImp 允许加入 retain 项，但在我们的设置下加入后训练基本不动，WMDP 与 MMLU 的准确率都几乎不变，因此两者不使用 retain 项。

**目标函数与超参数。** 记 $\ell_t=-\log\pi_\theta(x_t\mid x_{<t})$、$p_t=e^{-\ell_t}$ 分别为 token 级损失与概率，$\bar\ell(x)$ 为按长度归一化的序列损失，$\mathcal{L}_r$ 为 retain 平均损失。GA 最小化 $-\mathbb{E}_f[\ell_t]$，GD 在此基础上以权重 $\alpha=1$ 加上 $\mathcal{L}_r$。WGA 与 SatImp 用 token 概率的（截断梯度的）函数为每个 token 的上升项加权：WGA 为 $p_t^{\beta}$（$\beta=2$），SatImp 为 $p_t^{\beta_1}(1-p_t)^{\beta_2}$（$\beta_1=2$，$\beta_2=0.1$）。CEU 让每个位置的模型分布逼近去掉正确 token 后的重归一化分布 $\pi_\theta(v)/(1-\pi_\theta(x_t))$，并忽略第一个 token。PureNPO 使用按长度归一化的序列级形式 $-\tfrac{2}{\beta}\log\sigma(\beta(\bar\ell_\theta-\bar\ell_{\rm ref}))$，SimNPO 使用其无参考模型的变体 $-\tfrac{2}{\beta}\log\sigma(\beta(\bar\ell_\theta-\delta))$，两者均取 $\beta=0.1$（SimNPO 的 $\delta=0$），并以 $\alpha=1$ 加上 retain 项。PDU 在 forget token 上最小化 logit 离散度 $(\max_v z_t-\bar z_t)^2$，同时要求 $\mathcal{L}_r\le\varepsilon$，该约束通过对偶变量 $\lambda$ 实现，$\lambda$ 初始为 1，按 $\lambda\leftarrow\max(0,\lambda+\eta(\mathcal{L}_r-\varepsilon))$ 更新，$\eta=0.1$。RMU 把 forget token 在第 9 层的激活拉向一个范数为 $c$ 的固定随机方向，并以权重 $\alpha$ 使 retain token 的激活贴近冻结参考模型。所有方法 forget 项的权重均为 1。随骨干变化的超参数列于下表。

| | Qwen3.5 | Intern-S2 | Gemma 4 |
|---|---|---|---|
| PDU $\varepsilon$（Bio / Cyber） | 2.51 / 2.51 | 1.98 / 2.05 | 2.6 / 2.6 |
| RMU steering 系数 $c$ | 300 | 300 | 1200 |
| RMU retain 权重 $\alpha$ | 634 | 350 | 4.45 |

**训练调度与 checkpoint 选择。** 两个领域的调度不同。WMDP-Bio 的学习率为 $10^{-6}$（PureNPO、SimNPO 为 $5\times10^{-5}$，PDU 在 Qwen3.5 与 Gemma 4 上为 $4\times10^{-6}$），在 3 个 epoch（2142 步，Gemma 4 为 2133 步）内衰减到 0，报告最后一步的 checkpoint。WMDP-Cyber 的学习率为 $5\times10^{-6}$（PureNPO、SimNPO 同样为 $5\times10^{-5}$），在 10 个 epoch（每个 epoch 23 步）内衰减到 $10^{-6}$。每个 Cyber run 在每个 epoch 后在完整的 WMDP-Cyber 和每个子任务 35 题的 MMLU 上做 0-shot 评测，报告 MMLU 减去 WMDP-Cyber 差值最大（并列取最早）的那个 epoch 的 checkpoint，所选 epoch 见下表。该在线评测只用于绘制训练曲线和选取 Cyber 的 checkpoint；表 2、表 3 中的数字来自 4.1 节的离线评测协议。

| Cyber 所选 epoch | GA | GD | CEU | WGA | SatImp | PureNPO | SimNPO | PDU | RMU |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3.5 | 2 | 3 | 8 | 8 | 2 | 3 | 2 | 2 | 4 |
| Intern-S2 | 2 | 6 | 4 | 1 | 1 | 3 | 1 | 3 | 2 |
| Gemma 4 | 1 | 7 | 10 | 3 | 9 | 3 | 1 | 9 | 4 |
