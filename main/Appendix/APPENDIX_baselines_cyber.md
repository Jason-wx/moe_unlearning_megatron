# 附录草稿：Baseline 训练细节（WMDP-Cyber）

依据：`main/train_configs/{gemma,intern,qwen}/cyber/*.yaml`
（3 个模型 × 9 个 baseline；TRACE 不在此列）。以下内容均读自这些配置
以及 `src/megatron_unlearning/` 中的代码。

## 1. 模型

| 简称 | 模型 | 专家数 |
|---|---|---|
| Gemma | Gemma4-26B-A4B（`gemma-26b-a4b-nocap`） | 128 |
| Intern | Intern-S2-base05（35B MoE） | 256 |
| Qwen | Qwen3.5-35B-A3B | 256 |

## 2. 数据

- **Forget 集**：WMDP-Cyber forget 语料，用各模型自己的 tokenizer 分词后，
  软打包（soft pack）成长度 4096 的序列（末尾加 EOS；Gemma 另外在开头加 BOS）。
  打包种子 43。Intern/Qwen 共 743 个 pack，Gemma 为 748 个，
  即每个 epoch 23 个优化步。
- **Retain 集**（仅在使用 retain 的方法中，见第 4 节）：FineWeb-Edu
  （5B token 子集），同样打包为 4096，种子 42。
- 数据顺序固定（`shuffle: false`）；使用 retain 时，每个 forget pack
  会配一个按种子随机抽取的 retain pack。

## 3. 共同训练设置（27 个 run 完全一致）

| 项目 | 取值 |
|---|---|
| 框架 | Megatron-Core 后端，8 卡 |
| 并行 | TP=1，PP=1，CP=1，EP=8，分布式优化器，不开序列并行 |
| 可训练参数 | 全参数（attention、experts、router）；RMU 仅训练第 7–9 层 |
| 精度 | bf16 |
| 优化器 | AdamW，betas (0.9, 0.999)，eps 1e-8，weight decay 0 |
| 梯度裁剪 | 1.0 |
| Global batch size | 32 条序列（每条 4096 token） |
| 训练长度 | 10 epoch = 230 步（三个模型相同） |
| 学习率调度 | cosine，前 5% 线性 warmup（从 0 起），下限 1e-6 |
| 随机种子 | 42 |
| 参考模型（NPO 类、RMU） | 冻结的初始模型副本 |

所有 run 的 global batch 都是 32。micro batch 只因显存不同而不同：
GradAscent、RMU、SatImp、WGA 用 micro batch 4（4 × 8 卡，无梯度累积）；
CEU、GradDiff、PDU、PureNPO、SimNPO 用 micro batch 1（1 × 8 卡 × 4 次梯度累积），
因为它们用到全词表 logits 或 forget+retain 成对 pack。两者优化数学上等价。

**学习率**：除 PureNPO 和 SimNPO 为 5e-5 外，其余方法均为 5e-6。

## 4. 各方法的超参数

“使用 retain”指 retain 集是否真正参与损失计算。

| 方法 | 损失（forget / retain） | 超参数 | 使用 retain |
|---|---|---|---|
| GradAscent (GA) | forget 上的 −NLL | 无 | 否 |
| GradDiff (GD) | −NLL(forget) + alpha · NLL(retain) | gamma=1，alpha=1 | 是 |
| PDU | forget 上的 logit 离散度 + λ · (NLL(retain) − eps)，λ 通过对偶上升更新（下限 0） | gamma=1，λ₀=1，对偶步长 0.1，eps：Gemma 2.6 / Intern 2.05 / Qwen 2.51 | 是 |
| PureNPO | 长度归一化的 NPO（forget）+ alpha · NLL(retain) | beta=0.1，gamma=1，alpha=1 | 是 |
| SimNPO | 无参考模型的序列级 NPO + alpha · NLL(retain) | beta=0.1，delta=0，gamma=1，alpha=1 | 是 |
| RMU | 第 9 层激活对固定随机控制向量的 MSE（forget）+ alpha · 对冻结参考模型激活的 MSE（retain） | 目标层 9，训练层 7–9，gamma=1；steering 系数 / alpha：Gemma 1200 / 4.45，Intern 300 / 350，Qwen 300 / 634；固定范数控制向量 | 是 |
| CEU | forget 上对“去掉正确 token 后的分布”的交叉熵 | gamma=1，忽略第 1 个 token | 否 |
| SatImp | 饱和度/重要性加权的梯度上升 | beta1=2，beta2=0.1，alpha=0 | 否 |
| WGA | 按 token 概率加权的梯度上升 | beta=2，alpha=0 | 否 |

说明：GradAscent 的配置里写了 FineWeb retain 缓存，但代码对该方法不会加载
（见 `algorithm_uses_retain`），因此 GA 实际只用 forget 数据。SatImp 和 WGA
因 alpha=0，不计算 retain 项。

## 5. Checkpoint 选择

每个 run 先带逐 epoch 在线评测（WMDP-Cyber 全量；MMLU 每个子任务 35 题；0-shot）
跑出完整曲线。最终保存的 checkpoint 是 margin = MMLU − WMDP-Cyber 最大的那个
epoch 对应的步（并列时取最早），不保存最终步。各方法选中的步数（每 epoch 23 步）：

| 方法 | Gemma | Intern | Qwen |
|---|---|---|---|
| CEU | 230 (ep10) | 92 (ep4) | 184 (ep8) |
| GradAscent | 23 (ep1) | 46 (ep2) | 46 (ep2) |
| GradDiff | 161 (ep7) | 138 (ep6) | 69 (ep3) |
| PDU | 207 (ep9) | 69 (ep3) | 46 (ep2) |
| PureNPO | 69 (ep3) | 69 (ep3) | 69 (ep3) |
| RMU | 92 (ep4) | 46 (ep2) | 92 (ep4) |
| SatImp | 207 (ep9) | 23 (ep1) | 46 (ep2) |
| SimNPO | 23 (ep1) | 23 (ep1) | 46 (ep2) |
| WGA | 69 (ep3) | 23 (ep1) | 184 (ep8) |

## 6. 提交前需核对

- 评测 batch size（32，GradDiff/PureNPO/SimNPO 为 16）只影响评测吞吐，不影响结果。
- PDU 的 eps、RMU 的 alpha/steering 等超参数的调参过程，yaml 中除注释外没有记录；
  如审稿人需要，请补充搜索范围。
- FineWeb-Edu 子集的抽取方式和 WMDP-Cyber 语料的确切来源，请对照
  `configs/data_processing/` 下的数据处理配置再确认。
- 以上代码读自当前工作区，`src/megatron_unlearning/` 下有未提交的改动，
  请确认与训练这批权重时的版本一致。

---

# 附录正文草稿（学术行文版，WMDP-Cyber）

> 以下为可直接改写进论文的详细表述，后期再精简。符号约定：$\mathcal{D}_f$ 为 forget 集，$\mathcal{D}_r$ 为 retain 集，$\pi_\theta$ 为当前模型，$\pi_{\mathrm{ref}}$ 为冻结的初始模型；对序列 $x$ 的第 $t$ 个 token，$\ell_t(\theta)=-\log \pi_\theta(x_t\mid x_{<t})$，$p_t=\exp(-\ell_t)$。

## A. Baseline 实现与训练设置

**A.1 总体设置。** 我们在三个 MoE 模型（Gemma4-26B-A4B、Intern-S2-35B、Qwen3.5-35B-A3B）上，针对 WMDP-Cyber 实现并训练了九个遗忘 baseline：GradAscent (GA)、GradDiff (GD)、CEU、WGA、SatImp、PureNPO、SimNPO、PDU 与 RMU。所有 baseline 共用同一套训练框架（Megatron-Core，8 卡，专家并行度为 8，张量/流水线/上下文并行度均为 1）、同一套数据流水线与同一套评测协议，以保证方法间的差异仅来自遗忘目标本身。

**A.2 数据。** Forget 数据为 WMDP-Cyber forget 语料，使用各模型自带的 tokenizer 分词，并以软打包方式组织为长度 4096 的序列（序列末尾追加 EOS；Gemma 额外在开头追加 BOS），Intern/Qwen 共 743 个序列，Gemma 为 748 个。需要 retain 数据的方法使用 FineWeb-Edu（约 5B token 子集）作为 retain 语料，同样打包为 4096 token；训练时每个 forget 序列配对一个按固定种子随机抽取的 retain 序列。数据顺序固定，不做打乱。

**A.3 优化与调度。** 所有方法均使用 AdamW（$\beta_1=0.9$，$\beta_2=0.999$，$\epsilon=10^{-8}$，权重衰减为 0），梯度裁剪阈值为 1.0，bf16 精度，随机种子 42。全局 batch size 统一为 32 条序列（即每步 32×4096 个 forget token）。学习率为 $5\times10^{-6}$，其中 SimNPO 与 PureNPO 使用 $5\times10^{-5}$（NPO 类目标的梯度被 sigmoid 权重进一步衰减，需要更大的步长）。学习率采用 cosine 调度，前 5% 步数从 0 线性 warmup，下限为 $10^{-6}$。每个 run 训练 10 个 epoch，对应 230 个优化步（每个 epoch 23 步）。除 RMU 仅更新第 7–9 层外，其余方法均更新模型全部参数（attention、专家与 router）。

由于显存限制，全局 batch 通过不同的 micro-batch 配置实现：GA、RMU、SatImp、WGA 使用 micro-batch 4（4×8 卡，无梯度累积）；CEU、GD、PDU、PureNPO、SimNPO 使用 micro-batch 1，并累积 4 步（1×8 卡×4）。两种配置的每步更新在数学上等价。

**A.4 Checkpoint 选择。** 每个 epoch 结束时在线评测 WMDP-Cyber（全部 1987 题）与 MMLU（每个子任务抽取 35 题），均为 0-shot。我们以 $\text{margin}=\text{Acc}_{\text{MMLU}}-\text{Acc}_{\text{WMDP-Cyber}}$ 作为选择准则，即在遗忘效果（WMDP-Cyber 准确率越低越好）与通用能力保留（MMLU 准确率越高越好）之间取平衡，选取 margin 最大的 epoch（并列时取最早者）作为该方法在该模型上的最终权重。选定后，我们以相同配置复跑并只在该 epoch 保存 checkpoint，不保存训练末尾的权重。选中的 epoch 见 A.6。

## B. 是否使用 Retain 数据

按照遗忘损失中是否出现 retain 项，九个 baseline 分为两类：

- **仅使用 forget 数据（forget-only）**：GA、CEU、WGA、SatImp。其中 WGA 与 SatImp 的实现含有 retain 权重 $\alpha$，但在本文设置中取 $\alpha=0$，因此不构造 retain 项；GA 与 CEU 的目标函数本身不含 retain 项。这四种方法在训练中不接触 retain 语料。
- **同时使用 forget 与 retain 数据**：GD、PureNPO、SimNPO、PDU、RMU。前三者在遗忘损失之外加上 $\alpha\cdot\mathcal{L}_r$，其中 $\alpha=1$；PDU 将 retain 损失作为约束，通过对偶变量动态加权；RMU 则约束 retain 样本的中间层表示不偏离冻结的参考模型。

需要说明的是，GA 的配置文件中虽然列有 retain 缓存路径，但其实现并不加载该数据，因此 GA 实际上是 forget-only。

**WGA、SatImp 取 forget-only 的原因。** 这两种方法的实现允许配合 retain 项使用，但在我们的实验设置下，加入 retain 项后训练基本不动：WMDP 与 MMLU 的准确率几乎都不发生变化，遗忘无法发生。因此本文按 forget-only 形式运行这两种方法（取 $\alpha=0$）。

## C. 各方法的目标函数

记 forget 序列上的 token 级平均为 $\mathbb{E}_f[\cdot]$，retain 上的平均 NLL 为 $\mathcal{L}_r=\mathbb{E}_{r}[\ell_t]$，序列级平均 NLL 为 $\bar\ell(x)=\frac1{|x|}\sum_t\ell_t$。下列各式均为被最小化的损失，$\mathrm{sg}(\cdot)$ 表示停止梯度。

1. **GA**：$\mathcal{L}=-\gamma\,\mathbb{E}_f[\ell_t]$，即在 forget 数据上直接做梯度上升。
2. **GD**：$\mathcal{L}=-\gamma\,\mathbb{E}_f[\ell_t]+\alpha\,\mathcal{L}_r$，在梯度上升的同时在 retain 数据上做标准语言模型训练。
3. **WGA**：$\mathcal{L}=-\mathbb{E}_f\big[\mathrm{sg}(p_t^{\beta})\,\ell_t\big]$，以 token 概率的 $\beta$ 次幂加权，抑制对已被"遗忘"（概率已很低）的 token 继续上升，从而减轻对模型的破坏。
4. **SatImp**：$\mathcal{L}=-\mathbb{E}_f\big[\mathrm{sg}\big(p_t^{\beta_1}(1-p_t)^{\beta_2}\big)\,\ell_t\big]$，在 WGA 的饱和度权重基础上乘以重要性权重 $(1-p_t)^{\beta_2}$。
5. **CEU**：对每个位置，将"去掉正确 token 后重新归一化"的分布 $q(v)=\pi_\theta(v)/(1-\pi_\theta(x_t))$（$v\neq x_t$）作为目标（梯度不回传到目标），最小化 $\pi_\theta$ 与 $q$ 之间的交叉熵，忽略序列第一个 token。该目标在抹去正确答案概率的同时保持其余词表上的相对分布。
6. **PureNPO**：$\mathcal{L}=\gamma\cdot\mathbb{E}_{x\sim\mathcal{D}_f}\Big[-\tfrac{2}{\beta}\log\sigma\big(\beta(\bar\ell_\theta(x)-\bar\ell_{\mathrm{ref}}(x))\big)\Big]+\alpha\,\mathcal{L}_r$。这里采用长度归一化的序列级 NLL，参考模型为初始模型。
7. **SimNPO**：$\mathcal{L}=\gamma\cdot\mathbb{E}_{x\sim\mathcal{D}_f}\Big[-\tfrac{2}{\beta}\log\sigma\big(\beta(\bar\ell_\theta(x)-\delta)\big)\Big]+\alpha\,\mathcal{L}_r$，不依赖参考模型；本文取 $\delta=0$。
8. **PDU**：forget 项为 logit 离散度 $\mathcal{L}_f=\mathbb{E}_f\big[(\max_v z_t(v)-\overline{z_t})^2\big]$（$z_t$ 为词表 logits，$\overline{z_t}$ 为其词表均值），retain 项作为约束 $\mathcal{L}_r\le\varepsilon$。总损失为 $\mathcal{L}=\gamma\mathcal{L}_f+\lambda(\mathcal{L}_r-\varepsilon)$，对偶变量每步更新 $\lambda\leftarrow\max\big(0,\ \lambda+\eta(\mathcal{L}_r-\varepsilon)\big)$（在数据并行各卡间取平均），$\lambda_0=1$，$\eta=0.1$。约束阈值 $\varepsilon$ 针对各模型分别设定。
9. **RMU**：取第 9 层输出激活 $h_\theta^{(9)}$，forget 样本的激活被拉向固定随机控制向量 $c\,u$（$u$ 为固定种子生成的单位向量，$c$ 为 steering 系数，范数固定），retain 样本的激活被约束在冻结参考模型的激活 $h_{\mathrm{ref}}^{(9)}$ 附近：$\mathcal{L}=\gamma\,\mathrm{MSE}\big(h_\theta^{(9)},c\,u\big)_{f}+\alpha\,\mathrm{MSE}\big(h_\theta^{(9)},h_{\mathrm{ref}}^{(9)}\big)_{r}$。只更新第 7–9 层参数。

## D. 关键超参数一览

通用设置：global batch 32；10 个 epoch（230 步）；学习率 $5\times10^{-6}$（SimNPO、PureNPO 为 $5\times10^{-5}$）；$\gamma=1$。

| 方法 | 使用 retain | 关键超参数 |
|---|---|---|
| GA | 否 | — |
| CEU | 否 | 忽略首个 token |
| WGA | 否 | $\beta=2$，$\alpha=0$ |
| SatImp | 否 | $\beta_1=2$，$\beta_2=0.1$，$\alpha=0$ |
| GD | 是 | $\alpha=1$ |
| PureNPO | 是 | $\beta=0.1$，$\alpha=1$，参考模型为初始模型 |
| SimNPO | 是 | $\beta=0.1$，$\delta=0$，$\alpha=1$ |
| PDU | 是 | $\lambda_0=1$，$\eta=0.1$；$\varepsilon$：Gemma 2.6 / Intern 2.05 / Qwen 2.51 |
| RMU | 是 | 目标层 9，训练层 7–9；steering 系数 $c$：Gemma 1200 / Intern 300 / Qwen 300；$\alpha$：Gemma 4.45 / Intern 350 / Qwen 634 |

## E. 各方法最终选定的 epoch

（每个 epoch 23 步；括号内为 epoch 序号）

| 方法 | Gemma | Intern | Qwen |
|---|---|---|---|
| GA | ep1 | ep2 | ep2 |
| GD | ep7 | ep6 | ep3 |
| CEU | ep10 | ep4 | ep8 |
| WGA | ep3 | ep1 | ep8 |
| SatImp | ep9 | ep1 | ep2 |
| PureNPO | ep3 | ep3 | ep3 |
| SimNPO | ep1 | ep1 | ep2 |
| PDU | ep9 | ep3 | ep2 |
| RMU | ep4 | ep2 | ep4 |

**撰写时需注意：** (1) 上面各目标函数的符号与归一化方式（token 级 vs. 序列级平均）依据代码实现整理，引用原论文公式时请与原文对照；(2) $\varepsilon$、$\alpha$、$c$ 的选取过程目前没有文档记录，正文如声称"经调参得到"，需要补充搜索范围；(3) 以 margin 选 epoch 使用了评测集本身，论文中应如实说明这一点（属于 checkpoint 层面的 oracle 选择）。
