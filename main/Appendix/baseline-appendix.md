# Appendix: Baseline Training Details

We implement 9 unlearning baselines (GA, GD, CEU, WGA, SatImp, NPO, SimNPO, PDU and RMU) on Gemma-4-26B-A4B, Intern-S2-35B and Qwen3.5-35B-A3B for both WMDP-Bio and WMDP-Cyber ($9 \times 3 \times 2 = 54$ runs), with FineWeb-Edu as the retain data. All runs use a Megatron-Core backend on 8 GPUs with expert parallelism 8, tensor / pipeline / context parallelism 1, bf16 precision and seed 42.

## Objective functions

We denote the forget set by $\mathcal{D}_f$, the retain set by $\mathcal{D}_r$, the current model by $\pi_\theta$ and the frozen initial model by $\pi_{\mathrm{ref}}$. For token $t$ in sequence $x$, let $z_t \in \mathbb{R}^{|V|}$ be the vocabulary logits output by $\pi_\theta$, and let $\ell_t(\theta) = -\log \pi_\theta(x_t \mid x_{<t})$ be the per-token negative log-likelihood (NLL), with $p_t = \pi_\theta(x_t \mid x_{<t}) = \exp(-\ell_t)$ the corresponding token probability. The sequence-level mean NLL is $\bar\ell(x) = \frac{1}{|x|}\sum_t \ell_t$. All formulas below are losses to be minimized. We write $\mathbb{E}_f[\cdot]$ for the token-level average over $\mathcal{D}_f$, and $\mathcal{L}_r = \mathbb{E}_r[\ell_t]$ for the mean NLL on $\mathcal{D}_r$. We also use $\mathrm{sg}(\cdot)$ for the stop-gradient operator, which equals the identity in the forward pass but blocks gradient flow in the backward pass, so that its argument acts as a constant when differentiating the loss.

1. **GA** (Gradient Ascent; Jang et al., 2023). $\mathcal{L} = -\gamma\,\mathbb{E}_f[\ell_t]$. The model is updated to increase the NLL on the forget data.

2. **GD** (Gradient Difference; Maini et al., 2024). $\mathcal{L} = -\gamma\,\mathbb{E}_f[\ell_t] + \alpha\,\mathcal{L}_r$. Gradient ascent on $\mathcal{D}_f$ is combined with standard language-model training on $\mathcal{D}_r$, where $\alpha$ balances the two objectives.

3. **WGA** (Weighted Gradient Ascent; Wang et al., 2025b). $\mathcal{L} = -\mathbb{E}_f\!\big[\mathrm{sg}(w_t)\,\ell_t\big]$, where $w_t = p_t^{\beta}$ and $\mathrm{sg}(\cdot)$ treats the weight as a constant during optimization. Each token's NLL is scaled by the $\beta$-th power of its current probability, so high-probability tokens contribute more and already-low-probability tokens are down-weighted. We set the retain-loss weight $\alpha=0$ as including the retain term fails to yield effective forgetting in our setting.

4. **SatImp** (Saturation–Importance; Yang et al., 2025). $\mathcal{L} = -\mathbb{E}_f\!\big[\mathrm{sg}(w_t)\,\ell_t\big]$, where $w_t = p_t^{\beta_1}(1-p_t)^{\beta_2}$. The weight is the product of a saturation factor $p_t^{\beta_1}$, which down-weights tokens that are already low-probability (same role as in WGA), and an importance factor $(1-p_t)^{\beta_2}$, which up-weights tokens that the model has not yet confidently learned. Together they concentrate the unlearning signal on tokens of intermediate probability. As with WGA, we set $\alpha=0$ for the same reason.

5. **CEU** (Cross-Entropy Unlearning; Yang, 2025). For each position $t$, a target distribution $q_t$ is constructed by removing the correct token $x_t$ from the vocabulary and renormalizing:
$$q_t(v) = \frac{\exp(z_t(v))}{\sum_{v'\neq x_t}\exp(z_t(v'))}\;\text{for } v \neq x_t,\qquad q_t(x_t)=0.$$
The loss is the cross-entropy between $q_t$ (treated as a fixed target) and the model's output: $\mathcal{L} = -\mathbb{E}_f\!\big[\sum_v q_t(v)\log\pi_\theta(v\mid x_{<t})\big]$, ignoring the first token of each sequence.

6. **NPO** (Negative Preference Optimization; Zhang et al., 2024). $\mathcal{L} = \gamma\,\mathbb{E}_{x\sim\mathcal{D}_f}\!\Big[-\tfrac{2}{\beta}\log\sigma\!\big(\beta(\bar\ell_\theta(x)-\bar\ell_{\mathrm{ref}}(x))\big)\Big]+\alpha\,\mathcal{L}_r$, where $\sigma$ is the sigmoid function. The original NPO operates on unnormalized sequence-level NLL; we use the length-normalized variant $\bar\ell(x)=\frac{1}{|x|}\sum_t\ell_t$ because the unnormalized form leads to excessive forgetting and model collapse in our setting.

7. **SimNPO** (Simplified NPO; Fan et al., 2025b). $\mathcal{L} = \gamma\,\mathbb{E}_{x\sim\mathcal{D}_f}\!\Big[-\tfrac{2}{\beta}\log\sigma\!\big(\beta(\bar\ell_\theta(x)-\delta)\big)\Big]+\alpha\,\mathcal{L}_r$. A reference-free variant: the reference model's NLL is replaced by a constant margin $\delta$ (set to 0 in our experiments), removing the need to maintain and forward-pass a frozen reference.

8. **PDU** (Primal-Dual Unlearning; Entesari et al., 2025). The forget loss is a logit-margin flattening term $\mathcal{L}_f = \mathbb{E}_f\!\big[(\max_v z_t(v) - \overline{z_t})^2\big]$, where $\overline{z_t} = \frac{1}{|V|}\sum_v z_t(v)$, which drives the output distribution toward uniformity. Retain performance is formulated as a constraint $\mathcal{L}_r \le \varepsilon$ and enforced via Lagrangian relaxation: $\mathcal{L} = \gamma\,\mathcal{L}_f + \lambda(\mathcal{L}_r - \varepsilon)$, where the dual variable $\lambda$ is updated each step by $\lambda \leftarrow \max(0,\;\lambda + \eta(\mathcal{L}_r - \varepsilon))$ to automatically balance forgetting and retention.

9. **RMU** (Representation Misdirection for Unlearning; Li et al., 2024). $\mathcal{L} = \gamma\,\mathrm{MSE}\!\big(h_\theta^{(9)},\;c\,u\big)_f + \alpha\,\mathrm{MSE}\!\big(h_\theta^{(9)},\;h_{\mathrm{ref}}^{(9)}\big)_r$, where $u$ is a fixed random unit vector scaled by a steering coefficient $c$. On forget data, the activations at layer 9 are pushed toward $c\,u$; on retain data, they are anchored to the frozen reference model. In our setting, we update layers 7–9 and freeze all other parameters.

## Training setup

**Data and optimization.** Forget and retain corpora are tokenized per model and soft-packed into 4096-token sequences; Gemma additionally prepends BOS to each sequence. All methods use AdamW ($\beta_1\!=\!0.9$, $\beta_2\!=\!0.999$, $\epsilon\!=\!10^{-8}$, weight decay 0) with gradient clipping 1.0, global batch size 32, cosine learning-rate decay with 0.05 linear warmup from 0, and train all parameters (attention, experts, routers) except RMU which updates only layers 7–9. For methods with a retain term (GD, NPO, SimNPO), $\alpha=1$. Table C.1 lists the remaining per-method hyperparameters; Table C.2 gives the model-specific values for PDU and RMU.

**Table C.1 | Per-method hyperparameters for the nine unlearning baselines.** All methods share a global batch size of 32 and a forget-loss weight of $\gamma = 1$. Methods not listed with specific hyperparameters use only the shared training configuration described in the text.

| Method | Key hyperparameters |
|---|---|
| GA | — |
| GD | — |
| WGA | $\beta\!=\!2$, $\alpha\!=\!0$ |
| SatImp | $\beta_1\!=\!2$, $\beta_2\!=\!0.1$, $\alpha\!=\!0$ |
| CEU | — |
| NPO | $\beta\!=\!0.1$; ref = initial model |
| SimNPO | $\beta\!=\!0.1$, $\delta\!=\!0$ |
| PDU | $\lambda_0\!=\!1$, $\eta\!=\!0.1$; $\varepsilon$ per model (Table C.2) |
| RMU | $l\!=\!9$, train layers 7–9; $c$, $\alpha$ per model (Table C.2) |

**Table C.2 | Model-specific hyperparameters for PDU and RMU across three backbones.** RMU's steering coefficient $c$ and retain weight $\alpha$ are shared across the Bio and Cyber domains for each model; PDU's constraint threshold $\varepsilon$ differs slightly for Intern-S2.

| | | Gemma 4 | Intern-S2 | Qwen3.5 |
|---|---|---|---|---|
| PDU | $\varepsilon$ (Cyber / Bio) | 2.6 / 2.6 | 2.05 / 1.98 | 2.51 / 2.51 |
| RMU | Steering $c$ | 1200 | 300 | 300 |
| | Retain weight $\alpha$ | 4.45 | 350 | 634 |

**Bio vs. Cyber differences.** The two domains differ substantially in forget-set size: the Bio forget set contains roughly 30$\times$ more packed training sequences than the Cyber one, leading to different training configurations. Bio runs train for 3 epochs (2142 steps) at a base learning rate of $1 \times 10^{-6}$ with cosine decaying to 0; Cyber runs train for 10 epochs (230 steps) at $5 \times 10^{-6}$ with a cosine floor of $10^{-6}$. NPO and SimNPO use $5 \times 10^{-5}$ in both domains; PDU uses $4 \times 10^{-6}$ on Gemma / Qwen for Bio. The two domains use different checkpoint strategies: Bio runs use the final-step checkpoint without selection, while Cyber runs are evaluated at each of the 10 epochs on WMDP-Cyber and MMLU (0-shot), and we select the epoch maximizing $\mathrm{Acc}_\mathrm{MMLU} - \mathrm{Acc}_\mathrm{WMDP\text{-}Cyber}$. Table C.3 lists the selected epochs.

**Table C.3 | Selected training epoch for each baseline on WMDP-Cyber (23 steps per epoch), chosen by maximizing $\mathrm{Acc}_\mathrm{MMLU} - \mathrm{Acc}_\mathrm{WMDP\text{-}Cyber}$ over 10 epochs.** WMDP-Bio uses the final-step checkpoint for all methods without epoch selection.

| Method | Gemma 4 | Intern-S2 | Qwen3.5 |
|---|---|---|---|
| GA | 1 | 2 | 2 |
| GD | 7 | 6 | 3 |
| CEU | 10 | 4 | 8 |
| WGA | 3 | 1 | 8 |
| SatImp | 9 | 1 | 2 |
| NPO | 3 | 3 | 3 |
| SimNPO | 1 | 1 | 2 |
| PDU | 9 | 3 | 2 |
| RMU | 4 | 2 | 4 |