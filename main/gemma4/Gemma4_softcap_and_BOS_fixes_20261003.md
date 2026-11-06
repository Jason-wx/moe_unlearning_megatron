# Gemma4 修复记录：BOS（训练/测评）与 softcap

**日期**：2026-10-03
**模型**：gemma-26b-a4b（30 层 / hidden 2816 / 128 experts / vocab 262144 / pad_token_id 0）

gemma-26b-a4b 是三条模型线（intern-s2 / qwen35 / gemma4）中唯一带
`final_logit_softcapping`、且 tokenizer 默认在句首加 `<bos>`（id=2）的模型。
本轮调试定案了三个问题，**按严重程度排序**：

1. **训练数据缺 BOS**（最重）：训练缓存一条 BOS 都没有，训练输入和评测输入
   不在同一个计算 regime——升向族算法（GA/GD/SatImp/WGA/CEU）在 gemma4 上的
   全部早期结果因此零迁移、整体作废；
2. **在线测评缺 BOS**：在线打分管线把 BOS 丢掉，读数系统性偏低 ~27pt，
   "测出来的分是错的"；
3. **softcap（cap 机制）**：测评侧打分漏掉一次 tanh 变换（~0.5pt，已修）；
   训练侧 cap 确实改变升向算法的优化动力学，但**不是**零迁移的根因，
   最终以 nocap 影子模型处置。

前两个是同一个模型特性（Gemma 句首恒有 `<bos>`）在两个环节缺位的两份账：
**训练侧缺 BOS 让"训出来的破坏测不到"，测评侧缺 BOS 让"测出来的分是错的"**。

---

## 0. 先说清楚：BOS 到底加几个？

**是——每条序列开头恰好一个 `<bos>`（id=2），没有别的。** Gemma 预训练与
推理的口径就是"序列首 token 恒为 `<bos>`"，少了它模型直接掉到接近随机
（base 0.79 → 0.46，见 §2）。要的不是一个特殊格式、不是每句加、不是加多个——
就是**每条序列一个 token**。

关键是"**序列**"的定义在两个环节不同：

| 环节 | 喂给模型的"一条序列" | BOS 应该加在哪 | 怎么加 |
|---|---|---|---|
| 测评 | 一条 prompt（100-300 token） | prompt 开头一个 | tokenizer 默认行为，去掉代码里的 override 即可（§2） |
| 训练 | 一个 4096-token 的 **pack**（多段文档拼接/切分装进一个容器） | **每个 pack 头上一个** | 建缓存时逐 pack 前插（§1.3） |

训练侧为什么不能靠 tokenizer 的 `add_bos_token`：tokenizer 是在**建缓存**时
离线跑的，它看到的单位是**文档**，只在文档开头加一个 BOS；而 soft packing
把长文档切成多段、把多段文档拼进同一个 pack，forget 语料平均 ~13k token/文档、
每个 4096 pack 平均只有 0.31 个文档边界——**约 2/3 的 pack 是文档中段的续块，
头部根本没有文档开头**，文档级 BOS 落不到它们头上。所以要逐 pack 插（实测每
pack 恰好一个 BOS、原末 token 让位，不会出现双 BOS）。

---

## 1. 训练数据缺 BOS（最重：升向族零迁移的真根因）

### 1.1 症状：训练端把 NLL 打崩，评测端纹丝不动

main/gemma4 十二配置（=interns2 主线逐字段迁移）里，所有以"抬升 forget NLL"
为驱动力的算法在 gemma4 上全部冻结在 0.78-0.80，而同配置在 interns2 上要么
深度遗忘、要么彻底崩塌：

| | 训练 forget NLL（首→末） | 评测 bio/mmlu（首→末） |
|---|---|---|
| interns2 GA | 1.52 → 177（仍在涨） | 0.841 → **0.237/0.251** |
| gemma GA nocap | 3.93 → **3639** | 0.795 → 0.794/0.760（不动） |
| gemma WGA/SatImp nocap | → 52.9 / 62.2 | 不动 |
| gemma GD nocap | 4.05 → 2535 | 0.792 → 0.789/0.758（不动） |

最干净的是 GD 这一行：retain 用的是 fineweb 通用文本，它的 NLL 从 4.91 涨到
9.20（中途 23）——**训练输入上的通用语言能力已经被打坏一半，MMLU 却只掉
1pt**。这说明被破坏的计算和评测输入走的计算基本不重叠，而且被门控的是
**输入的格式，不是内容**（retain 语料和 WMDP 毫无关系，照样打坏；照样不迁移）。

训练端优化本身没问题（NLL 推得比 Intern 高 20 倍），问题在"训练输入上的
破坏迁移不到评测输入"。

### 1.2 根因：训练缓存里一条 BOS 都没有

缓存配置照抄了 Intern 模板的 `add_bos_token: false`（interns2 的 tokenizer
本就不加 BOS，抄过来无害；Gemma 却是句首恒有 BOS 的模型）。实测抽 199 个
pack，**无一含 id 2**。于是训练与评测喂的是两个 regime 的输入：

| | 训练输入（Gemma 缓存） | 评测输入 |
|---|---|---|
| BOS | **无** | **有**（tokenizer 默认） |
| 长度 | 4096 pack | ~100-300 token |

判据来自 §2 的修复：**同一个 base，去掉 BOS 在线读 0.46/0.40，带 BOS
0.79/0.77**——无 BOS 不是轻微的分布偏移，是另一套计算。

机制解读：升向目标没有上界，Adam 会沿最便宜的方向把 loss 推上去。无 BOS 的
长 pack 上存在评测输入没有的脆弱结构（被迫充当 attention sink 的首 token/EOS、
滑窗层看不到序列头的长程位置），"拨一个局部开关"就能让整条 pack 的预测全崩、
满足目标，完全不必碰通用知识；Intern/Qwen 的训练与评测在同一个 regime，没有
这条捷径，只能破坏通用计算，破坏于是能迁移到评测。

为什么拖到 10 月才发现：门控不是绝对的。RMU（中层激活目标，不走 logit）在
同一批无 BOS 数据上正常遗忘（0.79→0.346）；PureNPO 用大步长 lr 5e-5 也能迁移
（破坏面铺开）。**小步长 + logit 级升向目标**才是最致命的组合，而这恰好是
GA 族三模型统一的 lr 1e-6 设定——早期"RMU 能动、GA 不动"被读成了算法差异。

### 1.3 修复：`bos_per_pack`，每个 pack 一个 BOS

- 数据段新字段 `packing.bos_per_pack: true`（与 `add_bos_token` 互斥）：每条
  4096 pack 头部前插一个 `<bos>`（id=2），原末 token 让位、pack 长度不变。
- 为什么不能只把 `add_bos_token` 改 true：见 §0——文档级 BOS 覆盖不到约 2/3
  的中段续块 pack。
- 验证方式：逐 pack 对比新旧缓存，**新缓存 = 旧缓存右移一位 + 头部 BOS**，
  其余 token 完全一致——修复对数据只改这一个 token，消融才干净。
- BOS 缓存清单（`configs/data_processing/*_bos.yaml`，同源同 seed 重建）：
  `gemma4_wmdp_bio_softpack4096_bos/forget`、
  `gemma4_wmdp_cyber_softpack4096_bos/forget`、
  `fineweb_edu_5b_gemma4_softpack4096_bos_seed42`。

### 1.4 定案：BOS 消融（10-03）+ cyber 语料外推

`bos-ablation/` 五臂：nocap 模型 + BOS 缓存，**其余字段与冻结基线逐字段
一致**——pack 仍是 4096 长、算法超参不变，唯一变量就是 BOS。结果（无 BOS
基线全线钉死 0.78-0.80）：

| 算法 | step 300 | 终态 2133 | 判读 |
|---|---|---|---|
| **GD** | 0.578/0.755 | **0.273/0.725** | ✅ **成立**：单调下坠，margin 0.452 |
| GA | 0.247/0.248 | 0.270/0.238 | 全崩（MMLU 同崩；gemma4 上 GA 不可用） |
| SatImp | 0.577/0.596 | 0.661/0.609 | step300 过冲后反扑 + MMLU 侵蚀 ~16pt |
| WGA | 0.588/0.626 | 0.654/0.639 | 同上 |
| CEU | 0.781/0.759 | 0.782/0.760 | 仍冻——CEU 自身另有 fp32 归一化 bug（10-03 已改 log 域修复，另案） |

一个 token 的差别：GD 从"全程 0.79 不动"变成"300 步内 0.578、2133 步
0.273/0.725"——与 interns2 同配置的形态对上了。

跨语料外推（cyber，10-03 R1，`main/main-cyber/gemma4/`）：GD
0.499→**0.259/0.721**（MMLU 保持），PDU 六臂全活且 ε 单调分化（最优
eps2.60@207 = 0.249/0.759）。BOS 门控结论在 cyber 语料上同样成立，logit 级
目标在 gemma4 上的迁移自此打通。

---

## 2. 在线测评缺 BOS

### 2.1 症状

- 未训练的 base 全量在线读数 **wmdp_bio 46.19% / mmlu 40.25%**，离线 HF 真值
  **79.26% / 74.24%**——没做任何遗忘却掉 33pt ⇒ 在线前向对 Gemma 算错；
- 四个 RMU 跑（lr/mbs/full-expert 各不同）在线读数几乎一样（≈0.43/0.39）、
  对超参完全不敏感——是管线地板噪声，不是遗忘效果。

### 2.2 根因

`src/megatron_unlearning/online_eval.py::split_context_continuation` 两处 encode
写死 `add_special_tokens=False`，而 Gemma tokenizer 默认在句首加 `<bos>`（id=2）：

```
encode(ctx)                          -> [2, 818, 2269, ...]   # 默认带 BOS
encode(ctx, add_special_tokens=False) -> [818, 2269, ...]    # 在线路径，丢了 BOS
```

也就是把"每条序列开头那一个 BOS"删掉了——对 Gemma 这一个 token 值 ~27pt。

**为什么只有 Gemma 炸**：Intern-S2 / Qwen3.5 的 tokenizer 默认不加 BOS，同一条
`add_special_tokens=False` 路径对它们无害。这正是此前只在 Gemma4 上出问题、
Intern 在线↔离线 parity 一直过关的原因。

### 2.3 修复与验证

去掉 override，改用 tokenizer 默认（与离线 lm_eval 的 `HFLM._encode_pair`
同口径）。BOS 同时进 whole/context 两侧、continuation 切片不变，所以修复
真的只是"留住了那一个 token"。八卡 H200 实测，**同 limit=100 头部切片**的
apples-to-apples 对照：

| 修前（在线） | 修后（在线） | 离线 HFLM（同 limit） |
|---|---|---|
| 0.41 | **0.68** | **0.67** |

（全量 1273 题的 0.79 与 limit=100 切片不可比，正确对照是同 limit 的 0.67。）
单测 46 个全过，含 `test_split_uses_tokenizer_default_bos` 守回归；BOS-bug
时代旧 jsonl 归档至 `online-evaluation/_broken_bos_bug/`。

这个修复反过来成为 §1 的判据：无 BOS 的 Gemma 是另一套计算，训练缓存无 BOS
的问题由此锁定。

---

## 3. final_logit_softcapping（cap 机制）

### 3.1 机制

gemma 的 config.json（text_config）带 `final_logit_softcapping: 30.0`
（intern/qwen35 均无）。训练前向输出层做 `ℓ = 30·tanh(x/30)`，logit 值域压死
在 (−30, +30)。两条数学后果：

- **NLL 有顶**：gold 压到 −30、单个垃圾 token 顶到 +30 时 NLL → ≈60
  nats/token（严格上界 60+ln(V−1)≈72.5，需其余 26 万 token 全部顶格；实测
  平台 59.99 对应"单 token 塌缩"态）；
- **饱和区梯度死亡**：`dℓ/dx = sech²(x/30)`，logit −29.9 时梯度衰减 ~150×，
  −29.99 时 ~1500×。

### 3.2 测评侧：选择性打分漏 cap（09-20 已修，~0.5pt）

为省显存，在线选择性打分（`_score_batch`）设 `post_process=False` 提前返回、
手工 `F.linear` 打 LM head，**绕开了** MCore `_forward_output_layer` 里的
softcap。已在 `reduce_selected_logprobs` 补回同样的 tanh 封顶；`_score_batch`
用 `getattr(gpt, "final_logit_softcapping", None)` 从活模型取值（非 Gemma /
nocap 模型 → None → 恒等，向后兼容且口径自动跟随模型）。量级 ~0.5pt，是次因；
回归单测 `test_selective_matches_full_ce_with_final_logit_softcapping` 守住。

### 3.3 训练侧：cap 真实改变动力学，但不是零迁移根因

GA 在 capped 模型上 forget NLL 1.66 → 59.99 钉死（=2×30 渐近线）、grad_norm
465 → 0.02、评测全程平线——09-30 曾据此把 cap 判为零迁移根因
（`GA_family_softcap_analysis_20260930.md`）。10-02 复盘推翻：**关掉 cap 后
训练破坏放大 20 倍（NLL → 3639，Intern 才 177），评测依然零动**（§1.1 表）。
卡住的是迁移不是优化；cap 决定的只是 loss 曲线形状（撞顶 + 梯度死）。

### 3.4 处置：nocap 影子目录（模型路径即开关）

- 影子目录 `.../wangxiao1/checkpoints/gemma/gemma-26b-a4b-nocap`：**权重全符号
  链接**，仅 config.json 置 `final_logit_softcapping: null`。
- `model.path` 指向影子目录即自动关 cap：`model_config.py` 读 HF config →
  bridge 赋 None → 跳过 tanh。零代码改动（bridge 里 softcap 仅 4 行、无
  `or 30` 回退）；在线测评从活模型取值自动同口径；ckpt 导出逐字复制该
  config → 下游 lm-eval 自动 uncapped；参考模型同源同口径。
- **base 评测零损耗实测**（同一份权重仅改 config；MCQ 的 argmax 对逐元素
  单调变换严格不变，所以选择题读数理论上就不该动）：

| task | capped | nocap | Δ |
|---|---|---|---|
| wmdp_bio | 0.7918 | 0.7879 | −0.39pp（±1.1pp） |
| wmdp_cyber | 0.4962 | 0.4982 | +0.20pp |
| wmdp_chem | 0.5735 | 0.5784 | +0.49pp |
| mmlu | 0.7441 | 0.7449 | +0.08pp（±0.34pp） |

全部 |Δ|≤0.5pp、≤0.2σ。注意：**nocap 是另一个（失校准的）模型**——step-1
forget NLL 1.66→3.93、retain 2.28→4.91，nocap 结果与 capped 不可直接互比；
旧 capped run 的续测不要指影子目录。当前 gemma4 终线统一 nocap+BOS。

---

## 4. 总结

| # | 问题 | 严重度 | 位置 | 修复 | 验证 |
|---|---|---|---|---|---|
| 1 | 训练缓存无 BOS → 升向族零迁移、早期结果整体作废 | 最重 | 数据处理配置 + 缓存 builder | `bos_per_pack: true` 逐 pack 前插（每 pack 一个），重建三个缓存 | GD 0.79→0.273/0.725（bio）、0.499→0.259/0.721（cyber） |
| 2 | 在线 encode 丢 BOS → 读数低 ~27pt | 中 | `online_eval.py::split_context_continuation` | 去掉 `add_special_tokens=False`，用 tokenizer 默认 | 在线 0.68 ≈ 离线 0.67（同切片） |
| 3 | 选择性打分漏 softcap ~0.5pt；训练侧 cap 改变升向动力学 | 轻（测评）/ 已绕开（训练） | `online_eval.py` 打分路径；nocap 影子模型 | 打分补 tanh cap；`model.path` 指影子目录关 cap | softcap parity 单测；base 四任务零损耗表 |

**当前终线口径**：nocap 影子模型 + BOS 缓存（即 §1.4 消融验证过的组合）。
bio 终线 `configs-final/`（9 算法、存 final ckpt）与 cyber R2 存点轮均已提交
（10-03），跑完后以离线 lm_eval 复核。

**相关文件**：`bos-ablation/`（§1.4 数据）·
`configs/training/megatron/gemma4/README_online_eval_bos_fix.md`（§2 详版）·
`GA_family_nocap_rootcause_20261002.md`（§1 根因复盘）·
`GA_family_softcap_analysis_20260930.md`（softcap 首判，结论已被 §3.3 修正）
