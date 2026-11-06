# Gemma4 GA 族失效复盘（关 softcap 之后）— 2026-10-02

> 这份是对 `GA_family_softcap_analysis_20260930.md` 的复核。数据来源：`main/{interns2,qwen35,gemma4}/logs/*.log`、
> `online-evaluation/**/*.jsonl`、Gemma 训练缓存抽样，以及 CPU 单测。§5 的判别实验还没跑。

## TL;DR

1. **根因不是 softcap。** 带 cap 的 GA 在训练输入上已经**完全崩了**：NLL 停在 ≈60，对应的是单 token 塌缩，并不是 cap 的上限 72.5。
   关 cap 之后训练 NLL 涨到 3639，是 Intern 的 20 倍。两种情况下评测都基本没动。
   卡住的是"训练输入上的破坏能不能迁移到评测输入"，优化本身没有问题。
2. **首要嫌疑：训练和评测输入不在同一个 regime。** Gemma 的缓存全部**没有 BOS**（配置照抄了 Intern 模板的 `add_bos_token: false`），
   pack 长 4096，其中 75% 的 token 落在 1024 滑窗之外。评测输入则是 BOS 加上 100–300 个 token。
   Gemma 训练时句首恒有 BOS，基座去掉 BOS 后 MCQ 只剩 0.46/0.40，说明无 BOS 走的是另一套计算。
   Intern/Qwen 的分词器本来就不加 BOS，训练和评测是同一个 regime，所以同样的配置能迁移过去。
3. **实现 bug：CEU 的 fp32 归一化。** 三个模型都受影响，已用数值单测复现（§4.1）。
   另外还有几处问题：nocap 实验本身有混杂；滑窗超过 1024 的部分从没做过 parity；Gemma 的梯度量级异常。
4. **判别实验（§5）**：一个 1 卡、约 1 小时的只读 HF 探针，就能区分 BOS 门控和长度门控。

## 1. 核心证据：训练端"成功"了，评测端零迁移

| 模型 / 算法 | 训练端 forget NLL（s1→末） | 评测 bio/mmlu（s1→末） |
|---|---|---|
| Intern GA | 1.52 → 177 | 0.841/0.864 → **0.237/0.251**（s300 已经到 0.348/0.468） |
| Qwen GA | 1.52 → 215 | 0.833/0.834 → **0.247/0.262**（s700 0.327/0.491） |
| Gemma GA capped | 1.66 → 59.9999 | 0.794/0.766 → 0.783/0.759 |
| Gemma GA nocap | 3.93 → **3639** | 0.795/0.767 → 0.794/0.760 |
| Intern GD | 1.52 → 170（retain 2.10→1.86） | 0.841/0.870 → 0.711/0.850（最低 0.589@700） |
| Gemma GD nocap | 4.05 → 2535（**retain 4.91→9.20**，s100 时 23.2） | 0.792/0.769 → 0.789/0.758 |
| Intern WGA / SatImp | 1.52 → 9.5 | 0.840 → 0.737 / 0.844 → 0.738 |
| Gemma WGA / SatImp nocap | 3.9 → 52.9 / 62.2 | 0.789 → 0.785 / 0.791 → 0.791 |

- **训练端破坏比 Intern 大一个数量级以上，评测端却是 0。**
  SatImp/WGA 的对比最干净：Intern 只把 NLL 推到 9.5，评测就掉了 10pt；Gemma 推到 53–62（每个 token 的概率约 e⁻⁶⁰），评测还是 0。
- **GD nocap 最能说明问题。** retain 是 fineweb 通用文本，它的 NLL 从 4.9 涨到 9.2，中途一度到 23。
  也就是说，训练输入上的通用语言能力被打坏了一半，而 MMLU 只掉了 1pt。
  这说明被破坏的计算和评测输入走的计算基本不重叠，而且门控和 forget 领域无关：**门控的是输入格式，不是内容**。
- **冻结约束。** LM head、所有 norm 和 embedding 都是冻结的。NLL 要到几千，最终 hidden 必须几乎完全对齐到某个垃圾方向；
  如果评测输入上也发生了同样的事，MCQ 必然掉到 0.25。所以这次塌缩一定由某个"训练输入有、评测输入没有"的特征控制。

## 2. 嫌疑为什么落在 BOS 和长上下文上

| | 训练输入（Gemma 缓存） | 评测输入 |
|---|---|---|
| BOS | **无**（抽 199 个 pack，都不含 id 2；首 token 102/199 是 `Introduction`） | **有**（分词器默认） |
| 长度 | 4096 pack | ~100–300 |
| 滑窗层能否看到序列头 | 75% 的 token 看不到（位置 >1024） | 都看得到 |
| 对照：Intern/Qwen | 无 BOS | 无 BOS，同一 regime |

- **无 BOS 是另一套计算，不是轻微的分布偏移。** 基座去掉 BOS 后在线读数是 0.46/0.40，带 BOS 是 0.79/0.77（见 `README_online_eval_bos_fix.md`）。
- **机制猜想。** GA 的目标没有上界，Adam 会沿最便宜的方向把 loss 往上推。
  Gemma 的训练输入里有一个评测输入里没有的脆弱结构：无 BOS 时被迫充当 attention sink 的首 token 或 EOS，以及滑窗之外的长程位置。
  只要打坏它，整条 pack 的预测就全崩了。这相当于拨一个局部开关就满足了目标，不用碰通用知识。
  Intern/Qwen 的训练和评测在同一个 regime，没有这条捷径，只能去破坏通用计算，于是破坏能迁移到评测。
- **旁证：RMU 诊断里，chunk1024 的迁移是 softpack4096 的 2.6 倍**（−11.0 vs −4.2pt），指向的也是"训练输入的 regime 离评测太远"。
  那份诊断说"BOS 不是变量"，但只对 RMU（中层激活目标）验证过，不能外推到 logit 级目标。
- **反例与边界：门控不是绝对的。** 同样的无 BOS 数据上，lr 5e-5 的 PureNPO 能迁移，最低 0.521（MMLU 也掉到 0.694，破坏面广）。
  而 lr1e-6 的 NPO/SimNPO 臂在 Gemma 上不动（0.786/0.785）。所以更准确的说法是：
  **小步长时，优化器会优先走最便宜的方向，而在 Gemma 上这个方向恰好只作用于训练 regime**；大步长时破坏面会铺开，也就能迁移了。
  GA 族在三个模型上都是 lr 1e-6，Gemma 是唯一一个有这条捷径的。
- **尚未排除：长度/滑窗门控。** 它和 BOS 缺失叠加在一起，要靠 §5 的探针来分开。
  还有一个更糟的可能：MCore 在超过 1024 的长度上和 HF 不等价（见 §4.3）。
  那样的话，模型学会破坏的是一条只存在于训练前向里的计算路径。

## 3. 对 09-30 文档的修正

1. **"NLL 上限 60"不对。** 每个 token 的真实上界是 60+ln(V−1)≈**72.48**。
   ≈60 的平台对应的状态是：label 在 −30，**恰好一个**垃圾 token 在 +30，其余都在 −30。
   也就是训练输入上已经塌缩成单 token 输出。可见 capped GA **已经把训练输入彻底打崩**；
   梯度死亡是"到达退化最优点"的结果，不是失败的原因。cap 只决定了 loss 曲线的形状。
2. **"base 关 cap 零损耗，所以 nocap 是干净对照"不成立。**
   - MCQ 的 argmax 对任何严格单调变换都不变，所以那个探针在构造上就测不出差异。
   - 实际上 nocap 会让训练输入的 step-1 NLL 明显变化：forget 从 1.66 变成 3.93，retain 从 2.28 变成 4.91。
     这说明 it 模型的原始 logit 经常远超 ±30，依赖 cap 来校准。
   - 结论：nocap 线是另一个（失校准的）模型，结论有混杂。
3. **SatImp/WGA 冻结不是 cap 造成的。** nocap 下它们照样冻结：NLL 在 50–62，权重 p^β 在 1e-17 到 1e-20。
   这是算法自身的饱和，Intern/Qwen 后期梯度约 1e-4，也是同一现象，只是 Gemma 上来得更极端。
4. **TRACE "同根因已证实"也要撤回。** 两个 lr 臂的 NLL 都钉在 60.00，按第 1 条的逻辑，同样是训练输入已塌缩、只是没迁移。
5. **PDU "ε 需向上扫"应等修完 BOS 后再判。** PDU 同样是 logit 级目标，可能受同一个迁移问题影响。

## 4. 实现审计

### 4.1 【确认 bug】CEU 的 fp32 归一化（三个模型都受影响）

问题出在 `src/megatron_unlearning/objectives.py:278`：

```
target = target / (1.0 - correct_probability).clamp_min(1e-12)
```

当 gold token 领先其余 token 超过约 17 nats 时，fp32 下 p_gold 会舍入成 1.0，于是 `1−p` 变成 0，分母被 clamp 到 1e-12。
后果是：`target_mass` 不再等于 1，最多会被放大 1e12 倍乘上非 label 质量。
HF 参考实现（`src/trainer/unlearn/ceu.py:46-54`）是先把 label 填成 −inf 再做 softmax，没有这个问题。

CPU 单测，单行，V=262144，和 HF 对比梯度范数：

| gold 领先（nats） | 0 | 5 | 10 | 15 | 20 | 25 | 35 | 60 |
|---|---|---|---|---|---|---|---|---|
| \|g_ours\|/\|g_HF\| | 1 | 1 | 1.51 | **6.4e3** | 27 | 0.26 | 1e-5 | 1e-16 |

- **含义：** 中高置信 token 的梯度被放大到最多约 6000 倍，极高置信 token 的梯度则被抹掉。
  clip 1.0 之后，前约 100 步的更新方向几乎全由少数 token 决定。
- **日志吻合：**
  - step 5 的 loss 尖峰：Intern 53.3、Qwen 11.6、Gemma capped 2780（正常值约 4）。
  - step 1–50 的 grad：Intern 1.8e3、Qwen 750、Gemma 9e4；nocap 时 1e6–1.6e7，loss 2.8 万。
- **修法：** target 改在 log 域构造，即 label 置 −inf，在其余 token 上做 softmax；loss = logZ − Σ target·shifted。
  CPU 实测和 HF 在 gap 0–60 范围内 cos=1.000001、范数比 1.0000；batch 均值 1.628440 vs 1.628440。
  **尚未改代码。** TP>1 时要用 `tests/megatron/test_tp_vocab_parallel_objectives.py` 的 gloo 测试回归。
- **对现有结果的影响：** 三个模型的 CEU 都等于"CEU + 前 100 步的随机暖启动"，修完应重跑。
  这个 bug 不能解释 Gemma 不迁移的问题，因为别的算法也一样不迁移。

### 4.2 【配置缺陷，大概率是主因】Gemma 训练缓存缺 BOS

- `configs/data_processing/gemma4_{fineweb_edu_5b,wmdp_bio_forget}_xtuner_softpack.yaml:25` 写的是 `add_bos_token: false`，从 Intern 模板照抄而来。
  两个缓存实测都没有 id 2。
- 已核实：改成 `true` 后，builder 的路径（文本前拼 `<bos>`，再以 `add_special_tokens=False` encode）只会得到**单个** BOS，即 `[2, …, 1]`，不会出现双 BOS。
- **单改这一项还不够。** forget 缓存平均每个 pack 只有 0.31 个 EOS，说明约 2/3 的 pack 是长文档的中段续块；
  文档级 BOS 只能覆盖文档开头那一块。要让每条训练序列都以 BOS 开头，得在 collate 里逐序列前插 BOS（插 id 2，去掉末 token），或者改用逐 chunk 加 BOS 的切法。

### 4.3 【未验证】序列长度超过 1024 时 MCore 与 HF 的 parity

- 滑窗的链路是：bridge 给出 `window_size=(1023,0)`，`window_attn_skip_freq` 按层下发 → TE 的 `window_size`。这条链路读代码看是对的。
  bridge 自带的手工 mask 分支（`use_bidirectional_attention=='vision'`）没有走。
- **但 09-20 的 parity 只用了短 prompt，滑窗从来没生效过。**
  训练 75% 的 token 落在 1024 之外，若这部分前向和 HF 不一致，模型学会破坏的就是一条只存在于训练前向里的路径。
- **便宜的验证方法：** 用 HF 的 capped 基座，算随机抽的 forget pack（4096、无 BOS）的 NLL，跟 MCore step 1–5 的读数比。
  MCore 读数是 1.66/1.73/1.62/1.79/1.54；step 1 的 lr=0，所以 step 1 就是纯基座。

### 4.4 其他（次要）

- **nocap 线有混杂（见 §3.2）。** 建议主线回到 capped 原模型。nocap 改变的是模型本身，结论不再是关于 Gemma4 的。
- **Gemma 梯度量级是 Intern/Qwen 的 30–570 倍**（GA step 1：466 vs 0.82），每步都被 clip。
  Adam 近似尺度不变，所以这不是原因，但说明梯度可能被少数离群 token 或张量主导。缺按参数分组的 grad-norm 日志，建议补上。
- **fineweb 的 pack 每条约 4.5 个文档，只用 EOS 隔开，attention 跨文档可见。** 三个模型都这样，不是差异项。
- **流水线本身没问题。** 同一条 Gemma 在线评测管线上，RMU、NPO 都能把读数拉下来，说明评测确实读到了更新后的权重，不存在读旧权重的问题。

## 5. 判别实验（需确认后提交）

**E1：HF 只读探针。** 1 卡 H200，约 1 小时。ckpt 是 root 0600 权限，只能在 rjob 容器里跑。

- 模型：nocap GA 的 ckpt-2133；对照组为 nocap 基座和 capped 基座。
- 输入：64 个 forget pack，4 种条件：{原样 4096 无 BOS, 前插 BOS} × {全长, 截前 256}；另加 WMDP 题面。
- 输出：按位置分桶的 NLL，桶为 [0,64)、[64,256)、[256,1024)、[1024,4096)。

判读：

| 观测 | 结论 |
|---|---|
| 无 BOS 短序列 NLL 爆，加 BOS 后恢复正常 | **BOS 门控** |
| 短序列都正常，只有超过 1024 的段爆 | **长度/滑窗门控** |
| capped 基座的 HF NLL 明显 ≠ 1.66 | **MCore 长序列 parity bug**（§4.3） |

**E2：capped GA 的 2×2 小跑。** 每臂 8 卡，约 3.5 s/step × 300 步 ≈ 25 分钟，在线评测点放在 100/200/300。

- 两个因子：{无 BOS, 逐序列 BOS} × {softpack4096, chunk1024}。
- 预期：如果 BOS 是门控，BOS 臂在 300 步内应该像 Intern 一样崩到 0.35 左右。
- 需要一个小改动：data 段加一个 `prepend_bos` 开关，collate 里插 id 2。

E1 决定 E2 先跑哪一臂。两个都是 gemma4 线的 GPU 作业，不在自主授权范围内，需确认后再提交。

