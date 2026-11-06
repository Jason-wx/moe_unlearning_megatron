# Gemma4 升向算法(GA 族)集体失效根因分析 — final_logit_softcapping

日期:2026-09-30 · 结论:**不是超参迁移问题,是模型输出端 softcap 的机制性梯度绞杀**

## 现象

main/gemma4 十二配置(=interns2 主线逐字段迁移)中,所有以"抬升 forget NLL"为
驱动力的算法在 gemma-26b-a4b 上全部冻结,而 interns2 上同配置要么深度遗忘、要么
彻底崩塌:

| 算法 | gemma4 (step1→2133) wmdp/mmlu | interns2 终态 |
|---|---|---|
| GA | 0.794/0.766 → 0.783/0.759 | 0.841/0.864 → 0.237/0.251(崩塌) |
| SatImp | 0.789/0.768 → 0.793/0.757 | 深度遗忘 |
| WGA | 0.790/0.765 → 0.788/0.757 | 深度遗忘 |
| GradDiff | 0.792/0.769 → 0.783/0.759 | 深度遗忘 |
| CEU | 0.791/0.764 → 0.789/0.759 | — |
| **RMU(不走 logit)** | 0.791/0.763 → **0.346**/0.726 | 0.370/0.842(有效) |
| PureNPO 5e-5 | 0.793/0.769 → 0.653/0.718 | 有效 |

## 根因

**gemma-26b-a4b 的 config.json 带 `final_logit_softcapping: 30.0`**
(`checkpoints/gemma/gemma-26b-a4b/config.json` text_config;intern/qwen35 均无此
字段)。训练前向的输出层做 `ℓ = 30·tanh(x/30)`,logit 值域被压死在 (-30, +30)。

两条数学后果:

1. **NLL 有硬上限 ≈ 2×30 = 60 nats/token**(目标 token 压到 -30、其余质量顶到
   +30 时取到)。tanh 渐近不可达,故只能无限逼近 60。
2. **饱和区梯度死亡**:`dℓ/dx = sech²(x/30)`。capped logit 到 -29.9 时梯度衰减
   ~150×,到 -29.99 时 ~1500×。

## 证据链(全部实测)

1. **GA 的 forget_nll 精确撞上天花板**:1.66 → step≈150 即 59.89 → 此后 2000
   步钉死在 59.985~59.9999(= 2×30 的渐近线)。**目标函数已达全局最大值,梯度
   归零,再多的 epoch/lr 都无济于事**(step 300→2133 的 eval 全程平线)。
   对照 interns2:forget_nll 同样从 1.5 起步,但无 cap,一路涨到 **176 nats**
   @2142 步仍在涨 —— 崩塌正是 60→176 这段"超限破坏"造成的。
2. **grad_norm 塌缩印证梯度死亡**:gemma4 GA 465(step1)→ 14.7(151)→
   0.02~1(后期);SatImp 更是掉到 **3e-4~1e-3**。interns2 GA 后期仍有 12~19,
   持续重塑全模型 → MMLU 0.86→0.25。gemma4 梯度死后权重基本不动 → MMLU 只
   -0.7pp。
3. **绕开 logit 的算法不受影响**:RMU(激活空间 MSE)0.791→0.346,正常;
   NPO 族(有界 sigmoid 目标)5e-5 下能动到 0.65 后自限回弹——同样是"cap +
   有界损失"的自限味道,但不像纯 ascent 那样一步死绝。
4. 代码侧 `src/megatron_unlearning/online_eval.py` 注释已明确训练前向在
   `_forward_output_layer`(post_process=True)内应用该 softcap,在线测评的手工
   LM head 为 parity 也复刻了同一变换——训练/评测口径一致,不存在评测错配。

## TRACE(Safe_MoE/TRACE 仓库)同一根因(09-30 追加)

TRACE 的目标函数 `src/trace_unlearning/losses.py:121` 是
`forget_objective = -forget_nll`——forget-only(λ_r=0)时**数学上就是限定在
selected experts 上的 GA**,纯 logit 域上升,同样被 softcap 绞杀:

| | intern-S2 TRACE | gemma4 TRACE |
|---|---|---|
| forget_nll 轨迹 | 1.52 → 80 → 124 → **~145**(无顶,持续爬) | 1.65 → 55 → **60.00 钉死**(=2×30) |
| 撞顶后 grad_norm | 维持 11~130(活着) | 塌到 0.01~7(死亡) |
| wmdp_bio | 0.84 → **0.286**(MMLU 仅 -2pp) | 0.79 → 0.75-0.77(两 lr 臂同终点) |
| lr 敏感性 | — | 5 倍 lr(1e-6 vs 5e-6)无效:天花板是目标函数全局最大值,lr 只改变撞墙速度 |

关键量化:intern 完成外科遗忘所需的 NLL 位移是 **~80-145 nats**,gemma4 的
可表达上限是 **60 nats** 且梯度在顶上归零——不是"差一点",是结构性不可达。
TRACE eval 明细:`evaluation/Gemma4-26B-A4B/gemma4_..._lr{1e-6,5e-6}_ep3_.../
checkpoint-{711,1422,2133}`(711→2133 全程冻结,无 intern 上观察到的中期最优点
或后期反弹,因为根本没有可用的优化压力)。横截面佐证:TRACE 跑过的另两个
模型 DeepSeek-V2-Lite、Qwen3.5(均无 softcap)都能遗忘,gemma4 是唯一带
`final_logit_softcapping` 的,也是唯一死的。

注:softcap 是 TRACE 在 gemma4 失效的主因,但非唯一结构差异(gemma4 128
experts top-8 vs intern 256 experts top-8,selection 集合不同);不过 nll 精确
撞 60.00 的直接证据说明优化动态被 cap 主导,调 lr/epoch/专家数都无法绕开。

## 与 qwen35「遗忘墙」的区别

qwen3p5-35b-a3b config **无** softcapping。qwen35 升向族停在 0.79-0.81 是优化
几何/表征层面的墙(MMLU 还在付代价);gemma4 这次是输出非线性的**机制性梯度
死亡**(连 MMLU 都不用付)。两者数值巧合地都在 ~0.79,但成因不同,勿混为一谈。

## 影响与建议

- gemma4 上 GA/SatImp/WGA/GD/CEU 这批结果**测的是 softcap,不是算法**,不能与
  intern/qwen35 横向对比;重跑/加 epoch 无意义(目标函数已在全局最大值)。
- 若要在 gemma4 上做有效遗忘,路线只有:(a) 有界目标类——RMU(已验证 0.346/
  0.726)、NPO 族加压(lr/β)、PDU 加大 ε(本次 ε 网格 2.0-2.5 全没动,唯一
  3.0 动到 0.639,说明网格偏低,应向上扫);(b) 或训练时摘掉 softcap——但这
  改变了模型输出函数,ckpt 与 capped 评测口径失配,需慎重定义 parity。
- PDU ε 语义在 cap 下基本不变(2~5 nats 处 tanh 近线性),锚点 2.0945 仍有效。

## 探针:base 模型关掉 softcap 直接测,掉不掉?(09-30 实测,结论=不掉)

担心点:基模预训练时开着 cap,若测评时关掉,logit 值域改变可能让性能塌掉。
lm-eval(HF,accelerate 2 卡 H200,auto batch≈25-28)对**同一份 jincheng 权重**仅改
config(`final_logit_softcapping: 30.0 → null`,影子目录
`wangxiao1/checkpoints/gemma/gemma-26b-a4b-nocap`,权重全符号链接):

| task | capped(09-20,jincheng) | nocap(09-30,影子) | Δ | stderr |
|---|---|---|---|---|
| wmdp_bio | 0.7918 | 0.7879 | **-0.39pp** | ±1.1pp |
| wmdp_cyber | 0.4962 | 0.4982 | +0.20pp | ±1.1pp |
| wmdp_chem | 0.5735 | 0.5784 | +0.49pp | ±2.4pp |
| mmlu | 0.7441 | 0.7449 | +0.08pp | ±0.34pp |

全部 |Δ| ≤ 0.5pp、≤0.2σ,与理论一致:MC 选择题的 argmax 选择对逐元素单调变换
严格不变,只有 loglikelihood 打分受二阶影响。08-04 的旧基线(zskj-hub 副本,
0.7926/0.7424)与 09-20 jincheng capped 各任务差 <1pp,佐证同一权重同一口径。

**结论:关 cap 的读出对 base 知识零损耗 → uncapped 训练线(B 口径:训练关 cap
+ 测评关 cap)可行**,且 uncapped 线的基线参照就是这份 nocap 读数
(bio 0.7879 / mmlu 0.7449)。作业:`gemma4-baseline-nocap-wmdp-mmlu-eval-31894145`
(safewt 2 卡);注意 batch=64 在 262k 词表下 OOM(单次 logits 分配 64GB),
必须 auto。结果:
`evaluations/results/baselines/lm_eval_results_gemma4_base_nocap_wmdp_mmlu`。

**训练侧同样零代码(09-30 核实)**:`model.path` 直接指向影子目录即自动关 cap——
`model_config.py:21` 的 `AutoConfig.from_pretrained(run.model.path)` 读到 null →
bridge `gemma4.py:534` 赋值 None → `:707` 跳过 tanh;权重走符号链接;在线测评
从活模型对象取 softcap 值自动同口径;导出 `copy_hf_checkpoint_metadata` 逐字复制
该 config → ckpt 下游 lm-eval 自动 uncapped;参考模型同源同口径。bridge 全包仅
4 行 softcap 代码,无 `or 30` 回退。原 env 门控方案作废,**模型路径即开关**。
新跑臂统一 `-nocap` 后缀;旧 capped run 的续测不要指影子目录。
