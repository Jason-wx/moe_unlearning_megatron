# 评测结果

本目录只保存正式汇总和对应的 `lm-eval` 原始 JSON。
运行与评测方法见 [Runbook](../RUNBOOK.md)。

```text
results/baselines/                 Gemma 4、Intern-S2、Qwen3.5 基础模型结果
results/rmu/                       三个模型的 RMU 结果与汇总
unlearning/interns2/               Intern-S2 各轮遗忘实验
```

- 基础模型性能见 `results/baselines/`。
- RMU 多模型对比见 `results/rmu/README.md`。
- Intern-S2 方法对比见 `unlearning/interns2/RESULTS_SUMMARY.md`。
- 新评测目录统一使用 `lm_eval_results_<experiment>_<tasks>/result.json`。

训练日志、临时评测和仅用于 checkpoint roundtrip 的结果不放在这里。
