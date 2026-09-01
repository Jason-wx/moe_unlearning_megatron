# 文档索引

使用仓库请先阅读根目录的 [RUNBOOK](../RUNBOOK.md)。训练配置和算法能力见
[configs/README.md](../configs/README.md)。

本目录只保留长期维护的工程文档：

- [experiment-progress.md](experiment-progress.md)：正式实验的完成情况和结果入口。
- [architecture.md](architecture.md)：双后端边界、稳定入口和代码分层。
- [data-processing.md](data-processing.md)：两套离线 token cache 的格式与语义。
- [runtime-layout.md](runtime-layout.md)：环境、cache、checkpoint、日志和评测目录。
- [hyperparameter-sources.md](hyperparameter-sources.md)：Intern-S2 正式配置的参数来源。
- [contributing.md](contributing.md)：修改代码、算法和配置的贡献流程。
- [links.md](links.md)：算法、评测和相关项目的参考资料。

原始实验结果不放在 `docs/`：

- [Intern-S2 遗忘结果](../evaluations/unlearning/interns2/RESULTS_SUMMARY.md)
- [RMU 多模型结果](../evaluations/results/rmu/README.md)
