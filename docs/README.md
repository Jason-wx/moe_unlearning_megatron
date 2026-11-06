# 文档索引

使用仓库请先阅读根目录的 [RUNBOOK](../RUNBOOK.md)：构造 data cache、运行算法和评测。

本目录保存长期维护的工程文档：

- [architecture.md](architecture.md)：双后端边界、稳定入口和代码分层。
- [data-processing.md](data-processing.md)：两套离线 token cache 的格式与语义
  （与 [configs/data_processing/README.md](../configs/data_processing/README.md) 内容相同）。
- [runtime-layout.md](runtime-layout.md)：环境、cache、checkpoint、日志和评测目录。
- [hyperparameter-sources.md](hyperparameter-sources.md)：早期 Intern-S2 V3 配置的超参数来源。
- [contributing.md](contributing.md)：修改代码、算法和配置的流程。
- [links.md](links.md)：算法、评测和相关项目的参考资料。

最终实验的训练配置在
[configs/training/megatron/final-training-configs/](../configs/training/megatron/final-training-configs/)，
评测结果见 [evaluations/README.md](../evaluations/README.md)。
