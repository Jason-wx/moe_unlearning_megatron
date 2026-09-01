# 数据处理与离线 Token Cache

使用命令见根目录 [RUNBOOK](../RUNBOOK.md)。本文只说明两套 cache 的格式和训练语义。

## Document chunk

document_chunk_v1 对每篇文档单独 tokenize，再在文档内部按 max length 和 stride
滑窗切块，不会把不同文档拼到同一个样本中。默认保留最后一个不足长度的 chunk。

构建入口：

~~~text
scripts/build_cache.sh
src/data/build_document_chunk_cache.py
~~~

支持 Parquet 和 JSONL。主要参数是 tokenizer、输入文件、文本字段、max length、stride、
worker 数和输出目录。

输出文件：

| 文件 | 内容 |
| --- | --- |
| meta.json | tokenizer、数据源、切分参数和样本统计 |
| summary.json | chunk 数和 token 数 |
| chunks.jsonl | 每个 chunk 的 token offset、长度和原文档索引 |
| token_ids.uint32.bin | 连续存储的 token ids |

## XTuner soft-pack

xtuner_softpack_v1 用于复现 XTuner 的纯文本 soft packing。它按 seed 固定文档顺序，
把短文档组合成定长 pack，并截断超过 pack max length 的单篇长文档。

构建入口和配置：

~~~text
scripts/build_xtuner_pack_cache.sh
src/data/build_xtuner_packed_cache.py
configs/data_processing/
~~~

配置分为 tokenizer、dataset、packing、build 和 output 五部分。当前支持纯文本 Parquet
和 JSONL，不支持图片或视频。

输出文件：

| 文件 | 内容 |
| --- | --- |
| meta.json | tokenizer hash、数据源、seed 和 packing 参数 |
| summary.json | pack 数、有效 token 数和 packing efficiency |
| chunks.jsonl | pack offset、长度、源文档索引和文档起点 |
| token_ids.uint32.bin | padding 后的 pack token ids |
| labels.int32.bin | 文档边界与 padding 对应的 -100 mask |
| attention_mask.uint8.bin | 区分有效 token 和 padding 的 mask |

正式 Intern-S2 Megatron 配置使用 4096-token XTuner soft-pack，并设置
data.shuffle: false，因为顺序已经由 cache seed 固定。

## 训练读取

两套格式都由 src/data/cached_pretraining.py 中的 CachedDocumentChunkDataset 读取。
训练 YAML 通过下面两个字段决定 cache 的角色：

~~~yaml
data:
  forget_cache: /path/to/forget-cache
  retain_cache: /path/to/retain-cache
~~~

forget cache 是训练锚点；只有需要 retain loss 的算法才读取 retain cache。cache 与
tokenizer、原始数据和切分或 packing 参数绑定，这些输入改变时应构建新的 cache。
