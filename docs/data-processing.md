# 数据处理与离线 Token Cache

使用命令见根目录 [RUNBOOK](../RUNBOOK.md)。本文只说明两套 cache 的格式和训练语义。

## Document chunk

document_chunk_v1 对每篇文档单独 tokenize，再在文档内部按 max length 和 stride
滑窗切块，不会把不同文档拼到同一个样本中。默认保留最后一个不足长度的 chunk。

构建入口：

~~~text
scripts/build_document_chunk_cache_example.sh
src/data/build_document_chunk_cache.py
configs/data_processing/document_chunk/<model>/<dataset>.yaml
~~~

配置目录与 XTuner 相同（interns2、qwen35、gemma4 × wmdp_bio_forget、wmdp_cyber_forget、
fineweb_edu_5b）。builder 只接受命令行参数，YAML 记录每个 cache 的参数，文件头给出完整命令；示例见 `scripts/build_document_chunk_cache_example.sh`。

支持 Parquet 和 JSONL。主要参数是 tokenizer、输入文件、文本字段、max length、stride、
worker 数和输出目录。

默认每个 chunk 都会加上 tokenizer 编码时的特殊 token，并计入 max length：Gemma4
每个 chunk 以 `<bos>` 开头、正文 1023 个 token；Intern-S2 与 Qwen3.5 不加特殊 token、
正文 1024 个 token。省略 stride 时取正文长度，相邻 chunk 首尾相接；显式 stride 超过
正文长度会因丢 token 而报错，因此 Gemma4 应省略 stride 或使用 1023。

旧版 builder 对 Gemma4 预留了 BOS 位置却没有写入 BOS，并在每个 chunk 边界漏掉 1 个
token。`data_cache/gemma4/` 下的 `*_max1024_stride1024` cache 由旧版构建，没有 BOS，
如需使用应重新构建。

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
scripts/build_xtuner_cache_example.sh
src/data/build_xtuner_packed_cache.py
configs/data_processing/xtuner/<model>/<dataset>.yaml
~~~

`<model>` 为 interns2、qwen35、gemma4，`<dataset>` 为 wmdp_bio_forget、
wmdp_cyber_forget、fineweb_edu_5b。配置分为 tokenizer、dataset、packing、build 和
output 五部分。当前支持纯文本 Parquet 和 JSONL，不支持图片或视频。

`packing.bos_per_pack: true` 会让每个 pack 的第一个 token 为 `<bos>`。Gemma4 的三份
配置默认开启：WMDP-Bio 文档平均约 13k tokens，仅靠文档级 `add_bos_token` 会让约
2/3 的 pack 没有 BOS，而 Gemma4 在评测时始终以 BOS 开头。

输出文件：

| 文件 | 内容 |
| --- | --- |
| meta.json | tokenizer hash、数据源、seed 和 packing 参数 |
| summary.json | pack 数、有效 token 数和 packing efficiency |
| chunks.jsonl | pack offset、长度、源文档索引和文档起点 |
| token_ids.uint32.bin | padding 后的 pack token ids |
| labels.int32.bin | 文档边界与 padding 对应的 -100 mask |
| attention_mask.uint8.bin | 区分有效 token 和 padding 的 mask |

三个模型的最终 Megatron 配置都使用 4096-token XTuner soft-pack，并设置
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
