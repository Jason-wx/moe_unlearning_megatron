# 数据处理与离线 Token Cache

本文单独记录 WMDP/RMU 训练的数据处理策略。模型适配、hook 层和可训练参数见
`docs/qwen35-moe-rmu-notes.md`。

## 背景

当前 WMDP Bio 本地 corpus 单条文本较长。用 Qwen3.5 MoE tokenizer 对本地 parquet 随机抽样 1000 条，结果如下：

| split | 总行数 | 抽样数 | token mean | token median | token p90 | token p95 | token max |
|---|---:|---:|---:|---:|---:|---:|---:|
| bio-forget | 24,453 | 1,000 | 6,965.8 | 6,457.5 | 11,448 | 13,437 | 53,129 |
| bio-retain | 60,887 | 1,000 | 7,890.9 | 7,596.0 | 14,488 | 17,640 | 42,894 |

字符和词数：

| split | char mean | char median | word mean | word median |
|---|---:|---:|---:|---:|
| bio-forget | 29,227.9 | 27,937.0 | 4,380.1 | 4,215.5 |
| bio-retain | 30,507.4 | 29,205.5 | 4,005.4 | 3,851.0 |

如果沿用官方 WMDP/RMU 的逐条样本截断，例如 `max_length=512`，平均每条样本只保留约 6%-8% 的开头 token。对长文版 WMDP corpus，这个假设不稳健。

## 在线 Chunk 策略

当前在线 dataset 使用 `DocumentChunkDataset`：

```text
每条原始文档独立 tokenizer
-> 在该文档内部按 max_length / stride 切 chunk
-> 不跨文档拼接
-> 每个 chunk 独立作为一条训练样本
-> 默认保留文档尾部不足 max_length 的 chunk
```

这比直接截断保留更多文本，也避免旧式全 corpus 拼接后跨文档切 chunk 的语义混杂。

相关配置通常在训练 config 里定义，例如：

```text
configs/data/datasets/WMDP_forget.yaml   -> DocumentChunkDataset
configs/data/datasets/WMDP_retain.yaml   -> DocumentChunkDataset
configs/qwen_moe_rmu/bio_l10_fixed.yaml  -> max_length=1024
```

`tokenize_batch_size` 只表示每次送进 tokenizer 的文档列表大小。它不是 Python 多进程，也不会让不同训练 rank 共享处理结果。分布式训练时，每个 rank 都会各自构造 dataset，所以在线 tokenize 会被重复执行多次。

## 推荐：离线 Token Cache

更稳的做法是把数据处理拆成两步：

1. 离线阶段只负责读取原始数据、tokenizer、切 chunk，并写出 cache。
2. 训练阶段只传入 cache 目录，由 `CachedDocumentChunkDataset` 直接读 token 和 chunk 索引。

这样训练启动不再重复 tokenizer，也方便不同实验复用同一份 token cache。

## 第一部分：构建 Cache

通用入口：

```bash
python scripts/build_document_chunk_cache.py \
  --model-path /path/to/tokenizer_or_model \
  --data-files '/path/to/corpus/*.parquet' \
  --dataset-path parquet \
  --split train \
  --text-key text \
  --output-dir /path/to/cache_dir \
  --max-length 1024 \
  --stride 1024 \
  --min-length 1 \
  --tokenize-batch-size 1024
```

关键输入参数：

| 参数 | 含义 |
|---|---|
| `--model-path` | tokenizer 或模型目录；cache 和 tokenizer 强绑定 |
| `--data-files` | 输入数据文件或 glob |
| `--dataset-path` | `datasets.load_dataset()` 的 path，parquet 数据用 `parquet` |
| `--text-key` | 文本列名，当前 WMDP 为 `text` |
| `--max-length` | 每个 chunk 最大 token 数 |
| `--stride` | 文档内滑窗步长；等于 `max_length` 表示不重叠 |
| `--min-length` | 过滤过短 chunk |
| `--drop-last` | 丢弃文档尾部短 chunk，默认不丢弃 |
| `--tokenize-batch-size` | 批量 tokenizer 的文档数，只影响离线构建速度和内存 |
| `--output-dir` | cache 输出目录 |
| `--overwrite` | 允许覆盖已有 cache |

Intern-S2 WMDP-Bio 已有封装脚本：

```bash
PYTHON_BIN=/home/wangxiao1/softmare/miniconda3/envs/unlearning/bin/python \
OVERWRITE=true \
bash scripts/build_interns2_wmdp_bio_chunk_cache.sh
```

默认输入和输出：

```text
MODEL_PATH=/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/jincheng/sgtm/checkpoints/Intern-S2-base05
WMDP_DATA_ROOT=/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/jincheng/data/wmdp
CACHE_ROOT=/mnt/shared-storage-user/safethm-share/wangxiao1/Safe_MoE/open-unlearning/data_cache/interns2_wmdp_bio_max1024_stride1024
FORGET_CACHE_DIR=${CACHE_ROOT}/forget
RETAIN_CACHE_DIR=${CACHE_ROOT}/retain
MAX_LENGTH=1024
CHUNK_STRIDE=1024
MIN_CHUNK_LENGTH=1
TOKENIZE_BATCH_SIZE=1024
```

输出目录内包含：

| 文件 | 内容 |
|---|---|
| `meta.json` | tokenizer、数据源和切分参数 |
| `token_ids.uint32.bin` | 所有 chunk token 连续存储的 uint32 token ids |
| `chunks.jsonl` | 每个训练样本在 token 文件中的 offset、length 和源文档信息 |
| `summary.json` | chunk 数和 token 数摘要 |

只要 tokenizer、原始数据、`max_length`、`stride`、`min_length`、`drop_last` 或 special token 设置改变，就应重新构建 cache。

## 第二部分：训练直接读取 Cache

训练时不再传原始数据处理参数，只传 cache 位置：

```bash
bash scripts/interns2_moe_rmu_bio_l10_fixed_wmdp_forget_fineweb_retain_cached.sh
```

该脚本实际覆盖的 dataset 配置是：

```text
data.forget.WMDP_forget.handler=CachedDocumentChunkDataset
data.forget.WMDP_forget.args.cache_dir=${FORGET_CACHE_DIR}
data.retain.WMDP_retain.handler=CachedDocumentChunkDataset
data.retain.WMDP_retain.args.cache_dir=${RETAIN_CACHE_DIR}
```

也可以显式指定两个目录：

```bash
FORGET_CACHE_DIR=/path/to/forget_cache \
RETAIN_CACHE_DIR=/path/to/retain_cache \
bash scripts/interns2_moe_rmu_bio_l10_fixed_wmdp_forget_fineweb_retain_cached.sh
```

## 速度注意点

离线构建 cache 仍然是单进程 Python 调度，但 heavy tokenizer 调用会走 fast tokenizer/Rayon。可以通过这些环境变量提高离线阶段利用率：

```bash
TOKENIZERS_PARALLELISM=true \
RAYON_NUM_THREADS=32 \
PYTHON_BIN=/home/wangxiao1/softmare/miniconda3/envs/unlearning/bin/python \
OVERWRITE=true \
bash scripts/build_interns2_wmdp_bio_chunk_cache.sh
```

如果 CPU 和内存资源很充足，优先增大离线阶段的 `TOKENIZE_BATCH_SIZE` 和 `RAYON_NUM_THREADS`。训练阶段不建议再做在线 tokenizer 优化，因为训练进程只应读取已经固定的 token cache。
