#!/usr/bin/env bash
set -euo pipefail

# Example: build one document-chunk cache (Qwen3.5, WMDP-Bio forget).
# The parameters are those of
# configs/data_processing/document_chunk/qwen35/wmdp_bio_forget.yaml; to build
# another cache, copy the flags from the header of another file in that folder.
#
#   bash scripts/build_document_chunk_cache_example.sh
#
# The builder refuses to replace an existing cache; add --overwrite to rebuild.

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${OU_PYTHON:-${REPO_ROOT}/envs/open-unlearning/bin/python}
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${REPO_ROOT}"

"${PYTHON_BIN}" -m data.build_document_chunk_cache \
  --model-path /mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/zskj-hub/models-Qwen-Qwen3.5-35B-A3B-Base \
  --data-files '/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/jincheng/data/wmdp/bio-forget-corpus/*.parquet' \
  --dataset-path parquet \
  --text-key text \
  --max-length 1024 \
  --stride 1024 \
  --min-length 1 \
  --workers 8 \
  --tokenize-batch-size 1024 \
  --output-dir data_cache/qwen35/qwen35_wmdp_bio_max1024_stride1024/forget \
  "$@"
