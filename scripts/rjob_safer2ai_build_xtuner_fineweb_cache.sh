#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${OU_REPO_ROOT:-/mnt/shared-storage-user/safethm-share/wangxiao1/Safe_MoE/open-unlearning}
CONFIG=${CONFIG:-configs/data_processing/interns2_fineweb_edu_5b_xtuner_softpack.yaml}
JOB_NAME=${JOB_NAME:-ou-build-xtuner-fineweb-edu-5b}
IMAGE=${IMAGE:-registry.h.pjlab.org.cn/ailab-safer2ai-safer2ai_gpu/unlearning:unlearning-cu126-torch271-v1}
GPUS=${GPUS:-1}
CPUS=${CPUS:-128}
MEMORY_MB=${MEMORY_MB:-102400}
TOKENIZE_WORKERS=${TOKENIZE_WORKERS:-128}

rjob submit \
  --priority="${PRIORITY:-9}" \
  --name="${JOB_NAME}" \
  --namespace="${NAMESPACE:-ailab-safer2ai}" \
  --charged-group="${CHARGED_GROUP:-safer2ai_gpu}" \
  --private-machine="${PRIVATE_MACHINE:-group}" \
  --gpu="${GPUS}" \
  --cpu="${CPUS}" \
  --memory="${MEMORY_MB}" \
  --custom-resources "brainpp.cn/fuse=1" \
  --termination-grace-period-seconds="${TERMINATION_GRACE_PERIOD_SECONDS:-600}" \
  --mount=gpfs://gpfs1/safethm-share:/mnt/shared-storage-user/safethm-share \
  --mount=gpfs://gpfs2/wenxiaoyu-gpfs02:/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02 \
  --image-pull-policy=IfNotPresent \
  --image="${IMAGE}" \
  -e DISTRIBUTED_JOB=true \
  -- bash -c "set -euo pipefail
cd '${REPO_ROOT}'
export HF_HOME=/tmp/open-unlearning-hf-home
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
bash scripts/build_xtuner_pack_cache.sh \\
  --config '${CONFIG}' \\
  --tokenize-workers '${TOKENIZE_WORKERS}'"
