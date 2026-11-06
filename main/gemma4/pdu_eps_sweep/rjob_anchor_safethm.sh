#!/usr/bin/env bash
set -euo pipefail

# Measure the gemma4 baseline retain-NLL anchor for PDU's retain_loss_eps
# calibration -> safethm resource group, 4-GPU forward-only job.
# Protocol (2026-09-28, confirmed with 王潇): same methodology as the qwen35
# anchor but 4 GPUs and 200 randomly-sampled packs (~0.8M valid tokens;
# qwen35 used 800 packs / 2 GPUs, anchor 2.0082, sigma 0.140, SE 0.010).
# Tool: src/tools/measure_retain_nll.py (same label/NLL semantics as the
# Megatron PDU retain loss; --sample-mode random is seeded, without
# replacement; device_map=auto shards across all 4 GPUs).
# Overrides vs the tool's gemma4 preset:
#   --model-path -> the sgtm gemma-26b-a4b weights the main line trains
#     from (the preset points at the HF public google/gemma-4-26B-A4B —
#     the anchor must match the actual training init; path is the
#     post-reorg sgtm layout, gemma/ subdirectory);
#   --cache -> the softpack4096 retain cache the main line trains on
#     (the preset points at a 1024-token doc-chunk cache);
#   --sequence-length 4096 to match.
# Usage: bash rjob_anchor_safethm.sh   (PREVIEW=true prints the rjob cmd)

REPO_ROOT=${OU_REPO_ROOT:-/mnt/shared-storage-user/safethm-share/wangxiao1/Safe_MoE/open-unlearning}
SWEEP_DIR=${REPO_ROOT}/main/gemma4/pdu_eps_sweep
JOB_NAME=${JOB_NAME:-gemma4-pdu-anchor-safethm}
NAMESPACE=ailab-safethm
CHARGED_GROUP=sfaethm_gpu
IMAGE=registry.h.pjlab.org.cn/ailab-safethm-sfaethm_cpu/safemoe:jc-work-20260506163353

MODEL_PATH=/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/jincheng/sgtm/checkpoints/gemma/gemma-26b-a4b
RETAIN_CACHE=${REPO_ROOT}/data_cache/Xtuner/fineweb_edu_5b_gemma4_softpack4096_seed42
OUTPUT=${SWEEP_DIR}/retain_nll_anchor_gemma4.json

GPUS=4
CPUS=48
MEMORY_MB=128000
FUSE_COUNT=1
RDMA_COUNT=4
TERMINATION_GRACE_PERIOD_SECONDS=600

RJOB_ARGS=(
  submit
  --priority=9
  --name="${JOB_NAME}"
  --gpu="${GPUS}"
  --memory="${MEMORY_MB}"
  --cpu="${CPUS}"
  --charged-group="${CHARGED_GROUP}"
  --namespace="${NAMESPACE}"
  --private-machine=group
  --custom-resources "brainpp.cn/fuse=${FUSE_COUNT}"
  --custom-resources "rdma/mlnx_shared=${RDMA_COUNT}"
  --termination-grace-period-seconds "${TERMINATION_GRACE_PERIOD_SECONDS}"
  --mount=gpfs://gpfs1/safethm-share:/mnt/shared-storage-user/safethm-share
  --mount=gpfs://gpfs2/gpfs2-shared-public:/mnt/shared-storage-gpfs2/gpfs2-shared-public
  --mount=gpfs://gpfs2/wenxiaoyu-gpfs02:/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02
  --image-pull-policy=IfNotPresent
  --image="${IMAGE}"
  -- bash -c "set -euo pipefail
cd '${REPO_ROOT}'
export PYTHONUNBUFFERED=1
export HF_HOME=\"\${HF_HOME:-/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/jincheng/.cache/huggingface}\"
export HF_DATASETS_CACHE=\"\${HF_HOME}/datasets\"
export HF_HUB_CACHE=\"\${HF_HOME}/hub\"
export HUGGINGFACE_HUB_CACHE=\"\${HF_HUB_CACHE}\"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
mkdir -p '${SWEEP_DIR}'
PYTHONPATH=src envs/open-unlearning/bin/python -m tools.measure_retain_nll \\
  --model gemma4 \\
  --model-path '${MODEL_PATH}' \\
  --cache '${RETAIN_CACHE}' \\
  --sequence-length 4096 \\
  --num-samples 200 \\
  --sample-mode random \\
  --batch-size 4 \\
  --seed 42 \\
  --output '${OUTPUT}'"
)

if [[ "${PREVIEW:-false}" == "true" ]]; then
  printf 'rjob'
  printf ' %q' "${RJOB_ARGS[@]}"
  printf '\n'
  exit 0
fi

exec rjob "${RJOB_ARGS[@]}"
