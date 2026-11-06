#!/usr/bin/env bash
set -euo pipefail

# qwen35 unlearning-wall diagnosis (2026-09-29) on safer2ai, 7 GPUs max:
# output-shortcut test (MCQ logit scale / logit lens / cloze / NLL + routing)
# and weight-diff localization vs base, one model per GPU (run_diag.sh).
# Runs in the container because the main ckpts are root 0600.
# Cluster params copied from V4/SimNPO/rjob_simnpo_eval_safe2ai.sh, scaled to 7.
#   bash main/qwen35/diagnosis/rjob_diag_safer2ai.sh

JOB_NAME=${JOB_NAME:-qwen35-diag-shortcut-wdiff}
REPO_ROOT=${OU_REPO_ROOT:-/mnt/shared-storage-user/safethm-share/wangxiao1/Safe_MoE/open-unlearning}
RUN_SCRIPT=main/qwen35/diagnosis/run_diag.sh
RUN_ROOT=${REPO_ROOT}/outputs

GPUS=7
CPUS=84
MEMORY_MB=224000
CHARGED_GROUP=safer2ai_gpu
NAMESPACE=ailab-safer2ai
FUSE_COUNT=1
RDMA_COUNT=7
TERMINATION_GRACE_PERIOD_SECONDS=600
IMAGE=registry.h.pjlab.org.cn/ailab-safer2ai-safer2ai_gpu/unlearning:unlearning-cu126-torch271-v1

rjob submit \
  --priority=9 \
  --name="${JOB_NAME}" \
  --gpu="${GPUS}" \
  --memory="${MEMORY_MB}" \
  --cpu="${CPUS}" \
  --charged-group="${CHARGED_GROUP}" \
  --namespace="${NAMESPACE}" \
  --private-machine=group \
  --custom-resources "brainpp.cn/fuse=${FUSE_COUNT}" \
  --custom-resources "rdma/mlnx_shared=${RDMA_COUNT}" \
  --termination-grace-period-seconds "${TERMINATION_GRACE_PERIOD_SECONDS}" \
  --mount=gpfs://gpfs1/safethm-share:/mnt/shared-storage-user/safethm-share \
  --mount=gpfs://gpfs2/gpfs2-shared-public:/mnt/shared-storage-gpfs2/gpfs2-shared-public \
  --mount=gpfs://gpfs2/wenxiaoyu-gpfs02:/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02 \
  --image-pull-policy=IfNotPresent \
  --image="${IMAGE}" \
  -e DISTRIBUTED_JOB=true \
  -- bash -c "set -euo pipefail
cd '${REPO_ROOT}'
export OU_REPO_ROOT='${REPO_ROOT}'
export NUM_GPUS='${GPUS}'
export CPU_THREADS_PER_PROCESS=12
RUN_TS=\$(date +%Y%m%d_%H%M%S)
export LOG_DIR='${RUN_ROOT}/rjob_logs/${JOB_NAME}'_\${RUN_TS}
mkdir -p \"\${LOG_DIR}\"
echo 'job=${JOB_NAME} run_script=${RUN_SCRIPT}' | tee \"\${LOG_DIR}/metadata.log\"
(bash '${RUN_SCRIPT}') 2>&1 | tee \"\${LOG_DIR}/run.log\""
