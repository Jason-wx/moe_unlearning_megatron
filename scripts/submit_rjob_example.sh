#!/usr/bin/env bash
set -euo pipefail

# This is the only maintained RJob example. All experiment semantics live in
# CONFIG; this file only declares cluster resources and the selected backend.
BACKEND=${BACKEND:-megatron}
WORKFLOW=${WORKFLOW:-single}
CONFIG=${CONFIG:-configs/experiments/interns2_pure_npo.yaml}
GPUS=${GPUS:-8}
CPUS=${CPUS:-96}
MEMORY_MB=${MEMORY_MB:-900000}
JOB_NAME=${JOB_NAME:-ou-${BACKEND}-example}
REPO_ROOT=${OU_REPO_ROOT:-/mnt/shared-storage-user/safethm-share/wangxiao1/Safe_MoE/open-unlearning}
if [[ -n "${OU_RUN_ROOT:-}" ]]; then
  RUN_ROOT=${OU_RUN_ROOT}
elif [[ -n "${OU_RUNTIME_ROOT:-}" ]]; then
  RUN_ROOT=${OU_RUNTIME_ROOT}/runs
else
  RUN_ROOT=${REPO_ROOT}/outputs
fi
IMAGE=${IMAGE:-registry.h.pjlab.org.cn/ailab-safer2ai-safer2ai_gpu/unlearning:unlearning-cu126-torch271-v1}
USE_IMAGE=${USE_IMAGE:-true}
USE_NAMESPACE=${USE_NAMESPACE:-true}
FUSE_COUNT=${FUSE_COUNT:-}
RDMA_COUNT=${RDMA_COUNT:-}
TERMINATION_GRACE_PERIOD_SECONDS=${TERMINATION_GRACE_PERIOD_SECONDS:-}

if [[ "${BACKEND}" != "deepspeed" && "${BACKEND}" != "megatron" ]]; then
  echo "BACKEND must be deepspeed or megatron" >&2
  exit 2
fi
if [[ "${WORKFLOW}" != "single" && "${WORKFLOW}" != "train-eval-group" ]]; then
  echo "WORKFLOW must be single or train-eval-group" >&2
  exit 2
fi
if [[ "${WORKFLOW}" == "train-eval-group" && "${BACKEND}" != "megatron" ]]; then
  echo "train-eval-group currently requires BACKEND=megatron" >&2
  exit 2
fi
if [[ "${USE_IMAGE}" != "true" && "${USE_IMAGE}" != "false" ]]; then
  echo "USE_IMAGE must be true or false" >&2
  exit 2
fi
if [[ "${USE_NAMESPACE}" != "true" && "${USE_NAMESPACE}" != "false" ]]; then
  echo "USE_NAMESPACE must be true or false" >&2
  exit 2
fi

IMAGE_ARGS=()
if [[ "${USE_IMAGE}" == "true" ]]; then
  IMAGE_ARGS+=(--image-pull-policy=IfNotPresent --image="${IMAGE}")
fi
NAMESPACE_ARGS=()
if [[ "${USE_NAMESPACE}" == "true" ]]; then
  NAMESPACE_ARGS+=(--namespace="${NAMESPACE:-ailab-safer2ai}")
fi
CUSTOM_RESOURCE_ARGS=()
if [[ -n "${FUSE_COUNT}" ]]; then
  CUSTOM_RESOURCE_ARGS+=(--custom-resources "brainpp.cn/fuse=${FUSE_COUNT}")
fi
if [[ -n "${RDMA_COUNT}" ]]; then
  CUSTOM_RESOURCE_ARGS+=(--custom-resources "rdma/mlnx_shared=${RDMA_COUNT}")
fi
TERMINATION_ARGS=()
if [[ -n "${TERMINATION_GRACE_PERIOD_SECONDS}" ]]; then
  TERMINATION_ARGS+=(
    --termination-grace-period-seconds "${TERMINATION_GRACE_PERIOD_SECONDS}"
  )
fi

EXTRA_ARGS_Q=
if (( $# )); then
  printf -v EXTRA_ARGS_Q ' %q' "$@"
fi
if [[ "${WORKFLOW}" == "single" ]]; then
  REMOTE_COMMAND="bash 'scripts/train_${BACKEND}.sh' '${CONFIG}'${EXTRA_ARGS_Q}"
else
  REMOTE_COMMAND="bash scripts/run_megatron_train_eval_group.sh${EXTRA_ARGS_Q}"
fi

rjob submit \
  --priority="${PRIORITY:-9}" \
  --name="${JOB_NAME}" \
  --gpu="${GPUS}" \
  --memory="${MEMORY_MB}" \
  --cpu="${CPUS}" \
  --charged-group="${CHARGED_GROUP:-safer2ai_gpu}" \
  "${NAMESPACE_ARGS[@]}" \
  --private-machine="${PRIVATE_MACHINE:-group}" \
  "${CUSTOM_RESOURCE_ARGS[@]}" \
  "${TERMINATION_ARGS[@]}" \
  --mount=gpfs://gpfs1/safethm-share:/mnt/shared-storage-user/safethm-share \
  --mount=gpfs://gpfs2/gpfs2-shared-public:/mnt/shared-storage-gpfs2/gpfs2-shared-public \
  --mount=gpfs://gpfs2/wenxiaoyu-gpfs02:/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02 \
  "${IMAGE_ARGS[@]}" \
  -e DISTRIBUTED_JOB=true \
  -- bash -c "set -euo pipefail
cd '${REPO_ROOT}'
export OU_REPO_ROOT='${REPO_ROOT}'
export OU_RUNTIME_ROOT='${OU_RUNTIME_ROOT:-}'
export OU_MEGATRON_CHECKPOINT_ROOT='${OU_MEGATRON_CHECKPOINT_ROOT:-/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/wangxiao1/checkpoints/unlearn/megatron}'
export NUM_GPUS='${GPUS}'
export NUM_PROCESSES='${EVAL_NUM_PROCESSES:-${GPUS}}'
export CPU_THREADS_PER_PROCESS='${EVAL_CPU_THREADS_PER_PROCESS:-$((CPUS / GPUS))}'
export PYTORCH_CUDA_ALLOC_CONF='${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}'
export CONFIG='${CONFIG}'
export SWEEP_NAME='${SWEEP_NAME:-}'
export INTERMEDIATE_STEP='${INTERMEDIATE_STEP:-300}'
export EVAL_TASKS='${EVAL_TASKS:-wmdp,mmlu}'
RUN_TS=\$(date +%Y%m%d_%H%M%S)
LOG_DIR='${RUN_ROOT}/rjob_logs/${JOB_NAME}'_\${RUN_TS}
mkdir -p \"\${LOG_DIR}\"
echo 'backend=${BACKEND} workflow=${WORKFLOW} config=${CONFIG}' | tee \"\${LOG_DIR}/metadata.log\"
${REMOTE_COMMAND} 2>&1 | tee \"\${LOG_DIR}/train.log\""
