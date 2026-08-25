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
  --namespace="${NAMESPACE:-ailab-safer2ai}" \
  --private-machine="${PRIVATE_MACHINE:-group}" \
  --mount=gpfs://gpfs1/safethm-share:/mnt/shared-storage-user/safethm-share \
  --mount=gpfs://gpfs2/gpfs2-shared-public:/mnt/shared-storage-gpfs2/gpfs2-shared-public \
  --mount=gpfs://gpfs2/wenxiaoyu-gpfs02:/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02 \
  --image-pull-policy=IfNotPresent \
  --image="${IMAGE}" \
  -e DISTRIBUTED_JOB=true \
  -- bash -lc "set -euo pipefail
cd '${REPO_ROOT}'
export OU_REPO_ROOT='${REPO_ROOT}'
export OU_RUNTIME_ROOT='${OU_RUNTIME_ROOT:-}'
export OU_MEGATRON_CHECKPOINT_ROOT='${OU_MEGATRON_CHECKPOINT_ROOT:-/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/wangxiao1/checkpoints/unlearn/megatron}'
export NUM_GPUS='${GPUS}'
export CONFIG='${CONFIG}'
export SWEEP_NAME='${SWEEP_NAME:-}'
export INTERMEDIATE_STEP='${INTERMEDIATE_STEP:-300}'
export EVAL_TASKS='${EVAL_TASKS:-wmdp,mmlu}'
RUN_TS=\$(date +%Y%m%d_%H%M%S)
LOG_DIR='${RUN_ROOT}/rjob_logs/${JOB_NAME}'_\${RUN_TS}
mkdir -p \"\${LOG_DIR}\"
echo 'backend=${BACKEND} workflow=${WORKFLOW} config=${CONFIG}' | tee \"\${LOG_DIR}/metadata.log\"
${REMOTE_COMMAND} 2>&1 | tee \"\${LOG_DIR}/train.log\""
