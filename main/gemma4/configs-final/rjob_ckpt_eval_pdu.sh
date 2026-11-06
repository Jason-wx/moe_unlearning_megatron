#!/usr/bin/env bash
set -euo pipefail

# Submit ONE safer2ai rjob (user allocation 2026-10-04: 2 GPUs, safer2ai pool)
# that offline-evaluates the LAST missing gemma4 bio FINAL-line ckpt - the PDU
# run's step-2133 export - with the official scripts/eval.sh
# (full wmdp_bio + mmlu, 0-shot, bf16). The other eight step-2133 ckpts were
# evaluated by rjob_ckpt_eval_safewt.sh (g2..g5) on safewt; this closes the
# 9/9 record. Layout: single ckpt, BOTH GPUs as accelerate data-parallel
# replicas (NUM_PROCESSES=2) - the profile proven by
# scripts/rjob_gemma4_baseline_eval_nocap_safewt.sh and the bio eval round.
# LM_EVAL_BATCH_SIZE pinned to 16 (NOT auto): auto picked 25 on the 140G H200
# baseline but safer2ai cards are 80G-class - weights ~49G + bs16 worst-case
# logits ~21G fits, bs25 (~33G logits) is marginal. bs does not change
# results, only throughput.
# Results land DIRECTLY beside the eight siblings via OU_EVAL_ROOT:
#   main/gemma4/offline-eval/lm_eval_results_gemma4-final-pdu-nocap-step2133_wmdp_bio_mmlu/result.json
# Already-evaluated (result.json present) => skip. Params follow
# scripts/rjob_safer2ai_example.sh (mounts/env layout).
#
# Example:
#   bash main/gemma4/configs-final/rjob_ckpt_eval_pdu.sh                      # 默认 safethm
#   RESOURCE_GROUP=safer2ai bash main/gemma4/configs-final/rjob_ckpt_eval_pdu.sh
#   PREVIEW=true bash main/gemma4/configs-final/rjob_ckpt_eval_pdu.sh
# History: first submitted to safer2ai 10-04 15:52 (34169318), still Inqueue
# when the user re-allocated to safethm - stopped clean, no results produced,
# resubmitted here.

REPO_ROOT=${OU_REPO_ROOT:-/mnt/shared-storage-user/safethm-share/wangxiao1/Safe_MoE/open-unlearning}

RESOURCE_GROUP=${RESOURCE_GROUP:-safethm}
case "${RESOURCE_GROUP}" in
  safethm)
    NAMESPACE=ailab-safethm
    CHARGED_GROUP=sfaethm_gpu
    IMAGE=registry.h.pjlab.org.cn/ailab-safethm-sfaethm_cpu/safemoe:jc-work-20260506163353
    ;;
  safer2ai)
    NAMESPACE=ailab-safer2ai
    CHARGED_GROUP=safer2ai_gpu
    IMAGE=registry.h.pjlab.org.cn/ailab-safer2ai-safer2ai_gpu/unlearning:unlearning-cu126-torch271-v1
    ;;
  *)
    echo "ERROR: unknown RESOURCE_GROUP '${RESOURCE_GROUP}' (safethm|safer2ai)" >&2
    exit 1
    ;;
esac

CKPT_ROOT=/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/wangxiao1/checkpoints/unlearn/megatron/main
EVAL_ROOT=${REPO_ROOT}/main/gemma4/offline-eval

# PDU run stopped feeding online evals at step 1400 (WMDP rebound after
# 0.249@1000, see main/gemma4/GA_family_nocap_rootcause_20261002.md lineage);
# the exported final ckpt on disk is checkpoint-step-2133 like the other 8.
CKPT_NAME=gemma4-final-pdu-nocap-step2133
CKPT_PATH=${CKPT_ROOT}/Gemma4/PDU

JOB_NAME=${JOB_NAME:-gemma4-final-ckpt-eval-pdu}
GPUS=2
CPUS=32
MEMORY_MB=128000
FUSE_COUNT=1
RDMA_COUNT=2
TERMINATION_GRACE_PERIOD_SECONDS=600
RUN_ROOT=${OU_RUN_ROOT:-${REPO_ROOT}/outputs}
BATCH_SIZE=${BATCH_SIZE:-16}

if [[ ! -d "${CKPT_PATH}" ]]; then
  echo "ERROR: ckpt dir missing: ${CKPT_PATH}" >&2
  exit 1
fi

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
  -e DISTRIBUTED_JOB=true
  -- bash -c "set -euo pipefail
cd '${REPO_ROOT}'
export OU_REPO_ROOT='${REPO_ROOT}'
export OU_EVAL_ROOT='${EVAL_ROOT}'
export NUM_PROCESSES='${GPUS}'
export CPU_THREADS_PER_PROCESS='$((CPUS / GPUS))'
export LM_EVAL_BATCH_SIZE='${BATCH_SIZE}'
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export PYTHONUNBUFFERED=1
export HF_HOME='/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/jincheng/.cache/huggingface'
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
RUN_TS=\$(date +%Y%m%d_%H%M%S)
LOG_DIR='${RUN_ROOT}/rjob_logs/${JOB_NAME}'_\${RUN_TS}
mkdir -p \"\${LOG_DIR}\" '${EVAL_ROOT}'
out_dir='${EVAL_ROOT}/lm_eval_results_${CKPT_NAME}_wmdp_bio_mmlu'
echo 'job=${JOB_NAME} layout=2gpu-accelerate-dp tasks=wmdp_bio,mmlu bs=${BATCH_SIZE} gpus=${GPUS} ckpt=${CKPT_NAME}' | tee \"\${LOG_DIR}/metadata.log\"
if [[ -f \"\${out_dir}/result.json\" ]]; then
  echo 'SKIP: result.json already present' | tee -a \"\${LOG_DIR}/summary.log\"
  exit 0
fi
if bash '${REPO_ROOT}/scripts/eval.sh' '${CKPT_PATH}' 'wmdp_bio,mmlu' \"\${out_dir}\" > \"\${LOG_DIR}/eval-${CKPT_NAME}.log\" 2>&1; then
  echo 'OK ${CKPT_NAME}' | tee -a \"\${LOG_DIR}/summary.log\"
else
  echo 'FAILED ${CKPT_NAME} (exit '\$?')' | tee -a \"\${LOG_DIR}/summary.log\"
  exit 1
fi
echo '${JOB_NAME} finished successfully' | tee -a \"\${LOG_DIR}/summary.log\"
"
)

if [[ "${PREVIEW:-false}" == "true" ]]; then
  printf 'rjob'
  printf ' %q' "${RJOB_ARGS[@]}"
  printf '\n'
  exit 0
fi

exec rjob "${RJOB_ARGS[@]}"
