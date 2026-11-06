#!/usr/bin/env bash
set -euo pipefail

# Qwen3.5 main-cyber RJob submission - R2 round (2026-10-01): rerun with a
# single best-point checkpoint save per config (save_steps = the margin-
# best step, mmlu - wmdp_cyber argmax on the R1 curve; see SAVE_STEPS in
# make_configs.py). PDU keeps ONLY the eps 2.51 arm this round (the 2.16/
# 2.30 configs stay on disk unsubmitted - R1 showed the three arms near-
# identical), so NINE configs in THREE jobs, ALL on safewt (user
# allocation 2026-10-01; eps-sweep + R1-proven cluster). Groups balanced
# on measured R1 wall-clock (ep1->ep10 eval span) with the longest member
# first in each serial chain:
#   r2-pdu-gd-rmu          pdu2p51 1.37h + grad_diff 1.08h + rmu 0.69h = 3.14h
#   r2-pnpo-simnpo-satimp  pure_npo 1.23h + simnpo 1.11h + satimp 0.80h = 3.14h
#   r2-ceu-ga-wga          ceu 1.12h + grad_ascent 0.85h + wga 0.81h = 2.78h
# Each export is ~70G (intern point saves measured 66G); 9 exports ~630G
# against 5.7T free on the output_root filesystem (checked 2026-10-01).
# R1's six-batch cross-cluster allocation is history (README + git); those
# batch tags were retired when R2 landed.
# The resource group is selected with RESOURCE_GROUP={safethm,safewt,safer2ai}
# and the batch with BATCH={r2-pdu-gd-rmu,r2-pnpo-simnpo-satimp,r2-ceu-ga-wga}
# (or pass explicit YAML paths / RUN_CONFIGS to override). Parameters follow
# main/main-cyber/interns2/rjob_main.sh
# (mounts, env exports, log layout), which in turn followed
# scripts/rjob_safer2ai_example.sh.
#
# Examples:
#   RESOURCE_GROUP=safewt BATCH=r2-pdu-gd-rmu bash rjob_main.sh
#   RESOURCE_GROUP=safewt BATCH=r2-ceu-ga-wga PREVIEW=true bash rjob_main.sh

REPO_ROOT=${OU_REPO_ROOT:-/mnt/shared-storage-user/safethm-share/wangxiao1/Safe_MoE/open-unlearning}
RUN_ROOT=${REPO_ROOT}/outputs
RUN_SCRIPT=${RUN_SCRIPT:-main/main-cyber/qwen35/main_run_train.sh}
RESOURCE_GROUP=${RESOURCE_GROUP:-safewt}

case "${RESOURCE_GROUP}" in
  safethm)
    NAMESPACE=ailab-safethm
    CHARGED_GROUP=sfaethm_gpu
    IMAGE=registry.h.pjlab.org.cn/ailab-safethm-sfaethm_cpu/safemoe:jc-work-20260506163353
    ;;
  safewt)
    NAMESPACE=ailab-safewt
    CHARGED_GROUP=safewt_gpu
    IMAGE=registry.h.pjlab.org.cn/ailab-safethm-sfaethm_cpu/safemoe:jc-work-20260506163353
    ;;
  safer2ai)
    NAMESPACE=ailab-safer2ai
    CHARGED_GROUP=safer2ai_gpu
    IMAGE=registry.h.pjlab.org.cn/ailab-safer2ai-safer2ai_gpu/unlearning:unlearning-cu126-torch271-v1
    ;;
  *)
    echo "ERROR: unknown RESOURCE_GROUP '${RESOURCE_GROUP}' (safethm|safewt|safer2ai)" >&2
    exit 1
    ;;
esac

BATCH=${BATCH:-}
if (( $# )); then
  RUN_CONFIG_ARGS=("$@")
elif [[ -n "${RUN_CONFIGS:-}" ]]; then
  read -r -a RUN_CONFIG_ARGS <<< "${RUN_CONFIGS}"
elif [[ -n "${BATCH}" ]]; then
  case "${BATCH}" in
    r2-pdu-gd-rmu) RUN_CONFIG_ARGS=(main/main-cyber/qwen35/pdu_eps2p51.yaml main/main-cyber/qwen35/grad_diff.yaml main/main-cyber/qwen35/rmu.yaml) ;;
    r2-pnpo-simnpo-satimp) RUN_CONFIG_ARGS=(main/main-cyber/qwen35/pure_npo.yaml main/main-cyber/qwen35/simnpo.yaml main/main-cyber/qwen35/satimp_forgetonly.yaml) ;;
    r2-ceu-ga-wga) RUN_CONFIG_ARGS=(main/main-cyber/qwen35/ceu.yaml main/main-cyber/qwen35/grad_ascent.yaml main/main-cyber/qwen35/WGA_forgetonly.yaml) ;;
    *) echo "ERROR: unknown BATCH '${BATCH}' (r2-pdu-gd-rmu|r2-pnpo-simnpo-satimp|r2-ceu-ga-wga)" >&2; exit 1 ;;
  esac
else
  echo "ERROR: select a batch with BATCH=..., pass YAML paths, or set RUN_CONFIGS." >&2
  exit 1
fi

JOB_NAME=${JOB_NAME:-qwen35-main-cyber-${BATCH:-custom}-${RESOURCE_GROUP}}
RUN_CONFIG_DISPLAY="${RUN_CONFIG_ARGS[*]}"

REMOTE_RUN_ARGS=""
for config in "${RUN_CONFIG_ARGS[@]}"; do
  printf -v quoted_config '%q ' "${config}"
  REMOTE_RUN_ARGS+="${quoted_config}"
done

GPUS=8
CPUS=96
MEMORY_MB=256000
FUSE_COUNT=1
RDMA_COUNT=8
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
  -e DISTRIBUTED_JOB=true
  -- bash -c "set -euo pipefail
cd '${REPO_ROOT}'
export OU_REPO_ROOT='${REPO_ROOT}'
export NUM_GPUS='${GPUS}'
export NUM_PROCESSES='${GPUS}'
export CPU_THREADS_PER_PROCESS='$((CPUS / GPUS))'
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export PYTHONUNBUFFERED=1
export HF_HOME=\"\${HF_HOME:-/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/jincheng/.cache/huggingface}\"
export HF_DATASETS_CACHE=\"\${HF_HOME}/datasets\"
export HF_HUB_CACHE=\"\${HF_HOME}/hub\"
export HUGGINGFACE_HUB_CACHE=\"\${HF_HUB_CACHE}\"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
RUN_TS=\$(date +%Y%m%d_%H%M%S)
LOG_DIR='${RUN_ROOT}/rjob_logs/${JOB_NAME}'_\${RUN_TS}
mkdir -p \"\${LOG_DIR}\"
echo 'job=${JOB_NAME} resource_group=${RESOURCE_GROUP} run_script=${RUN_SCRIPT} configs=${RUN_CONFIG_DISPLAY}' | tee \"\${LOG_DIR}/metadata.log\"
(bash '${RUN_SCRIPT}' ${REMOTE_RUN_ARGS}) 2>&1 | tee \"\${LOG_DIR}/run.log\""
)

if [[ "${PREVIEW:-false}" == "true" ]]; then
  printf 'rjob'
  printf ' %q' "${RJOB_ARGS[@]}"
  printf '\n'
  exit 0
fi

exec rjob "${RJOB_ARGS[@]}"
