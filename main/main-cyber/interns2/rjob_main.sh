#!/usr/bin/env bash
set -euo pipefail

# Intern-S2 main-cyber RJob submission. Ten algorithms are split into five
# serial groups of two (same pairing as the bio main line); each group is
# one 8-GPU RJob. Original split 1:1:3 (safethm:safer2ai:safewt); on
# 2026-09-30 the safethm g1 job was cancelled before start and resubmitted
# to safewt, so the final allocation is 0:1:4:
#   safethm   -> (none; g1 job cancelled, moved to safewt)
#   safer2ai  -> g2 (pdu, grad_ascent)
#   safewt    -> g1 (rmu, grad_diff), g3 (pure_npo, simnpo),
#                g4 (satimp, WGA), g5 (ceu, undial)
# The resource group is selected with RESOURCE_GROUP={safethm,safewt,safer2ai}
# and the batch with BATCH={g1..g5} (or pass explicit YAML paths / RUN_CONFIGS
# to override). Parameters follow the bio line's main/interns2/rjob_main.sh
# (mounts, env exports, log layout), which in turn followed
# scripts/rjob_safer2ai_example.sh.
#
# Examples:
#   RESOURCE_GROUP=safewt  BATCH=g3 bash rjob_main.sh
#   RESOURCE_GROUP=safethm BATCH=g1 PREVIEW=true bash rjob_main.sh
#
# Round 3 (2026-09-30 ~19:30, full R1 schedule + point ckpt saves):
#   safer2ai: pdu (1 job, single config)
#   safewt:   4 jobs of two configs each (serial):
#             ceu-rmu, gd-simnpo, ga-satimp, pnpo-wga
#   e.g. RESOURCE_GROUP=safewt BATCH=gd-simnpo bash rjob_main.sh \
#          main/main-cyber/interns2/grad_diff.yaml \
#          main/main-cyber/interns2/simnpo.yaml

REPO_ROOT=${OU_REPO_ROOT:-/mnt/shared-storage-user/safethm-share/wangxiao1/Safe_MoE/open-unlearning}
RUN_ROOT=${REPO_ROOT}/outputs
RUN_SCRIPT=${RUN_SCRIPT:-main/main-cyber/interns2/main_run_train.sh}
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
    g1) RUN_CONFIG_ARGS=(main/main-cyber/interns2/rmu.yaml main/main-cyber/interns2/grad_diff.yaml) ;;
    g2) RUN_CONFIG_ARGS=(main/main-cyber/interns2/pdu.yaml main/main-cyber/interns2/grad_ascent.yaml) ;;
    g3) RUN_CONFIG_ARGS=(main/main-cyber/interns2/pure_npo.yaml main/main-cyber/interns2/simnpo.yaml) ;;
    g4) RUN_CONFIG_ARGS=(main/main-cyber/interns2/satimp_forgetonly.yaml main/main-cyber/interns2/WGA_forgetonly.yaml) ;;
    g5) RUN_CONFIG_ARGS=(main/main-cyber/interns2/ceu.yaml main/main-cyber/interns2/undial_fixed.yaml) ;;
    *) echo "ERROR: unknown BATCH '${BATCH}' (g1..g5)" >&2; exit 1 ;;
  esac
else
  echo "ERROR: select a batch with BATCH=g1..g5, pass YAML paths, or set RUN_CONFIGS." >&2
  exit 1
fi

JOB_NAME=${JOB_NAME:-interns2-main-cyber-${BATCH:-custom}-${RESOURCE_GROUP}}
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
