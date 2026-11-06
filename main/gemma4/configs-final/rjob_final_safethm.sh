#!/usr/bin/env bash
set -euo pipefail

# Gemma4 FINAL-line RJob submission -> safethm resource group.
# [RETIRED 2026-10-03] The g2 submission from this script was cancelled before
# training started; all five final-line groups now ride safewt via
# rjob_final_safewt.sh (BATCH=g2). Kept for reference only.
# Nine configs (main/gemma4/configs-final/: nocap shadow model + BOS-anchored
# caches, final-ckpt saving, PDU eps=2.60 single arm) split into five serial
# groups of one 8-GPU RJob each; pool split safethm:safer2ai:safewt = 1:1:3
# (user-confirmed 2026-10-03). This script owns the safethm slot: g2 (CEU +
# RMU). CEU full-logits is proven on safethm H200 (bos-ablation round,
# H200-0699); RMU is the light mbs-4 profile.
# Usage:
#   bash rjob_final_safethm.sh                  # BATCH=g2 default slot
#   bash rjob_final_safethm.sh main/gemma4/configs-final/ceu.yaml   # explicit
#   PREVIEW=true bash rjob_final_safethm.sh
# Parameters follow main/gemma4/rjob_main_safethm.sh (charged-group=
# sfaethm_gpu / namespace=ailab-safethm) and scripts/rjob_safethm_example.sh
# (mounts, env exports, log layout).

REPO_ROOT=${OU_REPO_ROOT:-/mnt/shared-storage-user/safethm-share/wangxiao1/Safe_MoE/open-unlearning}
RUN_ROOT=${REPO_ROOT}/outputs
RUN_SCRIPT=${RUN_SCRIPT:-main/gemma4/main_run_train.sh}
RESOURCE_GROUP=safethm
NAMESPACE=ailab-safethm
CHARGED_GROUP=sfaethm_gpu
IMAGE=registry.h.pjlab.org.cn/ailab-safethm-sfaethm_cpu/safemoe:jc-work-20260506163353

BATCH=${BATCH:-g2}
if (( $# )); then
  RUN_CONFIG_ARGS=("$@")
elif [[ -n "${RUN_CONFIGS:-}" ]]; then
  read -r -a RUN_CONFIG_ARGS <<< "${RUN_CONFIGS}"
else
  case "${BATCH}" in
    # Final-line plan (2026-10-03): g1 pdu (safewt) | g2 ceu+rmu (safethm) |
    # g3 pure_npo+satimp (safewt) | g4 grad_diff+WGA (safer2ai) |
    # g5 simnpo+grad_ascent (safewt).
    g2) RUN_CONFIG_ARGS=(main/gemma4/configs-final/ceu.yaml main/gemma4/configs-final/rmu.yaml) ;;
    *) echo "ERROR: unknown BATCH '${BATCH}' for safethm (g2)" >&2; exit 1 ;;
  esac
fi

JOB_NAME=${JOB_NAME:-gemma4-final-${BATCH:-custom}-${RESOURCE_GROUP}}
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
# PDU int32-overflow guard: env override > config.json auto-detection; no-op
# for non-PDU algorithms (gated on algorithm.name == 'pdu').
export OU_PDU_SPLIT_FORWARD=1
export HF_HOME=\"\${HF_HOME:-/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/jincheng/.cache/huggingface}\"
export HF_DATASETS_CACHE=\"\${HF_HOME}/datasets\"
export HF_HUB_CACHE=\"\${HF_HOME}/hub\"
export HUGGINGFACE_HUB_CACHE=\"\${HF_HUB_CACHE}\"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
POD_FQDN=\$(envs/open-unlearning/bin/python -c 'import socket; print(socket.getfqdn())' 2>/dev/null || hostname)
POD_IP=\$(envs/open-unlearning/bin/python -c 'import socket; s=socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.connect((\"10.255.255.255\",80)); print(s.getsockname()[0])' 2>/dev/null || hostname -i | awk '{print \$1}')
if [ -n \"\${POD_FQDN}\" ] && [ -n \"\${POD_IP}\" ] && ! getent hosts \"\${POD_FQDN}\" >/dev/null 2>&1; then
  echo \"\${POD_IP} \${POD_FQDN} \$(hostname -s)\" >> /etc/hosts
  echo \"pinned \${POD_FQDN} -> \${POD_IP} in /etc/hosts (pod FQDN unresolvable via cluster DNS)\"
fi
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
