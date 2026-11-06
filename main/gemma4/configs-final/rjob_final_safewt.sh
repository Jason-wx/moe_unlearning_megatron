#!/usr/bin/env bash
set -euo pipefail

# Gemma4 FINAL-line RJob submission -> safewt resource group.
# Nine configs (main/gemma4/configs-final/: nocap shadow model + BOS-anchored
# caches, final-ckpt saving, PDU eps=2.60 single arm) split into five serial
# groups of one 8-GPU RJob each; pool split safethm:safer2ai:safewt = 1:1:3
# (user-confirmed 2026-10-03; adjusted minutes later: ALL five groups ride
# safewt — the safethm/safer2ai submissions were cancelled and g2/g4
# resubmitted here). This script owns all five groups: g1 (PDU alone),
# g2 (ceu+rmu), g3, g4, g5. Pool compatibility is proven: GD+CEU ran 2133
# steps on the safewt image in the 09-28 main line (g2-safewt-46004055) and
# the PDU eps sweep rode safewt too.
# Usage:
#   BATCH=g1 bash rjob_final_safewt.sh          # g1|g2|g3|g4|g5
#   bash rjob_final_safewt.sh main/gemma4/configs-final/pdu.yaml   # explicit
#   BATCH=g1 PREVIEW=true bash rjob_final_safewt.sh
# Parameters follow main/gemma4/rjob_main_safewt.sh (charged-group=safewt_gpu
# / namespace=ailab-safewt) and scripts/rjob_safer2ai_example.sh (mounts,
# env exports, log layout).

REPO_ROOT=${OU_REPO_ROOT:-/mnt/shared-storage-user/safethm-share/wangxiao1/Safe_MoE/open-unlearning}
RUN_ROOT=${REPO_ROOT}/outputs
RUN_SCRIPT=${RUN_SCRIPT:-main/gemma4/main_run_train.sh}
RESOURCE_GROUP=safewt
NAMESPACE=ailab-safewt
CHARGED_GROUP=safewt_gpu
IMAGE=registry.h.pjlab.org.cn/ailab-safethm-sfaethm_cpu/safemoe:jc-work-20260506163353

BATCH=${BATCH:-}
if (( $# )); then
  RUN_CONFIG_ARGS=("$@")
elif [[ -n "${RUN_CONFIGS:-}" ]]; then
  read -r -a RUN_CONFIG_ARGS <<< "${RUN_CONFIGS}"
elif [[ -n "${BATCH}" ]]; then
  case "${BATCH}" in
    # Final-line plan (2026-10-03): g1 pdu | g2 ceu+rmu | g3 pure_npo+satimp
    # | g4 grad_diff+WGA | g5 simnpo+grad_ascent, all on safewt. PDU rides
    # alone: single-arm mbs-1 run with the per-microbatch dual update.
    g1) RUN_CONFIG_ARGS=(main/gemma4/configs-final/pdu.yaml) ;;
    g2) RUN_CONFIG_ARGS=(main/gemma4/configs-final/ceu.yaml main/gemma4/configs-final/rmu.yaml) ;;
    g3) RUN_CONFIG_ARGS=(main/gemma4/configs-final/pure_npo.yaml main/gemma4/configs-final/satimp_forgetonly.yaml) ;;
    g4) RUN_CONFIG_ARGS=(main/gemma4/configs-final/grad_diff.yaml main/gemma4/configs-final/WGA_forgetonly.yaml) ;;
    g5) RUN_CONFIG_ARGS=(main/gemma4/configs-final/simnpo.yaml main/gemma4/configs-final/grad_ascent.yaml) ;;
    *) echo "ERROR: unknown BATCH '${BATCH}' for safewt (g1..g5)" >&2; exit 1 ;;
  esac
else
  echo "ERROR: select a batch with BATCH=g1..g5, pass YAML paths, or set RUN_CONFIGS." >&2
  exit 1
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
# PDU int32-overflow guard: env override > config.json auto-detection. The
# auto path returns True for gemma4 offline but has not run on GPU without
# this env; it is a no-op for every non-PDU algorithm (gated on
# algorithm.name == 'pdu' in forward_step.py).
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
