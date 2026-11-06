#!/usr/bin/env bash
set -euo pipefail

# Gemma4 x PDU crash debug RJob submission -> safewt resource group
# (2026-09-29: both safewt eps arms died with a TE rmsnorm_bwd illegal
# memory access at the first backward; only the gemma4 x PDU combination is
# affected — see pdu_eps_sweep/README.md). Runs on the same safemoe image
# that produced the crash. Two 2-step arms under CUDA_LAUNCH_BLOCKING=1:
# reproduce/pinpoint, then the activation_recompute: null fix candidate.
# (First submitted to safer2ai as gemma4-pdu-debug-safer2ai-44949101,
# deleted while queued and moved to safewt per Wang Xiao 09-29.)
# Usage:
#   bash rjob_debug_safewt.sh
#   PREVIEW=true bash rjob_debug_safewt.sh   # print rjob cmd only
# Parameters follow the verified rjob_pdu_eps_sweep_safewt.sh template.

REPO_ROOT=${OU_REPO_ROOT:-/mnt/shared-storage-user/safethm-share/wangxiao1/Safe_MoE/open-unlearning}
RUN_ROOT=${REPO_ROOT}/outputs
RUN_SCRIPT=main/gemma4/pdu_eps_sweep/debug_run_train.sh
DEBUG_DIR=${REPO_ROOT}/main/gemma4/pdu_eps_sweep/debug
RESOURCE_GROUP=safewt
NAMESPACE=ailab-safewt
CHARGED_GROUP=safewt_gpu
IMAGE=registry.h.pjlab.org.cn/ailab-safethm-sfaethm_cpu/safemoe:jc-work-20260506163353

for cfg in pdu_debug_blocking.yaml pdu_debug_norecompute.yaml; do
  if [[ ! -f "${DEBUG_DIR}/${cfg}" ]]; then
    echo "ERROR: debug config missing: ${DEBUG_DIR}/${cfg}" >&2
    exit 1
  fi
done

JOB_NAME=${JOB_NAME:-gemma4-pdu-debug-${RESOURCE_GROUP}}
RUN_CONFIG_DISPLAY="debug/pdu_debug_blocking.yaml debug/pdu_debug_norecompute.yaml"

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
export CUDA_LAUNCH_BLOCKING=1
export PYTHONUNBUFFERED=1
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
echo 'job=${JOB_NAME} resource_group=${RESOURCE_GROUP} run_script=${RUN_SCRIPT} configs=${RUN_CONFIG_DISPLAY} cuda_launch_blocking=1' | tee \"\${LOG_DIR}/metadata.log\"
(bash '${RUN_SCRIPT}') 2>&1 | tee \"\${LOG_DIR}/run.log\""
)

if [[ "${PREVIEW:-false}" == "true" ]]; then
  printf 'rjob'
  printf ' %q' "${RJOB_ARGS[@]}"
  printf '\n'
  exit 0
fi

exec rjob "${RJOB_ARGS[@]}"
