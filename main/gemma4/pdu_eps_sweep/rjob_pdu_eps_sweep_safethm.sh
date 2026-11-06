#!/usr/bin/env bash
set -euo pipefail

# Gemma4 PDU eps-sweep RJob submission -> safethm resource group.
# Six eps arms (absolute grid fixed by Wang Xiao 2026-09-28: 2.0/2.1/2.2/
# 2.25/2.5/3.0; anchor retain NLL 2.0945) are split into three serial groups
# of two arms; one 8-GPU rjob per group (2026-09-28 plan: g1 -> safethm,
# g2 -> safewt, g3 -> safer2ai). Preconditions:
#   1. retain_nll_anchor_gemma4.json exists (rjob_anchor_safethm.sh finished);
#   2. make_eps_configs.py has been run to emit the pdu_eps*.yaml arms.
# Usage:
#   bash rjob_pdu_eps_sweep_safethm.sh               # BATCH defaults to g1
#   BATCH=g2 bash rjob_pdu_eps_sweep_safethm.sh      # any group on any cluster
#   PREVIEW=true bash rjob_pdu_eps_sweep_safethm.sh  # print rjob cmd only
# Parameters follow the verified main/gemma4/rjob_main_safewt.sh template.

REPO_ROOT=${OU_REPO_ROOT:-/mnt/shared-storage-user/safethm-share/wangxiao1/Safe_MoE/open-unlearning}
RUN_ROOT=${REPO_ROOT}/outputs
RUN_SCRIPT=main/gemma4/pdu_eps_sweep/pdu_run_train.sh
SWEEP_DIR=${REPO_ROOT}/main/gemma4/pdu_eps_sweep
RESOURCE_GROUP=safethm
NAMESPACE=ailab-safethm
CHARGED_GROUP=sfaethm_gpu
IMAGE=registry.h.pjlab.org.cn/ailab-safethm-sfaethm_cpu/safemoe:jc-work-20260506163353

# g1 tight pair, g2 loose pair, g3 diagnostic pair.
BATCH=${BATCH:-g1}
case "${BATCH}" in
  g1) RUN_CONFIG_ARGS=(main/gemma4/pdu_eps_sweep/pdu_eps2p00.yaml main/gemma4/pdu_eps_sweep/pdu_eps2p10.yaml) ;;
  g2) RUN_CONFIG_ARGS=(main/gemma4/pdu_eps_sweep/pdu_eps2p20.yaml main/gemma4/pdu_eps_sweep/pdu_eps2p25.yaml) ;;
  g3) RUN_CONFIG_ARGS=(main/gemma4/pdu_eps_sweep/pdu_eps2p50.yaml main/gemma4/pdu_eps_sweep/pdu_eps3p00.yaml) ;;
  *) echo "ERROR: unknown BATCH '${BATCH}' (g1..g3)" >&2; exit 1 ;;
esac

if [[ ! -f "${SWEEP_DIR}/retain_nll_anchor_gemma4.json" ]]; then
  echo "ERROR: anchor JSON missing — rjob_anchor_safethm.sh has not finished." >&2
  exit 1
fi
for cfg in "${RUN_CONFIG_ARGS[@]}"; do
  if [[ ! -f "${REPO_ROOT}/${cfg}" ]]; then
    echo "ERROR: arm config missing: ${cfg} — run: python3 ${SWEEP_DIR}/make_eps_configs.py" >&2
    exit 1
  fi
done
echo "group ${BATCH} arms: ${RUN_CONFIG_ARGS[*]##*/}"

JOB_NAME=${JOB_NAME:-gemma4-pdu-eps-sweep-${BATCH}-${RESOURCE_GROUP}}
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
