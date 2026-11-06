#!/usr/bin/env bash
set -euo pipefail

# Gemma4 main-cyber R1 RJob submission -> safethm resource group.
# Fourteen configs-r1/ yamls (nine algorithms; PDU split into six
# absolute-eps arms) in FOUR serial groups over two resource groups:
# g1/g2 -> safethm (this script), g3/g4 -> safer2ai
# (rjob_r1_safer2ai.sh). Each group is one 8-GPU RJob. G1/R1 = pure
# curve round: no checkpoints anywhere. Groups are time-balanced from
# the gemma4 bio/nocap per-arm measurements projected to 230 steps
# (PDU ~1h, mbs1 ~0.6-0.8h, mbs4 ~0.3-0.4h with cyber eval x1.6),
# longest-first inside each group.
# Usage:
#   BATCH=g1 bash rjob_r1_safethm.sh      # g1|g2 selects the group
#   bash rjob_r1_safethm.sh <yaml...>     # explicit YAML paths
#   BATCH=g1 PREVIEW=true bash rjob_r1_safethm.sh  # print rjob cmd only
# safethm parameters follow main/gemma4/rjob_main_safethm.sh
# (charged-group=sfaethm_gpu / namespace=ailab-safethm) which follows
# scripts/rjob_safethm_example.sh (mounts, env exports, log layout).

REPO_ROOT=${OU_REPO_ROOT:-/mnt/shared-storage-user/safethm-share/wangxiao1/Safe_MoE/open-unlearning}
RUN_ROOT=${REPO_ROOT}/outputs
RUN_SCRIPT=${RUN_SCRIPT:-main/main-cyber/gemma4/main_run_train.sh}
RESOURCE_GROUP=safethm
NAMESPACE=ailab-safethm
CHARGED_GROUP=sfaethm_gpu
IMAGE=registry.h.pjlab.org.cn/ailab-safethm-sfaethm_cpu/safemoe:jc-work-20260506163353

BATCH=${BATCH:-}
if (( $# )); then
  RUN_CONFIG_ARGS=("$@")
elif [[ -n "${RUN_CONFIGS:-}" ]]; then
  read -r -a RUN_CONFIG_ARGS <<< "${RUN_CONFIGS}"
elif [[ -n "${BATCH}" ]]; then
  CFG=main/main-cyber/gemma4/configs-r1
  case "${BATCH}" in
    # Submission plan (2026-10-03): g1/g2 -> safethm, g3/g4 -> safer2ai.
    # g1/g2: each pairs two PDU arms (~1h each) with the mbs-4 fast
    # configs; g3: the loose PDU pair alone; g4: the mbs-1 non-PDU
    # quartet (full-logit trio at eval bs 16 + CEU).
    g1) RUN_CONFIG_ARGS=(${CFG}/pdu_eps2p00.yaml ${CFG}/pdu_eps2p10.yaml ${CFG}/rmu.yaml ${CFG}/grad_ascent.yaml) ;;
    g2) RUN_CONFIG_ARGS=(${CFG}/pdu_eps2p20.yaml ${CFG}/pdu_eps2p40.yaml ${CFG}/satimp_forgetonly.yaml ${CFG}/WGA_forgetonly.yaml) ;;
    *) echo "ERROR: unknown BATCH '${BATCH}' (this script holds g1|g2; g3|g4 live in rjob_r1_safer2ai.sh)" >&2; exit 1 ;;
  esac
else
  echo "ERROR: select a batch with BATCH=g1|g2, pass YAML paths, or set RUN_CONFIGS." >&2
  exit 1
fi

JOB_NAME=${JOB_NAME:-gemma4-cyber-r1-${BATCH:-custom}-${RESOURCE_GROUP}}
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
