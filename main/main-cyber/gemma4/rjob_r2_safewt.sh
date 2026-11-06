#!/usr/bin/env bash
set -euo pipefail

# Gemma4 main-cyber R2 RJob submission -> safewt resource group.
# Nine configs-r2/ yamls (PDU collapsed to its single winning arm,
# eps 2.60) in FIVE serial 8-GPU jobs, all on safewt (2026-10-03 plan):
# PDU alone (slowest single run, mbs1 + paired split-forward) and four
# time-balanced pairs measured from the R1 eval timestamps (per-run
# totals: pdu ~120m, ceu ~79m, pnpo ~72m, gd ~68m, simnpo ~67m,
# ga ~58m, wga ~56m, satimp ~55m, rmu ~50m; pairs land at 124-129m),
# longest-first inside each group.
# R2 = save-point round: each run exports exactly ONE checkpoint at its
# R1 margin-best step (see configs-r2/ headers). Online-eval jsonls land
# in main/main-cyber/gemma4/online-evaluation/ as gemma4-main-cyber-r2-*.
# Usage:
#   BATCH=g1 bash rjob_r2_safewt.sh      # g1..g5 selects the group
#   bash rjob_r2_safewt.sh <yaml...>     # explicit YAML paths
#   BATCH=g1 PREVIEW=true bash rjob_r2_safewt.sh  # print rjob cmd only
# Parameters follow the verified main/gemma4/bos-ablation/rjob_bos_safewt.sh
# (namespace=ailab-safewt / charged-group=safewt_gpu / safemoe image;
# mounts, env exports, log layout identical).

REPO_ROOT=${OU_REPO_ROOT:-/mnt/shared-storage-user/safethm-share/wangxiao1/Safe_MoE/open-unlearning}
RUN_ROOT=${REPO_ROOT}/outputs
RUN_SCRIPT=${RUN_SCRIPT:-main/main-cyber/gemma4/main_run_train.sh}
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
  CFG=main/main-cyber/gemma4/configs-r2
  case "${BATCH}" in
    # Submission plan (2026-10-03): five jobs, all safewt. PDU alone;
    # the remaining eight in measured-time pairs (mbs1 long + mbs4 short).
    g1) RUN_CONFIG_ARGS=(${CFG}/pdu.yaml) ;;
    g2) RUN_CONFIG_ARGS=(${CFG}/ceu.yaml ${CFG}/rmu.yaml) ;;
    g3) RUN_CONFIG_ARGS=(${CFG}/pure_npo.yaml ${CFG}/satimp_forgetonly.yaml) ;;
    g4) RUN_CONFIG_ARGS=(${CFG}/grad_diff.yaml ${CFG}/WGA_forgetonly.yaml) ;;
    g5) RUN_CONFIG_ARGS=(${CFG}/simnpo.yaml ${CFG}/grad_ascent.yaml) ;;
    *) echo "ERROR: unknown BATCH '${BATCH}' (expected g1..g5)" >&2; exit 1 ;;
  esac
else
  echo "ERROR: select a batch with BATCH=g1..g5, pass YAML paths, or set RUN_CONFIGS." >&2
  exit 1
fi

JOB_NAME=${JOB_NAME:-gemma4-cyber-r2-${BATCH:-custom}-${RESOURCE_GROUP}}
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
