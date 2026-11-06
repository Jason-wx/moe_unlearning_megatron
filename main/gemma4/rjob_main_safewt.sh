#!/usr/bin/env bash
set -euo pipefail

# Gemma4 main-line RJob submission -> safewt resource group.
# Twelve configs (seven algorithms copied from the Intern-S2 main line via
# main/qwen35, the RMU best-arm re-run with ckpt saving, plus
# PureNPO/SimNPO lr arms 1e-5/1e-6) are split into five serial groups;
# each group is one 8-GPU RJob. PDU (pending retain-loss-eps anchor +
# sweep) and UNDIAL (pending debugging) are intentionally excluded.
# Usage:
#   bash rjob_main_safewt.sh            # BATCH=g1..g6 selects the group
#   BATCH=g2 bash rjob_main_safewt.sh
#   bash rjob_main_safewt.sh main/gemma4/ceu.yaml   # explicit YAML paths
#   BATCH=g1 PREVIEW=true bash rjob_main_safewt.sh  # print rjob cmd only
# safewt parameters follow the verified qwen35 main-line script
# main/qwen35/rjob_main_safewt.sh (charged-group=safewt_gpu /
# namespace=ailab-safewt) and scripts/rjob_safer2ai_example.sh (mounts,
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
    # Submission plan (2026-09-28, confirmed): g1..g3 -> safewt, g4..g5 ->
    # safer2ai. g1/g3 hold the mbs-4 fast configs; g2 pairs the mbs-1 heavy
    # configs; g4/g5 are the PureNPO/SimNPO lr brackets (3 arms each).
    g1) RUN_CONFIG_ARGS=(main/gemma4/grad_ascent.yaml main/gemma4/satimp_forgetonly.yaml) ;;
    g2) RUN_CONFIG_ARGS=(main/gemma4/grad_diff.yaml main/gemma4/ceu.yaml) ;;
    g3) RUN_CONFIG_ARGS=(main/gemma4/WGA_forgetonly.yaml main/gemma4/rmu.yaml) ;;
    g4) RUN_CONFIG_ARGS=(main/gemma4/pure_npo.yaml main/gemma4/pure_npo_lr1e-5.yaml main/gemma4/pure_npo_lr1e-6.yaml) ;;
    g5) RUN_CONFIG_ARGS=(main/gemma4/simnpo.yaml main/gemma4/simnpo_lr1e-5.yaml main/gemma4/simnpo_lr1e-6.yaml) ;;
    *) echo "ERROR: unknown BATCH '${BATCH}' (g1..g5)" >&2; exit 1 ;;
  esac
else
  echo "ERROR: select a batch with BATCH=g1..g5, pass YAML paths, or set RUN_CONFIGS." >&2
  exit 1
fi

JOB_NAME=${JOB_NAME:-gemma4-main-${BATCH:-custom}-${RESOURCE_GROUP}}
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
