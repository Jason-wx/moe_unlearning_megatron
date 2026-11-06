#!/usr/bin/env bash
set -euo pipefail

# Gemma4 main-cyber R2-RERUN RJob submission -> safewt resource group.
# Repeats the five algorithms whose R1->R2 readouts drifted hardest
# (simnpo -0.043 / wga mmlu -0.059 / pure-npo +0.127 / grad-diff -0.057 /
# ceu +0.065) to get a third draw on identical configs. The configs are the
# configs-r2/ files UNCHANGED except evaluation.output_dir ->
# online-evaluation/r3/ (fresh curve files); run names and save_steps stay
# as-is, so each rerun re-saves its picked-step ckpt into the (already
# deleted) gemma4-main-cyber-r2-* dir. CEU here is the FIRST run on the
# fixed log-space leave-gold-out code (objectives.py 10-03 14:15) - R1/R2
# both ran the buggy normalization.
# Submission plan (2026-10-03, user-confirmed): THREE serial groups, all on
# safewt, balanced by R2-measured runtimes (pnpo 72 + simnpo 67 = 139m,
# ceu 79 + wga 56 = 135m, gd 68m). Prior R2 training logs archived to
# logs/r2/ (the runner overwrites configs-r2-*.log in place).
# Usage:
#   BATCH=h1 bash rjob_r2r_safewt.sh        # h1|p2|... (h1|h2|h3)
#   bash rjob_r2r_safewt.sh main/main-cyber/gemma4/configs-r2/simnpo.yaml
#   BATCH=h1 PREVIEW=true bash rjob_r2r_safewt.sh
# Parameters follow the verified cyber-line script rjob_r2_safewt.sh
# (namespace=ailab-safewt / charged-group=safewt_gpu / safemoe image; mounts,
# env exports, log layout identical) - per [[do-not-modify-team-scripts]] this
# is a standalone script, team examples untouched.

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
  case "${BATCH}" in
    h1) RUN_CONFIG_ARGS=(main/main-cyber/gemma4/configs-r2/pure_npo.yaml main/main-cyber/gemma4/configs-r2/simnpo.yaml) ;;
    h2) RUN_CONFIG_ARGS=(main/main-cyber/gemma4/configs-r2/ceu.yaml main/main-cyber/gemma4/configs-r2/WGA_forgetonly.yaml) ;;
    h3) RUN_CONFIG_ARGS=(main/main-cyber/gemma4/configs-r2/grad_diff.yaml) ;;
    *) echo "ERROR: unknown BATCH '${BATCH}' (h1|h2|h3)" >&2; exit 1 ;;
  esac
else
  echo "ERROR: select a batch with BATCH=h1|h2|h3, pass YAML paths, or set RUN_CONFIGS." >&2
  exit 1
fi

JOB_NAME=${JOB_NAME:-gemma4-cyber-r2r-${BATCH:-custom}-${RESOURCE_GROUP}}
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
