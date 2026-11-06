#!/usr/bin/env bash
set -euo pipefail

# Submit one RJob that trains one or more configs in order.
#
#   CLUSTER=safethm scripts/submit_rjob_example.sh CONFIG [CONFIG...]
#
#   CLUSTER  safer2ai (default) | safethm | safewt
#
# The backend (megatron / deepspeed) and GPU count come from each config; all
# configs of one job must use the same GPU count. Configs run one after another
# and a failed config does not stop the next one. Logs go to
# outputs/rjob_logs/<JOB_NAME>_<time>/<config name>.log.
# Optional: JOB_NAME (default: first config's name), CPUS (96), MEMORY_MB
# (256000), PRIORITY (9). DRY_RUN=1 prints the rjob command without submitting.

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${OU_PYTHON:-${REPO_ROOT}/envs/open-unlearning/bin/python}

if (( $# == 0 )); then
  sed -n '4,15p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' >&2
  exit 2
fi

CLUSTER=${CLUSTER:-safer2ai}
case "${CLUSTER}" in
  safer2ai)
    NAMESPACE=ailab-safer2ai
    CHARGED_GROUP=safer2ai_gpu
    IMAGE=registry.h.pjlab.org.cn/ailab-safer2ai-safer2ai_gpu/unlearning:unlearning-cu126-torch271-v1
    ;;
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
  *)
    echo "CLUSTER must be safer2ai, safethm, or safewt" >&2
    exit 2
    ;;
esac

# Read "<backend> <world size> <name>" from each config and build the job steps.
STEPS=
GPUS=
FIRST_NAME=
for config in "$@"; do
  if [[ ! -f "${config}" ]]; then
    echo "config not found: ${config}" >&2
    exit 2
  fi
  config=$(realpath "${config}")
  backend= world_size= name=
  read -r backend world_size name < <(
    "${PYTHON_BIN}" - "${config}" <<'PY'
import sys

import yaml

with open(sys.argv[1], encoding="utf-8") as handle:
    config = yaml.safe_load(handle)
((backend, settings),) = config["backends"].items()
world_size = settings["parallel"]["world_size"] if backend == "megatron" else settings["world_size"]
print(backend, world_size, config["name"])
PY
  )
  if [[ -z "${name}" ]]; then
    echo "could not read backend / world size / name from ${config}" >&2
    exit 2
  fi
  if [[ -n "${GPUS}" && "${world_size}" != "${GPUS}" ]]; then
    echo "all configs of one job must use the same GPU count: ${config} needs ${world_size}, not ${GPUS}" >&2
    exit 2
  fi
  GPUS=${world_size}
  FIRST_NAME=${FIRST_NAME:-${name}}
  printf -v step 'run_config %q %q %q\n' "${backend}" "${config}" "${name}"
  STEPS+=${step}
done

JOB_NAME=${JOB_NAME:-${FIRST_NAME}}
CPUS=${CPUS:-96}
MEMORY_MB=${MEMORY_MB:-256000}
if [[ -n "${OU_RUN_ROOT:-}" ]]; then
  RUN_ROOT=${OU_RUN_ROOT}
elif [[ -n "${OU_RUNTIME_ROOT:-}" ]]; then
  RUN_ROOT=${OU_RUNTIME_ROOT}/runs
else
  RUN_ROOT=${REPO_ROOT}/outputs
fi

REMOTE_COMMAND="set -uo pipefail
cd '${REPO_ROOT}'
export OU_REPO_ROOT='${REPO_ROOT}'
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG_DIR='${RUN_ROOT}/rjob_logs/${JOB_NAME}'_\$(date +%Y%m%d_%H%M%S)
mkdir -p \"\${LOG_DIR}\"
failed=0
run_config() {
  echo \"==> \$1 \$2\" | tee -a \"\${LOG_DIR}/job.log\"
  if bash \"scripts/train_\$1.sh\" \"\$2\" 2>&1 | tee \"\${LOG_DIR}/\$3.log\"; then
    echo \"<== ok \$3\" | tee -a \"\${LOG_DIR}/job.log\"
  else
    echo \"<== FAILED \$3\" | tee -a \"\${LOG_DIR}/job.log\"
    failed=1
  fi
}
${STEPS}exit \${failed}"

command=(
  rjob submit
  --priority="${PRIORITY:-9}"
  --name="${JOB_NAME}"
  --namespace="${NAMESPACE}"
  --charged-group="${CHARGED_GROUP}"
  --private-machine=group
  --gpu="${GPUS}"
  --cpu="${CPUS}"
  --memory="${MEMORY_MB}"
  --custom-resources "brainpp.cn/fuse=1"
  --custom-resources "rdma/mlnx_shared=${GPUS}"
  --termination-grace-period-seconds 600
  --mount=gpfs://gpfs1/safethm-share:/mnt/shared-storage-user/safethm-share
  --mount=gpfs://gpfs2/gpfs2-shared-public:/mnt/shared-storage-gpfs2/gpfs2-shared-public
  --mount=gpfs://gpfs2/wenxiaoyu-gpfs02:/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02
  --image-pull-policy=IfNotPresent
  --image="${IMAGE}"
  -e DISTRIBUTED_JOB=true
  -- bash -c "${REMOTE_COMMAND}"
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '%q ' "${command[@]}"
  echo
  exit 0
fi
"${command[@]}"
