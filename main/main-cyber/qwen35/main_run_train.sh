#!/usr/bin/env bash
set -Eeuo pipefail

# Qwen3.5 main-cyber generic serial training runner. Pass one or more
# complete YAML paths as arguments (or set RUN_CONFIGS in the RJob wrapper).
# The configs of a group execute strictly serially; a failed run (e.g.
# CUDA OOM) is logged and the sequence continues, exiting non-zero at the
# end if any config failed. Mirrors main/main-cyber/interns2/main_run_train.sh.
export PYTHONUNBUFFERED=1

REPO_ROOT="/mnt/shared-storage-user/safethm-share/wangxiao1/Safe_MoE/open-unlearning"
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TRAIN_SCRIPT="${REPO_ROOT}/scripts/train_megatron.sh"
LOG_DIR="${SCRIPT_DIR}/logs"

CONFIGS=(
)

if (( $# )); then
  CONFIGS=("$@")
elif [[ -n "${RUN_CONFIGS:-}" ]]; then
  # Used by rjob_main.sh. Paths in this repository contain no spaces.
  read -r -a CONFIGS <<< "${RUN_CONFIGS}"
fi

if (( ${#CONFIGS[@]} == 0 )); then
  echo "ERROR: no configs selected. Pass YAML paths or set CONFIGS/RUN_CONFIGS." >&2
  exit 1
fi

mkdir -p "${LOG_DIR}" "${SCRIPT_DIR}/online-evaluation"
cd "${REPO_ROOT}"

if [[ ! -x "${TRAIN_SCRIPT}" ]]; then
  echo "ERROR: training script is missing or not executable: ${TRAIN_SCRIPT}" >&2
  exit 1
fi

for config in "${CONFIGS[@]}"; do
  if [[ "${config}" != /* ]]; then
    config="${REPO_ROOT}/${config}"
  fi
  if [[ ! -f "${config}" ]]; then
    echo "ERROR: missing config: ${config}" >&2
    exit 1
  fi
done

idx=0
total=${#CONFIGS[@]}
failed_tags=()
for config in "${CONFIGS[@]}"; do
  idx=$((idx + 1))
  # Include the parent directory so serial runs with same-named configs
  # elsewhere do not overwrite one another's logs.
  tag="$(basename "$(dirname "${config}")")-$(basename "${config}" .yaml)"
  log_file="${LOG_DIR}/${tag}.log"

  echo "Starting ${idx}/${total}: ${tag}"
  echo "Config: ${config}"
  echo "Log: ${log_file}"
  echo "Command: bash ${TRAIN_SCRIPT} ${config}"
  if ! stdbuf -oL -eL bash "${TRAIN_SCRIPT}" "${config}" >"${log_file}" 2>&1; then
    echo "ERROR: ${tag} failed; continuing with the remaining configs; see ${log_file}" >&2
    failed_tags+=("${tag}")
    continue
  fi
  echo "Finished ${idx}/${total}: ${tag}"
done

if (( ${#failed_tags[@]} )); then
  echo "ERROR: ${#failed_tags[@]} of ${total} run(s) failed: ${failed_tags[*]}" >&2
  exit 1
fi

echo "All ${total} main-cyber algorithm runs completed."
echo "Training logs: ${LOG_DIR}"
echo "Online evaluation JSONL: ${SCRIPT_DIR}/online-evaluation"
