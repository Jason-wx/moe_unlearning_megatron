#!/usr/bin/env bash
set -uo pipefail

# Run self-contained training configs serially on one allocation. Training and checkpoint
# behavior comes entirely from each YAML; this runner evaluates its epoch-1 export.
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${OU_PYTHON:-${REPO_ROOT}/envs/open-unlearning/bin/python}
SWEEP_NAME=${SWEEP_NAME:-interns2-megatron-$(date +%Y%m%d_%H%M%S)}
EVAL_TASKS=${EVAL_TASKS:-wmdp,mmlu}
DRY_RUN=${DRY_RUN:-false}
WORKFLOW_MODE=${WORKFLOW_MODE:-group}
CHECKPOINT_ROOT=${OU_MEGATRON_CHECKPOINT_ROOT:-/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/wangxiao1/checkpoints/unlearn/megatron/v3}

if [[ "${WORKFLOW_MODE}" == "group" && "$#" -eq 0 ]]; then
  echo "Usage: scripts/run_megatron_train_eval_group.sh CONFIG.yaml..." >&2
  exit 2
fi
if [[ ! "${SWEEP_NAME}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "Unsafe SWEEP_NAME: ${SWEEP_NAME}" >&2
  exit 2
fi

if [[ -n "${OU_EVAL_ROOT:-}" ]]; then
  EVAL_ROOT=${OU_EVAL_ROOT}
elif [[ -n "${OU_RUNTIME_ROOT:-}" ]]; then
  EVAL_ROOT=${OU_RUNTIME_ROOT}/evaluations
else
  EVAL_ROOT=${REPO_ROOT}/evaluations
fi
EVAL_TASK_LABEL=${EVAL_TASKS//,/_}
EVAL_TASK_LABEL=${EVAL_TASK_LABEL// /_}
if [[ ! "${EVAL_TASK_LABEL}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "Unsafe EVAL_TASKS label: ${EVAL_TASK_LABEL}" >&2
  exit 2
fi

if [[ "${WORKFLOW_MODE}" != "group" ]]; then
  echo "Unsupported WORKFLOW_MODE: ${WORKFLOW_MODE}" >&2
  exit 2
fi

configs=("$@")

failures=0
completed=0
total=${#configs[@]}
cd "${REPO_ROOT}"

for config in "${configs[@]}"; do
  if [[ ! -f "${config}" ]]; then
    echo "Training config not found: ${config}" >&2
    failures=$((failures + 1))
    continue
  fi
  algorithm=$("${PYTHON_BIN}" -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1], encoding="utf-8"))["algorithm"]["name"])' "${config}")
  output_name=$("${PYTHON_BIN}" -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1], encoding="utf-8"))["training"]["output_name"])' "${config}")
  completed=$((completed + 1))
  model_root="${CHECKPOINT_ROOT}/${output_name}"
  checkpoint_epoch="${model_root}/checkpoint-epoch-1"
  echo "workflow_train_start=${completed}/${total} algorithm=${algorithm} output=${model_root}"

  train_args=("${config}")
  if [[ "${DRY_RUN}" == "true" ]]; then
    train_args+=(--dry-run)
  fi

  if ! bash scripts/train_megatron.sh "${train_args[@]}"; then
    echo "workflow_train_failed algorithm=${algorithm}" >&2
    failures=$((failures + 1))
    continue
  fi

  for checkpoint_spec in "epoch1:${checkpoint_epoch}"
  do
    checkpoint_label=${checkpoint_spec%%:*}
    checkpoint_dir=${checkpoint_spec#*:}
    result_dir="${EVAL_ROOT}/unlearning/interns2/${SWEEP_NAME}/lm_eval_results_${algorithm}_${checkpoint_label}_${EVAL_TASK_LABEL}"
    echo "workflow_eval_start algorithm=${algorithm} checkpoint=${checkpoint_label} model=${checkpoint_dir}"
    if [[ "${DRY_RUN}" == "true" ]]; then
      printf 'dry_run_eval_command='
      printf '%q ' bash scripts/lm_eval_accelerate.sh "${checkpoint_dir}" "${EVAL_TASKS}" "${result_dir}"
      printf '\n'
      continue
    fi
    if [[ ! -d "${checkpoint_dir}" ]]; then
      echo "workflow_checkpoint_missing algorithm=${algorithm} checkpoint=${checkpoint_dir}" >&2
      failures=$((failures + 1))
      continue
    fi
    if ! bash scripts/lm_eval_accelerate.sh "${checkpoint_dir}" "${EVAL_TASKS}" "${result_dir}"; then
      echo "workflow_eval_failed algorithm=${algorithm} checkpoint=${checkpoint_label}" >&2
      failures=$((failures + 1))
      continue
    fi
    echo "workflow_eval_complete algorithm=${algorithm} checkpoint=${checkpoint_label} result=${result_dir}"
  done
done

echo "workflow_complete algorithms=${total} failures=${failures} sweep=${SWEEP_NAME}"
if (( failures > 0 )); then
  exit 1
fi
