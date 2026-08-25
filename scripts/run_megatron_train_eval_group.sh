#!/usr/bin/env bash
set -uo pipefail

# Run a group of algorithms serially on one allocation. Each algorithm trains
# for one complete epoch, exports step 300 and epoch 1, and evaluates both
# checkpoints before the next algorithm starts.
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONFIG=${CONFIG:-configs/experiments/interns2_pure_npo.yaml}
SWEEP_NAME=${SWEEP_NAME:-interns2-megatron-$(date +%Y%m%d_%H%M%S)}
INTERMEDIATE_STEP=${INTERMEDIATE_STEP:-300}
EVAL_TASKS=${EVAL_TASKS:-wmdp,mmlu}
DRY_RUN=${DRY_RUN:-false}
CHECKPOINT_ROOT=${OU_MEGATRON_CHECKPOINT_ROOT:-/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/wangxiao1/checkpoints/unlearn/megatron}

if [[ "$#" -eq 0 ]]; then
  echo "Usage: scripts/run_megatron_train_eval_group.sh ALGORITHM..." >&2
  exit 2
fi
if [[ ! "${SWEEP_NAME}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "Unsafe SWEEP_NAME: ${SWEEP_NAME}" >&2
  exit 2
fi
if [[ ! "${INTERMEDIATE_STEP}" =~ ^[1-9][0-9]*$ ]]; then
  echo "INTERMEDIATE_STEP must be a positive integer" >&2
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

normalize_algorithm() {
  case "$1" in
    grad_ascent|grad_diff|pure_npo|token_npo|token_simnpo|wga|satimp|ceu|pdu|undial|rmu)
      printf '%s\n' "$1"
      ;;
    npo)
      printf '%s\n' pure_npo
      ;;
    simnpo)
      printf '%s\n' token_simnpo
      ;;
    *)
      echo "Unsupported Megatron algorithm component: $1" >&2
      return 2
      ;;
  esac
}

algorithms=()
for requested in "$@"; do
  if ! normalized=$(normalize_algorithm "${requested}"); then
    exit 2
  fi
  algorithms+=("${normalized}")
done

failures=0
completed=0
total=${#algorithms[@]}
cd "${REPO_ROOT}"

for algorithm in "${algorithms[@]}"; do
  completed=$((completed + 1))
  run_name="${SWEEP_NAME}-${algorithm}"
  model_root="${CHECKPOINT_ROOT}/${run_name}"
  checkpoint_step="${model_root}/checkpoint-step-${INTERMEDIATE_STEP}"
  checkpoint_epoch="${model_root}/checkpoint-epoch-1"
  echo "workflow_train_start=${completed}/${total} algorithm=${algorithm} output=${model_root}"

  train_args=(
    "${CONFIG}"
    --algorithm "${algorithm}"
    --train-epochs 1
    --save-at-step "${INTERMEDIATE_STEP}"
    --save-at-epoch-end
    --output-dir "saves/${run_name}"
  )
  if [[ "${DRY_RUN}" == "true" ]]; then
    train_args+=(--dry-run)
  fi

  if ! bash scripts/train_megatron.sh "${train_args[@]}"; then
    echo "workflow_train_failed algorithm=${algorithm}" >&2
    failures=$((failures + 1))
    continue
  fi

  for checkpoint_spec in \
    "step${INTERMEDIATE_STEP}:${checkpoint_step}" \
    "epoch1:${checkpoint_epoch}"
  do
    checkpoint_label=${checkpoint_spec%%:*}
    checkpoint_dir=${checkpoint_spec#*:}
    result_dir="${EVAL_ROOT}/unlearning/interns2/${SWEEP_NAME}/lm_eval_results_${algorithm}_${checkpoint_label}_${EVAL_TASK_LABEL}"
    echo "workflow_eval_start algorithm=${algorithm} checkpoint=${checkpoint_label} model=${checkpoint_dir}"
    if [[ "${DRY_RUN}" == "true" ]]; then
      printf 'dry_run_eval_command='
      printf '%q ' bash scripts/evaluate.sh "${checkpoint_dir}" "${EVAL_TASKS}" "${result_dir}"
      printf '\n'
      continue
    fi
    if [[ ! -d "${checkpoint_dir}" ]]; then
      echo "workflow_checkpoint_missing algorithm=${algorithm} checkpoint=${checkpoint_dir}" >&2
      failures=$((failures + 1))
      continue
    fi
    if ! bash scripts/evaluate.sh "${checkpoint_dir}" "${EVAL_TASKS}" "${result_dir}"; then
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
