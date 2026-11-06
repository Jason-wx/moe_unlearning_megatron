#!/usr/bin/env bash
set -euo pipefail

# Offline lm-eval of one HF checkpoint on all visible GPUs (accelerate).
#
#   scripts/eval.sh MODEL_DIR TASKS OUTPUT_DIR
#   scripts/eval.sh <output_root>/<output_name>/checkpoint-step-2142 wmdp_bio,mmlu \
#       evaluations/unlearning/<run>/lm_eval_results_<run>_wmdp_bio_mmlu
#
# OUTPUT_DIR is replaced on every run, so it must live under evaluations/ and its
# name must start with lm_eval_results_. The result is OUTPUT_DIR/result.json.
# Optional: NUM_PROCESSES (default 8), LM_EVAL_BATCH_SIZE (default auto),
# LM_EVAL_LIMIT (questions per task, for quick checks).

if [[ "$#" -ne 3 ]]; then
    echo "Usage: scripts/eval.sh MODEL_DIR TASKS OUTPUT_DIR" >&2
    exit 1
fi

MODEL_DIR="$1"
TASKS="$2"
OUTPUT_DIR="$3"

if [[ "${MODEL_DIR}" == /gpfs2-shared-public/* ]]; then
    MODEL_DIR="/mnt/shared-storage-gpfs2/${MODEL_DIR#/}"
fi
MODEL_DIR="$(realpath "${MODEL_DIR}")"
OUTPUT_DIR="$(realpath -m "${OUTPUT_DIR}")"

OPEN_UNLEARNING_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [[ -n "${OU_EVAL_ROOT:-}" ]]; then
    EVAL_ROOT="${OU_EVAL_ROOT}"
elif [[ -n "${OU_RUNTIME_ROOT:-}" ]]; then
    EVAL_ROOT="${OU_RUNTIME_ROOT}/evaluations"
else
    EVAL_ROOT="${OPEN_UNLEARNING_DIR}/evaluations"
fi
EVAL_ROOT="$(realpath -m "${EVAL_ROOT}")"
if [[ "${OUTPUT_DIR}" == "${EVAL_ROOT}" || "${OUTPUT_DIR}" == "${OPEN_UNLEARNING_DIR}" || "${OUTPUT_DIR}" == "/" ]]; then
    echo "Refusing unsafe OUTPUT_DIR: ${OUTPUT_DIR}" >&2
    echo "Use a dedicated lm_eval_results_* subdirectory under ${EVAL_ROOT}." >&2
    exit 2
fi
if [[ "${OUTPUT_DIR}" != "${EVAL_ROOT}/"* ]]; then
    echo "Refusing OUTPUT_DIR outside evaluation root: ${OUTPUT_DIR}" >&2
    echo "Expected a subdirectory under ${EVAL_ROOT}" >&2
    exit 2
fi
if [[ "$(basename "${OUTPUT_DIR}")" != lm_eval_results_* ]]; then
    echo "Refusing OUTPUT_DIR that is not an lm_eval result directory: ${OUTPUT_DIR}" >&2
    echo "Expected basename to start with lm_eval_results_" >&2
    exit 2
fi
if [[ -e "${OUTPUT_DIR}" && ! -d "${OUTPUT_DIR}" ]]; then
    echo "Refusing OUTPUT_DIR because it exists and is not a directory: ${OUTPUT_DIR}" >&2
    exit 2
fi

TMP_DIR="${OUTPUT_DIR}.tmp_${TASKS//,/_}_$$"
ENV_DIR="${ENV_DIR:-${OPEN_UNLEARNING_DIR}/envs/open-unlearning}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
CPU_THREADS_PER_PROCESS="${CPU_THREADS_PER_PROCESS:-20}"
LM_EVAL_BATCH_SIZE="${LM_EVAL_BATCH_SIZE:-auto}"
LM_EVAL_LIMIT="${LM_EVAL_LIMIT:-}"

if [[ ! -d "${MODEL_DIR}" ]]; then
    echo "Model directory does not exist: ${MODEL_DIR}" >&2
    exit 1
fi
if [[ ! "${NUM_PROCESSES}" =~ ^[1-9][0-9]*$ ]]; then
    echo "NUM_PROCESSES must be a positive integer: ${NUM_PROCESSES}" >&2
    exit 1
fi
PYTHON_BIN="${ENV_DIR}/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Missing executable python under ${ENV_DIR}/bin" >&2
    exit 1
fi
if [[ -f "${ENV_DIR}/bin/activate" ]]; then
    source "${ENV_DIR}/bin/activate"
fi

if ! "${PYTHON_BIN}" - "${NUM_PROCESSES}" <<'PY'
from __future__ import annotations

import sys

try:
    import accelerate
    import datasets
    import lm_eval
    import torch
    import transformers
except Exception as exc:
    raise SystemExit(f"Evaluation environment import failed: {exc}") from exc

def version_tuple(value: str) -> tuple[int, ...]:
    parts: list[int] = []
    for part in value.split("."):
        digits = "".join(character for character in part if character.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)

if version_tuple(datasets.__version__) < (4, 8, 4):
    raise SystemExit(
        f"datasets>=4.8.4 is required for the shared offline cache; got {datasets.__version__}"
    )

required_gpus = int(sys.argv[1])
visible_gpus = torch.cuda.device_count()
if not torch.cuda.is_available() or visible_gpus < required_gpus:
    raise SystemExit(
        f"Insufficient CUDA devices: required {required_gpus}, visible {visible_gpus}"
    )

print(
    "Evaluation environment OK: "
    f"python={sys.version.split()[0]} "
    f"torch={torch.__version__} "
    f"transformers={transformers.__version__} "
    f"accelerate={accelerate.__version__} "
    f"lm_eval={getattr(lm_eval, '__version__', 'unknown')} "
    f"datasets={datasets.__version__} "
    f"gpus={visible_gpus}"
)
PY
then
    echo "Evaluation environment preflight failed: ${ENV_DIR}" >&2
    exit 1
fi

cd "${OPEN_UNLEARNING_DIR}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-${PYTORCH_CUDA_ALLOC_CONF}}"

CACHE_ROOT="/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/jincheng/.cache"
KERNEL_CACHE_ROOT="${CACHE_ROOT}/sgtm/kernel-cache"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CACHE_ROOT}}"
export HF_HOME="${HF_HOME:-${CACHE_ROOT}/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HUB_CACHE}}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TILELANG_CACHE_DIR="${TILELANG_CACHE_DIR:-${KERNEL_CACHE_ROOT}/tilelang}"
export TILELANG_TMP_DIR="${TILELANG_TMP_DIR:-${KERNEL_CACHE_ROOT}/tilelang/tmp}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${KERNEL_CACHE_ROOT}/triton}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${KERNEL_CACHE_ROOT}/torch_extensions}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${KERNEL_CACHE_ROOT}/torchinductor}"

safe_remove_eval_dir() {
    local target
    local target_name
    target="$(realpath -m "$1")"
    target_name="$(basename "${target}")"
    if [[ "${target}" != "${EVAL_ROOT}/"* || "${target_name}" != lm_eval_results_* ]]; then
        echo "Refusing to remove unsafe eval path: ${target}" >&2
        exit 2
    fi
    rm -rf "${target}"
}

safe_remove_eval_dir "${OUTPUT_DIR}"
safe_remove_eval_dir "${TMP_DIR}"
mkdir -p \
    "${OUTPUT_DIR}" \
    "${TMP_DIR}" \
    "${XDG_CACHE_HOME}" \
    "${HF_HOME}" \
    "${HF_DATASETS_CACHE}" \
    "${HUGGINGFACE_HUB_CACHE}" \
    "${TILELANG_CACHE_DIR}" \
    "${TILELANG_TMP_DIR}" \
    "${TRITON_CACHE_DIR}" \
    "${TORCH_EXTENSIONS_DIR}" \
    "${TORCHINDUCTOR_CACHE_DIR}"
trap 'safe_remove_eval_dir "${TMP_DIR}"' EXIT

run_lm_eval() {
    local model_dir="$1"
    local output_dir="$2"
    local -a limit_args=()
    if [[ -n "${LM_EVAL_LIMIT}" ]]; then
        limit_args=(--limit "${LM_EVAL_LIMIT}")
    fi
    echo "Running lm-eval: model=${model_dir} tasks=${TASKS} output=${output_dir}"
    "${PYTHON_BIN}" -m accelerate.commands.accelerate_cli launch \
        --num_processes "${NUM_PROCESSES}" \
        --num_cpu_threads_per_process "${CPU_THREADS_PER_PROCESS}" \
        -m lm_eval run \
        --model hf \
        --model_args "pretrained=${model_dir},dtype=bfloat16,trust_remote_code=True" \
        --tasks "${TASKS}" \
        --batch_size "${LM_EVAL_BATCH_SIZE}" \
        "${limit_args[@]}" \
        --output_path "${output_dir}"
}

latest_result_json() {
    local result
    result="$(find "$1" -type f -name 'results_*.json' -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)"
    if [[ -z "${result}" ]]; then
        echo "No lm-eval results_*.json found under $1" >&2
        exit 1
    fi
    printf '%s\n' "${result}"
}

run_lm_eval "${MODEL_DIR}" "${TMP_DIR}/run"
RESULT_JSON="$(latest_result_json "${TMP_DIR}/run")"
cp "${RESULT_JSON}" "${OUTPUT_DIR}/result.json"
echo "Wrote lm-eval result to ${OUTPUT_DIR}/result.json"
