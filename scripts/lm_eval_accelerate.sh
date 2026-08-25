#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 3 ]]; then
    echo "Usage: scripts/evaluate.sh MODEL_DIR TASKS OUTPUT_DIR" >&2
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
REPO_DIR="${REPO_DIR:-${OPEN_UNLEARNING_DIR}}"
ENV_DIR="${ENV_DIR:-${OPEN_UNLEARNING_DIR}/envs/open-unlearning}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
CPU_THREADS_PER_PROCESS="${CPU_THREADS_PER_PROCESS:-20}"
LM_EVAL_BATCH_SIZE="${LM_EVAL_BATCH_SIZE:-auto}"
LM_EVAL_LIMIT="${LM_EVAL_LIMIT:-}"

if [[ ! -d "${MODEL_DIR}" ]]; then
    echo "Model directory does not exist: ${MODEL_DIR}" >&2
    exit 1
fi
if [[ ! -d "${REPO_DIR}" ]]; then
    echo "Evaluation repository does not exist: ${REPO_DIR}" >&2
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

cd "${REPO_DIR}"

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

run_lm_eval "${MODEL_DIR}" "${TMP_DIR}/full"
FULL_JSON="$(latest_result_json "${TMP_DIR}/full")"

if [[ ! -f "${MODEL_DIR}/expert_expansion_manifest.json" ]]; then
    cp "${FULL_JSON}" "${OUTPUT_DIR}/result.json"
    echo "No expert_expansion_manifest.json found; wrote full-only result to ${OUTPUT_DIR}/result.json"
    exit 0
fi

ABLATION_SCRIPT="${REPO_DIR}/scripts/ablate_harmful_experts.py"
if [[ ! -f "${ABLATION_SCRIPT}" ]]; then
    echo "Model requires expert-expansion ablation, but helper is missing: ${ABLATION_SCRIPT}" >&2
    echo "Set REPO_DIR to a repository that provides scripts/ablate_harmful_experts.py." >&2
    exit 1
fi

"${PYTHON_BIN}" "${ABLATION_SCRIPT}" \
    --model-dir "${MODEL_DIR}" \
    --output-dir "${TMP_DIR}/ablated_model" \
    --overwrite

run_lm_eval "${TMP_DIR}/ablated_model" "${TMP_DIR}/ablated"
ABLATED_JSON="$(latest_result_json "${TMP_DIR}/ablated")"

"${PYTHON_BIN}" - "${FULL_JSON}" "${ABLATED_JSON}" "${OUTPUT_DIR}" "${MODEL_DIR}" "${TASKS}" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

full_results = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
ablated_results = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
output_dir = Path(sys.argv[3])
model_dir = sys.argv[4]
tasks = sys.argv[5]

def metric_base(name: str) -> str:
    return name.split(",", 1)[0]

def result_sections(results: dict[str, Any]) -> dict[str, dict[str, Any]]:
    sections: dict[str, dict[str, Any]] = {}
    for section_name in ("results", "groups"):
        for task, values in (results.get(section_name) or {}).items():
            if isinstance(values, dict):
                sections.setdefault(task, values)
    return sections

def summary_score(task: str, values: dict[str, Any]) -> tuple[str, float] | None:
    preferred = {"mmlu": ("acc",), "mmlu_pro": ("exact_match",), "wmdp": ("acc",)}.get(task, ("acc", "exact_match"))
    for base in preferred:
        for name, value in values.items():
            if metric_base(str(name)) == base and ",stderr" not in str(name) and not isinstance(value, bool) and isinstance(value, (int, float)):
                return base, float(value)
    return None

full_tasks = result_sections(full_results)
ablated_tasks = result_sections(ablated_results)
summary: dict[str, dict[str, float | str]] = {}
for task in sorted(set(full_tasks) & set(ablated_tasks)):
    full_score = summary_score(task, full_tasks[task])
    ablated_score = summary_score(task, ablated_tasks[task])
    if full_score is None or ablated_score is None or full_score[1] == 0.0:
        continue
    base, full_value = full_score
    ablated_value = ablated_score[1]
    summary[task] = {"metric": base, "full": full_value, "ablated": ablated_value, "drop": (full_value - ablated_value) / abs(full_value)}

result = {"model_dir": model_dir, "tasks": tasks, "summary": summary, "full": full_results, "ablated": ablated_results}
output_path = output_dir / "result.json"
output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
for task, values in summary.items():
    print(f"{task}/{values['metric']}_drop={values['drop']:.6f}")
print(f"Wrote lm-eval result to {output_path}")
PY
