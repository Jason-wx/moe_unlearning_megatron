#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONFIG=${1:?usage: scripts/train_deepspeed.sh CONFIG [extra ou arguments]}
shift
PYTHON_BIN=${OU_PYTHON:-${REPO_ROOT}/envs/open-unlearning/bin/python}

export OU_REPO_ROOT=${OU_REPO_ROOT:-${REPO_ROOT}}
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_MODULES_CACHE=${HF_MODULES_CACHE:-/tmp/open-unlearning-hf-modules}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/tmp/open-unlearning-triton}
export WANDB_MODE=offline

cd "${REPO_ROOT}"
exec "${PYTHON_BIN}" -m open_unlearning.cli train \
  --backend deepspeed --config "${CONFIG}" "$@"

