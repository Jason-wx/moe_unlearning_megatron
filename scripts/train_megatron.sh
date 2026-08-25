#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONFIG=${1:?usage: scripts/train_megatron.sh CONFIG [extra ou arguments]}
shift
PYTHON_BIN=${OU_PYTHON:-${REPO_ROOT}/envs/open-unlearning/bin/python}
TORCHRUN_BIN=${OU_TORCHRUN:-${REPO_ROOT}/envs/open-unlearning/bin/torchrun}
NUM_GPUS=${NUM_GPUS:-8}

export OU_REPO_ROOT=${OU_REPO_ROOT:-${REPO_ROOT}}
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_MODULES_CACHE=${HF_MODULES_CACHE:-/tmp/open-unlearning-hf-modules}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export USE_MCORE_GDN=${USE_MCORE_GDN:-1}
export TORCHINDUCTOR_COMPILE_THREADS=${TORCHINDUCTOR_COMPILE_THREADS:-2}
export WANDB_MODE=offline

ENV_SITE=$("${PYTHON_BIN}" -c 'import site; print(site.getsitepackages()[0])')
export LD_LIBRARY_PATH="${ENV_SITE}/nvidia/cudnn/lib:${ENV_SITE}/nvidia/nccl/lib:${ENV_SITE}/torch/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

cd "${REPO_ROOT}"
if [[ " ${*} " == *" --dry-run "* ]]; then
  exec "${PYTHON_BIN}" -m open_unlearning.cli train \
    --backend megatron --config "${CONFIG}" "$@"
fi
exec "${TORCHRUN_BIN}" --standalone --nproc-per-node "${NUM_GPUS}" \
  -m open_unlearning.cli train --backend megatron --config "${CONFIG}" "$@"

