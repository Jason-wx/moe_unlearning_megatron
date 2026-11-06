#!/usr/bin/env bash
set -Eeuo pipefail

# Debug round 3: bisect which side of the PDU objective triggers the
# gemma4 x PDU rmsnorm_bwd illegal access (round 2 ruled out tensor layout).
#   arm 1: OU_PDU_BISECT=forget — loss = gamma * dispersion only
#          (retain CE detached; CEU proves forget-only full-logit works)
#   arm 2: OU_PDU_BISECT=retain — loss = lambda * violation only
#          (dispersion detached)
# Both 2 steps, recompute full (the crashing layout), CUDA_LAUNCH_BLOCKING=1.
REPO_ROOT=${OU_REPO_ROOT:-/mnt/shared-storage-user/safethm-share/wangxiao1/Safe_MoE/open-unlearning}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export CUDA_LAUNCH_BLOCKING=1

CONFIG1=main/gemma4/pdu_eps_sweep/debug/pdu_debug_forgetonly.yaml
CONFIG2=main/gemma4/pdu_eps_sweep/debug/pdu_debug_retainonly.yaml
for cfg in "${CONFIG1}" "${CONFIG2}"; do
  if [[ ! -f "${REPO_ROOT}/${cfg}" ]]; then
    echo "ERROR: debug config not found: ${REPO_ROOT}/${cfg}" >&2
    exit 1
  fi
done

echo "=== arm 1/2: OU_PDU_BISECT=forget (dispersion only) ==="
OU_PDU_BISECT=forget bash "${SCRIPT_DIR}/main_run_train.sh" "${CONFIG1}" || echo "arm 1 failed; continuing"

echo "=== arm 2/2: OU_PDU_BISECT=retain (dual retain term only) ==="
OU_PDU_BISECT=retain bash "${SCRIPT_DIR}/main_run_train.sh" "${CONFIG2}" || echo "arm 2 failed"
