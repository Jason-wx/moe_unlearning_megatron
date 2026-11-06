#!/usr/bin/env bash
set -Eeuo pipefail

# Debug runner for the gemma4 x PDU first-backward illegal-memory-access:
# runs the two 2-step debug configs serially under CUDA_LAUNCH_BLOCKING=1 so
# the async CUDA error surfaces at the true faulting kernel instead of the
# first victim (TE rmsnorm bwd).
#   1) pdu_debug_blocking.yaml   — exact reproduction of the sweep layout
#      (activation_recompute: full), pinpoint the faulting kernel
#   2) pdu_debug_norecompute.yaml — fix candidate (activation_recompute:
#      null, the RMU precedent on gemma4)
# Paths are repo-root-relative (the rjob cds to REPO_ROOT first).
REPO_ROOT=${OU_REPO_ROOT:-/mnt/shared-storage-user/safethm-share/wangxiao1/Safe_MoE/open-unlearning}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export CUDA_LAUNCH_BLOCKING=1
echo "CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING} (async errors now surface at the faulting kernel)"

CONFIGS=(
  main/gemma4/pdu_eps_sweep/debug/pdu_debug_blocking.yaml
  main/gemma4/pdu_eps_sweep/debug/pdu_debug_norecompute.yaml
)
for cfg in "${CONFIGS[@]}"; do
  if [[ ! -f "${REPO_ROOT}/${cfg}" ]]; then
    echo "ERROR: debug config not found: ${REPO_ROOT}/${cfg}" >&2
    exit 1
  fi
done

exec bash "${SCRIPT_DIR}/main_run_train.sh" "${CONFIGS[@]}"
