#!/usr/bin/env bash
set -Eeuo pipefail

# Debug round 2 for the gemma4 x PDU first-backward illegal-memory-access.
# Two 2-step arms on the same safemoe image, both under CUDA_LAUNCH_BLOCKING=1:
#   1) pdu_debug_blocking.yaml — plain reproduction (recompute full)
#   2) pdu_debug_contig.yaml   — OU_PDU_CONTIG_LOSS=1: contiguous fp32 CE /
#      dispersion in objectives.py (env-gated code path, default off)
# A failed arm is logged and the sequence continues (main_run_train.sh
# semantics) so arm 2 still runs after arm 1's expected crash.
REPO_ROOT=${OU_REPO_ROOT:-/mnt/shared-storage-user/safethm-share/wangxiao1/Safe_MoE/open-unlearning}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export CUDA_LAUNCH_BLOCKING=1

CONFIG1=main/gemma4/pdu_eps_sweep/debug/pdu_debug_blocking.yaml
CONFIG2=main/gemma4/pdu_eps_sweep/debug/pdu_debug_contig.yaml
for cfg in "${CONFIG1}" "${CONFIG2}"; do
  if [[ ! -f "${REPO_ROOT}/${cfg}" ]]; then
    echo "ERROR: debug config not found: ${REPO_ROOT}/${cfg}" >&2
    exit 1
  fi
done

echo "=== arm 1/2: plain reproduction (recompute full) ==="
bash "${SCRIPT_DIR}/main_run_train.sh" "${CONFIG1}" || echo "arm 1 failed (expected); continuing"

echo "=== arm 2/2: OU_PDU_CONTIG_LOSS=1 layout fix ==="
OU_PDU_CONTIG_LOSS=1 bash "${SCRIPT_DIR}/main_run_train.sh" "${CONFIG2}" || echo "arm 2 failed"
