#!/usr/bin/env bash
set -Eeuo pipefail

# Debug round 4: test the PDUState.update() hypothesis. Rounds 1-3 ruled out
# recompute, loss-tensor layout, and loss composition. The two things every
# crashing arm shares that no working run (GD / CEU) has: (a) mixed
# forget+retain packs through the full-logit path, (b) PDUState.update()'s
# mid-forward dist.all_reduce + .item() host sync. This round removes (b):
# OU_PDU_NO_DUAL_UPDATE=1 makes update() a no-op (lambda pinned at alpha),
# leaving the full two-term PDU loss and the mixed-pack batch untouched.
# 2 steps, recompute full (the crashing layout), CUDA_LAUNCH_BLOCKING=1.
REPO_ROOT=${OU_REPO_ROOT:-/mnt/shared-storage-user/safethm-share/wangxiao1/Safe_MoE/open-unlearning}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export CUDA_LAUNCH_BLOCKING=1

CONFIG=main/gemma4/pdu_eps_sweep/debug/pdu_debug_blocking.yaml
if [[ ! -f "${REPO_ROOT}/${CONFIG}" ]]; then
  echo "ERROR: debug config not found: ${REPO_ROOT}/${CONFIG}" >&2
  exit 1
fi

echo "=== arm 1/1: OU_PDU_NO_DUAL_UPDATE=1 (skip PDUState.update) ==="
OU_PDU_NO_DUAL_UPDATE=1 bash "${SCRIPT_DIR}/main_run_train.sh" "${CONFIG}" || echo "arm 1 failed"
