#!/usr/bin/env bash
set -Eeuo pipefail

# Debug round 5: BRANCH-level bisect. Round 3's gate was flawed — it only
# reweighted the loss AFTER both branches (logits[forget_rows] dispersion AND
# logits[retain_rows] + full-vocab NLL) had already executed their forward
# ops, so the retain branch was never actually removed. CEU, which runs clean
# on gemma4, early-returns and never touches retain_rows at all. This round
# skips each branch's computation entirely:
#   arm 1: OU_PDU_BISECT=forget — structurally identical to CEU's clean path
#          (full logits -> forget rows -> objective -> return); no retain_rows
#          indexing, no dual update
#   arm 2: OU_PDU_BISECT=retain — only logits[retain_rows] + full-vocab NLL;
#          no dispersion branch
# Both 2 steps, recompute full (the crashing layout), CUDA_LAUNCH_BLOCKING=1.
REPO_ROOT=${OU_REPO_ROOT:-/mnt/shared-storage-user/safethm-share/wangxiao1/Safe_MoE/open-unlearning}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export CUDA_LAUNCH_BLOCKING=1

CONFIG=main/gemma4/pdu_eps_sweep/debug/pdu_debug_blocking.yaml
if [[ ! -f "${REPO_ROOT}/${CONFIG}" ]]; then
  echo "ERROR: debug config not found: ${REPO_ROOT}/${CONFIG}" >&2
  exit 1
fi

echo "=== arm 1/2: OU_PDU_BISECT=forget (skip retain branch entirely; CEU-shaped) ==="
OU_PDU_BISECT=forget bash "${SCRIPT_DIR}/main_run_train.sh" "${CONFIG}" || echo "arm 1 failed; continuing"

echo "=== arm 2/2: OU_PDU_BISECT=retain (skip dispersion branch entirely) ==="
OU_PDU_BISECT=retain bash "${SCRIPT_DIR}/main_run_train.sh" "${CONFIG}" || echo "arm 2 failed"
