#!/usr/bin/env bash
set -Eeuo pipefail

# Debug round 6: int32-overflow hypothesis test and fix candidate.
# Root-cause analysis of rounds 1-5: a PDU microbatch mixes one forget and
# one retain pack, so gemma4's full-logits output is 2 x 4096 x 262144 =
# 2**31 elements (exactly INT32_MAX + 1). Every run that materializes a
# >=2**31-element logit tensor crashes; CEU (2**30, forget-only) and qwen35
# PDU (smaller vocab) run clean. OU_PDU_SPLIT_FORWARD=1 forwards each role
# separately (two CEU-sized 2**30 forwards) and combines the PDU loss.
# 2 steps, recompute full (the crashing layout), CUDA_LAUNCH_BLOCKING=1.
REPO_ROOT=${OU_REPO_ROOT:-/mnt/shared-storage-user/safethm-share/wangxiao1/Safe_MoE/open-unlearning}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export CUDA_LAUNCH_BLOCKING=1

CONFIG=main/gemma4/pdu_eps_sweep/debug/pdu_debug_blocking.yaml
if [[ ! -f "${REPO_ROOT}/${CONFIG}" ]]; then
  echo "ERROR: debug config not found: ${REPO_ROOT}/${CONFIG}" >&2
  exit 1
fi

echo "=== arm 1/1: OU_PDU_SPLIT_FORWARD=1 (per-role forwards, no 2^31 tensor) ==="
OU_PDU_SPLIT_FORWARD=1 bash "${SCRIPT_DIR}/main_run_train.sh" "${CONFIG}" || echo "arm 1 failed"
