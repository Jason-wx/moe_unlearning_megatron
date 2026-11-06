#!/usr/bin/env bash
set -Eeuo pipefail

# Debug round 8: validate the AUTOMATIC int32 guard (no env vars).
# The g2 sweep resubmit crashed identically to rounds 1-5 despite the round-6b
# fix — but 6b forced the split via OU_PDU_SPLIT_FORWARD=1 while the sweep
# relies on _pdu_needs_split_forward's model introspection. This arm runs the
# 2-step config with NO env override: the [pdu-split-guard] log line prints
# the resolved vocab and the guard decision, proving whether introspection
# works through the real DDP -> Gemma4GPTModel -> language_model.output_layer
# chain (and whether the split then runs clean).
REPO_ROOT=${OU_REPO_ROOT:-/mnt/shared-storage-user/safethm-share/wangxiao1/Safe_MoE/open-unlearning}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

unset OU_PDU_SPLIT_FORWARD OU_PDU_BISECT OU_PDU_NO_DUAL_UPDATE OU_PDU_CONTIG_LOSS
export CUDA_LAUNCH_BLOCKING=1

CONFIG=main/gemma4/pdu_eps_sweep/debug/pdu_debug_blocking.yaml
if [[ ! -f "${REPO_ROOT}/${CONFIG}" ]]; then
  echo "ERROR: debug config not found: ${REPO_ROOT}/${CONFIG}" >&2
  exit 1
fi

echo "=== arm 1/1: AUTO GUARD ONLY (no env overrides) ==="
bash "${SCRIPT_DIR}/main_run_train.sh" "${CONFIG}" || echo "arm 1 failed"
