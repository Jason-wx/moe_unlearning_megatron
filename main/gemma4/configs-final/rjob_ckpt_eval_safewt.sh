#!/usr/bin/env bash
set -euo pipefail

# Submit safewt rjobs that offline-evaluate the EIGHT gemma4 bio FINAL-line
# final checkpoints (step 2133, all except PDU whose run stopped at 1400 with
# a WMDP rebound) with the official scripts/eval.sh (full
# wmdp_bio + mmlu, 0-shot, bf16). Four jobs of two ckpts each, mirroring the
# training grouping minus PDU (rjob_final_safewt.sh g2..g5); ckpts within a
# job run SERIALLY, each eval using BOTH GPUs as accelerate data-parallel
# replicas (NUM_PROCESSES=2) - the profile proven by
# scripts/rjob_gemma4_baseline_eval_nocap_safewt.sh. LM_EVAL_BATCH_SIZE=auto:
# bs64 OOMs even on a 140G H200 for gemma4 MMLU long prompts (see that
# script's note; auto picked 25 on the baseline).
# Results land under main/gemma4/configs-final/ckpt-eval/ via OU_EVAL_ROOT:
#   ckpt-eval/lm_eval_results_<ckpt-name>_wmdp_bio_mmlu/result.json
# Already-evaluated ckpts (result.json present) are skipped on resubmit; a
# failed ckpt does not stop its group's second ckpt. Params follow
# scripts/rjob_gemma4_baseline_eval_nocap_safewt.sh (safewt 2-GPU tier) and
# scripts/rjob_safer2ai_example.sh (mounts/env layout).
#
# Examples:
#   BATCH=g1 bash main/gemma4/configs-final/rjob_ckpt_eval_safewt.sh
#   PREVIEW=true BATCH=g1 bash main/gemma4/configs-final/rjob_ckpt_eval_safewt.sh

REPO_ROOT=${OU_REPO_ROOT:-/mnt/shared-storage-user/safethm-share/wangxiao1/Safe_MoE/open-unlearning}

RESOURCE_GROUP=safewt
NAMESPACE=ailab-safewt
CHARGED_GROUP=safewt_gpu
IMAGE=registry.h.pjlab.org.cn/ailab-safethm-sfaethm_cpu/safemoe:jc-work-20260506163353

CKPT_ROOT=/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/wangxiao1/checkpoints/unlearn/megatron/main
EVAL_ROOT=${REPO_ROOT}/main/gemma4/configs-final/ckpt-eval

# name=path entries (arrays deliberately NOT named GROUPS; the cluster
# injects a GROUPS array into bash sessions and assignments to it are
# silently swallowed).
# Paths follow the 10-04 reorg: weights flattened into main/Gemma4/<Algo>/
# (HF files directly under the algo dir, no checkpoint-step-N subfolder).
# name= prefixes keep the original run-step identity so result dir names stay
# aligned with the already-archived lm_eval_results_* directories.
CKPT_CEU="gemma4-final-ceu-nocap-step2133=${CKPT_ROOT}/Gemma4/CEU"
CKPT_RMU="gemma4-final-rmu-nocap-step2133=${CKPT_ROOT}/Gemma4/RMU"
CKPT_PNPO="gemma4-final-pure-npo-nocap-step2133=${CKPT_ROOT}/Gemma4/PureNPO"
CKPT_SATIMP="gemma4-final-satimp-forgetonly-nocap-step2133=${CKPT_ROOT}/Gemma4/SatImp"
CKPT_GD="gemma4-final-grad-diff-nocap-step2133=${CKPT_ROOT}/Gemma4/GradDiff"
CKPT_WGA="gemma4-final-wga-forgetonly-nocap-step2133=${CKPT_ROOT}/Gemma4/WGA"
CKPT_SIMNPO="gemma4-final-simnpo-nocap-step2133=${CKPT_ROOT}/Gemma4/SimNPO"
CKPT_GA="gemma4-final-grad-ascent-nocap-step2133=${CKPT_ROOT}/Gemma4/GradAscent"

BATCH=${BATCH:-}
if (( $# )); then
  CKPT_SPECS=("$@")
elif [[ -n "${BATCH}" ]]; then
  case "${BATCH}" in
    g1) CKPT_SPECS=("${CKPT_CEU}" "${CKPT_RMU}") ;;
    g2) CKPT_SPECS=("${CKPT_PNPO}" "${CKPT_SATIMP}") ;;
    g3) CKPT_SPECS=("${CKPT_GD}" "${CKPT_WGA}") ;;
    g4) CKPT_SPECS=("${CKPT_SIMNPO}" "${CKPT_GA}") ;;
    *) echo "ERROR: unknown BATCH '${BATCH}' (g1|g2|g3|g4)" >&2; exit 1 ;;
  esac
else
  echo "ERROR: select a batch with BATCH=g1..g4 or pass name=path specs." >&2
  exit 1
fi

JOB_NAME=${JOB_NAME:-gemma4-final-ckpt-eval-${BATCH:-custom}}
GPUS=2
CPUS=32
MEMORY_MB=128000
FUSE_COUNT=1
RDMA_COUNT=2
TERMINATION_GRACE_PERIOD_SECONDS=600
RUN_ROOT=${OU_RUN_ROOT:-${REPO_ROOT}/outputs}

# Static name=path list baked into the job at submit time.
CKPT_LIST=$(printf '%s\n' "${CKPT_SPECS[@]}")

# Quick local sanity: every ckpt dir must exist before we ask for GPUs.
while IFS='=' read -r ckpt_name ckpt_path; do
  [[ -z "${ckpt_name}" ]] && continue
  if [[ ! -d "${ckpt_path}" ]]; then
    echo "ERROR: ckpt dir missing: ${ckpt_path}" >&2
    exit 1
  fi
done <<< "${CKPT_LIST}"

RJOB_ARGS=(
  submit
  --priority=9
  --name="${JOB_NAME}"
  --gpu="${GPUS}"
  --memory="${MEMORY_MB}"
  --cpu="${CPUS}"
  --charged-group="${CHARGED_GROUP}"
  --namespace="${NAMESPACE}"
  --private-machine=group
  --custom-resources "brainpp.cn/fuse=${FUSE_COUNT}"
  --custom-resources "rdma/mlnx_shared=${RDMA_COUNT}"
  --termination-grace-period-seconds "${TERMINATION_GRACE_PERIOD_SECONDS}"
  --mount=gpfs://gpfs1/safethm-share:/mnt/shared-storage-user/safethm-share
  --mount=gpfs://gpfs2/gpfs2-shared-public:/mnt/shared-storage-gpfs2/gpfs2-shared-public
  --mount=gpfs://gpfs2/wenxiaoyu-gpfs02:/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02
  --image-pull-policy=IfNotPresent
  --image="${IMAGE}"
  -e DISTRIBUTED_JOB=true
  -- bash -c "set -euo pipefail
cd '${REPO_ROOT}'
export OU_REPO_ROOT='${REPO_ROOT}'
export OU_EVAL_ROOT='${EVAL_ROOT}'
export NUM_PROCESSES='${GPUS}'
export CPU_THREADS_PER_PROCESS='$((CPUS / GPUS))'
export LM_EVAL_BATCH_SIZE='auto'
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export PYTHONUNBUFFERED=1
export HF_HOME='/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/jincheng/.cache/huggingface'
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
RUN_TS=\$(date +%Y%m%d_%H%M%S)
LOG_DIR='${RUN_ROOT}/rjob_logs/${JOB_NAME}'_\${RUN_TS}
mkdir -p \"\${LOG_DIR}\" '${EVAL_ROOT}'
echo 'job=${JOB_NAME} layout=serial-2ckpts-2gpu-dp tasks=wmdp_bio,mmlu bs=auto gpus=${GPUS} eval_root=${EVAL_ROOT}' | tee \"\${LOG_DIR}/metadata.log\"
mapfile -t SPECS <<'CKPTS'
${CKPT_LIST}
CKPTS
FAILED=''
for ckpt_spec in \"\${SPECS[@]}\"; do
  IFS='=' read -r ckpt_name ckpt_path <<< \"\${ckpt_spec}\"
  out_dir='${EVAL_ROOT}/lm_eval_results_'\${ckpt_name}'_wmdp_bio_mmlu'
  echo \"=== [\$(date +%H:%M:%S)] ckpt=\${ckpt_name} -> \${out_dir}\" | tee -a \"\${LOG_DIR}/summary.log\"
  if [[ -f \"\${out_dir}/result.json\" ]]; then
    echo \"SKIP \${ckpt_name}: result.json already present\" | tee -a \"\${LOG_DIR}/summary.log\"
    continue
  fi
  if [[ ! -d \"\${ckpt_path}\" ]]; then
    echo \"FAILED \${ckpt_name}: ckpt dir missing\" | tee -a \"\${LOG_DIR}/summary.log\"
    FAILED=\"\${FAILED} \${ckpt_name}(missing)\"
    continue
  fi
  if bash '${REPO_ROOT}/scripts/eval.sh' \"\${ckpt_path}\" 'wmdp_bio,mmlu' \"\${out_dir}\" > \"\${LOG_DIR}/eval-\${ckpt_name}.log\" 2>&1; then
    echo \"OK \${ckpt_name}\" | tee -a \"\${LOG_DIR}/summary.log\"
  else
    echo \"FAILED \${ckpt_name} (exit \$?) - continuing to next ckpt\" | tee -a \"\${LOG_DIR}/summary.log\"
    FAILED=\"\${FAILED} \${ckpt_name}\"
  fi
done
echo 'job=${JOB_NAME} evaluated='\$(grep -c '^OK ' \"\${LOG_DIR}/summary.log\" || true)' skipped='\$(grep -c '^SKIP ' \"\${LOG_DIR}/summary.log\" || true)' failed=[\${FAILED}]' | tee -a \"\${LOG_DIR}/summary.log\"
if [[ -n \"\${FAILED}\" ]]; then
  echo '${JOB_NAME} finished WITH FAILURES' | tee -a \"\${LOG_DIR}/summary.log\"
  exit 1
fi
echo '${JOB_NAME} finished successfully' | tee -a \"\${LOG_DIR}/summary.log\"
"
)

if [[ "${PREVIEW:-false}" == "true" ]]; then
  printf 'rjob'
  printf ' %q' "${RJOB_ARGS[@]}"
  printf '\n'
  exit 0
fi

exec rjob "${RJOB_ARGS[@]}"
