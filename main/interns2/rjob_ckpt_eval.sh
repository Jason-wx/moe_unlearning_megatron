#!/usr/bin/env bash
set -euo pipefail

# Submit ONE safewt rjob that offline-evaluates up to 8 Intern-S2 main-line
# checkpoints with the official scripts/eval.sh, ONE GPU PER
# CKPT: the 8 ckpts of a group run in parallel, each pinned to its own GPU
# via CUDA_VISIBLE_DEVICES and evaluated single-process
# (NUM_PROCESSES=1, LM_EVAL_BATCH_SIZE=64, bf16, 0-shot, full-size
# wmdp_bio+mmlu). If a group ever holds more than 8 ckpts, extras run in
# follow-up waves of 8. Two complementary jobs cover all 16 ckpts on disk
# (2026-09-26): 11 finals (10 algorithms + pdu-lr4e-6) + gd {714,2142-rerun}
# + simnpo {300,714,2142-rerun}.
#   GROUP=a  -> 8 algorithm finals: rmu grad-diff grad-ascent pure-npo
#               simnpo satimp wga ceu
#   GROUP=b  -> pdu(lr1e-6) pdu-lr4e-6 undial-tp4 + the 5 saveckpts
# Results land under main/interns2/online-evaluation/ckpt-eval/ via the
# script's native OU_EVAL_ROOT override (official script stays unmodified):
#   ckpt-eval/lm_eval_results_<ckpt-name>_wmdp_bio_mmlu/result.json
# Already-evaluated ckpts (result.json present) are skipped, so a resubmit
# only evaluates what is missing. A failed ckpt does not stop the others;
# the job exits nonzero at the end and names the failures in summary.log.
# Layout history: the first submission (2026-09-26 22:35, jobs
# interns2-ckpt-eval-{a,b}-*) ran the serial 8-GPU-DDP-per-ckpt layout; this
# per-GPU-parallel layout is what later submissions use.
#   RESOURCE_GROUP=safewt GROUP=a bash main/interns2/rjob_ckpt_eval.sh

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${OU_REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}

GROUP=${GROUP:-a}
RESOURCE_GROUP=${RESOURCE_GROUP:-safewt}
CKPT_ROOT=/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/wangxiao1/checkpoints/unlearn/megatron/main
EVAL_ROOT=${REPO_ROOT}/main/interns2/online-evaluation/ckpt-eval

case "${RESOURCE_GROUP}" in
  safewt)
    NAMESPACE=ailab-safewt
    CHARGED_GROUP=safewt_gpu
    IMAGE=registry.h.pjlab.org.cn/ailab-safethm-sfaethm_cpu/safemoe:jc-work-20260506163353
    ;;
  safethm)
    NAMESPACE=ailab-safethm
    CHARGED_GROUP=sfaethm_gpu
    IMAGE=registry.h.pjlab.org.cn/ailab-safethm-sfaethm_cpu/safemoe:jc-work-20260506163353
    ;;
  *)
    echo "ERROR: unknown RESOURCE_GROUP '${RESOURCE_GROUP}' (safewt|safethm)" >&2
    exit 1
    ;;
esac

# name=path entries (arrays deliberately NOT named GROUPS; the cluster
# injects a GROUPS array into bash sessions and assignments to it are
# silently swallowed).
case "${GROUP}" in
  a)
    CKPT_SPECS=(
      "interns2-main-rmu-step2142=${CKPT_ROOT}/interns2-main-rmu/checkpoint-step-2142"
      "interns2-main-grad-diff-step2142=${CKPT_ROOT}/interns2-main-grad-diff/checkpoint-step-2142"
      "interns2-main-grad-ascent-step2142=${CKPT_ROOT}/interns2-main-grad-ascent/checkpoint-step-2142"
      "interns2-main-pure-npo-step2142=${CKPT_ROOT}/interns2-main-pure-npo/checkpoint-step-2142"
      "interns2-main-simnpo-step2142=${CKPT_ROOT}/interns2-main-simnpo/checkpoint-step-2142"
      "interns2-main-satimp-step2142=${CKPT_ROOT}/interns2-main-satimp-forgetonly/checkpoint-step-2142"
      "interns2-main-wga-step2142=${CKPT_ROOT}/interns2-main-wga-forgetonly/checkpoint-step-2142"
      "interns2-main-ceu-step2142=${CKPT_ROOT}/interns2-main-ceu/checkpoint-step-2142"
    )
    ;;
  b)
    CKPT_SPECS=(
      "interns2-main-pdu-step2142=${CKPT_ROOT}/interns2-main-pdu/checkpoint-step-2142"
      "interns2-main-pdu-lr4e-6-step2142=${CKPT_ROOT}/interns2-main-pdu-lr4e-6/checkpoint-step-2142"
      "interns2-main-undial-tp4-step2142=${CKPT_ROOT}/interns2-main-undial-tp4/checkpoint-step-2142"
      "interns2-main-grad-diff-saveckpts-step714=${CKPT_ROOT}/interns2-main-grad-diff-saveckpts/checkpoint-step-714"
      "interns2-main-grad-diff-saveckpts-step2142=${CKPT_ROOT}/interns2-main-grad-diff-saveckpts/checkpoint-step-2142"
      "interns2-main-simnpo-saveckpts-step300=${CKPT_ROOT}/interns2-main-simnpo-saveckpts/checkpoint-step-300"
      "interns2-main-simnpo-saveckpts-step714=${CKPT_ROOT}/interns2-main-simnpo-saveckpts/checkpoint-step-714"
      "interns2-main-simnpo-saveckpts-step2142=${CKPT_ROOT}/interns2-main-simnpo-saveckpts/checkpoint-step-2142"
    )
    ;;
  *)
    echo "ERROR: unknown GROUP '${GROUP}' (a|b)" >&2
    exit 1
    ;;
esac

JOB_NAME=${JOB_NAME:-interns2-ckpt-eval-${GROUP}}
GPUS=8
CPUS=96
MEMORY_MB=128000
FUSE_COUNT=1
RDMA_COUNT=8
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

rjob submit \
  --priority=9 \
  --name="${JOB_NAME}" \
  --gpu="${GPUS}" \
  --memory="${MEMORY_MB}" \
  --cpu="${CPUS}" \
  --charged-group="${CHARGED_GROUP}" \
  --namespace="${NAMESPACE}" \
  --private-machine=group \
  --custom-resources "brainpp.cn/fuse=${FUSE_COUNT}" \
  --custom-resources "rdma/mlnx_shared=${RDMA_COUNT}" \
  --termination-grace-period-seconds "${TERMINATION_GRACE_PERIOD_SECONDS}" \
  --mount=gpfs://gpfs1/safethm-share:/mnt/shared-storage-user/safethm-share \
  --mount=gpfs://gpfs2/gpfs2-shared-public:/mnt/shared-storage-gpfs2/gpfs2-shared-public \
  --mount=gpfs://gpfs2/wenxiaoyu-gpfs02:/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02 \
  --image-pull-policy=IfNotPresent \
  --image="${IMAGE}" \
  -e DISTRIBUTED_JOB=true \
  -- bash -c "set -euo pipefail
cd '${REPO_ROOT}'
export OU_REPO_ROOT='${REPO_ROOT}'
export OU_EVAL_ROOT='${EVAL_ROOT}'
export NUM_PROCESSES='1'
export CPU_THREADS_PER_PROCESS='$((CPUS / GPUS))'
export LM_EVAL_BATCH_SIZE='64'
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export PYTHONUNBUFFERED=1
export HF_HOME='/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/jincheng/.cache/huggingface'
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
RUN_TS=\$(date +%Y%m%d_%H%M%S)
LOG_DIR='${RUN_ROOT}/rjob_logs/${JOB_NAME}'_\${RUN_TS}
mkdir -p \"\${LOG_DIR}\"
echo 'job=${JOB_NAME} layout=1-gpu-per-ckpt tasks=wmdp_bio,mmlu bs=64 gpus=${GPUS} eval_root=${EVAL_ROOT}' | tee \"\${LOG_DIR}/metadata.log\"
mapfile -t SPECS <<'CKPTS'
${CKPT_LIST}
CKPTS
TOTAL=\${#SPECS[@]}
IDX=0
FAILED=''
DONE=0
SKIPPED=0
while (( IDX < TOTAL )); do
  WAVE_PIDS=()
  WAVE_NAMES=()
  for (( GPU=0; GPU<${GPUS} && IDX<TOTAL; GPU++,IDX++ )); do
    IFS='=' read -r ckpt_name ckpt_path <<< \"\${SPECS[IDX]}\"
    out_dir='${EVAL_ROOT}/lm_eval_results_'\${ckpt_name}'_wmdp_bio_mmlu'
    echo \"=== [\$(date +%H:%M:%S)] gpu=\${GPU} ckpt=\${ckpt_name} -> \${out_dir}\" | tee -a \"\${LOG_DIR}/summary.log\"
    if [[ -f \"\${out_dir}/result.json\" ]]; then
      echo \"SKIP \${ckpt_name}: result.json already present\" | tee -a \"\${LOG_DIR}/summary.log\"
      SKIPPED=\$((SKIPPED + 1))
      continue
    fi
    if [[ ! -d \"\${ckpt_path}\" ]]; then
      echo \"FAILED \${ckpt_name}: ckpt dir missing\" | tee -a \"\${LOG_DIR}/summary.log\"
      FAILED=\"\${FAILED} \${ckpt_name}(missing)\"
      continue
    fi
    (
      export CUDA_VISIBLE_DEVICES=\"\${GPU}\"
      if bash '${REPO_ROOT}/scripts/eval.sh' \"\${ckpt_path}\" 'wmdp_bio,mmlu' \"\${out_dir}\" > \"\${LOG_DIR}/eval-\${ckpt_name}.log\" 2>&1; then
        echo \"OK \${ckpt_name} gpu=\${GPU}\" >> \"\${LOG_DIR}/summary.log\"
      else
        echo \"FAILED \${ckpt_name} gpu=\${GPU} (exit \$?)\" >> \"\${LOG_DIR}/summary.log\"
        exit 1
      fi
    ) &
    WAVE_PIDS+=(\"\$!\")
    WAVE_NAMES+=(\"\${ckpt_name}\")
  done
  for J in \"\${!WAVE_PIDS[@]}\"; do
    if wait \"\${WAVE_PIDS[J]}\"; then
      DONE=\$((DONE + 1))
    else
      FAILED=\"\${FAILED} \${WAVE_NAMES[J]}\"
    fi
  done
done
echo \"evaluated=\${DONE} skipped=\${SKIPPED} failed=[\${FAILED}]\" | tee -a \"\${LOG_DIR}/summary.log\"
if [[ -n \"\${FAILED}\" ]]; then
  echo '${JOB_NAME} finished WITH FAILURES' | tee -a \"\${LOG_DIR}/summary.log\"
  exit 1
fi
echo '${JOB_NAME} finished successfully' | tee -a \"\${LOG_DIR}/summary.log\"
"
