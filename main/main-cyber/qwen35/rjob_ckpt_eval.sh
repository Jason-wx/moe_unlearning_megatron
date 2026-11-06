#!/usr/bin/env bash
set -euo pipefail

# Submit safewt rjobs that offline-evaluate the NINE qwen35 main-cyber R2
# point checkpoints with the official scripts/eval.sh (full
# wmdp_cyber + mmlu, 0-shot, bf16, LM_EVAL_BATCH_SIZE=16; see the OOM
# history note at the BATCH_SIZE definition). ONE job per R2 training group
# (user allocation 2026-10-02): 3 GPUs per job, ONE ckpt per GPU -
# distributed eval, no serial queue (each group has exactly 3 ckpts, the
# round-robin deal degenerates to one each). Pattern follows the bio line's
# main/main-cyber/interns2/rjob_ckpt_eval.sh (same harness, same bs16
# tier); params follow scripts/rjob_safer2ai_example.sh. Results land under
# main/main-cyber/qwen35/ckpt-eval/ via OU_EVAL_ROOT:
#   ckpt-eval/lm_eval_results_<ckpt-name>_wmdp_cyber_mmlu/result.json
# Already-evaluated ckpts (result.json present) are skipped on resubmit; a
# failed ckpt does not stop the other GPUs of its job.
#
# Examples:
#   RESOURCE_GROUP=safewt BATCH=r2-pdu-gd-rmu         bash rjob_ckpt_eval.sh
#   RESOURCE_GROUP=safewt BATCH=r2-pnpo-simnpo-satimp bash rjob_ckpt_eval.sh
#   RESOURCE_GROUP=safewt BATCH=r2-ceu-ga-wga         bash rjob_ckpt_eval.sh
#   PREVIEW=true BATCH=r2-pdu-gd-rmu bash rjob_ckpt_eval.sh   # print only

REPO_ROOT=${OU_REPO_ROOT:-/mnt/shared-storage-user/safethm-share/wangxiao1/Safe_MoE/open-unlearning}

RESOURCE_GROUP=${RESOURCE_GROUP:-safewt}

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
  safer2ai)
    NAMESPACE=ailab-safer2ai
    CHARGED_GROUP=safer2ai_gpu
    IMAGE=registry.h.pjlab.org.cn/ailab-safer2ai-safer2ai_gpu/unlearning:unlearning-cu126-torch271-v1
    ;;
  *)
    echo "ERROR: unknown RESOURCE_GROUP '${RESOURCE_GROUP}' (safewt|safethm|safer2ai)" >&2
    exit 1
    ;;
esac

CKPT_ROOT=/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/wangxiao1/checkpoints/unlearn/megatron/main-cyber
EVAL_ROOT=${REPO_ROOT}/main/main-cyber/qwen35/ckpt-eval

# name=path entries (arrays deliberately NOT named GROUPS; the cluster
# injects a GROUPS array into bash sessions and assignments to it are
# silently swallowed). The nine R2 margin-best ckpts (save point per
# make_configs.py SAVE_STEPS; ckpt dirs also hold their config yamls).
# Layout note (2026-10-02 reorg): weights live at ${CKPT_ROOT}/qwen35/<algo>/
# directly (HF files + config yaml one level down); the <algo>-stepN NAME part
# is kept so result dirs / skip-if-present match the original 2026-10-02 run.
CKPT_RMU="qwen35-main-cyber-rmu-step92=${CKPT_ROOT}/qwen35/RMU"
CKPT_PDU="qwen35-main-cyber-pdu-eps2p51-step115=${CKPT_ROOT}/qwen35/PDU"
CKPT_GD="qwen35-main-cyber-grad-diff-step69=${CKPT_ROOT}/qwen35/GradDiff"
CKPT_GA="qwen35-main-cyber-grad-ascent-step46=${CKPT_ROOT}/qwen35/GradAscent"
CKPT_PNPO="qwen35-main-cyber-pure-npo-step69=${CKPT_ROOT}/qwen35/PureNPO"
CKPT_SIMNPO="qwen35-main-cyber-simnpo-step46=${CKPT_ROOT}/qwen35/SimNPO"
CKPT_SATIMP="qwen35-main-cyber-satimp-forgetonly-step46=${CKPT_ROOT}/qwen35/SatImp"
CKPT_WGA="qwen35-main-cyber-wga-forgetonly-step184=${CKPT_ROOT}/qwen35/WGA"
CKPT_CEU="qwen35-main-cyber-ceu-step184=${CKPT_ROOT}/qwen35/CEU"

# Same grouping as the R2 training jobs (one job per training group).
BATCH=${BATCH:-}
if (( $# )); then
  CKPT_SPECS=("$@")
elif [[ -n "${BATCH}" ]]; then
  case "${BATCH}" in
    r2-pdu-gd-rmu)         CKPT_SPECS=("${CKPT_PDU}" "${CKPT_GD}" "${CKPT_RMU}") ;;
    r2-pnpo-simnpo-satimp) CKPT_SPECS=("${CKPT_PNPO}" "${CKPT_SIMNPO}" "${CKPT_SATIMP}") ;;
    r2-ceu-ga-wga)         CKPT_SPECS=("${CKPT_CEU}" "${CKPT_GA}" "${CKPT_WGA}") ;;
    *) echo "ERROR: unknown BATCH '${BATCH}' (r2-pdu-gd-rmu|r2-pnpo-simnpo-satimp|r2-ceu-ga-wga)" >&2; exit 1 ;;
  esac
else
  echo "ERROR: select with BATCH=..., pass name=path specs, or set CKPT_SPECS." >&2
  exit 1
fi

JOB_NAME=${JOB_NAME:-qwen35-main-cyber-ckpt-eval-${BATCH:-custom}}
# One ckpt per GPU (user layout 2026-10-02): GPUS must equal the ckpt count
# so the round-robin deal gives each card exactly one ckpt. 12 CPU threads
# per GPU.
# LM_EVAL_BATCH_SIZE: 64 OOMs on a single 140G H200 with wmdp_cyber's long
# prompts (one 64x~2.8k x 248k-vocab bf16 logits tensor is ~83 GiB on top of
# the ~72 GiB model; bio-line bs64 only fit because bio prompts are shorter).
# bs32 still OOM'd (2026-10-01 22:5x, intern run g3bs32-38718786) - some
# cyber prompts are longer than the ~2.8k estimate - so 16 is the working
# tier (~21 GiB worst-case logits, ample headroom). Batch size does not
# change lm_eval results, only throughput.
GPUS=${GPUS:-3}
BATCH_SIZE=${BATCH_SIZE:-16}
CPUS=${CPUS:-$((GPUS * 12))}
MEMORY_MB=128000
FUSE_COUNT=1
RDMA_COUNT=${RDMA_COUNT:-${GPUS}}
TERMINATION_GRACE_PERIOD_SECONDS=600
RUN_ROOT=${OU_RUN_ROOT:-${REPO_ROOT}/outputs}

if (( ${#CKPT_SPECS[@]} != GPUS )); then
  echo "ERROR: ${#CKPT_SPECS[@]} ckpts vs GPUS=${GPUS} - this layout is one ckpt per GPU." >&2
  exit 1
fi

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
export NUM_PROCESSES='1'
export CPU_THREADS_PER_PROCESS='$((CPUS / GPUS))'
export LM_EVAL_BATCH_SIZE='${BATCH_SIZE}'
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export PYTHONUNBUFFERED=1
export HF_HOME='/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/jincheng/.cache/huggingface'
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
RUN_TS=\$(date +%Y%m%d_%H%M%S)
LOG_DIR='${RUN_ROOT}/rjob_logs/${JOB_NAME}'_\${RUN_TS}
mkdir -p \"\${LOG_DIR}\" '${EVAL_ROOT}'
echo 'job=${JOB_NAME} layout=1-gpu-per-ckpt tasks=wmdp_cyber,mmlu bs=${BATCH_SIZE} gpus=${GPUS} eval_root=${EVAL_ROOT}' | tee \"\${LOG_DIR}/metadata.log\"
mapfile -t SPECS <<'CKPTS'
${CKPT_LIST}
CKPTS
TOTAL=\${#SPECS[@]}
FAILED=''
DONE=0
SKIPPED=0
# Round-robin deal: with TOTAL==GPUS each card takes exactly one ckpt.
for (( GPU=0; GPU<${GPUS}; GPU++ )); do
  (
    IDX=\${GPU}
    while (( IDX < TOTAL )); do
      IFS='=' read -r ckpt_name ckpt_path <<< \"\${SPECS[IDX]}\"
      out_dir='${EVAL_ROOT}/lm_eval_results_'\${ckpt_name}'_wmdp_cyber_mmlu'
      echo \"=== [\$(date +%H:%M:%S)] gpu=\${GPU} ckpt=\${ckpt_name} -> \${out_dir}\" >> \"\${LOG_DIR}/summary.log\"
      if [[ -f \"\${out_dir}/result.json\" ]]; then
        echo \"SKIP \${ckpt_name}: result.json already present\" >> \"\${LOG_DIR}/summary.log\"
        SKIPPED=\$((SKIPPED + 1))
        IDX=\$((IDX + ${GPUS}))
        continue
      fi
      if [[ ! -d \"\${ckpt_path}\" ]]; then
        echo \"FAILED \${ckpt_name}: ckpt dir missing\" >> \"\${LOG_DIR}/summary.log\"
        FAILED=\"\${FAILED} \${ckpt_name}(missing)\"
        IDX=\$((IDX + ${GPUS}))
        continue
      fi
      if (
        export CUDA_VISIBLE_DEVICES=\"\${GPU}\"
        bash '${REPO_ROOT}/scripts/eval.sh' \"\${ckpt_path}\" 'wmdp_cyber,mmlu' \"\${out_dir}\" > \"\${LOG_DIR}/eval-\${ckpt_name}.log\" 2>&1
      ); then
        echo \"OK \${ckpt_name} gpu=\${GPU}\" >> \"\${LOG_DIR}/summary.log\"
      else
        echo \"FAILED \${ckpt_name} gpu=\${GPU}\" >> \"\${LOG_DIR}/summary.log\"
        exit 1
      fi
      IDX=\$((IDX + ${GPUS}))
    done
  ) &
done
WAIT_STATUS=0
wait || WAIT_STATUS=\$?
DONE=\$(grep -c '^OK ' \"\${LOG_DIR}/summary.log\" || true)
SKIPPED=\$(grep -c '^SKIP ' \"\${LOG_DIR}/summary.log\" || true)
FAILED_LIST=\$(grep '^FAILED ' \"\${LOG_DIR}/summary.log\" | awk '{print \$2}' | tr '\n' ' ' || true)
echo \"evaluated=\${DONE} skipped=\${SKIPPED} failed=[\${FAILED_LIST}]\" | tee -a \"\${LOG_DIR}/summary.log\"
if [[ -n \"\${FAILED_LIST}\" ]]; then
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
