#!/usr/bin/env bash
set -euo pipefail

# Submit safethm rjobs that offline-evaluate the NINE qwen35 main(bio)-line
# point checkpoints (all step 2142) with the official scripts/
# lm_eval_accelerate.sh (full wmdp_bio + mmlu, 0-shot, bf16,
# LM_EVAL_BATCH_SIZE=16). User allocation 2026-10-04: TWO jobs on safethm -
# BATCH=a3 = 3 GPUs / 4 ckpts (2+1+1), BATCH=a4 = 4 GPUs / 5 ckpts (2+1+1+1);
# layout = ONE ckpt per GPU with round-robin overflow (the gemma4-main-cyber
# eval profile; NUM_PROCESSES=1, each GPU runs lm_eval on its own ckpt
# serially, a failed ckpt does not stop the job's other GPUs).
# bs16 is the proven tier across all lines (bs32/64 OOM history with long
# prompts); batch size does not change results, only throughput.
# Results land under main/qwen35/offline-eval via OU_EVAL_ROOT:
#   offline-eval/lm_eval_results_<ckpt-name>_wmdp_bio_mmlu/result.json
# Already-evaluated ckpts (result.json present) are skipped on resubmit.
# Params follow scripts/rjob_safer2ai_example.sh (mounts/env layout) and the
# safethm case of main/main-cyber/gemma4/rjob_ckpt_eval_safewt.sh (same
# safemoe image as safewt).
#
# Examples:
#   BATCH=a3 bash main/qwen35/rjob_ckpt_eval_safethm.sh
#   BATCH=a4 bash main/qwen35/rjob_ckpt_eval_safethm.sh
#   PREVIEW=true BATCH=a3 bash main/qwen35/rjob_ckpt_eval_safethm.sh

REPO_ROOT=${OU_REPO_ROOT:-/mnt/shared-storage-user/safethm-share/wangxiao1/Safe_MoE/open-unlearning}

NAMESPACE=ailab-safethm
CHARGED_GROUP=sfaethm_gpu
IMAGE=registry.h.pjlab.org.cn/ailab-safethm-sfaethm_cpu/safemoe:jc-work-20260506163353

CKPT_ROOT=/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/wangxiao1/checkpoints/unlearn/megatron/main
EVAL_ROOT=${REPO_ROOT}/main/qwen35/offline-eval

# name=path entries (arrays deliberately NOT named GROUPS; the cluster
# injects a GROUPS array into bash sessions and assignments to it are
# silently swallowed). Nine step-2142 exports: eight qwen35-main-* runs plus
# the PDU eps-sweep winning arm pulled into the line on 10-04 (kept its
# sweep name qwen35-pdu-eps2p51; only a step-2142 ckpt exists on disk).
M=qwen35-main
S=step2142
# Paths follow the 10-04 reorg: weights flattened into main/Qwen35/<Algo>/
# (HF files directly under the algo dir, no checkpoint-step-N subfolder).
# name= prefixes keep the original run-step identity so result dir names stay
# aligned with the already-archived lm_eval_results_* directories.
CKPT_RMU="${M}-rmu-${S}=${CKPT_ROOT}/Qwen35/RMU"
CKPT_PDU="qwen35-pdu-eps2p51-${S}=${CKPT_ROOT}/Qwen35/PDU"
CKPT_GA="${M}-grad-ascent-${S}=${CKPT_ROOT}/Qwen35/GradAscent"
CKPT_PNPO="${M}-pure-npo-${S}=${CKPT_ROOT}/Qwen35/PureNPO"
CKPT_GD="${M}-grad-diff-${S}=${CKPT_ROOT}/Qwen35/GradDiff"
CKPT_SIMNPO="${M}-simnpo-${S}=${CKPT_ROOT}/Qwen35/SimNPO"
CKPT_SATIMP="${M}-satimp-forgetonly-${S}=${CKPT_ROOT}/Qwen35/SatImp"
CKPT_WGA="${M}-wga-forgetonly-${S}=${CKPT_ROOT}/Qwen35/WGA"
CKPT_CEU="${M}-ceu-${S}=${CKPT_ROOT}/Qwen35/CEU"

BATCH=${BATCH:-}
if (( $# )); then
  CKPT_SPECS=("$@")
elif [[ -n "${BATCH}" ]]; then
  case "${BATCH}" in
    # 3-GPU job, 4 ckpts (headline trio + PureNPO); gpu0 takes the 2nd serial ckpt
    a3) CKPT_SPECS=("${CKPT_RMU}" "${CKPT_PDU}" "${CKPT_GA}" "${CKPT_PNPO}") ;;
    # 4-GPU job, 5 ckpts (wall-family + shallow family); gpu0 takes the 2nd serial ckpt
    a4) CKPT_SPECS=("${CKPT_GD}" "${CKPT_SIMNPO}" "${CKPT_SATIMP}" "${CKPT_WGA}" "${CKPT_CEU}") ;;
    *) echo "ERROR: unknown BATCH '${BATCH}' (a3|a4)" >&2; exit 1 ;;
  esac
else
  echo "ERROR: select with BATCH=a3|a4, or pass name=path specs." >&2
  exit 1
fi

JOB_NAME=${JOB_NAME:-qwen35-main-ckpt-eval-${BATCH:-custom}}
GPUS=${GPUS:-3}
if [[ "${BATCH}" == "a4" ]]; then GPUS=4; fi   # 分组固定卡数（用户配额 3+4）
BATCH_SIZE=${BATCH_SIZE:-16}
CPUS=${CPUS:-$((GPUS * 12))}
MEMORY_MB=128000
FUSE_COUNT=1
RDMA_COUNT=${RDMA_COUNT:-${GPUS}}
TERMINATION_GRACE_PERIOD_SECONDS=600
RUN_ROOT=${OU_RUN_ROOT:-${REPO_ROOT}/outputs}

if (( ${#CKPT_SPECS[@]} < GPUS )); then
  echo "ERROR: ${#CKPT_SPECS[@]} ckpts vs GPUS=${GPUS} - one-ckpt-per-GPU layout needs >=GPUS ckpts." >&2
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
echo 'job=${JOB_NAME} layout=1-gpu-per-ckpt-roundrobin tasks=wmdp_bio,mmlu bs=${BATCH_SIZE} gpus=${GPUS} ckpts=${#CKPT_SPECS[@]} eval_root=${EVAL_ROOT}' | tee \"\${LOG_DIR}/metadata.log\"
mapfile -t SPECS <<'CKPTS'
${CKPT_LIST}
CKPTS
TOTAL=\${#SPECS[@]}
FAILED=''
DONE=0
SKIPPED=0
# Round-robin deal: each GPU walks IDX, IDX+GPUS, ... (with <=2 ckpts per GPU
# here, so every card runs 1-2 ckpts serially).
for (( GPU=0; GPU<${GPUS}; GPU++ )); do
  (
    IDX=\${GPU}
    while (( IDX < TOTAL )); do
      IFS='=' read -r ckpt_name ckpt_path <<< \"\${SPECS[IDX]}\"
      out_dir='${EVAL_ROOT}/lm_eval_results_'\${ckpt_name}'_wmdp_bio_mmlu'
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
        bash '${REPO_ROOT}/scripts/eval.sh' \"\${ckpt_path}\" 'wmdp_bio,mmlu' \"\${out_dir}\" > \"\${LOG_DIR}/eval-\${ckpt_name}.log\" 2>&1
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
