#!/usr/bin/env bash
# In-container body of rjob_diag_safer2ai.sh: smoke test, then one model per
# GPU through a slot queue (NUM_GPUS slots). Each model: diag_model.py
# (mcq/lens/cloze/nll) + weight diff vs its base when a base is given.
set -uo pipefail

DIAG_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${OU_REPO_ROOT:-$(cd "${DIAG_DIR}/../../.." && pwd)}
PY=${REPO_ROOT}/envs/open-unlearning/bin/python
OUT=${DIAG_DIR}/results
LOGS=${LOG_DIR:-${OUT}/logs}
NUM_GPUS=${NUM_GPUS:-7}
mkdir -p "${OUT}" "${LOGS}"

CACHE_ROOT=/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/jincheng/.cache
KERNEL_CACHE_ROOT=${CACHE_ROOT}/sgtm/kernel-cache
export HF_HOME=${CACHE_ROOT}/huggingface
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_MODULES_CACHE=/tmp/hf_modules
export TRITON_CACHE_DIR=${KERNEL_CACHE_ROOT}/triton
export TORCHINDUCTOR_CACHE_DIR=${KERNEL_CACHE_ROOT}/torchinductor
export TILELANG_CACHE_DIR=${KERNEL_CACHE_ROOT}/tilelang
export OMP_NUM_THREADS=${CPU_THREADS_PER_PROCESS:-12}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

CK=/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/wangxiao1/checkpoints
QB=/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/jincheng/sgtm/checkpoints/qwen/qwen3p5-35b-a3b
IB=/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/jincheng/sgtm/checkpoints/intern/Intern-S2-base05
M=${CK}/unlearn/megatron/main
T=${CK}/Trace

# name|path|base (empty base = no weight diff). NOT named GROUPS.
MODEL_SPECS=(
  "qwen-base|${QB}|"
  "qwen35-grad-diff|${M}/Qwen35/GradDiff|${QB}"
  "qwen35-satimp|${M}/Qwen35/SatImp|${QB}"
  "qwen35-wga|${M}/Qwen35/WGA|${QB}"
  "qwen35-ceu|${M}/Qwen35/CEU|${QB}"
  "qwen35-grad-ascent|${M}/Qwen35/GradAscent|${QB}"
  "qwen35-rmu|${M}/Qwen35/RMU|${QB}"
  "qwen35-trace-2142|${M}/Qwen35/Trace|${QB}"
  "intern-base|${IB}|"
  "intern-trace-714|${T}/intern_s2_base05_forget1_lr1e-6_ep2_ep8_mb4_gb32_forgetonly/checkpoint-714|${IB}"
)

echo "[smoke] qwen-base --limit 8"
if ! CUDA_VISIBLE_DEVICES=0 "${PY}" "${DIAG_DIR}/diag_model.py" --name smoke-qwen-base --path "${QB}" \
    --out-dir "${OUT}/smoke" --limit 8 --n-docs 2 --doc-len 256 > "${LOGS}/smoke.log" 2>&1; then
  echo "[smoke] FAILED"; tail -40 "${LOGS}/smoke.log"; exit 1
fi
tail -12 "${LOGS}/smoke.log"

declare -A SLOT_PID SLOT_NAME
FAILED=()
reap() {  # wait for any slot to free; record failures
  while true; do
    for s in "${!SLOT_PID[@]}"; do
      if ! kill -0 "${SLOT_PID[$s]}" 2>/dev/null; then
        wait "${SLOT_PID[$s]}"; rc=$?
        [[ $rc -ne 0 ]] && FAILED+=("${SLOT_NAME[$s]}(rc=$rc)")
        echo "[done] ${SLOT_NAME[$s]} rc=$rc gpu=$s"
        unset "SLOT_PID[$s]" "SLOT_NAME[$s]"
        FREE=$s; return
      fi
    done
    sleep 20
  done
}

next_gpu=0
for spec in "${MODEL_SPECS[@]}"; do
  IFS='|' read -r name path base <<< "${spec}"
  want="${OUT}/${name}.json"; [[ -n "${base}" ]] && want="${OUT}/${name}.wdiff.json"
  if [[ -f "${want}" ]]; then echo "[skip] ${name}"; continue; fi
  if [[ ! -d "${path}" ]]; then echo "[missing] ${name} ${path}"; FAILED+=("${name}(missing)"); continue; fi
  if (( next_gpu < NUM_GPUS )); then gpu=$next_gpu; next_gpu=$((next_gpu + 1)); else reap; gpu=$FREE; fi
  skip=""; [[ -f "${OUT}/${name}.json" ]] && skip="mcq,cloze,nll"
  echo "[start] ${name} gpu=${gpu}"
  CUDA_VISIBLE_DEVICES=${gpu} "${PY}" "${DIAG_DIR}/diag_model.py" --name "${name}" --path "${path}" \
    --base "${base}" --out-dir "${OUT}" ${skip:+--skip ${skip}} > "${LOGS}/${name}.log" 2>&1 &
  SLOT_PID[$gpu]=$!; SLOT_NAME[$gpu]=${name}
done
while (( ${#SLOT_PID[@]} > 0 )); do reap; done

{
  echo "finished $(date)"; echo "failed: ${FAILED[*]:-none}"
  ls -la "${OUT}"/*.json 2>/dev/null
} | tee "${OUT}/summary.log"
"${PY}" "${DIAG_DIR}/summarize.py" "${OUT}" || true
[[ ${#FAILED[@]} -eq 0 ]]
