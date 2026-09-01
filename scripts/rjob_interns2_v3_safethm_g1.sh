#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export BACKEND=megatron
export WORKFLOW=train-eval-group
export CONFIG=configs/training/megatron/interns2/rmu.yaml
export JOB_NAME=ou-interns2-v3-safethm-g1
export SWEEP_NAME=interns2-Megatron-V3
export EVAL_TASKS=wmdp,mmlu
export OU_EVAL_ROOT=/mnt/shared-storage-user/safethm-share/wangxiao1/Safe_MoE/open-unlearning/evaluations
export GPUS=8
export CPUS=64
export MEMORY_MB=128000
export CHARGED_GROUP=sfaethm_gpu
export NAMESPACE=ailab-safethm
export FUSE_COUNT=1
export RDMA_COUNT=8
export TERMINATION_GRACE_PERIOD_SECONDS=600
export IMAGE=registry.h.pjlab.org.cn/ailab-safethm-sfaethm_cpu/safemoe:jc-work-20260506163353

exec "${SCRIPT_DIR}/submit_rjob_example.sh" \
  configs/training/megatron/interns2/rmu.yaml \
  configs/training/megatron/interns2/pure_npo.yaml \
  configs/training/megatron/interns2/grad_diff.yaml \
  configs/training/megatron/interns2/npo.yaml
