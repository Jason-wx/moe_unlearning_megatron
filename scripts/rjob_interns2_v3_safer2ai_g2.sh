#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export BACKEND=megatron
export WORKFLOW=train-eval-group
export CONFIG=configs/training/megatron/interns2/ceu.yaml
export JOB_NAME=ou-interns2-v3-safer2ai-resplit-g2
export SWEEP_NAME=interns2-Megatron-V3
export EVAL_TASKS=wmdp,mmlu
export OU_EVAL_ROOT=/mnt/shared-storage-user/safethm-share/wangxiao1/Safe_MoE/open-unlearning/evaluations
export GPUS=8
export CPUS=64
export MEMORY_MB=128000
export CHARGED_GROUP=safer2ai_gpu
export NAMESPACE=ailab-safer2ai
export IMAGE=registry.h.pjlab.org.cn/ailab-safer2ai-safer2ai_gpu/unlearning:unlearning-cu126-torch271-v1

exec "${SCRIPT_DIR}/submit_rjob_example.sh" \
  configs/training/megatron/interns2/ceu.yaml \
  configs/training/megatron/interns2/token_npo.yaml
