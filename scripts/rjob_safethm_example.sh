#!/usr/bin/env bash
set -euo pipefail

# Intern-S2 RMU: zero-based layers 7,8,9 use full trainable mode.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export BACKEND=megatron
export WORKFLOW=train-eval-group
export CONFIG=configs/training/megatron/interns2/rmu.yaml
export JOB_NAME=ou-interns2-rmu-lr1e5-layers8-10-8gpu
export SWEEP_NAME=interns2-rmu-lr1e5-layers8-10-8gpu
export GPUS=8
export CPUS=64
export MEMORY_MB=128000
export CHARGED_GROUP=sfaethm_gpu
export USE_NAMESPACE=true
export NAMESPACE=ailab-safethm
export FUSE_COUNT=1
export RDMA_COUNT=8
export TERMINATION_GRACE_PERIOD_SECONDS=600
export USE_IMAGE=true
export IMAGE=registry.h.pjlab.org.cn/ailab-safethm-sfaethm_cpu/safemoe:jc-work-20260506163353

exec "${SCRIPT_DIR}/submit_rjob_example.sh" \
  configs/training/megatron/interns2/rmu.yaml
