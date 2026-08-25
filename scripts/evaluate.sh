#!/usr/bin/env bash
set -euo pipefail

# Canonical evaluation wrapper.
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${REPO_ROOT}"
exec bash scripts/lm_eval_accelerate.sh "$@"
