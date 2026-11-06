#!/usr/bin/env bash
set -euo pipefail

# Example: build one XTuner soft-pack cache (Qwen3.5, WMDP-Bio forget).
# All build parameters live in the YAML; to build another cache, pick another
# file under configs/data_processing/xtuner/<model>/<dataset>.yaml.
#
#   bash scripts/build_xtuner_cache_example.sh
#
# The builder refuses to replace an existing cache; add --overwrite to rebuild.

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${OU_PYTHON:-${REPO_ROOT}/envs/open-unlearning/bin/python}
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${REPO_ROOT}"

"${PYTHON_BIN}" -m data.build_xtuner_packed_cache \
  --config configs/data_processing/xtuner/qwen35/wmdp_bio_forget.yaml \
  "$@"
