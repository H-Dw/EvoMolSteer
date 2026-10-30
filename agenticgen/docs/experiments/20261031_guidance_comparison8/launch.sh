#!/usr/bin/env bash
set -eo pipefail
repo=$1
plan=$2
source /opt/MolSteer/scripts/scnet/activate_dtk.sh >/dev/null 2>&1
export PYTHONPATH="$repo/agenticgen/src:$repo/src:/workspace/uniguide_deps:/opt/MolSteer/flowr_root"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "$repo"
exec /opt/miniforge3/envs/molsteer-flowr-dtk/bin/python -u \
  agenticgen/docs/experiments/20261031_guidance_comparison8/run_suite.py --plan "$plan"
