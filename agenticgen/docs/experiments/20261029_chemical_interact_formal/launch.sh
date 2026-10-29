#!/usr/bin/env bash
set -eo pipefail
repo=$1
config=$2
source /opt/MolSteer/scripts/scnet/activate_dtk.sh
export PYTHONPATH="$repo/agenticgen/src:$repo/src:/workspace/uniguide_deps:/opt/MolSteer/flowr_root"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "$repo"
exec /opt/miniforge3/envs/molsteer-flowr-dtk/bin/python -u \
  agenticgen/docs/experiments/20261029_chemical_interact_formal/run_campaign.py "$repo" "$config"
