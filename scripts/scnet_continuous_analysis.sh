#!/usr/bin/env bash
# Source is deployed by git pull before this script is launched.
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
python=${FLOWR_PYTHON:-/opt/miniforge3/envs/molsteer-flowr-dtk/bin/python}
flowr=${FLOWR_ROOT:-/root/private_data/MolSteer/flowr_root}
input=${INPUT_DATASET:-$flowr/experiments/ck2_clk3_lineage_20261003}
# Feature cache/shards use the large local disk; compact results are archived
# back to the private-data experiment after the run by the operator.
output=${ANALYSIS_OUTPUT:-/opt/evomolsteer-continuous-20261004/analysis_single_v3}
config=${ANALYSIS_CONFIG:-$repo/configs/continuous_single_target.json}
if [ -n "$(git -C "$repo" status --porcelain --untracked-files=no)" ]; then
  echo 'Tracked checkout is modified; refusing experiment' >&2
  exit 1
fi
source /opt/MolSteer/scripts/scnet/activate_dtk.sh
export LD_LIBRARY_PATH="/opt/miniforge3/envs/molsteer-flowr-dtk/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$repo/src:$flowr:$flowr/experiments/evomolsteer_online_20261004/runtime_deps:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
cd "$repo"
git rev-parse HEAD
"$python" -u scripts/run_pipeline.py --root "$input" --output "$output" --config "$config"
"$python" scripts/audit_continuous_analysis.py --analysis "$output"
"$python" scripts/audit_function_adequacy.py --analysis "$output"
"$python" scripts/plot_continuous_analysis.py --analysis "$output" --output "$output/figures"
