#!/usr/bin/env bash
set -euo pipefail
mode=${1:?pilot|comparison}
repo=$(cd "$(dirname "$0")/.." && pwd)
flowr=${FLOWR_ROOT:-/root/private_data/MolSteer/flowr_root}
root=${EXPERIMENT_ROOT:-/opt/evomolsteer-multistage-20261005}
python=${FLOWR_PYTHON:-/opt/miniforge3/envs/molsteer-flowr-dtk/bin/python}
source /opt/MolSteer/scripts/scnet/activate_dtk.sh
export LD_LIBRARY_PATH="/opt/miniforge3/envs/molsteer-flowr-dtk/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$repo/src:$flowr:$flowr/experiments/evomolsteer_online_20261004/runtime_deps:${PYTHONPATH:-}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
cd "$repo"
campaign="$root/generated/results/${mode}_v1"
output="$root/${mode}_evaluation"
args=(--campaign "$campaign" --output "$output")
if [ "$mode" = pilot ]; then args+=(--calibrate); fi
"$python" scripts/evaluate_multistage_runs.py "${args[@]}"
"$python" scripts/audit_generation.py --campaign "$campaign" --output "$output/generation_audit.json"
if [ "$mode" = comparison ]; then
  "$python" scripts/plot_multistage_results.py --reference configs/experiments/ck2_multistage_v1/reference_packet.json --evaluation "$output" --output "$output/figures"
fi
