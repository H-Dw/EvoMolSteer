#!/usr/bin/env bash
# Runtime setup belongs in this host wrapper, not in analysis or Agent Skills.
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
flowr=${FLOWR_ROOT:-/root/private_data/MolSteer/flowr_root}
python=${FLOWR_PYTHON:-/opt/miniforge3/envs/molsteer-flowr-dtk/bin/python}
activation=${FLOWR_ACTIVATE:-/opt/MolSteer/scripts/scnet/activate_dtk.sh}
if [ -f "$activation" ]; then source "$activation"; fi
export LD_LIBRARY_PATH="$(dirname "$python")/../lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$repo/src:$flowr:$flowr/experiments/evomolsteer_online_20261004/runtime_deps:${PYTHONPATH:-}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
cd "$repo"
exec "$python" -u scripts/run_unguided_targets.py "$@"
