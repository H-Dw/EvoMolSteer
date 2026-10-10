#!/usr/bin/env bash
set -euo pipefail
manifest=${1:?}; work=${2:?}
repo=$(cd "$(dirname "$0")/.." && pwd)
source /opt/MolSteer/scripts/scnet/activate_dtk.sh
python=/opt/miniforge3/envs/molsteer-flowr-dtk/bin/python
export LD_LIBRARY_PATH="/opt/miniforge3/envs/molsteer-flowr-dtk/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$repo/src:${FLOWR_ROOT:?}:$FLOWR_ROOT/experiments/evomolsteer_online_20261004/runtime_deps:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
cd "$repo"
"$python" -u scripts/run_outcome_round.py --manifest "$manifest" --work "$work" --flowr-root "$FLOWR_ROOT"
