#!/usr/bin/env bash
# Remote responsibilities: frozen inference + raw recording/transport only.
set -euo pipefail
campaign=${1:?campaign}
program=${2:?relative program path}
seed=${3:?seed}
arms=${4:-gradient}
n=${5:-32}
repo=$(cd "$(dirname "$0")/.." && pwd)
flowr=${FLOWR_ROOT:-/root/private_data/MolSteer/flowr_root}
experiment=${EXPERIMENT_ROOT:-/opt/evomolsteer-window-iter8-20261005}
python=${FLOWR_PYTHON:-/opt/miniforge3/envs/molsteer-flowr-dtk/bin/python}
if [ -n "$(git -C "$repo" status --porcelain --untracked-files=no)" ]; then
  echo 'Refusing modified tracked checkout' >&2; exit 1
fi
source /opt/MolSteer/scripts/scnet/activate_dtk.sh
export LD_LIBRARY_PATH="/opt/miniforge3/envs/molsteer-flowr-dtk/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$repo/src:$flowr:$flowr/experiments/evomolsteer_online_20261004/runtime_deps:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
cd "$repo"
mkdir -p "$experiment"
git rev-parse HEAD
"$python" -u scripts/generate_window_flowr.py --flowr-root "$flowr" \
  --input-dataset "$flowr/experiments/ck2_clk3_lineage_20261003" \
  --root "$experiment/generated" --checkpoint "$flowr/checkpoints/flowr_root_v2.ckpt" \
  --steps 100 --program "$program" --reference configs/experiments/ck2_window_iter8_v1/reference.json.gz \
  --campaign "$campaign" --n "$n" --batch 16 --seed "$seed" --arms "$arms"
"$python" scripts/archive_generation.py --dataset "$experiment/generated" --campaign "$campaign" \
  --output "$experiment/${campaign}.tar.gz"
