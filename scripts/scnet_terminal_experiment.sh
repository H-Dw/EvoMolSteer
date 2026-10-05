#!/usr/bin/env bash
set -euo pipefail
round=${1:?round}
mode=${2:?window or local}
program=${3:?program}
reference=${4:?reference}
arms=${5:-gradient}
n=${6:-100}
batch_indices=${7:-}
extra=()
if [ -n "$batch_indices" ]; then extra+=(--batch-indices "$batch_indices"); fi
repo=$(cd "$(dirname "$0")/.." && pwd)
flowr=${FLOWR_ROOT:-/root/private_data/MolSteer/flowr_root}
work=${EXPERIMENT_ROOT:-/opt/evomolsteer-terminal-seed42-20261005}
python=${FLOWR_PYTHON:-/opt/miniforge3/envs/molsteer-flowr-dtk/bin/python}
if [ -n "$(git -C "$repo" status --porcelain --untracked-files=no)" ]; then echo 'Dirty checkout' >&2; exit 1; fi
source /opt/MolSteer/scripts/scnet/activate_dtk.sh
export LD_LIBRARY_PATH="/opt/miniforge3/envs/molsteer-flowr-dtk/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$repo/src:$flowr:$flowr/experiments/evomolsteer_online_20261004/runtime_deps:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
cd "$repo"
"$python" scripts/check_terminal_round_ready.py --manifest "$work/round_ready.json" --campaign "$round" --program "$program"
case "$mode" in window|local) ;; *) exit 2;; esac
"$python" -u "scripts/generate_${mode}_flowr.py" --flowr-root "$flowr" \
 --input-dataset "$flowr/experiments/ck2_clk3_lineage_20261003" \
 --root "$work/generated" --checkpoint "$flowr/checkpoints/flowr_root_v2.ckpt" \
 --steps 100 --program "$program" --reference "$reference" --campaign "$round" \
 --n "$n" --batch 50 --seed 42 --arms "$arms" --export-terminal "${extra[@]}"
"$python" scripts/archive_generation.py --dataset "$work/generated" --campaign "$round" --output "$work/$round.tar.gz"
