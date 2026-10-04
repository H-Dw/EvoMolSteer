#!/usr/bin/env bash
# Execute only committed local-source changes pulled on this server.
set -euo pipefail
mode=${1:?pilot|comparison}
repo=$(cd "$(dirname "$0")/.." && pwd)
flowr=${FLOWR_ROOT:-/root/private_data/MolSteer/flowr_root}
experiment=${EXPERIMENT_ROOT:-/opt/evomolsteer-multistage-20261005}
python=${FLOWR_PYTHON:-/opt/miniforge3/envs/molsteer-flowr-dtk/bin/python}
if [ -n "$(git -C "$repo" status --porcelain --untracked-files=no)" ]; then
  echo 'Refusing a modified tracked checkout' >&2; exit 1
fi
source /opt/MolSteer/scripts/scnet/activate_dtk.sh
export LD_LIBRARY_PATH="/opt/miniforge3/envs/molsteer-flowr-dtk/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$repo/src:$flowr:$flowr/experiments/evomolsteer_online_20261004/runtime_deps:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
cd "$repo"
mkdir -p "$experiment"
git rev-parse HEAD
args=(--flowr-root "$flowr" --input-dataset "$flowr/experiments/ck2_clk3_lineage_20261003"
      --root "$experiment/generated" --checkpoint "$flowr/checkpoints/flowr_root_v2.ckpt"
      --steps 100 --window .5 --program configs/experiments/ck2_multistage_v1/reward_program.json
      --catalog configs/experiments/ck2_multistage_v1/reward_catalog.json
      --max-atom-step-A .025 --live-preflight --component-audit)
case "$mode" in
  pilot) args+=(--campaign pilot_v1 --n 12 --batch 12 --seed 20261015
               --arms unguided,gradient_zero,multistage_r005,multistage_r015,multistage_r030 --verify-zero);;
  comparison)
    ratio=$("$python" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["status"]=="selected"; print(d["selected_ratio"])' "$experiment/pilot_evaluation/calibration.json")
    args+=(--campaign comparison_v1 --n 64 --batch 16 --seed 20261105 --native-rms-ratio "$ratio"
           --arms unguided,single,multistage_full,multistage_window,static_full,gradient_legacy);;
  *) echo 'Unknown mode' >&2; exit 2;;
esac
"$python" -u scripts/generate_multistage_flowr.py "${args[@]}"
