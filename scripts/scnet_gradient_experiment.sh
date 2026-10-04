#!/usr/bin/env bash
# Run from the pulled EvoMolSteer checkout. No source mutation or credentials.
set -euo pipefail
mode=${1:?usage: scnet_gradient_experiment.sh analysis|pilot|comparison}
repo=$(cd "$(dirname "$0")/.." && pwd)
flowr=${FLOWR_ROOT:-/root/private_data/MolSteer/flowr_root}
experiment=${EXPERIMENT_ROOT:-$flowr/experiments/evomolsteer_single_guidance_20261004}
history=$flowr/experiments/ck2_clk3_lineage_20261003
python=${FLOWR_PYTHON:-/opt/miniforge3/envs/molsteer-flowr-dtk/bin/python}
if [ -n "$(git -C "$repo" status --porcelain --untracked-files=no)" ]; then
  echo 'Refusing experiment from a modified tracked checkout' >&2
  exit 1
fi
source /opt/MolSteer/scripts/scnet/activate_dtk.sh
export LD_LIBRARY_PATH="/opt/miniforge3/envs/molsteer-flowr-dtk/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$repo/src:$flowr:$flowr/experiments/evomolsteer_online_20261004/runtime_deps:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
cd "$repo"
mkdir -p "$experiment"
git rev-parse HEAD
if [ "$mode" = analysis ]; then
  "$python" scripts/run_pipeline.py --root "$history" --output "$experiment/analysis_single_v1" --config configs/single_target_analysis.json
  exit
fi
args=(--flowr-root "$flowr" --input-dataset "$history" --root "$experiment/generated"
      --checkpoint "$flowr/checkpoints/flowr_root_v2.ckpt" --steps 100 --window .5
      --program configs/experiments/ck2_single_guidance_v1/reward_program.json
      --catalog configs/experiments/ck2_single_guidance_v1/reward_catalog.json
      --strength .05 --max-atom-step-A .025 --live-preflight)
case "$mode" in
  pilot) args+=(--campaign pilot_v1 --n 8 --batch 8 --seed 20261004
                --arms gradient_zero,gradient_region,gradient_region_compact --verify-zero);;
  comparison) args+=(--campaign comparison_v1 --n 200 --batch 50 --seed 20261005
                     --arms unguided,single,gradient_region,gradient_region_compact);;
  *) echo 'Unknown mode' >&2;exit 2;;
esac
"$python" -u scripts/generate_gradient_flowr.py "${args[@]}"
