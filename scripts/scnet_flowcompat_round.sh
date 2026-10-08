#!/usr/bin/env bash
set -euo pipefail
number=${1:?}; campaign=${2:?}; program=${3:?}; reference=${4:?}; arms=${5:?}; n=${6:?}; batches=${7:?}; mode=${8:-flowcompat}
repo=$(cd "$(dirname "$0")/.." && pwd)
work=${EXPERIMENT_ROOT:?}; flowr=${FLOWR_ROOT:?}
printf -v tag 'round%02d' "$number"
trap 'printf "%s\n" "$?" > "$work/$tag.exit"' EXIT
source /opt/MolSteer/scripts/scnet/activate_dtk.sh
python=/opt/miniforge3/envs/molsteer-flowr-dtk/bin/python
export LD_LIBRARY_PATH="/opt/miniforge3/envs/molsteer-flowr-dtk/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$repo/src:$flowr:$flowr/experiments/evomolsteer_online_20261004/runtime_deps:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
cd "$repo"
"$python" scripts/check_terminal_round_ready.py --manifest "$work/round_ready.json" --campaign "$campaign" --program "$program"
source_args=(--input-dataset "$flowr"
 --relative-source flowr/models/fm_pocket.py --relative-source flowr/models/integrator.py
 --relative-source flowr/models/pocket.py --relative-source flowr/models/pocket_util.py)
"$python" scripts/attest_flowr_sources.py "${source_args[@]}" --output-record "$work/$tag.upstream.before.json"
"$python" -u "scripts/generate_${mode}_flowr.py" --flowr-root "$flowr" \
 --input-dataset "$flowr/experiments/ck2_clk3_lineage_20261003" --root "$work/generated" \
 --checkpoint "$flowr/checkpoints/flowr_root_v2.ckpt" --steps 100 --program "$program" --reference "$reference" \
 --campaign "$campaign" --n "$n" --batch 50 --seed 42 --arms "$arms" --batch-indices "$batches" --export-terminal
"$python" scripts/attest_flowr_sources.py "${source_args[@]}" --reference-record "$work/$tag.upstream.before.json" --output-record "$work/$tag.upstream.after.json"
"$python" scripts/bind_flowr_source_attestation.py --dataset "$work/generated" --campaign "$campaign" --before-record "$work/$tag.upstream.before.json" --after-record "$work/$tag.upstream.after.json"
"$python" scripts/archive_generation.py --dataset "$work/generated" --campaign "$campaign" --output "$work/$campaign.tar.gz" --evaluation-only
