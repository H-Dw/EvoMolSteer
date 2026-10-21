#!/usr/bin/env bash
# Reuse the audited scientific dependency overlays and inherit API/proxy settings.
set -eo pipefail
TASK_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /opt/MolSteer/scripts/scnet/activate_dtk.sh
export LD_LIBRARY_PATH="/opt/miniforge3/envs/molsteer-flowr-dtk/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$TASK_REPO/src:/opt/MolSteer/generated_datasets/plip_guidance_5nn4_5ost_s42_20261020/deps:/opt/MolSteer/generated_datasets/analysis_hiqbind_examples_20261020/deps:/opt/MolSteer/flowr_root:/opt/MolSteer/flowr_root/experiments/evomolsteer_online_20261004/runtime_deps:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
if [[ "${1:-}" == "test" ]]; then
    shift
    cd "$TASK_REPO"
    exec /opt/miniforge3/envs/molsteer-flowr-dtk/bin/python -B -m pytest "$@"
fi
exec /opt/miniforge3/envs/molsteer-flowr-dtk/bin/python -B "$TASK_REPO/scripts/plip_group_pipeline.py" "$@"
