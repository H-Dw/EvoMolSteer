#!/usr/bin/env bash
# Run from an immutable Git checkout on ssh.zzai.scnet.cn:11762. Credentials stay in the environment.
set -eo pipefail
TASK_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /opt/MolSteer/scripts/scnet/activate_dtk.sh
export LD_LIBRARY_PATH="/opt/miniforge3/envs/molsteer-flowr-dtk/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$TASK_REPO/src:/opt/MolSteer/generated_datasets/plip_guidance_5nn4_5ost_s42_20261020/deps:/opt/MolSteer/generated_datasets/analysis_hiqbind_examples_20261020/deps:/opt/MolSteer/flowr_root:/opt/MolSteer/flowr_root/experiments/evomolsteer_online_20261004/runtime_deps:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
if [[ "${1:-}" == "background" ]]; then
    shift
    exec /opt/miniforge3/envs/molsteer-flowr-dtk/bin/python -B "$TASK_REPO/scripts/run_plip_adaptive_background.py" "$@"
fi
if [[ "${1:-}" == "report" ]]; then
    shift
    exec /opt/miniforge3/envs/molsteer-flowr-dtk/bin/python -B "$TASK_REPO/scripts/analyze_plip_adaptive.py" "$@"
fi
if [[ "${1:-}" == "check-gradients" ]]; then
    shift
    exec /opt/miniforge3/envs/molsteer-flowr-dtk/bin/python -B "$TASK_REPO/scripts/check_plip_adaptive_gradients.py" "$@"
fi
if [[ "${1:-}" == "adjust-input-budget" ]]; then
    shift
    exec /opt/miniforge3/envs/molsteer-flowr-dtk/bin/python -B "$TASK_REPO/scripts/adjust_plip_prompt_budget.py" "$@"
fi
exec /opt/miniforge3/envs/molsteer-flowr-dtk/bin/python -B -m evomolsteer.adaptive_schedule.cli "$@"
