#!/usr/bin/env bash
set -euo pipefail
# Paramiko/ssh command sessions do not reach the exports below the interactive
# guard in root's .bashrc. Recover the existing key and proxy environment once
# before starting a formal Agent version; never copy secrets into this script.
if [[ "${1:-}" == "run-version" && -z "${DEEPSEEK_API_KEY:-}" ]]; then
  if [[ "${UG_STARTUP_ENV_LOADED:-0}" == "1" ]]; then
    echo "DEEPSEEK_API_KEY is unavailable after loading the existing interactive shell environment" >&2
    exit 2
  fi
  export UG_STARTUP_ENV_LOADED=1
  exec bash -ic 'exec bash "$@"' uniguide-startup "${BASH_SOURCE[0]}" "$@"
fi
source /opt/MolSteer/scripts/scnet/activate_dtk.sh
cd /workspace/EvoMolSteer
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export PYTHONPATH="/workspace/uniguide_deps:/workspace/EvoMolSteer/agenticgen/src:/workspace/EvoMolSteer/src:/opt/MolSteer/flowr_root${PYTHONPATH:+:$PYTHONPATH}"
exec /opt/miniforge3/envs/molsteer-flowr-dtk/bin/python -m agenticgen.uniguide \
  --config /workspace/EvoMolSteer/agenticgen/configs/uniguide_flowr.json "$@"
