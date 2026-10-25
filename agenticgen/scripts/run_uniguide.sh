#!/usr/bin/env bash
set -euo pipefail
source /opt/MolSteer/scripts/scnet/activate_dtk.sh
cd /workspace/EvoMolSteer
export PYTHONPATH="/workspace/uniguide_deps:/workspace/EvoMolSteer/agenticgen/src:/workspace/EvoMolSteer/src:/opt/MolSteer/flowr_root${PYTHONPATH:+:$PYTHONPATH}"
exec /opt/miniforge3/envs/molsteer-flowr-dtk/bin/python -m agenticgen.uniguide \
  --config /workspace/EvoMolSteer/agenticgen/configs/uniguide_flowr.json "$@"
