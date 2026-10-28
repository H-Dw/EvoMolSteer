#!/usr/bin/env bash
set -euo pipefail
checkout=/workspace/uniguide_audits/distance-system-da1b535a
source /opt/MolSteer/scripts/scnet/activate_dtk.sh > "${checkout}-acceptance-env.log" 2>&1
export PYTHONPATH="$checkout/agenticgen/src:$checkout/src:/workspace/uniguide_deps:/opt/MolSteer/flowr_root"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python=/opt/miniforge3/envs/molsteer-flowr-dtk/bin/python
cd "$checkout"
"$python" /workspace/uniguide_audits/geometry_acceptance_setup_20261028.py "$checkout" > "${checkout}-setup.json"
"$python" agenticgen/scripts/run_uniguide_geometry.py --config "${checkout}-config.json" prepare > "${checkout}-prepare.log" 2>&1
"$python" agenticgen/scripts/run_uniguide_geometry.py --config "${checkout}-config.json" acceptance > "${checkout}-acceptance.log" 2>&1
date -Iseconds > "${checkout}-acceptance-finished.txt"
