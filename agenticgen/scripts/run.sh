#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export AGENTICGEN_REPOSITORY="$(git -C "$PROJECT" rev-parse --show-toplevel)"
export AGENTICGEN_HOST_CONFIG="${AGENTICGEN_HOST_CONFIG:-$PROJECT/src/agenticgen/resources/config/scnet11762.json}"
PYTHON="${AGENTICGEN_PYTHON:-$PROJECT/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  echo 'Install the audited task environment first; see docs/operations.md.' >&2
  exit 2
fi
ACTIVATION="$($PYTHON -c 'import json,os; print(json.load(open(os.environ["AGENTICGEN_HOST_CONFIG"]))["activation"])')"
source "$ACTIVATION" >/dev/null 2>&1
# Preserve API/proxy variables, but never inherit a research checkout as code.
unset PYTHONPATH
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
exec "$PYTHON" -m agenticgen "$@"
