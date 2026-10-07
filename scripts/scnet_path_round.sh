#!/usr/bin/env bash
set -euo pipefail
number=${1:?number}; shift
repo=$(cd "$(dirname "$0")/.." && pwd)
work=${EXPERIMENT_ROOT:?EXPERIMENT_ROOT}
printf -v tag 'round%02d' "$number"
trap 'printf "%s\n" "$?" > "$work/$tag.exit"' EXIT
bash "$repo/scripts/scnet_terminal_experiment.sh" "$1" affinity_endpoint "$2" "$3" "$4" "$5" "$6"
