#!/usr/bin/env bash
set -euo pipefail
round=${1:?round}
campaign=${2:?campaign}
program=${3:?program}
reference=${4:?reference}
previous_report=${5:?previous retained report relative to repository}
arms=${6:-gradient}
batch_indices=${7:-}
n=${8:-100}
repo=$(cd "$(dirname "$0")/.." && pwd)
work=${EXPERIMENT_ROOT:-/root/private_data/MolSteer/flowr_root/experiments/evomolsteer_motif_seed42_20261006}
python=${FLOWR_PYTHON:-/opt/miniforge3/envs/molsteer-flowr-dtk/bin/python}
printf -v round_name 'round%02d' "$round"
test ! -e "$work/$round_name.exit"
trap 'printf "%s\n" "$?" > "$work/$round_name.exit"' EXIT
test -z "$(git -C "$repo" status --porcelain --untracked-files=no)"
"$python" - "$repo" "$work" "$round" "$campaign" "$previous_report" <<'PY'
import json,sys,hashlib
from pathlib import Path
repo,work=map(lambda v:Path(v).resolve(),sys.argv[1:3]);number=int(sys.argv[3]);campaign=sys.argv[4]
report=(repo/sys.argv[5]).resolve()
if not report.is_relative_to(repo/'docs') or not report.is_file():raise ValueError('Retained scientific report must be in repository docs')
if not json.loads(report.read_text()):raise ValueError('Nonempty scientific comparison required')
local=report.parent.parent/'local';retention=json.loads((local/'retention.json').read_text())
files={v['path']:v for v in retention['files']}
required={'terminal_report.json','candidate_metrics.csv','execution_report.json','coordinate_audit.json','window/report.json'}
if required-set(files) or retention['candidate_rows']<=0:raise ValueError('Incomplete retained scientific evidence')
for name,item in files.items():
    path=(local/name).resolve()
    if not path.is_relative_to(local) or not path.is_file() or path.stat().st_size==0 or path.stat().st_size!=item['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:
        raise ValueError('Retained scientific file checksum/size/path mismatch')
m=json.loads((work/'round_ready.json').read_text())
if m['round']+1!=number:raise ValueError('Sequential round required')
plan={'allowed_bases':[str(work)],'protected':[str(repo),'/root/private_data/MolSteer/flowr_root/experiments/ck2_clk3_lineage_20261003','/root/private_data/MolSteer/flowr_root/checkpoints'],
      'result_report':str(report),'targets':[str(work/'generated')]}
for old in m['campaigns']:
    if Path(old).name!=old:raise ValueError('Invalid previous campaign name')
    plan['targets'].extend(str(work/(old+suffix)) for suffix in ('.tar.gz','.tar.gz.json'))
(work/f'cleanup_before_round{number:02d}.plan.json').write_text(json.dumps(plan))
PY
cd "$repo"
"$python" scripts/retire_experiment_outputs.py --plan "$work/cleanup_before_$round_name.plan.json" --report "$work/cleanup_before_$round_name.json" --apply
"$python" - "$work" "$round" "$campaign" <<'PY'
import json,sys
from pathlib import Path
work=Path(sys.argv[1]);number=int(sys.argv[2]);m=json.loads((work/'round_ready.json').read_text())
m.update(status='active',round=number,campaigns=[sys.argv[3]],completed_rounds=number-1,cleanup_audit=str(work/f'cleanup_before_round{number:02d}.json'))
(work/'round_ready.json').write_text(json.dumps(m,indent=2))
PY
export EXPERIMENT_ROOT="$work"
bash "$repo/scripts/scnet_terminal_experiment.sh" "$campaign" coordinate "$program" "$reference" "$arms" "$n" "$batch_indices"
