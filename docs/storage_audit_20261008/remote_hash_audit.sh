set -eu
du -s -B1 /root/private_data/MolSteer/flowr_root/experiments/ck2_clk3_lineage_20261003/results/main1000_w050/single
du -s -B1 /root/private_data/MolSteer/flowr_root/experiments/ck2_clk3_lineage_20261003
/opt/miniforge3/envs/molsteer-flowr-dtk/bin/python - <<'PY'
from pathlib import Path
import hashlib,json
r=Path('/root/private_data/MolSteer/flowr_root/experiments/ck2_clk3_lineage_20261003')
rows=[]
for p in sorted((r/'results/main1000_w050/single').rglob('*')):
 if not p.is_file():continue
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 rows.append({'path':p.relative_to(r).as_posix(),'bytes':p.stat().st_size,'sha256':h.hexdigest()})
p=Path('/tmp/evomolsteer_single_all_hashes_20261008.json');p.write_text(json.dumps(rows,indent=2));print(json.dumps({'files':len(rows),'bytes':sum(x['bytes'] for x in rows),'report':str(p)}))
PY
