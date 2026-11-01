"""Add the matched fifty archived unguided references without model inference."""
from pathlib import Path
import json,hashlib,datetime,importlib.util
from rdkit import Chem
from rdkit.Contrib.SA_Score import sascorer

spec=importlib.util.spec_from_file_location('saved_statistics','/tmp/summarize_results_20261101_v2.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
out=Path('/workspace/uniguide_result_exports_20261101/summary/final');out.mkdir(exist_ok=True)
x=module.read(Path('/workspace/uniguide_result_exports_20261101/summary/attempt_002/all_results.json'))
for p in sorted(Path('/workspace/agenticgen_uniguide_flowr_20261026/baselines').glob('*/unguided.json')):
 d=module.read(p);rows=[r for r in d['records']if r['matched_formal_population']];assert len(rows)==50
 values={'rmsd':[r['condition_rmsd_A']for r in rows],'native_rmsd':[r['condition_rmsd_A']for r in rows],'pic50':[r.get('pic50_on_upstream')for r in rows],'rescore':[r.get('pic50_on_rescore')for r in rows],'sa':[]}
 for r in rows:
  mol=Chem.MolFromSmiles(r.get('smiles',''))if r.get('build_success')else None
  values['sa'].append(float(sascorer.calculateScore(Chem.RemoveHs(mol)))if mol and mol.GetNumAtoms()else None)
 stats={}
 for k,v in values.items():
  f,e,detail=module.flags(v,'rmsd'if k=='native_rmsd'else k);stats[k]=module.stats(v,k,f,e)
 x['experiments'].append(dict(experiment='历史无引导配对基线',teacher=p.parent.name,phase='历史参考',unique_saved_candidates=50,built=sum(bool(r.get('build_success'))for r in rows),connected=sum(bool(r.get('connected'))for r in rows),groups=1,stats=stats,posebusters=dict(checked=0,evaluable=0,passed=0,components={},not_extrapolated=True),reference_source=str(p),reference_sha256=module.sha(p),pic50_source='Archived pic50_on_upstream (original saved output head); no new prediction',rmsd_source='Archived nine-PLIP-atom injective Hungarian RMSD in original frame; no new fitting'))
x['historical_reference_population']=100;x['timestamp']=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
module.write(out/'all_results.json',x)
print(json.dumps(dict(output=str(out/'all_results.json'),experiment_rows=len(x['experiments']),historical_references=100,groups=len(x['groups'])),ensure_ascii=True))
