"""Reproduce the closed-stream hazard across two real PoseBusters processes."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
import pandas as pd
import pytest

spec=importlib.util.spec_from_file_location('campaign_process',Path(__file__).parents[1]/'scripts/run_sequential_path_campaign.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def test_two_posebusters_evaluations_use_live_process_streams(tmp_path):
    worker=tmp_path/'worker.py'
    worker.write_text('''
import json,sys
from rdkit import Chem
from rdkit.Chem import AllChem
from posebusters import PoseBusters
m=Chem.AddHs(Chem.MolFromSmiles('CC'))
assert AllChem.EmbedMolecule(m,randomSeed=42)==0
m=Chem.RemoveHs(m)
protein=Chem.MolFromPDBBlock('ATOM      1  CA  ALA A   1      20.000  20.000  20.000  1.00 20.00           C  \\nEND\\n',sanitize=False,removeHs=False)
checks=PoseBusters(config='dock_fast',max_workers=0).bust(mol_pred=m,mol_cond=protein,full_report=False)
assert len(checks)==1 and bool(checks.iloc[0]['mol_pred_loaded']) and bool(checks.iloc[0]['mol_cond_loaded'])
print(json.dumps({'checked':True}))
''',encoding='utf-8')
    for n in range(2):
        with (tmp_path/f'{n}.log').open('w',encoding='utf-8') as log:
            subprocess.run([sys.executable,str(worker)],stdout=log,stderr=subprocess.STDOUT,check=True)
        assert '"checked": true' in (tmp_path/f'{n}.log').read_text()
        assert 'closed file' not in (tmp_path/f'{n}.log').read_text()

def test_evaluation_error_blocks_retention_and_cleanup(tmp_path,monkeypatch):
    out=tmp_path/'evaluated';out.mkdir()
    pd.DataFrame({'pb_error':['Incomplete PoseBusters checks']}).to_csv(out/'candidate_metrics.csv',index=False)
    monkeypatch.setattr(module.subprocess,'run',lambda *args,**kw:None)
    with pytest.raises(ValueError,match='retain raw structures'):
        module.evaluate_isolated(tmp_path,tmp_path,'test',out,{'arms':'gradient','batch_indices':[30]},tmp_path/'log')
