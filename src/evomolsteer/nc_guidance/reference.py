"""Reuse historical Steer molecules with the current basic quality/strain assay.

No new generation or affinity inference. The small cache contains scalar assay
results only; original coordinates, graphs and lineage remain in the dataset.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from rdkit import Chem,RDLogger,rdBase
from evomolsteer.io import read_json,write_json,digest
from evomolsteer.agentic_steer import native


def run(repo,cfg):
    repo=Path(repo);root=repo/cfg['dataset']/'results'/cfg['campaign']
    files=sorted((root/cfg['arm']).glob('batch_*/molecules_raw_decodable.sdf'))
    initial=root/cfg['arm']/'batch_000/initial_state.pt.gz'
    provenance={'final_records_sha256':digest(root/'final_records.json'),
        'initial_state_sha256':digest(initial),'quality_assay_sha256':digest(native.__file__),
        'rdkit_version':rdBase.rdkitVersion,'SDFs':{str(p.relative_to(root)):digest(p) for p in files},
        'coordinate_source':'Original SDF precision in the fixed target receptor coordinate frame',
        'score_source':'Cached final pic50_on_rescore, never rerun an affinity head',
        'assay':'Current molecule_quality: sanitize, connected, zero receptor pairs below .8 A; AddHs then MMFF94s 500 steps',
        'pocket_source':'Common target pocket from the first original initial state; sample equality checked in that state'}
    key=hashlib.sha256(json.dumps(provenance,sort_keys=True).encode()).hexdigest()
    cache=repo/'data/analysis_cache'/Path(cfg['campaign_study']).name/'steer_quality.parquet';meta=cache.with_suffix('.json')
    if cache.exists() and meta.exists() and read_json(meta)['input_signature']==key:
        table=pd.read_parquet(cache)
    else:
        with gzip.open(initial,'rb') as f:init=torch.load(f,map_location='cpu',weights_only=False)
        pocket=init['pocket_target'];com=init['target_com'].reshape(-1,3)
        if not torch.allclose(com,com[0:1].expand_as(com)) or not torch.allclose(pocket['coords'],pocket['coords'][0:1].expand_as(pocket['coords'])):
            raise ValueError('Target pocket differs within the original initial batch')
        local={'coords':pocket['coords'][0:1],'mask':pocket['mask'][0:1]};scale=float(init['coord_scale'])
        scores={r['node_id']:r for r in read_json(root/'final_records.json') if r['arm']==cfg['arm']}
        if len(files)*50!=len(scores):raise ValueError('Expected complete original batches; inspect missing SDF files')
        RDLogger.DisableLog('rdApp.warning');rows=[];seen=set()
        for path in files:
            molecules={}
            for mol in Chem.SDMolSupplier(str(path),sanitize=False,removeHs=False):
                if mol is None:continue
                if not mol.HasProp('node_id'):raise ValueError('Original SDF lacks candidate identity')
                molecules[mol.GetProp('node_id')]=mol
            batch=int(path.parent.name.split('_')[-1])
            for node,r in scores.items():
                if r['batch']!=batch:continue
                q=native.molecule_quality(molecules.get(node),local,com[0].numpy(),scale)
                rows.append({'node_id':node,'batch':batch,'slot':r['slot'],'pic50':r['pic50_on_rescore'],
                    **{k:v for k,v in q.items() if k!='smiles'}});seen.add(node)
            print('historical_quality_batch',batch,flush=True)
        if seen!=set(scores):raise ValueError('Historical attempt coverage changed')
        table=pd.DataFrame(rows);cache.parent.mkdir(parents=True,exist_ok=True)
        table.to_parquet(cache,index=False,compression=None);write_json(meta,{'input_signature':key,'provenance':provenance})
    score=pd.to_numeric(table.pic50,errors='coerce');valid=table.qualified_structure&score.notna()
    result={'n_attempts':len(table),'n_scored':int(score.notna().sum()),'n_basic_qualified':int(valid.sum()),
        'mean_all_pic50':float(score.mean()),'best_basic_qualified_pic50':float(score[valid].max()),
        'median_MMFF_relief_per_heavy':float(table.mmff_relief_per_heavy.median()),
        'energy_coverage':int(table.mmff_relief_per_heavy.notna().sum()),
        'n_qualified_above_8p704806':int((score[valid]>=8.704806).sum()),
        'cache':str(cache.relative_to(repo)),'cache_sha256':digest(cache),'provenance':provenance,
        'comparison_limits':'Historical 1000 Steer finals are not budget-matched independent draws; repeated lineages remain repeated, not removed'}
    write_json(repo/cfg['campaign_study']/'summary/historical_steer_comparison.json',result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',default='.');p.add_argument('--config',required=True);a=p.parse_args()
    r=run(a.repo,read_json(a.config));print({k:v for k,v in r.items() if k!='provenance'})


if __name__=='__main__':main()
