"""Same-assay two-stage MMFF for saved original Steer, without regeneration."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem,rdBase
from evomolsteer.io import read_json,write_json,digest
from evomolsteer.nc_guidance.energy_probe import measure


def evaluate(repo,config,output):
    repo=Path(repo);m=read_json(config);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    base=repo/m['raw_dataset']/'results'/m['campaign']/m['arm'];inputs={}
    for folder in sorted(base.glob('batch_*')):
        for name in ('final_records.json','molecules_all_built.sdf','posebusters_checks.csv'):
            path=folder/name
            if path.exists():inputs[path.relative_to(repo).as_posix()]=digest(path)
    key={'inputs':inputs,'energy_code_sha256':digest(Path(__file__).parents[1]/'nc_guidance/energy_probe.py'),'rdkit':rdBase.rdkitVersion}
    target=output/'summary.json'
    if target.exists():
        old=read_json(target)
        if old['provenance']!=key:raise ValueError('Reference quality input/assay changed; use a new output version')
        return old
    rows=[];checks=None
    for folder in sorted(base.glob('batch_*')):
        records=read_json(folder/'final_records.json');source=folder/'molecules_all_built.sdf'
        mols={mol.GetProp('node_id'):mol for mol in Chem.SDMolSupplier(str(source),sanitize=False,removeHs=False) if mol is not None}
        if len(mols)!=sum(r['build_success'] for r in records):raise ValueError('Source molecule/record count differs')
        pb=pd.read_csv(folder/'posebusters_checks.csv');cols=[c for c in pb.columns if c not in ('file','molecule')]
        if checks is None:checks=cols
        elif cols!=checks:raise ValueError('Original PB selected tests changed across batches')
        tests={r['molecule']:{c:True if str(r[c]).lower()=='true' else False if str(r[c]).lower()=='false' else None for c in cols} for r in pb.to_dict('records')}
        for r in records:
            node=r['node_id'];test=tests.get(node);row={'node_id':node,'batch':r['batch'],'slot':r['slot'],'pic50':r['pic50_on_rescore'],
                'qualified':bool(r['build_success'] and r['connected'] and (r.get('min_protein_distance_A') or 0)>=.8),
                'legacy_PB_pass':bool(test and all(v is True for v in test.values())),
                'legacy_PB_unresolved':bool(test and any(v is None for v in test.values())),
                'status':'not_decodable'}
            if node in mols and r['connected']:
                try:row.update(measure(mols[node]))
                except Exception as exc:row.update(status='error',error_type=type(exc).__name__)
            rows.append(row)
    frame=pd.DataFrame(rows);frame.to_parquet(output/'per_seed.parquet',compression=None,index=False)
    vals=frame.heavy_relaxation_relief_after_H.dropna();qualified=frame[frame.qualified];top=qualified.sort_values('pic50',ascending=False).iloc[0].to_dict()
    result={'provenance':key,'n':len(frame),'n_qualified':int(frame.qualified.sum()),'energy_coverage':len(vals),
        'converged':int(frame.status.eq('converged').sum()),'median_heavy_relief_after_H_per_atom':float(vals.median()),
        'p90_heavy_relief_after_H_per_atom':float(vals.quantile(.9)),'best_qualified':top,
        'legacy_PB_pass':int(frame.legacy_PB_pass.sum()),'legacy_PB_unresolved':int(frame.legacy_PB_unresolved.sum()),'legacy_PB_columns':checks,
        'limits':['No new generation or affinity inference.',
            'Two-stage fixed-heavy H minimization then full MMFF94s minimization; local relaxation proxy per heavy atom, not binding energy.',
            'Legacy PB selected columns differ from current library defaults; compare explicitly shared columns or report separately.',
            'Original Steer particles have ancestry dependence.']}
    write_json(target,result);return result


def highest_current_pb(repo,config,protein,output,max_candidates=20):
    """Score-ordered quality audit; stop at first pass without new molecules."""
    from importlib.metadata import version
    from posebusters import PoseBusters
    from evomolsteer.local_motif.quality import binary_checks
    repo=Path(repo);m=read_json(config);protein=Path(protein);output=Path(output)
    if output.exists():return read_json(output)
    base=repo/m['raw_dataset']/'results'/m['campaign']/m['arm'];candidates=[]
    for folder in sorted(base.glob('batch_*')):
        for r in read_json(folder/'final_records.json'):
            if r['build_success'] and r['connected'] and (r.get('min_protein_distance_A') or 0)>=.8:candidates.append(r)
    candidates.sort(key=lambda r:(-r['pic50_on_rescore'],r['batch'],r['slot']))
    pb=PoseBusters(config='dock',max_workers=0);molecules={};rows=[]
    for r in candidates[:max_candidates]:
        batch=r['batch']
        if batch not in molecules:
            source=base/f'batch_{batch:03d}'/'molecules_all_built.sdf'
            molecules[batch]={v.GetProp('node_id'):v for v in Chem.SDMolSupplier(str(source),sanitize=False,removeHs=False) if v is not None}
        tests=binary_checks(pb.bust(molecules[batch][r['node_id']],mol_cond=protein,full_report=False).iloc[0].to_dict())
        row={'node_id':r['node_id'],'pic50':r['pic50_on_rescore'],'tests':tests,'passes_all':bool(tests and all(v is True for v in tests.values()))};rows.append(row)
        if row['passes_all']:break
    found=bool(rows and rows[-1]['passes_all'])
    result={'tool':'PoseBusters','version':version('posebusters'),'config':'dock','protein_sha256':digest(protein),
        'n_evaluated':len(rows),'best_passing_pic50':rows[-1]['pic50'] if found else None,'best_passing_node':rows[-1]['node_id'] if found else None,
        'rows':rows,'highest_passing_identified':found,'meaning':'All higher-scoring basic-qualified original candidates were tested with this same current PB assay; no affinity inference or regeneration.'}
    write_json(output,result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',default='.');p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();r=evaluate(Path(a.repo).resolve(),a.config,a.output);print({k:r[k] for k in ('n','n_qualified','energy_coverage','median_heavy_relief_after_H_per_atom','legacy_PB_pass')})


if __name__=='__main__':main()
