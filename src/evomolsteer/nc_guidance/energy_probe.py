"""Two-stage MMFF diagnostic separating added-H placement from heavy relaxation.

This is a local relaxation proxy, not binding energy or a replacement gate.
"""
import gzip
from pathlib import Path
from rdkit import Chem,rdBase
from rdkit.Chem import AllChem
from evomolsteer.io import read_json,write_json,digest


def measure(mol,max_iterations=500):
    Chem.SanitizeMol(mol);mh=Chem.AddHs(Chem.Mol(mol),addCoords=True)
    props=AllChem.MMFFGetMoleculeProperties(mh,mmffVariant='MMFF94s')
    if props is None:return {'status':'unsupported'}
    ff=AllChem.MMFFGetMoleculeForceField(mh,props);before=ff.CalcEnergy()
    for atom in mh.GetAtoms():
        if atom.GetAtomicNum()>1:ff.AddFixedPoint(atom.GetIdx())
    ff.Initialize();hcode=ff.Minimize(maxIts=max_iterations);after_h=ff.CalcEnergy()
    free=AllChem.MMFFGetMoleculeForceField(mh,props);code=free.Minimize(maxIts=max_iterations);after=free.CalcEnergy()
    n=mol.GetNumHeavyAtoms()
    return {'n_heavy':n,'raw_before':before,'H_relaxed_fixed_heavy':after_h,'all_relaxed':after,
        'H_only_relief_per_heavy':(before-after_h)/n,'heavy_relaxation_relief_after_H':(after_h-after)/n,
        'H_status':hcode,'all_status':code,'status':'converged' if hcode==code==0 else 'not_converged'}


def run(repo,cases,output):
    repo=Path(repo);rows=[]
    for case in cases:
        path=repo/case['path']
        if path.suffix=='.gz':
            with gzip.open(path,'rt') as f:mol=Chem.MolFromMolBlock(f.read().split('$$$$')[0],sanitize=False,removeHs=False)
        else:
            mol=next(m for m in Chem.SDMolSupplier(str(path),sanitize=False,removeHs=False)
                if m is not None and (not case.get('node_id') or m.GetProp('node_id')==case['node_id']))
        rows.append({'case':case['case'],'source_sha256':digest(path),**measure(mol)})
    result={'rdkit':rdBase.rdkitVersion,'source_code_sha256':digest(__file__),'cases':rows,
        'meaning':'Fixed-heavy H minimization then free MMFF94s minimization; local relaxation diagnostic, not binding free energy'}
    write_json(output,result);return result


def compare_confirmations(repo,master):
    """Reproduce cohort summaries with failed/unsupported attempts kept visible."""
    import pandas as pd
    from evomolsteer.io import write_table
    from .research_report import confirmation_paths
    repo=Path(repo);study=repo/master['campaign_study'];state=read_json(study/'state.json')
    cases=[];metadata=[]
    for entry in state['rounds']:
        if not entry.get('confirmation'):continue
        cfg=read_json(repo/master['config_dir']/(entry['round']+'.json'))
        paths={'tail_R05':repo/entry['result_folder'],**confirmation_paths(repo,master,cfg)}
        for arm,path in paths.items():
            for row in read_json(path/'records.json'):
                tag=f"{entry['round']}:{arm}:{row['slot']:03d}"
                metadata.append({'case':tag,'round':entry['round'],'arm':arm,'slot':row['slot'],
                    'eligible':row['valid_connected'],'qualified':row['qualified_structure'],
                    'pic50':row['pic50'],'legacy_relief':row['mmff_relief_per_heavy']})
                if row['valid_connected']:
                    cases.append({'case':tag,'path':(path/f"{row['slot']:03d}.sdf.gz").as_posix()})
    write_json(study/'energy_probe_confirmation_inputs.json',cases)
    result=run(repo,cases,study/'analysis/two_stage_mmff.json')
    frame=pd.DataFrame(metadata).merge(pd.DataFrame(result['cases']),on='case',how='left')
    frame.to_parquet(study/'analysis/two_stage_mmff.parquet',index=False,compression=None)
    summary=[]
    for (label,arm),g in frame.groupby(['round','arm']):
        values=g.heavy_relaxation_relief_after_H.dropna()
        summary.append({'round':label,'arm':arm,'n_attempts':len(g),'energy_coverage':len(values),
            'converged':int(g.status.eq('converged').sum()),'median_heavy_relief_after_H':float(values.median()),
            'p90_heavy_relief_after_H':float(values.quantile(.9)),'max_heavy_relief_after_H':float(values.max()),
            'median_legacy':float(g.legacy_relief.median())})
    return write_table(study/'summary/two_stage_mmff.csv',summary)
