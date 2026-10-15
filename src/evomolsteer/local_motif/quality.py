"""Posthoc PoseBusters dock diagnostics; no graph-identity or promotion gate."""
import argparse
import gzip
from importlib.metadata import version
from pathlib import Path
import numpy as np
from rdkit import Chem
from evomolsteer.io import read_json,write_json,digest


def binary_checks(values):
    # Failed modules can yield NaN. Never silently omit those tests from an
    # all-tests-passed claim. Numeric 0/1 is an allowed binary representation.
    return {k:bool(v) if isinstance(v,(bool,np.bool_)) or isinstance(v,(int,float,np.number)) and np.isfinite(v) and v in (0,1) else None
            for k,v in values.items()}


def read_molecule(path):
    with gzip.open(path,'rt') as f:return Chem.MolFromMolBlock(f.read().split('$$$$')[0],sanitize=False,removeHs=False)


def evaluate(folder,protein,output):
    from posebusters import PoseBusters
    folder=Path(folder);protein=Path(protein);output=Path(output)
    source=digest(folder/'records.json');protein_sha=digest(protein)
    value=read_json(output) if output.exists() else None
    if value:
        if value['source_sha256']!=source or value['protein_sha256']!=protein_sha:raise ValueError('Quality cache input changed')
        if value['version']!=version('posebusters'):raise ValueError('Quality cache library changed')
        if value.get('policy_version')==2:return value
    pb=PoseBusters(config='dock',max_workers=0);rows=[]
    if value:
        # Reuse complete cached binary results; only recompute rows whose v1
        # serializer may have omitted an unknown/nonboolean selected test.
        legacy=output.with_name(output.stem+'.policy_v1.json')
        if not legacy.exists():write_json(legacy,value)
        first=next(r for r in value['rows'] if r['status']=='evaluated')
        expected=pb.bust(read_molecule(folder/f"{first['slot']:03d}.sdf.gz"),mol_cond=protein,full_report=False).columns.tolist()
        for r in value['rows']:
            if r['status']=='evaluated' and set(r['tests'])!=set(expected):
                fresh=pb.bust(read_molecule(folder/f"{r['slot']:03d}.sdf.gz"),mol_cond=protein,full_report=False)
                r['tests']=binary_checks(fresh.iloc[0].to_dict());r['recomputed_incomplete_legacy_tests']=True
            if r['status']=='evaluated':r['passes_all']=all(v is True for v in r['tests'].values())
        value.update(policy_version=2,rows=value['rows'],n_pass_all=sum(r['passes_all'] for r in value['rows']),
            n_unresolved=sum(r['status']=='evaluated' and any(v is None for v in r['tests'].values()) for r in value['rows']),
            expected_test_columns=expected,legacy_policy_correction='Missing selected tests were previously omitted; unknowns now count as unresolved, never passed. Complete cached tests were reused.')
        write_json(output,value);return value
    for record in read_json(folder/'records.json'):
        row={'slot':record['slot'],'pic50':record['pic50'],'basic_qualified':record['qualified_structure'],
             'status':'not_decodable','passes_all':False}
        path=folder/f"{record['slot']:03d}.sdf.gz"
        if path.exists():
            try:
                mol=read_molecule(path)
                result=pb.bust(mol, mol_cond=protein,full_report=False)
                if len(result)!=1:raise ValueError('Expected one quality row')
                vals=result.iloc[0].to_dict()
                tests=binary_checks(vals)
                row.update(status='evaluated',tests=tests,passes_all=bool(tests and all(v is True for v in tests.values())))
            except Exception as error:row.update(status='error',error=type(error).__name__+': '+str(error)[:200])
        rows.append(row)
    value={'tool':'PoseBusters','version':version('posebusters'),'policy_version':2,'config':'dock','protein_sha256':protein_sha,
        'source_sha256':source,'n_attempts':len(rows),'n_evaluated':sum(r['status']=='evaluated' for r in rows),
        'n_pass_all':sum(r['passes_all'] for r in rows),'rows':rows,
        'n_unresolved':sum(r['status']=='evaluated' and any(v is None for v in r['tests'].values()) for r in rows),
        'scope':'Posthoc structural diagnostics. No reference ligand or graph-identity test; no new generation or promotion gate.'}
    write_json(output,value);return value


def main():
    p=argparse.ArgumentParser();p.add_argument('--folder',required=True);p.add_argument('--protein',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    r=evaluate(a.folder,a.protein,a.output);print({k:r[k] for k in ['n_attempts','n_evaluated','n_pass_all','n_unresolved']})


if __name__=='__main__':main()
