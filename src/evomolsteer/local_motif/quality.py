"""Posthoc PoseBusters dock diagnostics; no graph-identity or promotion gate."""
import argparse
import gzip
from importlib.metadata import version
from pathlib import Path
import numpy as np
from rdkit import Chem
from evomolsteer.io import read_json,write_json,digest


def evaluate(folder,protein,output):
    from posebusters import PoseBusters
    folder=Path(folder);protein=Path(protein);output=Path(output)
    source=digest(folder/'records.json');protein_sha=digest(protein)
    if output.exists():
        value=read_json(output)
        if value['source_sha256']!=source or value['protein_sha256']!=protein_sha:raise ValueError('Quality cache input changed')
        return value
    pb=PoseBusters(config='dock',max_workers=0);rows=[]
    for record in read_json(folder/'records.json'):
        row={'slot':record['slot'],'pic50':record['pic50'],'basic_qualified':record['qualified_structure'],
             'status':'not_decodable','passes_all':False}
        path=folder/f"{record['slot']:03d}.sdf.gz"
        if path.exists():
            try:
                with gzip.open(path,'rt') as f:mol=Chem.MolFromMolBlock(f.read().split('$$$$')[0],sanitize=False,removeHs=False)
                result=pb.bust(mol, mol_cond=protein,full_report=False)
                if len(result)!=1:raise ValueError('Expected one quality row')
                vals=result.iloc[0].to_dict()
                tests={k:bool(v) for k,v in vals.items() if isinstance(v,(bool,np.bool_))}
                row.update(status='evaluated',tests=tests,passes_all=bool(tests and all(tests.values())))
            except Exception as error:row.update(status='error',error=type(error).__name__+': '+str(error)[:200])
        rows.append(row)
    value={'tool':'PoseBusters','version':version('posebusters'),'config':'dock','protein_sha256':protein_sha,
        'source_sha256':source,'n_attempts':len(rows),'n_evaluated':sum(r['status']=='evaluated' for r in rows),
        'n_pass_all':sum(r['passes_all'] for r in rows),'rows':rows,
        'scope':'Posthoc structural diagnostics. No reference ligand or graph-identity test; no new generation or promotion gate.'}
    write_json(output,value);return value


def main():
    p=argparse.ArgumentParser();p.add_argument('--folder',required=True);p.add_argument('--protein',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    r=evaluate(a.folder,a.protein,a.output);print({k:r[k] for k in ['n_attempts','n_evaluated','n_pass_all']})


if __name__=='__main__':main()
