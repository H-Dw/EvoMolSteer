"""Attach protein-region context to coordinate fields without changing rewards."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from evomolsteer.io import write_json,digest
from evomolsteer.generation.window_reference import load_reference


def annotate(reference,effects,protein,output):
    ref=load_reference(reference);table=pd.read_parquet(effects).set_index('feature');atoms=[]
    for line in Path(protein).read_text().splitlines():
        if line[:6].strip()!='ATOM':continue
        atoms.append({'coords_A':[float(line[30:38]),float(line[38:46]),float(line[46:54])],
            'chain':line[21].strip(),'residue':line[17:20].strip(),'residue_id':line[22:26].strip(),
            'insertion_code':line[26].strip(),'atom_name':line[12:16].strip()})
    positions=np.array([a['coords_A'] for a in atoms]);records=[]
    for i,point in enumerate(ref['landmarks_A']):
        distances=np.linalg.norm(positions-np.array(point),axis=1);j=int(distances.argmin())
        if distances[j]>1e-5:raise ValueError('Reference landmark/protein frame mismatch')
        group=[]
        for kind in ('softmin','occupancy3','occupancy5'):
            name=f'landmark_{i:02d}_{kind}';row=table.loc[name].to_dict();scale=ref['feature_scale'][ref['features'].index(name)]
            group.append({'feature':name,'scale':scale,'scale_floor':scale<=1e-5,**row})
        records.append({'landmark':i,'protein_atom_context':atoms[j],'coordinate_associations':group})
    result={'schema_version':'affinity-landmark-context-1.0','regions':records,
        'sources':{'reference_sha256':digest(reference),'effects_sha256':digest(effects),'protein_sha256':digest(protein)},
        'interpretation':'Nearest protein atom labels locate coordinate fields. They are not ligand atom-type features, hydrogen-bond assignments or causal affinity mechanisms.'}
    write_json(output,result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('reference','effects','protein','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();annotate(a.reference,a.effects,a.protein,a.output)
