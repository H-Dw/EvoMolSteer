"""Small receptor-frame spatial diagnostics for paired success/failure cases.

Contact counts are distance proxies, not hydrogen-bond assignments or energies.
Only residue summaries are saved; atom-pair matrices stay in memory.
"""
import argparse
import gzip
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from rdkit import Chem
from rdkit.Chem import Lipinski

from evomolsteer.io import read_json,write_table,digest
from evomolsteer.storage.transactions import atomic_json


def protein_atoms(path):
    rows=[];seen=set()
    for line in Path(path).read_text().splitlines():
        if not line.startswith('ATOM  ') or line[16:17] not in (' ','A'):continue
        name=line[12:16].strip();element=line[76:78].strip() or name.lstrip('0123456789')[:1]
        if element.upper() in ('H','D'):continue
        residue=f'{line[21:22].strip()}:{line[17:20].strip()}:{line[22:26].strip()}{line[26:27].strip()}'
        key=(residue,name)
        if key in seen:continue
        seen.add(key)
        rows.append({'residue':residue,'atom':name,'element':element.upper(),
                     'xyz':[float(line[30:38]),float(line[38:46]),float(line[46:54])]})
    if not rows:raise ValueError('No protein heavy atoms in PDB')
    return rows


def run(work,protein,candidates,methods,output):
    work,output=Path(work),Path(output);output.mkdir(parents=True,exist_ok=True)
    atoms=protein_atoms(protein);xyz=np.asarray([a['xyz'] for a in atoms]);protein_polar=np.asarray([a['element'] in ('N','O','S') for a in atoms])
    residues=np.asarray([a['residue'] for a in atoms]);allrows=[];summaries=[];sources={str(protein):digest(protein)}
    for candidate in candidates:
        local=[];near=set()
        for method in methods:
            folder=work/'methods'/method/candidate;path=folder/'terminal.sdf.gz'
            if not path.exists():continue
            with gzip.open(path,'rt') as stream:mol=Chem.MolFromMolBlock(stream.read().split('$$$$')[0],removeHs=False)
            if mol is None:raise ValueError('Could not read '+str(path))
            sources[str(path)]=digest(path);record=read_json(folder/'result.json')
            positions=mol.GetConformer().GetPositions();elements=np.asarray([a.GetSymbol() for a in mol.GetAtoms()]);heavy=elements!='H'
            positions=positions[heavy];elements=elements[heavy];polar=np.isin(elements,['N','O','S'])
            distances=cdist(positions,xyz)
            summaries.append({'candidate':candidate,'method':method,'pic50':record['pic50'],'heavy_atoms':len(positions),
                'HBA':Lipinski.NumHAcceptors(mol),'HBD':Lipinski.NumHDonors(mol),
                'receptor_polar_pairs_lt3p5':int((distances[np.ix_(polar,protein_polar)]<3.5).sum()),
                'protein_heavy_pairs_lt4':int((distances<4).sum()),
                'unique_ligand_atoms_within4':int((distances.min(1)<4).sum()),
                'min_protein_distance_A':float(distances.min()),'smiles':Chem.MolToSmiles(mol)})
            for residue in np.unique(residues):
                selected=residues==residue;d=distances[:,selected];min_all=float(d.min())
                if min_all<5:near.add(residue)
                p=distances[np.ix_(polar,selected&protein_polar)]
                nearest={}
                if p.size:
                    i,j=np.unravel_index(p.argmin(),p.shape);lig=np.flatnonzero(polar)[i];protein_index=np.flatnonzero(selected&protein_polar)[j]
                    nearest={'closest_ligand_polar_element':elements[lig],'closest_ligand_heavy_index':int(lig),
                             'closest_protein_polar_atom':atoms[protein_index]['atom']}
                local.append({'candidate':candidate,'method':method,'residue':residue,'min_heavy_A':min_all,
                    'pairs_lt4':int((d<4).sum()),'ligand_atoms_lt4':int((d.min(1)<4).sum()),
                    'min_polar_A':float(p.min()) if p.size else None,'polar_pairs_lt3p5':int((p<3.5).sum()),**nearest})
        allrows += [r for r in local if r['residue'] in near]
    residue_df=pd.DataFrame(allrows);summary=pd.DataFrame(summaries)
    write_table(output/'residue_distance_proxies.csv',residue_df)
    write_table(output/'molecular_contact_proxies.csv',summary)
    atomic_json(output/'audit.json',{'sources_sha256':sources,'methods':methods,'candidates':candidates,
        'frame':'Original aligned receptor world frame; ligand not superposed',
        'limitations':'N/O/S distance counts do not establish donor/acceptor compatibility, orientation, actual hydrogen bonds, binding energies or causal head attribution.'})
    return residue_df,summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--work',required=True);p.add_argument('--protein',required=True)
    p.add_argument('--candidates',required=True);p.add_argument('--methods',default='unguided,M2,top10_global_0,blend_0p5');p.add_argument('--output',required=True)
    a=p.parse_args();run(a.work,a.protein,a.candidates.split(','),a.methods.split(','),a.output)


if __name__=='__main__':main()
