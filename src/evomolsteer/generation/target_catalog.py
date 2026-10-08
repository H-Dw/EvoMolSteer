"""Resolve an explicit target list or prepared CrossDocked pocket/ligand pairs."""
import hashlib
from pathlib import Path
import re

from ..io import digest,read_json
from ..storage.transactions import checked_path

FORMAT='evomolsteer.target_collection.v1'


def target_key(identifier):
    if not isinstance(identifier,str) or not identifier.strip():
        raise ValueError('A nonempty target_id is required')
    reserved={'CON','PRN','AUX','NUL'}|{p+str(i) for p in ('COM','LPT') for i in range(1,10)}
    if re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,99}',identifier) and identifier not in ('.','..') and identifier.split('.')[0].upper() not in reserved:
        return identifier
    stem=re.sub(r'[^A-Za-z0-9_-]+','_',identifier).strip('_')[:70] or 'target'
    return stem+'_'+hashlib.sha256(identifier.encode()).hexdigest()[:12]


def discover_targets(dataset,*,manifest=None,protein_glob='**/*_pocket10.pdb',
                     protein_suffix='_pocket10.pdb',ligand_suffix='.sdf',expected_targets=None):
    """No random selection, ambiguous pairs, silent omissions or pickle loading.

    Auto discovery identifies each pocket/ligand pair by its relative path
    without the protein suffix, including multiple complexes in one folder.
    A JSON target manifest can supply explicit identities for a general split.
    """
    root=Path(dataset).resolve()
    if not root.is_dir():raise NotADirectoryError(root)
    if manifest:
        spec=read_json(manifest)
        if spec.get('format')!=FORMAT or not isinstance(spec.get('targets'),list):
            raise ValueError('Unsupported target collection manifest')
        rows=spec['targets']
    else:
        if not protein_suffix or not ligand_suffix or any(s in protein_suffix+ligand_suffix for s in '/\\'):
            raise ValueError('Filename suffixes cannot include directories')
        rows=[]
        for protein in sorted(root.glob(protein_glob)):
            if not protein.is_file() or not protein.name.endswith(protein_suffix):
                raise ValueError('Discovery glob includes a nonmatching protein: '+str(protein))
            ligand=protein.with_name(protein.name[:-len(protein_suffix)]+ligand_suffix)
            identifier=protein.relative_to(root).as_posix()[:-len(protein_suffix)]
            rows.append({'target_id':identifier,'target_protein':str(protein.relative_to(root)),
                         'target_ligand':str(ligand.relative_to(root))})
    if not rows:raise ValueError('No target inputs discovered')
    resolved=[];ids=set();keys=set()
    for row in sorted(rows,key=lambda r:r['target_id']):
        identifier=row['target_id'];key=target_key(identifier)
        if identifier in ids or key.casefold() in keys:
            raise ValueError('Ambiguous/duplicate target; provide unique IDs in a target manifest: '+identifier)
        ids.add(identifier);keys.add(key.casefold())
        files={}
        for role in ('target_protein','target_ligand'):
            path=checked_path(root/row[role],root)
            if not path.is_file() or path.stat().st_size==0:raise FileNotFoundError(path)
            files[role]={'path':path.relative_to(root).as_posix(),'sha256':digest(path),'bytes':path.stat().st_size}
        resolved.append({'target_id':identifier,'key':key,'files':files})
    if expected_targets is not None and (type(expected_targets) is not int or len(resolved)!=expected_targets):
        raise ValueError(f'Expected {expected_targets} targets, discovered {len(resolved)}')
    return resolved
