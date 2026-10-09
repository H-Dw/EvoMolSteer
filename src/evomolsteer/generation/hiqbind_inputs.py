"""Prepare a test-only PDB/SDF collection from verified FLOWR HiQBind archives."""
import io
import json
from pathlib import Path,PurePosixPath
import pickle
import re
import shutil
import tarfile

import numpy as np

from ..io import digest
from ..storage.download import checksum
from ..storage.transactions import atomic_json
from .target_catalog import FORMAT,discover_targets

SOURCE='https://zenodo.org/records/20069589'
ARCHIVES={
    'hiqbind__final.tar.gz':(461105394,'ad56efdf798c043eacda395bcf182a7a'),
    'hiqbind__data_prepared.tar.gz':(3749912378,'f125550f99666677a714275aefe82bdb'),
}


class PlainMetadataUnpickler(pickle.Unpickler):
    def find_class(self,module,name):
        raise pickle.UnpicklingError('Only plain list/dict/string metadata is accepted')


def plain_metadata(data):
    return PlainMetadataUnpickler(io.BytesIO(data)).load()


def verify_archive(path,expected):
    path=Path(path);size,md5=expected
    if not path.is_file() or path.stat().st_size!=size or checksum(path)!=md5:
        raise ValueError('Official HiQBind archive checksum/size mismatch: '+str(path))
    return {'name':path.name,'bytes':size,'md5':md5,'verified':True}


def validate_test_indices(system_ids,splits):
    if not isinstance(system_ids,(list,tuple)) or not system_ids or any(
        not isinstance(s,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',s) for s in system_ids):
        raise ValueError('Expected a plain list of safe system identifiers')
    if len(set(system_ids))!=len(system_ids):raise ValueError('Duplicate dataset system IDs')
    groups={}
    for role in ('train','val','test'):
        array=np.asarray(splits['idx_'+role])
        if array.ndim!=1 or array.dtype.kind not in 'iu' or len(set(array.tolist()))!=len(array):
            raise ValueError('Split indices must be unique integer vectors')
        values=array.tolist()
        if values and (min(values)<0 or max(values)>=len(system_ids)):raise ValueError('Split index outside dataset')
        groups[role]=set(values)
    if any(groups[a]&groups[b] for a,b in (('train','test'),('val','test'),('train','val'))):
        raise ValueError('Official train/val/test indices overlap')
    if not groups['test']:raise ValueError('Empty official test split')
    return [system_ids[i] for i in np.asarray(splits['idx_test']).tolist()]


def unpack_final_metadata(archive,destination):
    """Keep plain published membership lists and split metadata, without LMDB."""
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    inventory=[]
    with tarfile.open(archive,'r:gz') as tar:
        for member in tar:
            inventory.append({'name':member.name,'bytes':member.size})
            name=PurePosixPath(member.name)
            wanted=name.name in ('splits.npz','system_ids.pkl','system_ids.json',
                'system_ids_train.pkl','system_ids_val.pkl','system_ids_test.pkl',
                'test_data.csv','test_data_processed.csv')
            if not wanted:continue
            if name.is_absolute() or '..' in name.parts or not member.isfile() or member.size>32*1024**3:
                raise ValueError('Unsupported official metadata member: '+member.name)
            output=destination/Path(*name.parts)
            if output.exists():raise FileExistsError(output)
            output.parent.mkdir(parents=True,exist_ok=True)
            tar.extract(member,path=destination,filter='data')
    return inventory


def read_official_test_ids(metadata_root):
    root=Path(metadata_root);splits=list(root.rglob('splits.npz'))
    if len(splits)!=1:raise ValueError('Exactly one published splits.npz is required')
    # This release has generation membership lists and separate affinity-local
    # indices. The affinity indices must not be applied to the generation lists.
    membership={role:list(root.rglob('system_ids_'+role+'.pkl')) for role in ('train','val','test')}
    if all(len(paths)==1 for paths in membership.values()):
        groups={role:plain_metadata(paths[0].read_bytes()) for role,paths in membership.items()}
        for role,ids in groups.items():
            if not isinstance(ids,list) or not ids or len(set(ids))!=len(ids) or any(
                not isinstance(s,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',s) for s in ids):
                raise ValueError('Invalid published generation membership: '+role)
        if any(set(groups[a])&set(groups[b]) for a,b in (('train','val'),('train','test'),('val','test'))):
            raise ValueError('Published generation memberships overlap')
        with np.load(splits[0],allow_pickle=False) as arrays:
            affinity_counts={role:len(arrays['idx_'+role]) for role in groups}
        proof={'method':'published per-split generation system_ids lists',
               'total_systems':sum(map(len,groups.values())),'test_systems':len(groups['test']),
               'generation_membership_counts':{role:len(ids) for role,ids in groups.items()},
               'affinity_index_counts':affinity_counts,'affinity_indices_used_for_selection':False,
               'generation_memberships_disjoint':True,'splits_sha256':digest(splits[0]),
               'membership_sha256':{role:digest(paths[0]) for role,paths in membership.items()}}
        return groups['test'],proof
    ids_json=list(root.rglob('system_ids.json'));ids_pickle=list(root.rglob('system_ids.pkl'))
    if len(ids_json)==1:ids=json.loads(ids_json[0].read_text())
    elif len(ids_pickle)==1:ids=plain_metadata(ids_pickle[0].read_bytes())
    else:
        raise ValueError('Published plain generation membership lists are missing')
    with np.load(splits[0],allow_pickle=False) as arrays:
        selected=validate_test_indices(ids,{key:arrays[key] for key in ('idx_train','idx_val','idx_test')})
    return selected,{'method':'official idx_test mapped through system_ids','total_systems':len(ids),
                     'test_systems':len(selected),'splits_sha256':digest(splits[0])}


def extract_test_structures(archive,target_ids,destination):
    destination=Path(destination)
    if destination.exists():raise FileExistsError(destination)
    expected={identifier+suffix for identifier in target_ids for suffix in ('.pdb','.sdf')}
    if len(expected)!=2*len(target_ids):raise ValueError('Duplicate selected systems')
    destination.mkdir(parents=True)
    seen=set()
    with tarfile.open(archive,'r|gz') as tar:
        for member in tar:
            name=PurePosixPath(member.name)
            if name.name not in expected:continue
            if name.is_absolute() or '..' in name.parts or not member.isfile() or not 0<member.size<64*1024**2:
                raise ValueError('Unsafe selected structure member: '+member.name)
            if name.name in seen:raise ValueError('Ambiguous structure paths for '+name.name)
            source=tar.extractfile(member)
            with (destination/name.name).open('wb') as output:shutil.copyfileobj(source,output)
            seen.add(name.name)
    if seen!=expected:raise ValueError('Missing selected structures: '+str(sorted(expected-seen)))
    manifest={'format':FORMAT,'dataset':'HiQBind','split':'test','source':SOURCE,'targets':[
        {'target_id':s,'target_protein':s+'.pdb','target_ligand':s+'.sdf'} for s in target_ids]}
    atomic_json(destination/'targets.json',manifest)
    catalog=discover_targets(destination,manifest=destination/'targets.json',expected_targets=len(target_ids))
    return {'targets':len(catalog),'input_bytes':sum(f['bytes'] for r in catalog for f in r['files'].values()),
            'files':catalog}


def prepare_hiqbind(cache,output,*,metadata_directory=None):
    cache=Path(cache);output=Path(output)
    if output.exists():raise FileExistsError('Prepared destination already exists; verify rather than overwrite: '+str(output))
    archives=[verify_archive(cache/name,expected) for name,expected in ARCHIVES.items()]
    metadata=Path(metadata_directory) if metadata_directory else cache/'split_metadata'
    inventory=unpack_final_metadata(cache/'hiqbind__final.tar.gz',metadata)
    identifiers,split_proof=read_official_test_ids(metadata)
    validation=extract_test_structures(cache/'hiqbind__data_prepared.tar.gz',identifiers,output)
    proof={'format':'evomolsteer.hiqbind_input_preparation.v1','source':SOURCE,'archives':archives,
           'split':split_proof,'final_archive_inventory':inventory,'structures':validation}
    atomic_json(output/'PREPARED.json',proof)
    return proof
