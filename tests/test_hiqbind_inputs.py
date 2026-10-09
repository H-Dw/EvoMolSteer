import io
import pickle
import tarfile

import numpy as np
import pytest

from evomolsteer.generation.hiqbind_inputs import (plain_metadata,validate_test_indices,
    extract_test_structures,read_official_test_ids)
from evomolsteer.io import read_json


def test_split_mapping_uses_official_order_and_rejects_leakage(tmp_path):
    ids=['system_A','system_B','system_C']
    splits={'idx_train':np.array([0]),'idx_val':np.array([1]),'idx_test':np.array([2])}
    assert validate_test_indices(ids,splits)==['system_C']
    (tmp_path/'system_ids.pkl').write_bytes(pickle.dumps(ids));np.savez(tmp_path/'splits.npz',**splits)
    assert read_official_test_ids(tmp_path)[0]==['system_C']
    with pytest.raises(ValueError,match='overlap'):validate_test_indices(ids,{**splits,'idx_test':np.array([0])})
    with pytest.raises(ValueError,match='outside'):validate_test_indices(ids,{**splits,'idx_test':np.array([3])})


def test_plain_metadata_rejects_executable_pickle():
    assert plain_metadata(pickle.dumps(['A','B']))==['A','B']
    with pytest.raises(pickle.UnpicklingError):plain_metadata(pickle.dumps(Exception('not plain metadata')))


def test_selected_structures_are_exact_bytes_and_training_is_not_extracted(tmp_path):
    archive=tmp_path/'input.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for name,data in [('data/system_A.pdb',b'train protein'),('data/system_A.sdf',b'train ligand'),
                          ('data/system_C.pdb',b'test protein'),('data/system_C.sdf',b'test ligand')]:
            item=tarfile.TarInfo(name);item.size=len(data);tar.addfile(item,io.BytesIO(data))
    output=tmp_path/'prepared';result=extract_test_structures(archive,['system_C'],output)
    assert result['targets']==1 and (output/'system_C.pdb').read_bytes()==b'test protein'
    assert not (output/'system_A.pdb').exists()
    assert read_json(output/'targets.json')['split']=='test'
    with pytest.raises(FileExistsError):extract_test_structures(archive,['system_C'],output)


def test_missing_or_ambiguous_selected_structures_abort(tmp_path):
    archive=tmp_path/'missing.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        item=tarfile.TarInfo('data/system_C.pdb');item.size=1;tar.addfile(item,io.BytesIO(b'p'))
    with pytest.raises(ValueError,match='Missing selected'):extract_test_structures(archive,['system_C'],tmp_path/'missing')
