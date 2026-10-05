import gzip
import torch
from evomolsteer.generation.zero_equivalence import compare_final


def test_final_zero_equivalence_reports_tensor_mismatch(tmp_path):
    native={key:torch.zeros(2,3) for key in ['coords','atomics','bonds','charges','mask']}
    native['affinity']={'pic50':torch.ones(2,1)}
    def save(name,data):
        path=tmp_path/name
        with gzip.open(path,'wb') as f:torch.save(data,f)
        return path
    a=save('native.gz',native);b=save('zero.gz',native)
    assert compare_final(a,b)['passed']
    changed=dict(native);changed['coords']=native['coords'].clone();changed['coords'][0,0]=.01
    b=save('zero.gz',changed);r=compare_final(a,b)
    assert not r['passed'] and not r['byte_equal']['coords']
    assert r['byte_equal']['affinity.pic50'] and r['coordinate_rms_model_units']>0
