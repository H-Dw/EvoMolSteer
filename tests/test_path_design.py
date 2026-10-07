import sys,json
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from compile_path_design import compile_design

def fixture(tmp_path):
    request={'input_sha256':'abc','instruction_sha256':'def','analyst_sha256':'ghi'}
    response={**request,'decision':'test_source','updates':{'reference_sha256':'xyz','target_definition':'decoded terminal'},
        'justification':'Observed credit is testable, not proven causal','validation_hypotheses':['affinity gain'],'evidence_ids':['E_credit']}
    (tmp_path/'Designer.request.json').write_text(json.dumps(request))
    (tmp_path/'Designer.response.json').write_text(json.dumps(response))
    (tmp_path/'Designer.input.json').write_text('{"candidate_reference_sha256":"xyz"}')
    parent=tmp_path/'parent.json';parent.write_text('{"native_rms_ratio":0.33,"window":[0,0.5]}')
    return parent,response
def test_source_only_compilation(tmp_path):
    parent,r=fixture(tmp_path);p=compile_design(tmp_path,parent,tmp_path/'output.json')
    assert p['native_rms_ratio']==.33 and p['window']==[0,.5] and p['reference_sha256']=='xyz'
def test_reject_simultaneous_scalar_changes(tmp_path):
    parent,r=fixture(tmp_path);r['updates']['native_rms_ratio']=.66
    (tmp_path/'Designer.response.json').write_text(json.dumps(r))
    with pytest.raises(ValueError):compile_design(tmp_path,parent,tmp_path/'out.json')
def test_reject_unbound_response(tmp_path):
    parent,r=fixture(tmp_path);r['instruction_sha256']='different'
    (tmp_path/'Designer.response.json').write_text(json.dumps(r))
    with pytest.raises(ValueError):compile_design(tmp_path,parent,tmp_path/'out.json')
