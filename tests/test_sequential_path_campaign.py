"""Policy tests do not start SSH sessions or perform molecular inference."""
import importlib.util
import json
from pathlib import Path
import pytest
from evomolsteer.io import digest

spec=importlib.util.spec_from_file_location('path_campaign',Path(__file__).parents[1]/'scripts/run_sequential_path_campaign.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf-8')

def driver(tmp_path):
    d=module.Driver.__new__(module.Driver)
    d.root=tmp_path;d.cfg=tmp_path/'configs/experiments/elite_path20_v1'
    d.docs=tmp_path/'docs/experiments/elite_path20_20261007'
    ref=d.cfg/'terminal_path_reference.json.gz';ref.parent.mkdir(parents=True);ref.write_bytes(b'fixture')
    p={'reference_sha256':digest(ref),'reward_view':'endpoint_path_value','native_rms_ratio':.33,
       'teacher_score_beta':2.,'mixture_temperature':.5,'teacher_endpoint_temperature_A2':4.,
       'window':[.1,.7],'derivative_path':'flowr_endpoint_vjp','seed':42,'round':6,
       'analyst_sha256':'historical'}
    for n in [4,5,6,7]:write(d.cfg/f'round{n:02d}.json',{**p,'round':n})
    return d,p

def test_strength_trial_restores_evaluated_parent_and_changes_one_axis(tmp_path):
    d,p=driver(tmp_path);d.best=lambda kernel_only:6
    path,_,batches,arms,n=d.freeze(7);proposal=json.loads(path.read_text())
    assert proposal['native_rms_ratio']==.66 and proposal['window']==[.1,.7]
    for key,value in p.items():
        if key not in ['round','native_rms_ratio','analyst_sha256']:assert proposal[key]==value
    assert proposal['derivation']['historical_parent_provenance']['analyst_sha256']=='historical'
    assert proposal['derivation']['single_axis_change']=={'native_rms_ratio':.66}
    assert (batches,arms,n)==('30','gradient',50)

def test_best_parent_backtracks_after_a_decline(tmp_path):
    d,_=driver(tmp_path)
    for n in [6,7]:write(d.docs/f'round{n:02d}/retention.json',{'status':'complete'})
    d.metrics=lambda n:n;d.rank=lambda m:10 if m==6 else 1
    assert d.best(kernel_only=True)==6

def test_independent_confirmation_remains_frozen(tmp_path):
    d,_=driver(tmp_path);d.best=lambda:6
    path,_,batches,arms,n=d.freeze(18)
    assert json.loads(path.read_text())['derivation']['parent_round']==6
    assert (batches,arms,n)==('32,33','unguided,gradient',100)
    d.best=lambda:7
    path,_,batches,arms,n=d.freeze(20)
    assert json.loads(path.read_text())['derivation']['parent_round']==6
    assert (batches,arms,n)==('34,35','unguided,gradient',100)

def test_formula_requires_a_bound_designer_response(tmp_path):
    d,p=driver(tmp_path)
    p['reward_view']='endpoint_pointcloud';write(d.cfg/'round05.json',p)
    request={'input_sha256':'evidence','instruction_sha256':'skill'}
    write(d.docs/'round06_agents/Designer.request.json',request)
    response={**request,'decision':'test_formula','reward_view':'endpoint_path_value','justification':['formula only']}
    write(d.docs/'round06_agents/Designer.response.json',response)
    path,_,_,_,_=d.freeze(6)
    assert json.loads(path.read_text())['reward_view']=='endpoint_path_value'
    response['input_sha256']='stale';write(d.docs/'round06_agents/Designer.response.json',response)
    with pytest.raises(ValueError,match='binding'):d.freeze(6)
