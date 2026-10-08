import json,sys
from pathlib import Path
import pytest
from evomolsteer.io import digest,write_json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_dynamic_contrast_campaign import Driver


def test_confirmation_freezes_candidate_before_labels_and_rejects_drift(tmp_path):
    d=Driver.__new__(Driver);d.root=tmp_path;d.cfg=tmp_path/'configs/experiments/dynamic_contrast10_v1'
    d.docs=tmp_path/'docs/experiments/dynamic_contrast10_20261008';d.cfg.mkdir(parents=True);d.docs.mkdir(parents=True)
    ref=d.cfg/'reference.json.gz';ref.write_bytes(b'immutable teacher data')
    p={'seed':42,'round':2,'window':[.1,.6],'reference_sha256':digest(ref),'regional_weight':.05}
    write_json(d.cfg/'round02.json',p);write_json(d.cfg/'round01.json',{**p,'round':1,'regional_weight':0.})
    write_json(d.docs/'protocol.json',{'selection_rule':'Frozen primary affinity rule'})
    d.best=lambda:2
    path,reference,batches,arms,n=d.freeze(8)
    assert batches=='37,38' and arms=='unguided,gradient' and n==100
    frozen=json.loads((d.docs/'frozen_validation.json').read_text())
    assert frozen['candidate_round']==2 and frozen['program_sha256']==digest(d.cfg/'round02.json')
    p['regional_weight']=9.;write_json(d.cfg/'round02.json',p)
    with pytest.raises(ValueError,match='Frozen candidate changed'):d.freeze(9)


def test_unsupported_designer_feature_rejected_before_inference(tmp_path):
    import gzip
    d=Driver.__new__(Driver);d.root=tmp_path;d.cfg=tmp_path/'configs/experiments/dynamic_contrast10_v1'
    d.docs=tmp_path/'docs/experiments/dynamic_contrast10_20261008';d.cfg.mkdir(parents=True);d.docs.mkdir(parents=True)
    folder=d.docs/'round02_agents';folder.mkdir()
    for role in ['Analyst','Designer']:
        write_json(folder/(role+'.input.json'),{'evidence':1});(folder/(role+'.instructions.md')).write_text('literal skill')
        binding={'input_sha256':digest(folder/(role+'.input.json')),'instruction_sha256':digest(folder/(role+'.instructions.md'))}
        write_json(folder/(role+'.request.json'),binding)
        write_json(folder/(role+'.response.json'),{**binding,'role':role,'decision':'test_formula',
            'reward_view':'endpoint_dynamic_region','selected_feature_indices':[999]})
    write_json(d.cfg/'round01.json',{'window':[.1,.6],'seed':42})
    (d.cfg/'rest_reference.json.gz').write_bytes(gzip.compress(json.dumps({'dynamic_cohort':{'selected_feature_indices':[14]}}).encode()))
    with pytest.raises(ValueError,match='unsupported regional fields'):d.freeze(2)
