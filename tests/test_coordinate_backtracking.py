import copy
import pytest
from evomolsteer.io import digest,read_json
from evomolsteer.generation.prototypes import write_json
from evomolsteer.generation.coordinate_backtracking import freeze


def setup_case(tmp_path):
    ref=tmp_path/'reference.gz';ref.write_bytes(b'frozen reference')
    parent=tmp_path/'parent.json';write_json(parent,{'round':3,'seed':42,'reference_sha256':digest(ref),'window':[.2,.7],
        'reward_view':'coordinate_mixture','native_rms_ratio':.05,'mixture_temperature':.25,'robust_delta':1.,'constraints':{'max_atom_step_A':.025}})
    campaign=tmp_path/'campaign.json';write_json(campaign,{'rounds':[{'round':3,'status':'completed','batches':[0,1],'n_per_arm':100}],
        'rounds_started':1,'rounds_completed':1,'maximum_rounds':30})
    return campaign,parent,ref


def test_rollback_preserves_parent_and_dynamic_window(tmp_path):
    c,p,r=setup_case(tmp_path);before=digest(p)
    q=freeze(c,p,r,tmp_path/'next.json',tmp_path/'plan.json',4,'replay',{},'Replay verified parent','exact_replay')
    assert digest(p)==before and q['window']==[.2,.7]
    functional=copy.deepcopy(q);functional.pop('backtracking');functional['round']=3
    assert functional==read_json(p)
    assert read_json(c)['rounds_started']==1  # freezing is not a launched inference


@pytest.mark.parametrize('changes,kind', [({'native_rms_ratio':.025,'dose_reference':'predictive_flow'},'single_factor'),
    ({'native_rms_ratio':.025},'exact_replay'), ({'native_rms_ratio':.05},'single_factor')])
def test_backtracking_rejects_confounded_or_unchanged_factor(tmp_path,changes,kind):
    c,p,r=setup_case(tmp_path);before=digest(c)
    with pytest.raises(ValueError):freeze(c,p,r,tmp_path/'next.json',tmp_path/'plan.json',4,'bad',changes,'Invalid design',kind)
    assert digest(c)==before and not (tmp_path/'next.json').exists()


def test_reference_and_budget_cannot_be_relaxed_by_rollback(tmp_path):
    c,p,r=setup_case(tmp_path);config=read_json(c);config['maximum_rounds']=3;write_json(c,config)
    with pytest.raises(ValueError):freeze(c,p,r,tmp_path/'next.json',tmp_path/'plan.json',4,'bad',{},'Budget exhausted','exact_replay')
    config['maximum_rounds']=30;write_json(c,config);r.write_bytes(b'changed reference')
    with pytest.raises(ValueError,match='reference/seed'):freeze(c,p,r,tmp_path/'next.json',tmp_path/'plan.json',4,'bad',{},'Changed input','exact_replay')
