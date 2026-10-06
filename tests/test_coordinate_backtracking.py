import copy
import pytest
from evomolsteer.io import digest,read_json
from evomolsteer.generation.prototypes import write_json
from evomolsteer.generation.coordinate_backtracking import freeze,paired_batch_outcomes,freeze_contrast,freeze_shape


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


def test_first_update_cap_restores_parent_later_cap_and_records_effective_default(tmp_path):
    c,p,r=setup_case(tmp_path)
    q=freeze(c,p,r,tmp_path/'next.json',tmp_path/'plan.json',4,'initialcap',
        {'constraints.initial_atom_step_A':.0125},'Isolate initial dose saturation')
    assert q['constraints']=={'max_atom_step_A':.025,'initial_atom_step_A':.0125}
    assert q['backtracking']['before']=={'constraints.initial_atom_step_A':.025}
    assert q['window']==[.2,.7] and q['native_rms_ratio']==.05


@pytest.mark.parametrize('cap',[0.,.025,.026,float('nan')])
def test_initial_cap_cannot_relax_or_silently_replay_default(tmp_path,cap):
    c,p,r=setup_case(tmp_path);before=digest(c)
    with pytest.raises(ValueError):freeze(c,p,r,tmp_path/'next.json',tmp_path/'plan.json',4,'bad',
        {'constraints.initial_atom_step_A':cap},'Invalid initial cap')
    assert digest(c)==before and not (tmp_path/'next.json').exists()


def test_paired_batch_evidence_rejects_missing_or_duplicate_units():
    # Missing/duplicated batches must fail before access to outcome values.
    a={'batch_results':[{'arm':'gradient','batch':0}]}
    n={'batch_results':[{'arm':'unguided','batch':1}]}
    with pytest.raises(ValueError,match='Paired batch'):paired_batch_outcomes(a,n,a,n)
    a['batch_results'].append(a['batch_results'][0].copy())
    with pytest.raises(ValueError,match='Duplicate batch'):paired_batch_outcomes(a,n,a,n)


def test_contrast_freeze_rejects_a_changed_selected_target(tmp_path):
    import gzip,json
    c,p,old=setup_case(tmp_path)
    reference={'window':[.2,.7],'times':[.2], 'features':['r::all::proposal_spread'],'batches':[0],
        'regions':{'r':{'points_A':[[0,0,0]]}},'channel':'all','sources':[{'sha256':'original'}],
        'spatial_anchor':'endpoint','control_representation':'proposal','required_input_sha256':{},
        'frames':[{'time':.2,'modes':[{'source_batch':0,'center_A':[1.],'covariance_A2':[[1.]]}]}]}
    old.write_bytes(gzip.compress(json.dumps(reference).encode(),mtime=0))
    parent=read_json(p);parent['reference_sha256']=digest(old);write_json(p,parent)
    new=tmp_path/'new.gz';changed=copy.deepcopy(reference);changed['frames'][0]['modes'][0]['center_A']=[2.]
    new.write_bytes(gzip.compress(json.dumps(changed).encode(),mtime=0))
    before=digest(c)
    with pytest.raises(ValueError,match='Selected moment parity'):
        freeze_contrast(c,p,old,new,tmp_path/'next.json',tmp_path/'plan.json',4,'contrast','Cannot silently change selected targets')
    assert digest(c)==before and not (tmp_path/'next.json').exists()


def shape_case(tmp_path):
    import gzip,json,numpy as np
    c,p,old=setup_case(tmp_path)
    shared={'window':[.2,.7],'times':[.2,.7],'batches':[0],'regions':{'r':{'points_A':[[0,0,0]]}},
        'channel':'all','sources':[{'path':'single/batch_000/trajectory.h5','sha256':'source'}],
        'required_input_sha256':{},'spatial_anchor':'endpoint','control_representation':'proposal','spatial_width_A':4.}
    shared['frames']=[{'time':t} for t in shared['times']]
    old.write_bytes(gzip.compress(json.dumps(shared).encode(),mtime=0))
    parent=read_json(p);parent['reference_sha256']=digest(old);write_json(p,parent)
    ref={**shared,'schema_version':'regional-shape-mixture-1.0','feature_unit':'A^2','feature_scale_A2':[1.]*6,
        'frames':[{'time':t,'modes':[{'source_batch':0,'center_scaled':[0.]*6,'covariance_dimensionless':np.eye(6).tolist()}]} for t in shared['times']]}
    new=tmp_path/'shape.gz';new.write_bytes(gzip.compress(json.dumps(ref).encode(),mtime=0))
    contract=tmp_path/'designer.json';write_json(contract,{'schema_version':'shape-designer-contract-1.0','agent':'Designer',
        'parent_reference_sha256':digest(old),'reference_sha256':digest(new),'availability_change_disclosed':True,
        'inherit_without_simultaneous_retuning':{**{k:parent[k] for k in ('native_rms_ratio','mixture_temperature','robust_delta','constraints')},
            'dose_reference':'observed_native','initial_update_dose':'native','preserve_native_rigid_pose':False}})
    return c,p,old,new,contract


def test_shape_ablation_requires_bound_parent_controls_and_records_availability(tmp_path):
    c,p,old,new,contract=shape_case(tmp_path);before=digest(p)
    q=freeze_shape(c,p,old,new,tmp_path/'next.json',tmp_path/'plan.json',4,'shape','Representation test',contract)
    assert digest(p)==before and q['reward_view']=='shape_mixture' and q['window']==[.2,.7]
    assert q['constraints']==read_json(p)['constraints'] and q['backtracking']['kind']=='representation_ablation'
    assert 'single NOS' in read_json(tmp_path/'plan.json')['availability_change']
    assert read_json(c)['rounds'][-1]['arms']==['unguided','gradient_zero','gradient']


def test_shape_ablation_rejects_simultaneous_dose_retuning_without_mutation(tmp_path):
    c,p,old,new,contract=shape_case(tmp_path);before=digest(c)
    d=read_json(contract);d['inherit_without_simultaneous_retuning']['native_rms_ratio']=.5;write_json(contract,d)
    with pytest.raises(ValueError,match='controls'):
        freeze_shape(c,p,old,new,tmp_path/'next.json',tmp_path/'plan.json',4,'shape','Invalid design',contract)
    assert digest(c)==before and not (tmp_path/'next.json').exists()
