import copy
import numpy as np
import pytest
import torch
from evomolsteer.partial_search.mining import regional_sets,residual_ranks,correlation,cluster_interval
from evomolsteer.partial_search.agents import compile_programs,signature,execution_contract,analyze
from evomolsteer.io import read_json,write_json
from evomolsteer.final_guidance.workflow import CONTROL_SEMANTICS
from evomolsteer.final_guidance.reward import HalfStateReward


def example():
    teacher={'candidate_id':'F1','time':.5,'representation':'current_world_A','active_slots':[0,1,2],
        'coords_A':[[0.,0.,0.],[2.,0.,0.],[0.,2.,0.]]}
    analyst={'candidate_id':'F1','region_ids':['P001'],'atom_slots':[0,1],
        'point_importance':[{'slot':0,'relative_weight':1.},{'slot':1,'relative_weight':2.}]}
    dose={'pointcloud_delta_A':1.,'native_rms_ratio':.33,'max_atom_step_A':.15,'max_cumulative_rms_A':6.}
    semantics=copy.deepcopy(CONTROL_SEMANTICS);semantics['teacher_weights']='analyst_importance_normalized'
    design={'candidate_id':'F1','decision':'activate','region_ids':['P001'],'atom_slots':[0,1],
        'atom_weights':[1/3,2/3],'teacher_time':.5,'dose':dose,'control_semantics':semantics}
    cfg={'round_id':'R01','teacher_time':.5,'guidance_end':.5,'guidance_starts':[0.,.1,.3],'steps':100,'dose':dose}
    return teacher,analyst,design,cfg


def test_region_catalog_covers_all_points_with_proper_unique_subsets():
    x=np.random.default_rng(42).normal(size=(25,3));regions=regional_sets(x,[4,8,12,16,20,24])
    assert {s for r in regions for s in r['atom_slots']}==set(range(25))
    assert len({tuple(r['atom_slots']) for r in regions})==len(regions)
    assert all(0<len(r['atom_slots'])<25 for r in regions)


def test_incremental_analysis_removes_common_global_similarity():
    rng=np.random.default_rng(42);g=rng.normal(size=240);signal=rng.normal(size=240);batch=np.repeat(np.arange(8),30)
    x=g+signal;y=g+signal+rng.normal(scale=.1,size=240)
    rx=residual_ranks(x,g,batch);ry=residual_ranks(y,g,batch)
    assert correlation(rx,ry)>.8 and abs(np.dot(rx,residual_ranks(g,np.zeros_like(g),batch)))<1e-7
    assert np.max(np.abs(residual_ranks(g,g,batch)))<1e-8


def test_root_cluster_intervals_are_repeatable_and_not_clone_row_bootstrap():
    x=np.arange(20,dtype=float);y=x+.5*np.sin(x);groups=np.repeat(np.arange(5),4)
    assert cluster_interval(x,y,groups)==cluster_interval(x,y,groups)
    low,high=cluster_interval(x,y,groups);assert .95<low<=high<=1


def test_designer_cannot_replace_analyst_region_or_weight():
    teacher,analyst,design,cfg=example();programs=compile_programs(cfg,analyst,design,teacher,'a','d','t')
    assert len(programs)==3 and [len(p['expected_active_steps']) for p in programs]==[50,40,20]
    changed=copy.deepcopy(design);changed['atom_slots']=[0,2]
    with pytest.raises(ValueError,match='detached'):compile_programs(cfg,analyst,changed,teacher,'a','d','t')
    changed=copy.deepcopy(design);changed['atom_weights']=[.5,.5]
    with pytest.raises(ValueError,match='importance'):compile_programs(cfg,analyst,changed,teacher,'a','d','t')


def test_weighted_partial_reward_derivative_matches_fixed_assignment_difference():
    teacher,analyst,design,cfg=example();p=compile_programs(cfg,analyst,design,teacher,'a','d','t')[0];reward=HalfStateReward(p,teacher)
    z=torch.tensor([[[.1,.1,0.],[2.2,.2,0.],[0.,3.,0.]]],dtype=torch.double,requires_grad=True);mask=torch.ones((1,3),dtype=torch.bool)
    value,detail=reward.value(z,mask);g=torch.autograd.grad(value.sum(),z)[0];d=torch.ones_like(z);epsilon=1e-5
    fd=(reward.value(z.detach()+epsilon*d,mask,detail['assignment'])[0]-reward.value(z.detach()-epsilon*d,mask,detail['assignment'])[0])/(2*epsilon)
    assert torch.allclose(fd,(g*d).sum((1,2)),atol=1e-8) and g[0,2].abs().max()==0


def test_numerical_signature_ignores_narrative_but_tracks_executed_weights():
    teacher,analyst,design,cfg=example();p=compile_programs(cfg,analyst,design,teacher,'a','d','t')[0]
    other=copy.deepcopy(p);other['program_id']='R99';other['analyst_sha256']='different'
    assert signature(p)==signature(other)
    other['atom_weights']=[.5,.5];assert signature(p)!=signature(other)


def test_designer_cannot_describe_inactive_view_or_mixture():
    teacher,analyst,design,cfg=example();design['endpoint_mix']=0.
    design['execution_contract']=execution_contract(cfg)
    assert design['execution_contract']['coordinate_views']==['actual_half']
    reference=compile_programs(cfg,analyst,design,teacher,'a','d','t')
    changed=copy.deepcopy(design);changed['endpoint_mix']=.5
    with pytest.raises(ValueError,match='feature-transfer'):compile_programs(cfg,analyst,changed,teacher,'a','d','t')
    changed=copy.deepcopy(design);changed['execution_contract']['coordinate_views'].append('actual_final')
    with pytest.raises(ValueError,match='inactive'):compile_programs(cfg,analyst,changed,teacher,'a','d','t')
    legacy=copy.deepcopy(design);legacy.pop('endpoint_mix');legacy.pop('execution_contract')
    assert [signature(p) for p in reference]==[signature(p) for p in compile_programs(cfg,analyst,legacy,teacher,'a','d','t')]


def test_role_handoff_is_canonical_and_does_not_mutate_raw_api_answer(tmp_path):
    teacher,_,design,cfg=example()
    cfg.update(study='round',campaign_study='campaign',base_study='base',candidate_id='F1',
        analyst_skill='a.md',designer_skill='d.md',quantitative_decision=True)
    for name in ('a.md','d.md'):(tmp_path/name).write_text('Frozen test instructions',encoding='utf-8')
    regions=[{'region_id':key,'atom_slots':slots,'evidence_id':key+'/geometry',
        'incremental_rank_correlation':.2,'nearest_rest_pic50_mean':8.,'incremental_cluster95':[-.1,.3],
        'free_teacher_slots':1} for key,slots in [('P001',[0,1]),('P002',[1,2])]]
    write_json(tmp_path/'campaign/evidence.json',{'candidate':{'candidate_id':'F1'},'regions':regions,
        'terminal_coords_A':teacher['coords_A']})
    write_json(tmp_path/'base/candidates/F1/teacher.json',teacher)
    raw={'role':'Analyst','candidate_id':'F1','region_ids':['P001'],
        'hypotheses':[{'region_id':'P001','evidence_id':'P001/geometry','advantage':'spatial hypothesis',
            'alternative':'shared geometry','falsifier':'no paired terminal improvement'}],
        'quantitative_comparison':{'chosen_region':'P001','alternative_region':'P002','tradeoff_reason':'test'},
        'selection_rationale':'test','free_region_rationale':'one free point','expected_final_response':'test'}
    design.update(role='Designer',atom_weights=[.5,.5],scientific_rationale='Bound region',
        observable_falsifier='No paired improvement',endpoint_mix=0.,execution_contract=execution_contract(cfg))
    requests=[]
    def caller(messages,schema,output,**kwargs):
        answer=copy.deepcopy(raw if schema['properties']['role']['const']=='Analyst' else design)
        # Real API response key order need not equal sorted order when cached JSON is read.
        if len(requests)>=2:answer=dict(reversed(list(answer.items())))
        before=copy.deepcopy(answer);kwargs['response_validator'](answer)
        assert answer==before
        requests.append(messages[1]['content']);write_json(output,answer);return answer
    first=analyze(tmp_path,cfg,{},caller=caller);second=analyze(tmp_path,cfg,{},caller=caller)
    assert requests[:2]==requests[2:]
    assert [signature(p) for p in first]==[signature(p) for p in second]
    folder=tmp_path/'round/candidates/F1'
    assert 'alternative_catalogue_eligible' not in read_json(folder/'Analyst.selection.json')['quantitative_comparison']
    assert read_json(folder/'Analyst.json')['quantitative_comparison']['alternative_catalogue_eligible'] is True


def test_completed_campaign_does_not_connect_or_overwrite_active_skills(tmp_path,monkeypatch):
    from evomolsteer.partial_search import engine
    master={'campaign_study':'campaign','max_rounds':20}
    state={'next_round':21,'best_round':'R13','rounds':[{}]*20}
    write_json(tmp_path/'config.json',master);write_json(tmp_path/'campaign/state.json',state)
    write_json(tmp_path/'campaign/COMPLETE.json',{'rounds':20})
    def denied():raise AssertionError('Completed campaign must not reconnect')
    monkeypatch.setattr(engine,'Remote',denied)
    assert engine.run(tmp_path,tmp_path/'config.json',20)==state
