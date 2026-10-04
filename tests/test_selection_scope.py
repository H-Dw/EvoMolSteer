import copy
import numpy as np
import pandas as pd
import pytest
import torch

from evomolsteer.scope import SCOPE,stage_edges,stage_for,discover_scope,validate_frame
from evomolsteer.statistics import selection_moments,save_metrics,summarize_clusters
from evomolsteer.evidence import target_distribution,weighted_quantile
from evomolsteer.contracts import validate
from evomolsteer.reward import RewardProgram
from evomolsteer.io import write_json,write_table,read_json
from test_scientific_invariants import fixture


def test_last_real_selection_event_is_inclusive_not_a_late_bin():
    times=np.linspace(0,1,101,dtype=np.float32)[:51]
    edges=stage_edges(times,.1)
    assert edges==[0,.1,.2,.3,.4,.5]
    assert np.bincount(stage_for(times,edges)).tolist()==[10,10,10,10,11]
    with pytest.raises(ValueError):stage_for([.51],edges)


def test_scope_is_discovered_from_flags_not_assumed_half_window(tmp_path):
    cfg={'selection_arms':['joint'],'stage_width':.1}
    cfg['background_arm']='unguided'
    p=tmp_path/'joint/batch_000';p.mkdir(parents=True)
    (p/'COMPLETE.json').write_text('{}')
    t=np.arange(100)/100
    np.savez(p/'trajectory.npz',resampled=(t>=.1)&(t<=.3),score_time=np.repeat(t[:,None],3,axis=1))
    bg=tmp_path/'unguided/batch_000';bg.mkdir(parents=True)
    (bg/'COMPLETE.json').write_text('{}')
    np.savez(bg/'trajectory.npz',resampled=np.zeros(100,bool),score_time=np.repeat(t[:,None],3,axis=1))
    scope=discover_scope(tmp_path,cfg)
    assert scope['steps']==list(range(10,31))
    assert (scope['window_start'],scope['window_end'])==(.1,.3)


def test_background_kept_but_postwindow_and_outcomes_rejected():
    scope={'steps':[0,1],'score_times':[0.,.1],'stage_edges':[0.,.1],
           'selection_arms':['joint'],'background_arm':'unguided'}
    d=pd.DataFrame({'step':[0,1,0,1],'score_time':[0,.1,0,.1],'stage':[0]*4,
                    'arm':['joint','joint','unguided','unguided'],'resampled':[True,True,False,False]})
    validate_frame(d,scope)
    for changed in [d.assign(step=[0,2,0,1]),d.assign(terminal_descendants=0),d.assign(resampled=True)]:
        with pytest.raises(ValueError):validate_frame(changed,scope)


def test_preference_and_multinomial_noise_are_distinct():
    x=np.array([[0.],[10.]])
    m=selection_moments(x,[.8,.2],[0,2])
    assert m['expected_selection_shift'][0]==pytest.approx(-3)
    assert m['realized_selection_shift'][0]==pytest.approx(5)
    assert m['selection_noise_shift'][0]==pytest.approx(8)
    # A randomly selected high value does not reverse the expected preference.
    uniform=selection_moments(x,[.5,.5],[1,1])
    assert uniform['expected_selection_shift'][0]==0
    missing=selection_moments(np.array([[np.nan],[10.]]),[.8,.2],[0,2],.8)
    assert np.isnan(missing['expected_selection_shift'][0])


def test_event_matching_avoids_time_composition_bias_and_corrects_rates(tmp_path):
    cfg={'minimum_inference_batches':4,'seed':42,'permutations':128}
    # Each step has exactly zero effect, although its raw background drifts rapidly.
    events=pd.DataFrame([{'representation':'predicted_endpoint','arm':'joint','batch':b,
        'step':i,'stage':int(i>=2),'time':t,'dt':.01,'feature':'x','effect_a':0.}
        for b in range(4) for i,t in enumerate([.40,.41,.49,.50])])
    save_metrics(tmp_path,events,['effect_a'],cfg)
    summary=pd.read_csv(tmp_path/'effects.csv')
    assert (summary.effect==0).all()
    rates=pd.read_csv(tmp_path/'batch_effect_rates.csv')
    np.testing.assert_allclose(rates.delta_time,.495-.405)
    assert (rates.rate==0).all()


def test_cluster_results_ignore_input_row_order():
    cfg={'minimum_inference_batches':4,'seed':42,'permutations':128}
    d=pd.DataFrame({'batch':list(range(8))*2,'feature':['a']*8+['b']*8,
                    'value':[1,2,3,4,3,2,1,2]+[-1,2,-3,4,-3,2,-1,2]})
    a=summarize_clusters(d,['feature'],'value',cfg)
    b=summarize_clusters(d.sample(frac=1,random_state=4),['feature'],'value',cfg)
    pd.testing.assert_frame_equal(a,b)


def test_probability_prototypes_use_equal_event_weights_and_no_outcomes():
    np.testing.assert_allclose(weighted_quantile([0,10],[.6,.4],[.25,.5,.75]),
                               weighted_quantile([0,0,10],[.3,.3,.4],[.25,.5,.75]))
    d=pd.DataFrame({'batch':[0,0,0,0,1,1],'step':[0,0,1,1,0,0],
                    'root_id':['a','b','a','b','c','d'],'x':[0,10,0,10,0,10],
                    'probability':[.9,.1,.9,.1,.9,.1]})
    r=target_distribution(d,'x',{'minimum_feature_fraction':.8})
    assert r['support_batches']==2 and r['support_events']==3
    assert r['weighted_median']<r['uniform_median']
    repeated=pd.concat([d,d[d.batch==0].assign(step=lambda g:g.step+2)],ignore_index=True)
    rr=target_distribution(repeated,'x',{'minimum_feature_fraction':.8})
    # Repeating equivalent events of only one batch must not change its batch weight.
    assert rr['weighted_median']==pytest.approx(r['weighted_median'])


def test_reward_has_exact_zero_gradient_after_selection_window():
    p,c=fixture();p['selection_window']['end']=.5;p['terms'][0]['stage_end']=.5
    r=RewardProgram(p,c)
    x=torch.tensor([[[5.,.5,0.],[6.,1.,0.]]],dtype=torch.float64,requires_grad=True)
    atom=torch.tensor([[4,3]]);mask=torch.ones((1,2),dtype=torch.bool)
    reward,_=r(x,atom,mask,.51)
    grad=torch.autograd.grad(reward.sum(),x)[0]
    assert reward.item()==0 and torch.count_nonzero(grad)==0
    assert r(x,atom,mask,.5)[0].item()<0 # inclusive last event
    earlier=copy.deepcopy(p);earlier['terms'][0]['stage_start']=.2;earlier['terms'][0]['stage_end']=.3
    earlier_runtime=RewardProgram(earlier,c)
    assert earlier_runtime(x,atom,mask,.199999988)[0].item()<0
    assert earlier_runtime(x,atom,mask,.299999982)[0].item()==0
    bad=copy.deepcopy(p);bad['terms'][0]['stage_end']=.75
    with pytest.raises(ValueError):RewardProgram(bad,c)


def test_agent_cannot_reuse_legacy_bundle_or_invent_late_stage():
    scope={'analysis_scope':SCOPE,'window_start':0.,'window_end':.5,
           'stages':[{'stage':0,'stage_start':0.,'stage_end':.5}]}
    bundle={'schema_version':'2.0','dataset_id':'test','scope':scope,'targets':[],
        'feature_catalog':{'features':{'f':{'region':'r'}}},
        'evidence':[{'evidence_id':'e','feature':'f','representation':'predicted_endpoint','stage':0}]}
    data={'schema_version':'2.0','agent':'Analyst','dataset_id':'test','observations':[],
        'selection_comparison':'selected/rejected','stage_dynamics':'window only','limitations':[],
        'rules':[{'rule_id':'r','region':'r','feature':'f','representation':'predicted_endpoint',
            'stage_start':0.,'stage_end':.5,'direction':'defer','target_id':None,'status':'deferred',
            'evidence_ids':['e'],'counterevidence_ids':[],'rationale':'association'}]}
    validate('Analyst',data,bundle)
    bad=copy.deepcopy(data);bad['rules'][0]['stage_end']=.75
    with pytest.raises(ValueError):validate('Analyst',bad,bundle)
    with pytest.raises(ValueError):validate('Analyst',data,{**bundle,'schema_version':'1.0'})


def test_four_methods_share_event_scope_and_same_time_background(tmp_path):
    from evomolsteer import enrichment,differential,trends,pca
    cfg={'analysis_scope':SCOPE,'analysis_representations':['predicted_endpoint'],
         'minimum_feature_fraction':.8,'minimum_inference_batches':4,'seed':42,
         'permutations':128,'pca_components':1}
    scope={'analysis_scope':SCOPE,'steps':[0,1,2],'score_times':[0.,.1,.2],
           'stage_edges':[0.,.1,.2],'window_start':0.,'window_end':.2,
           'selection_arms':['joint'],'background_arm':'unguided'}
    rows,edges=[],[]
    for arm in ['joint','unguided']:
        for batch in range(4):
            node=lambda t,s:f'{arm}:{batch}:{t}:{s}'
            for step,time in enumerate(scope['score_times']):
                for slot in range(4):
                    parent=0 if arm=='joint' else slot
                    rows.append({'node_id':node(step,slot),'parent_node_id':node(step-1,parent) if step else '',
                        'representation':'predicted_endpoint','arm':arm,'batch':batch,'split':'discovery',
                        'step':step,'stage':0 if step==0 else 1,'slot':slot,'score_time':time,
                        'geometry_time':time,'dt':.1,'resampled':arm=='joint',
                        'selected':slot==0 if arm=='joint' else True,
                        'offspring_count':(4 if slot==0 else 0) if arm=='joint' else 1,
                        'probability':([.7,.1,.1,.1][slot] if arm=='joint' else .25),
                        'weight_on':[.7,.1,.1,.1][slot],'weight_off':[.7,.1,.1,.1][slot],
                        'x':100*step+slot})
                    if step:
                        edges.append({'source':node(step-1,parent),'target':node(step,slot),
                                      'arm':arm,'batch':batch,'step':step-1,'stage':0 if step==1 else 1,'dt':.1})
    write_json(tmp_path/'config.json',cfg);write_json(tmp_path/'selection_scope.json',scope)
    write_json(tmp_path/'feature_catalog.json',{'features':{'x':{}}})
    write_table(tmp_path/'features.parquet',rows);write_table(tmp_path/'edges.parquet',edges)
    for module in [enrichment,differential,trends,pca]:module.run(tmp_path)
    threshold=pd.read_csv(tmp_path/'discovery/enrichment/thresholds.csv')
    np.testing.assert_allclose(threshold.threshold,[2.25,102.25,202.25])
    event=pd.read_parquet(tmp_path/'discovery/enrichment/events.parquet')
    np.testing.assert_allclose(event.expected_high_mass_shift,-.15)
    basis=read_json(tmp_path/'discovery/pca/basis.json')['predicted_endpoint']
    assert basis['n_fit']==48 and basis['fit_window']==[0,.2]
