import numpy as np
import pandas as pd
import pytest
from evomolsteer.continuous.lineage import window_copy_counts
from evomolsteer.continuous.functional import global_fit,evaluate_frozen,quadrature
from evomolsteer.chemical_features import extend_catalog,measure_chemistry
from evomolsteer.io import write_json,write_table,read_json


def test_window_ancestry_counts_copies_not_binary_survival_or_t1():
    groups=[pd.DataFrame({'node_id':['a','b','c'],'parent_node_id':['','',''],'offspring_count':[2,1,0]}),
        pd.DataFrame({'node_id':['d','e','f'],'parent_node_id':['a','a','b'],'offspring_count':[0,1,2]}),
        pd.DataFrame({'node_id':['g','h','i'],'parent_node_id':['e','f','f'],'offspring_count':[0,3,0]})]
    copies=window_copy_counts(groups)
    assert [x.tolist() for x in copies]==[[0,3,0],[0,0,3],[0,3,0]]
    assert all(x.sum()==3 for x in copies)


def test_continuous_fit_recovers_function_derivatives_and_reversal():
    t=np.linspace(0,.5,51); curve=(t-.25)+2*(t-.25)**2
    cfg={'max_degree':5,'seed':42,'permutations':128}
    values=np.repeat(curve[None,:],8,axis=0)
    fit=global_fit(t,values,cfg)
    assert fit['degree']==2
    np.testing.assert_allclose(fit['curves']['fitted'],curve,atol=1e-14)
    np.testing.assert_allclose(fit['curves']['derivative'],1+4*(t-.25),atol=1e-13)
    np.testing.assert_allclose(evaluate_frozen(fit,t),curve,atol=1e-14)
    # A zero-integral sign-changing preference must remain globally detectable.
    reversal=global_fit(t,np.repeat((t-.25)[None,:],8,axis=0),cfg)
    assert abs(reversal['window_mean_effect'])<1e-14
    assert reversal['global_curve_p']<.01
    with pytest.raises(ValueError):evaluate_frozen(fit,[-.01,.51])


def test_neutral_curves_stay_neutral_and_nonuniform_grid_uses_time_weights():
    cfg={'max_degree':3,'seed':42,'permutations':128}
    t=np.array([0,.01,.1,.3,.5]);y=np.zeros((5,len(t)))
    f=global_fit(t,y,cfg)
    assert f['degree']==0 and f['global_curve_p']==1
    np.testing.assert_allclose(f['curves']['derivative'],0)
    assert quadrature(t)@t==pytest.approx(.25)
    with pytest.raises(ValueError):global_fit(t,np.full_like(y,np.nan),cfg)


def test_observed_rates_pair_batches_and_do_not_turn_missingness_into_motion():
    from evomolsteer.continuous.summary import point_curve
    y=np.array([[0.,0.],[100.,np.nan],[10.,10.]])
    result=point_curve({},'x',np.array([0.,.5]),y)
    assert result['mean'].iloc[0]!=result['mean'].iloc[1]
    assert result.observed_derivative.iloc[1]==0
    assert result.derivative_n_paired_batches.iloc[1]==2


def test_chemistry_observables_preserve_mask_units_translation_and_slot_permutation():
    vocab={'<PAD>':0,'C':1,'N':2,'O':3,'S':4,'F':5,'Cl':6,'Br':7,'I':8}
    base={'features':{},'atom_vocabulary':vocab,'regions':{'r':{'points_A':[[0.,0.,0.]]}},
          'contact_midpoint_A':4.5,'contact_width_A':.5}
    cat=extend_catalog(base,{'<PAD>':0,'0':1,'1':2},
        {'bond_order_by_label':{'0':0.,'1':1.,'2':2.,'3':3.,'4':1.5}})
    x=np.array([[[1.,0.,0.],[2.,0.,0.],[99.,99.,99.]]]); mask=np.array([[True,True,False]])
    atoms=np.array([[1,2,0]]);q=np.array([[1,2,0]])
    b=np.array([[[0,2,0],[2,0,0],[0,0,0]]])
    v=measure_chemistry(x,atoms,mask,b,q,cat)
    assert v['ligand::bond_length_mean'][0]==pytest.approx(1)
    assert v['ligand::bond_order_mean'][0]==2
    assert v['ligand::formal_charge_mean'][0]==.5
    assert v['r::steric_overlap_proxy'][0]==.5 # A^2, not energy
    idx=[1,0,2]
    perm=measure_chemistry(x[:,idx],atoms[:,idx],mask[:,idx],b[:,idx][:,:,idx],q[:,idx],cat)
    for key in v:np.testing.assert_allclose(v[key],perm[key],equal_nan=True)
    cat['regions']['r']['points_A']=[[10.,10.,10.]]
    moved=measure_chemistry(x+10,atoms,mask,b,q,cat)
    for key in v:np.testing.assert_allclose(v[key],moved[key],equal_nan=True)
    assert cat['energy_status']['physical_energy_available'] is False


def test_entire_window_events_have_no_bins_and_exact_selection_transmission_identity(tmp_path):
    from evomolsteer.continuous.events import build_events
    from evomolsteer.continuous.summary import summarize_method
    scope={'analysis_scope':'actual_resampling_events','steps':[0,1,2],'score_times':[0.,.25,.5],
        'stage_edges':[0.,.5],'window_start':0.,'window_end':.5,'selection_arms':['single'],'background_arm':'unguided'}
    cfg={'analysis_scope':'actual_resampling_events','time_analysis':'continuous_window',
        'analysis_representations':['predicted_endpoint'],'minimum_feature_fraction':.8,
        'minimum_inference_batches':4,'seed':42,'permutations':128,'max_degree':1}
    rows=[]
    for arm in ['single','unguided']:
        for batch in range(4):
            for step,t in enumerate(scope['score_times']):
                for slot in range(4):
                    rows.append(dict(node_id=f'{arm}:{batch}:{step}:{slot}',
                        parent_node_id=f'{arm}:{batch}:{step-1}:{slot}' if step else '',
                        root_id=str(slot),representation='predicted_endpoint',arm=arm,batch=batch,split='discovery',
                        step=step,slot=slot,stage=0,score_time=t,geometry_time=t,dt=.25,
                        resampled=arm=='single',selected=True,probability=.25,offspring_count=1,
                        pic50_on=7.,x=slot+100*t))
    write_json(tmp_path/'selection_scope.json',scope);write_json(tmp_path/'config.json',cfg)
    write_json(tmp_path/'feature_catalog.json',{'features':{'x':{}}});write_table(tmp_path/'features.parquet',rows)
    build_events(tmp_path)
    dest=tmp_path/'discovery/continuous'
    trends=pd.read_parquet(dest/'trends/batch_event_curves.parquet')
    assert 'stage' not in trends and set(trends.time)=={0,.25,.5}
    np.testing.assert_allclose(trends.expected_selection_shift,0)
    enrich=pd.read_parquet(dest/'enrichment/batch_event_curves.parquet')
    np.testing.assert_allclose(enrich.expected_low_mass_shift,0)
    dyn=pd.read_parquet(dest/'dynamics/batch_event_curves.parquet')
    np.testing.assert_allclose(dyn.dropna().population_change_rate,100)
    np.testing.assert_allclose(dyn.dropna().selection_component_rate+dyn.dropna().transmission_component_rate,100)
    summarize_method(tmp_path,'trends')
    funcs=read_json(dest/'trends/functions.json')
    model=next(v for v in funcs.values() if v['contrast']=='expected_selection_shift')
    assert model['degree']==0 and model['global_curve_p']==1
