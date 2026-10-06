import pytest
from evomolsteer.continuous.coordinate_agents import validate,DESIGNER


def test_design_contract_prohibits_unprovided_view_window_and_evidence():
    bundle={'window':[.1,.4],'evidence':[{'evidence_id':'e','feature':'r::all::spread'}],
            'features':{'r::all::spread':{'region':'r'}}}
    d={'schema_version':'coordinate-1.0','agent':'Designer','design_status':'exploratory','architecture':'spread_upper',
       'regions':['r'],'channel':'all','window':[.1,.4],'native_rms_ratio':.2,'mixture_temperature':.25,'robust_delta':1.,
       'rationale':'Association hypothesis','evidence_ids':['e'],'limitations':['Not causal']}
    assert validate(d,DESIGNER,bundle)
    assert validate({**d,'dose_reference':'predictive_flow','preserve_native_rigid_pose':False},DESIGNER,bundle)
    assert validate({**d,'initial_update_dose':'cap_to_flow'},DESIGNER,bundle)
    with pytest.raises(ValueError):validate({**d,'dose_reference':'predictive_flow','initial_update_dose':'cap_to_flow'},DESIGNER,bundle)
    for field,value in [('window',[0.,.5]),('regions',['unmeasured']),('evidence_ids',['unknown'])]:
        with pytest.raises(ValueError):validate({**d,field:value},DESIGNER,bundle)
    bundle.update(control_representation='proposal',features={'r::all::proposal_spread':{'region':'r'}})
    assert validate(d,DESIGNER,bundle)
    bundle['features']={'r::all::spread':{'region':'r'}}
    with pytest.raises(ValueError):validate(d,DESIGNER,bundle)


def test_supplementary_evidence_rejects_other_source_and_influence(tmp_path):
    import pandas as pd
    from evomolsteer.io import write_json
    from evomolsteer.continuous.coordinate_agents import export
    main=tmp_path/'main';other=tmp_path/'other';main.mkdir();other.mkdir()
    manifest={'window':[0.,.1],'times':[0.,.1],'splits':{'discovery':[0,1]},'sources':[{'sha256':'same'}],
        'spatial_anchor':'endpoint','control_representation':'proposal','interpretation':[]}
    write_json(main/'manifest.json',manifest)
    catalog={'features':{'r::all::proposal_spread':{'region':'r'}}}
    for p in (main,other):write_json(p/'feature_catalog.json',catalog)
    pd.DataFrame([{'split':'discovery','feature':'r::all::proposal_spread','metric':'selection_shift','q':1.}]).to_csv(main/'whole_window_evidence.csv',index=False)
    pd.DataFrame([{'batch':b,'time':.1,'unique_roots':1} for b in (0,1)]).to_csv(main/'lineage_diagnostics.csv',index=False)
    write_json(main/'continuous_functions.json',{})
    write_json(other/'manifest.json',{**manifest,'sources':[{'sha256':'different'}]})
    with pytest.raises(ValueError,match='support mismatch'):export(main,'Analyst',landmarks=('r',),transport=other)
    write_json(other/'manifest.json',{'window':[0.,.1],'source_manifest_sha256':'different'})
    with pytest.raises(ValueError,match='source/window mismatch'):export(main,'Analyst',landmarks=('r',),influence=other)


def test_contrast_requires_bound_and_every_observable():
    bundle={'window':[.2,.6],'evidence':[{'evidence_id':'e','feature':'r::all::proposal_spread'}],
        'control_representation':'proposal','features':{f'r::all::proposal_{k}':{'region':'r'} for k in ('centroid_x','centroid_y','centroid_z','spread')}}
    d={'schema_version':'coordinate-1.0','agent':'Designer','design_status':'exploratory','architecture':'selection_contrast',
       'regions':['r'],'channel':'all','window':[.2,.6],'native_rms_ratio':.05,'mixture_temperature':.25,'robust_delta':1.,
       'rationale':'Incremental preference hypothesis','evidence_ids':['e'],'limitations':['Background has previous SMC history']}
    with pytest.raises(ValueError,match='Contrast bound'):validate(d,DESIGNER,bundle)
    assert validate({**d,'contrast_bound_nats':1.},DESIGNER,bundle)
    bundle['features'].pop('r::all::proposal_centroid_z')
    with pytest.raises(ValueError,match='Missing controlled'):validate({**d,'contrast_bound_nats':1.},DESIGNER,bundle)
