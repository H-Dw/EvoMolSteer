import pytest
from agenticgen.uniguide.io import write, sha
from agenticgen.uniguide.fixed_scope import validate_fixed_call, validate_fixed_reference


def fixture(tmp_path):
    old=tmp_path/'baseline';source=old/'calibration/frozen/population_50'
    g={'attempts':50,'effective_guidance_scale_schedule':[70]*4}
    write(source/'aggregate.json',g)
    write(source/'COMPLETE.json',{'files':{'aggregate.json':sha(source/'aggregate.json')}})
    domain={'condition_sha256':'same','evidence_revision':0,'initial_source':'calibration/frozen/population_50',
            'calibrated_guidance_reference_scale':70}
    write(old/'calibration/domains.json',{'teacher':domain})
    cfg={'experiment_mode':'fixed_gradient_atom_scope','rounds':0,
         'guidance_atom_scope':{'mode':'condition_hops','hops':0},'windows':[[10,25],[26,50],[51,75],[76,99]],
         'fixed_guidance_schedule':[70]*4,'fixed_scale_reference':{'baseline_root':str(old),
             'domains_file_sha256':sha(old/'calibration/domains.json'),'source_complete_sha256':sha(source/'COMPLETE.json')}}
    return cfg,{'target':'teacher'},{'condition_sha256':'same'}


def test_precalibration_values_are_parameters_not_imported_measurements(tmp_path):
    cfg,t,c=fixture(tmp_path)
    r=validate_fixed_reference(cfg,t,c)
    assert not r['old_measurements_imported'] and not r['new_calibration_performed']
    validate_fixed_call(cfg,t,c,[70]*4,'formal')


@pytest.mark.parametrize('change',[{'rounds':1},{'rounds':False},{'fixed_guidance_schedule':[80]*4},
                                  {'prefix_inpainting':{'enabled':True}},
                                  {'experiment_mode':'other'}])
def test_implicit_search_or_other_interventions_cannot_enter(tmp_path,change):
    cfg,t,c=fixture(tmp_path);cfg.update(change)
    with pytest.raises(ValueError):validate_fixed_reference(cfg,t,c)


def test_reference_change_and_unrequested_schedule_fail(tmp_path):
    cfg,t,c=fixture(tmp_path)
    with pytest.raises(ValueError):validate_fixed_call(cfg,t,c,[90]*4,'formal')
    write(tmp_path/'baseline/calibration/domains.json',{})
    with pytest.raises(ValueError):validate_fixed_reference(cfg,t,c)


def test_agent_entrypoint_is_blocked_before_model_or_provider(tmp_path):
    from agenticgen.uniguide.absolute_runtime import run_version
    with pytest.raises(ValueError,match='cannot invoke'):
        run_version({'experiment_mode':'fixed_gradient_atom_scope'},'GD01')


@pytest.mark.parametrize('hops',[0,1,2])
def test_saved_trace_recomputes_scope_and_keeps_native_motion(tmp_path,hops):
    import h5py
    import numpy as np
    from agenticgen.uniguide.fixed_scope import verify_trace
    path=tmp_path/'trace.h5';mask=np.zeros((100,2,4),dtype=bool);mask[:,:,:hops+1]=True
    raw=np.ones((100,2,4,3));applied=np.where(mask[:,:,:,None],raw,0)
    native=np.full_like(raw,5);labels=np.zeros((100,2,4,4),dtype=int)
    labels[:,:,0,1]=1;labels[:,:,1,2]=1;labels[:,:,2,3]=1
    with h5py.File(path,'w') as f:
        for key,v in {'guidance_atom_mask':mask,'mask':np.ones_like(mask),'assignment':np.zeros((100,2,1),dtype=int),
                      'guidance_delta_A':applied,'native_proposal':native,'guided_proposal':native+applied,
                      'unmasked_input_gradient':raw,'input_gradient':applied,'guidance_scope_bond_classes':labels}.items():f[key]=v
    result=verify_trace(path,hops,tmp_path/'verified.json')
    assert result['outside_extra_increment_exact_zero'] and result['graph_radius_recomputed_exact']
    with h5py.File(path,'a') as f:f['guidance_delta_A'][11,0,3,0]=1
    with pytest.raises(ValueError,match='outside'):verify_trace(path,hops,tmp_path/'bad.json')
