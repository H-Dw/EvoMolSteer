import numpy as np
import pytest
from evomolsteer.continuous.coordinate_mining import boundary_descendants


def test_ancestry_is_preselection_mass_and_excludes_outside_edge():
    selected=np.array([[0,0,2],[0,2,2],[1,1,1]],int)
    ids,mass=boundary_descendants([True]*3,[.25,.5,.75],selected,[0,.5])
    assert ids.tolist()==[0,1]
    assert mass.tolist()==[[1,0,2],[1,0,2]]
    assert np.all(mass.sum(1)==3)
    modified=selected.copy();modified[-1]=[0,0,0]
    np.testing.assert_array_equal(mass,boundary_descendants([True]*3,[.25,.5,.75],modified,[0,.5])[1])


def test_dynamic_boundary_and_malformed_ancestry():
    s=np.array([[0,0],[0,1],[1,1]],int)
    ids,mass=boundary_descendants([True]*3,[.11,.21,.31],s,[.1,.31])
    assert ids.tolist()==[0,1,2] and mass.tolist()==[[2,0],[0,2],[0,2]]
    with pytest.raises(ValueError):boundary_descendants([True,False,True],[.1,.2,.3],s,[0,.3])
    with pytest.raises(ValueError):boundary_descendants([True]*3,[.1,.2,.3],s.astype(float),[0,.3])


def test_reference_scale_cannot_use_outside_state_variance(tmp_path,monkeypatch):
    from evomolsteer.generation import motif_reward as module
    from evomolsteer.generation.prototypes import write_json
    dataset=tmp_path/'data';mining=tmp_path/'mining';source=dataset/'results'/'c'
    write_json(source/'config.json',{'coord_scale':1.})
    write_json(source/'frame_batch_000.json',{'target_com':[[0,0,0]]})
    write_json(mining/'manifest.json',{'feature_family':'motif','control_representation':'proposal','spatial_anchor':'endpoint',
        'window':[0,.5],'times':[0.,.25,.5],'splits':{'discovery':[0]}})
    write_json(mining/'feature_catalog.json',{'regions':{},'atom_vocabulary':{},'spatial_width_A':4.})
    class Package:
        def __init__(self,path):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def read(self,key,i=None):
            if key=='resampled':return np.ones(3,bool)
            if key=='state_time':return np.array([.25,.5,.75])
            if key=='score_time':return np.array([0.,.25,.5])[:,None]
            if key=='selected_indices':return np.tile(np.arange(3),(3,1))
            if key=='selection_probability':return np.ones(3)/3
            if key in ('proposal_coords','predicted_coords'):
                return np.repeat((np.arange(3)*(100 if i==2 else i+1))[:,None,None],3,axis=2).astype(float)
            return np.ones((3,1))
    monkeypatch.setattr(module,'TrajectoryPackage',Package)
    monkeypatch.setattr(module,'digest',lambda path:'0'*64)
    monkeypatch.setattr(module,'numpy_motifs',lambda x,*args:(x[:,0,:2],['r::motif_centroid_x','r::motif_centroid_y'],{}))
    retained=module.build(dataset,'c',mining,tmp_path/'survival.gz',target='boundary_survival')
    immediate=module.build(dataset,'c',mining,tmp_path/'immediate.gz')
    assert retained['scale_score_times']==[0.,.25] and immediate['scale_score_times']==[0.,.25,.5]
    np.testing.assert_allclose(retained['feature_scale'],np.sqrt(5/3))
    assert np.min(immediate['feature_scale'])>40
