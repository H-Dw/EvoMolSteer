import gzip,json
from pathlib import Path
import numpy as np
import pytest
from evomolsteer.continuous.affinity_geometry import names
from evomolsteer.continuous.dynamic_regions import mine,observed_integral,masked_curve_fit
from evomolsteer.io import digest


def test_missing_evidence_is_not_a_zero_contrast():
    curves=np.array([[[2.],[np.nan],[2.]]])
    np.testing.assert_allclose(observed_integral(curves,np.array([0.,.2,.4])),[[2.]])


def test_global_fit_uses_observed_batch_cells_and_derivative():
    times=np.linspace(.2,.7,8);curves=np.tile(3*times+2,(4,1));curves[0,3]=np.nan
    f=masked_curve_fit(times,curves)
    assert f['degree']==1 and f['observed_grid_fraction_by_batch'][0]==.875
    assert f['legendre_coefficients'][1]*2/(times[-1]-times[0])==pytest.approx(3.)


def fixture(root):
    source=root/'results/demo';source.mkdir(parents=True);(source/'config.json').write_text('{"coord_scale":1}')
    times=np.arange(6)*.01;points=[[0,0,0]];fields=names(points)
    frames=[{'time':float(t),'teacher_endpoint_A':[[[1,0,0],[1,1,0]]],
        'teacher_scores':[2.],'teacher_batches':[0]} for t in times]
    ref={'schema_version':'affinity-endpoint-library-1.0','window':[0.,.06],'times':times.tolist(),
        'landmarks_A':points,'origin_A':[0,0,0],'features':fields,'feature_scale':[1.]*len(fields),'frames':frames}
    reference=root/'incumbent.json.gz';reference.write_bytes(gzip.compress(json.dumps(ref).encode(),mtime=0))
    for batch in range(3):
        folder=source/'single'/f'batch_{batch:03d}';folder.mkdir(parents=True)
        (source/f'frame_batch_{batch:03d}.json').write_text(json.dumps({'target_com':[[0,0,0]]*8}))
        x=np.zeros((6,8,2,3));x[:,:4,:,0]=1+batch*.03;x[:,4:,:,0]=4+batch*.01;x[:,:,:,1]=[0,1]
        score=np.tile(np.r_[np.linspace(2,3,4),np.linspace(0,1,4)],(6,1))
        np.savez_compressed(folder/'trajectory.npz',score_time=np.tile(times[:,None],(1,8)),
            state_time=times+.01,resampled=np.ones(6,bool),predicted_coords=x,pic50_on=score,
            root_slot=np.tile(np.arange(8),(6,1)),offspring_count=np.ones((6,8),int),mask=np.ones((6,8,2),bool))
    return reference


def test_dataset_interface_clock_correct_and_deterministic(tmp_path):
    reference=fixture(tmp_path)
    a=mine(tmp_path,'demo',reference,tmp_path/'a',batches=range(3))
    b=mine(tmp_path,'demo',reference,tmp_path/'b',batches=range(3))
    assert a['actual_update_end']==pytest.approx(.06)
    assert a['reference_sha256']==b['reference_sha256'] and a['unidentifiable_event_count']==0
    assert digest(tmp_path/'a/whole_window_effects.parquet')==digest(tmp_path/'b/whole_window_effects.parquet')
    assert set(a['policy']['selected_feature_indices']).issubset({14,15,16})
    with np.load(tmp_path/'a/cohort_codes.npz') as z:
        assert z['codes'].dtype==np.int8 and z['codes'].nbytes==3*6*8
    assert not (tmp_path/'a/particle_features.parquet').exists()
