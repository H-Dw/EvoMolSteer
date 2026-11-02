"""Full saved-model trajectory checks for the configured first-box dose shape."""
import functools
from pathlib import Path
import numpy as np
import h5py
from .box_search import early_shape
from .box_gates import step_enabled
from .guidance_windows import expand
from .geometry_sampling import GeometryControl,scale_binding
from .sampler import sample
from .io import read,frozen,sha


def run_teacher(model,config,teacher,condition,folder):
    folder=Path(folder)
    shape=early_shape(config)
    binding=scale_binding([50]*6,window_bounds=config['windows'],early_shape=shape)
    factory=functools.partial(GeometryControl,condition=condition,scale_binding=binding)
    sample(model,config,teacher,[50]*6,folder,10,control_factory=factory,check_affinity=False)
    expected=expand([50]*6,binding);rows=read(folder/'steps.json')
    if [r['step'] for r in rows]!=list(range(100)):
        raise ValueError('Configured early dose acceptance did not complete all intervals')
    with h5py.File(folder/'trace.h5','r') as trace:
        for step,row in enumerate(rows):
            actual=expected[step] if step_enabled(config,step) else 0
            if row['applied_guidance_scales']!=[actual]*10:
                raise ValueError('Actual applied dose differs from the frozen early shape')
            if row['chemical_inpainting_applied']!=[step_enabled(config,step)]*10:
                raise ValueError('Chemical phase differs from the declared gate')
            if actual==0 and (np.any(trace['input_gradient'][step]) or np.any(trace['guidance_delta_A'][step])):
                raise ValueError('Zero applied dose computed or applied extra guidance')
            if step in (0,99) and not np.array_equal(trace['native_proposal'][step],trace['guided_proposal'][step]):
                raise ValueError('Early shape changed a native endpoint')
    result=dict(shape=shape,full_intervals=100,first_box_actual_scales=expected[:11],
        native_endpoints_exact=True,chemical_phase_matches_gate=True,steps_sha256=sha(folder/'steps.json'),
        real_model_trajectory=True,engineering_only=True,scientific_candidates=0)
    frozen(folder/'EARLY_SHAPE_VERIFIED.json',result)
    return result
