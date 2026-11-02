"""Real full-trajectory acceptance for native-only time boxes; no scientific budget."""
import functools
from pathlib import Path
import numpy as np
import h5py
from .box_search import early_shape
from .io import read, frozen
from .native_tools import load_pt, tensor_digest
from .sampler import sample
from .geometry_sampling import GeometryControl, scale_binding


def run_teacher(model, config, teacher, condition, folder):
    from .absolute_acceptance import UnguidedRaw999
    folder=Path(folder)
    count=len(config['windows'])
    cases=[('NativeControl',[False]*count,[0]*count),('EarlyNative',[False,*[True]*(count-1)],[0,*[50]*(count-1)])]
    if count==7:
        for index,label in ((5,'LateMiddleNative'),(6,'LateFinalNative')):
            gates=[True]*count;gates[index]=False
            actual=[50]*count;actual[index]=0
            cases.append((label,gates,actual))
    result={}
    for label,gates,actual in cases:
        local=dict(config,guidance_box_mask=gates)
        binding=scale_binding(actual,window_bounds=config['windows'],early_shape=early_shape(config))
        factory=functools.partial(GeometryControl,condition=condition,scale_binding=binding)
        sample(model,local,teacher,actual,folder/label,10,control_factory=factory,check_affinity=False)
        steps=read(folder/label/'steps.json')
        with h5py.File(folder/label/'trace.h5','r') as trace:
            for step in range(100):
                allowed=1<=step<=98 and any(gates[i] and lo<=step<=hi for i,(lo,hi) in enumerate(config['windows']))
                if bool(any(steps[step]['chemical_inpainting_applied']))!=allowed:
                    raise ValueError('Chemical overlay differs from the actual box gate')
                if not allowed:
                    if np.any(trace['input_gradient'][step]) or np.any(trace['guidance_delta_A'][step]):
                        raise ValueError('Native-only box calculated/applied extra gradient')
                    if not np.array_equal(trace['native_proposal'][step],trace['guided_proposal'][step]):
                        raise ValueError('Native-only proposal changed')
        result[label]=dict(intervals=100,native_gradient_and_chemical_checks=True)
    native=dict(config,chemical_inpainting=dict(config['chemical_inpainting'],enabled=False))
    sample(model,native,teacher,[0]*count,folder/'IndependentNative',10,control_factory=UnguidedRaw999,check_affinity=False)
    a=load_pt(folder/'NativeControl/terminal.pt.gz');b=load_pt(folder/'IndependentNative/terminal.pt.gz')
    if tensor_digest(a)!=tensor_digest(b):raise ValueError('All-native gate differs from independent raw999 native output')
    result['independent_native_terminal_exact']=True
    frozen(folder/'BOX_GATES_VERIFIED.json',result)
    return result
