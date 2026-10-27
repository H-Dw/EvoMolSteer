"""Remote-only engineering checks; synthetic compiler checks are not experiment evidence."""
import inspect
import textwrap
import types

import pytest
import torch

from agenticgen.uniguide.joint_endpoint import project_coordinates,compile_endpoint,ClockAudit,FactorialBudget,execute_variant
from agenticgen.uniguide.io import read


def test_coordinate_projection_uses_world_to_native_and_preserves_other_atoms():
    coords=torch.arange(24.).reshape(2,4,3);state=dict(coords=coords,mask=torch.ones(2,4))
    ids=torch.tensor([[0,2],[1,3]]);core=[[10.,20.,30.],[12.,22.,32.]];com=torch.tensor([[2.,4.,6.],[4.,6.,8.]])
    result,audit=project_coordinates(state,ids,core,com,2.)
    for b in range(2):
        assert torch.equal(result['coords'][b,ids[b]],(torch.tensor(core)-com[b])/2)
        other=[i for i in range(4) if i not in ids[b]]
        assert torch.equal(result['coords'][b,other],coords[b,other])
    assert torch.equal(state['coords'],torch.arange(24.).reshape(2,4,3))
    assert audit['coordinates_replaced'] and audit['unselected_coordinates_unchanged']


class Observer:
    def __init__(self):self.raw_times=[]
    def capture_loop_prediction(self,a,b,i,times):
        self.last_loop_target={k:v.clone() for k,v in a.items()};self.last_loop_untarget={k:v.clone() for k,v in b.items()}
        self.raw_times.append(float(times[0]))


class Dummy:
    def __init__(self):self._adaptive=Observer();self.final_calls=0
    def _get_predictions(self,x):return x,None
    def _generate_selective(self,steps=100):
        time_points=torch.linspace(0,1,steps+1)
        step_sizes=[t1-t0 for t0,t1 in zip(time_points[:-1],time_points[1:])]
        times=[torch.tensor(0.)]
        for i,step_size in enumerate(step_sizes):
            out_target={'coords':torch.tensor(float(i))}
            predicted_target,cond_batch=self._get_predictions(out_target)
            out_untarget={'coords':torch.tensor(float(i))}
            predicted_untarget,_=self._get_predictions(out_untarget)
            # Simulate an in-loop state overlay after the capture point.
            predicted_target['coords'].fill_(-100.)
            times=[times[0]+step_size]
        eps=-1e-4
        times=[times[0]+eps]
        self.final_calls+=1
        out_target={'coords':torch.tensor(999.)}
        predicted_target,cond_batch=self._get_predictions(out_target)
        predicted_target={k:v.cpu().detach() for k,v in predicted_target.items()}
        return predicted_target


def test_raw999_returns_last_uncovered_prediction_without_extra_terminal_forward():
    original=Dummy()._generate_selective
    fn,rendered=compile_endpoint(original,textwrap.dedent(inspect.getsource(Dummy._generate_selective)),True)
    model=Dummy();result=types.MethodType(fn,model)()
    assert result['coords'].item()==99. and model.final_calls==0
    assert len(model._adaptive.raw_times)==100
    assert model._adaptive.raw_times[-1]==pytest.approx(.98901,abs=1e-6)
    assert '0.999' in rendered


def test_native_policy_retains_original_final_forward():
    import textwrap
    fn,_=compile_endpoint(Dummy()._generate_selective,textwrap.dedent(inspect.getsource(Dummy._generate_selective)),False)
    model=Dummy();result=types.MethodType(fn,model)()
    assert model.final_calls==1 and result['coords'].item()==999.


def test_clock_audit_records_effective_atomic_times_separately():
    clock=ClockAudit();clock.phase='integration:99'
    clock.observe([torch.tensor([.9999]),torch.tensor([.9999]),torch.tensor([.9999])],
                  [torch.tensor([[[.9999]]]),torch.tensor([[[1.0]]])])
    assert clock.events[0]['effective_atomic_network_time']==(1.,1.)
    assert clock.events[0]['global_times'][1]<1.


def test_factorial_budget_is_six_groups_and_failures_stay_charged(tmp_path):
    b=FactorialBudget(tmp_path)
    for i in range(6):b.reserve('groups/'+str(i),'formal',50,str(i))
    b.reserve('groups/0','formal',50,'0')
    with pytest.raises(ValueError):b.reserve('groups/6','formal',50,'6')
    assert sum(r['count'] for r in read(tmp_path/'attempts.json').values())==300


def test_execution_record_does_not_mislabel_uncovered_output_as_final_corrector():
    from agenticgen.uniguide import sampler
    function=execute_variant(sampler,lambda r:None,'last loop uncovered prediction')
    assert 'last loop uncovered prediction' in function.__code__.co_consts
    assert 'native final corrector prediction returned by sampler; separately preserve state at step 100' not in function.__code__.co_consts
