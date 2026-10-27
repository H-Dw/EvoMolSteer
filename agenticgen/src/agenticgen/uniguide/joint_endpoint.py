"""Joint chemical/coordinate inpainting and uncovered last-loop output policy."""
import ast
import inspect
from pathlib import Path
import types

import torch

from .chem_inpainting import project_chemistry
from .condition import condition_map, loss
from .io import read, write
from .native_tools import tree, save_pt, tensor_digest
from .sampler import Control

POLICY = "joint-inpainting-endpoint-factorial-v1"


def project_coordinates(state, assignment, core, com=None, coord_scale=1.):
    ids=assignment.to(state['coords'].device)
    target=torch.as_tensor(core,dtype=state['coords'].dtype,device=state['coords'].device)
    target=target[None].expand(len(ids),-1,-1)
    if com is not None:
        target=(target-com[:,None])/coord_scale
    result=dict(state);result['coords']=state['coords'].clone()
    batch=torch.arange(len(ids),device=ids.device)[:,None]
    result['coords'][batch,ids]=target
    mask=torch.zeros_like(state['mask'],dtype=torch.bool);mask[batch,ids]=True
    if not torch.equal(result['coords'][~mask],state['coords'][~mask]):
        raise ValueError('Coordinate projection modified unconditioned atoms')
    return result,dict(coordinates_replaced=True,selected_atoms_per_molecule=ids.shape[1],
        max_replacement_A=float(((result['coords']-state['coords'])*coord_scale).norm(dim=-1).max()),
        unselected_coordinates_unchanged=True)


class ClockAudit:
    def __init__(self):
        self.phase='initialization';self.events=[];self.seen=set()
    def observe(self, global_times, packed):
        values=tuple(float(t.flatten()[0]) for t in global_times)
        ranges=tuple((float(t.min()),float(t.max())) for t in packed)
        key=(self.phase,values,ranges)
        if key not in self.seen:
            self.seen.add(key);self.events.append(dict(phase=self.phase,global_times=values,
                effective_coordinate_network_time=ranges[0],effective_atomic_network_time=ranges[1]))


class EndpointControl(Control):
    def __init__(self,*args,chemical_features,joint,clock,baseline_trace=None,**kwargs):
        super().__init__(*args,**kwargs)
        self.features,self.joint,self.clock=chemical_features,joint,clock
        self.baseline_trace=baseline_trace
        self.projections=[];self.last_loop_target=None;self.last_loop_untarget=None
    def forward(self,*args,**kwargs):
        step=args[-2] if len(args)>=8 else kwargs['step']
        self.clock.phase='integration:'+str(step)
        return super().forward(*args,**kwargs)
    def checkpoint(self,step,*args):
        result=super().checkpoint(step,*args)
        if step==100:self.clock.phase='native_terminal_forward'
        return result
    def capture_loop_prediction(self,target,untarget,step,times):
        self.last_loop_target=tree(target,lambda x:x.detach().clone())
        self.last_loop_untarget=tree(untarget,lambda x:x.detach().clone())
        self.last_loop=dict(step=step,network_times=[float(t[0]) for t in times],
            output_is_uncovered_prediction=True)
    def after(self,curr,step):
        result=super().after(curr,step)
        if step==10 and self.baseline_trace is not None:
            import h5py
            import numpy as np
            with h5py.File(self.baseline_trace,'r') as baseline:
                comparison={key:bool(np.array_equal(self.trace[key][step],baseline[key][step]))
                    for key in ['current_A','guidance_delta_A','native_noise','guided_proposal','assignment']}
            write(self.folder/'PAIRED_PREFIX_ACCEPTANCE.json',comparison)
            if not all(comparison.values()):raise ValueError('Matched-grid prefix differs before coordinate intervention')
        if step>=10:
            before=tensor_digest(self.random_streams.states)
            result,audit=project_chemistry(result,self.last_assignment,self.features)
            if self.joint:
                result,coordinate=project_coordinates(result,self.last_assignment,self.teacher['core_A'],self.com,self.model.coord_scale)
                audit['coordinate_inpainting']=coordinate
            audit['step']=step;self.projections.append(audit)
            self.store('post_inpainting_coordinates',step,result['coords'])
            if before!=tensor_digest(self.random_streams.states):raise ValueError('Projection consumed RNG')
            if step%10==9:write(self.folder/'inpainting_steps.json',self.projections)
        return result
    def close(self):
        super().close()
        write(self.folder/'inpainting_steps.json',self.projections)
        if hasattr(self,'last_loop'):write(self.folder/'LAST_LOOP.json',self.last_loop)


def compile_endpoint(original, source, short):
    """Modify only the native time grid, capture point and terminal source."""
    parsed=ast.parse(source);fn=parsed.body[0]
    loops=[n for n in ast.walk(fn) if isinstance(n,ast.For) and ast.unparse(n.iter)=='enumerate(step_sizes)']
    if len(loops)!=1:raise ValueError('Ambiguous native loop')
    loop=loops[0];capture=[]
    for i,n in enumerate(loop.body):
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Tuple) and any(isinstance(v,ast.Name) and v.id=='predicted_untarget' for v in t.elts) for t in n.targets):capture.append(i)
    if len(capture)!=1:raise ValueError('Ambiguous uncovered prediction')
    loop.body[capture[0]+1:capture[0]+1]=ast.parse('self._adaptive.capture_loop_prediction(predicted_target,predicted_untarget,i,times)').body
    if short:
        grids=[n for n in ast.walk(fn) if isinstance(n,ast.Call) and ast.unparse(n.func)=='torch.linspace']
        if len(grids)!=1 or ast.literal_eval(grids[0].args[1])!=1:raise ValueError('Unsupported native grid')
        grids[0].args[1]=ast.Constant(.999)
        start=[i for i,n in enumerate(fn.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='eps' for t in n.targets)]
        end=[i for i,n in enumerate(fn.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='predicted_target' for t in n.targets) and isinstance(n.value,ast.DictComp)]
        if len(start)!=1 or len(end)!=1 or start[0]>=end[0]:raise ValueError('Unsupported native finalizer')
        fn.body[start[0]:end[0]]=ast.parse('predicted_target=self._adaptive.last_loop_target\npredicted_untarget=self._adaptive.last_loop_untarget').body
    ast.fix_missing_locations(parsed);rendered=ast.unparse(parsed)+'\n'
    scope=dict(original.__func__.__globals__)
    exec(compile(rendered,'joint_endpoint_native.py','exec'),scope)
    return scope['_generate_selective'],rendered


def finish_output(output,teacher,features,folder,joint,short,clock,model):
    folder=Path(folder)
    save_pt(folder/'terminal_uncovered_before_projection.pt.gz',output)
    value,_,_,ids=loss(output['coords'],teacher['core_A'],output['mask'].bool())
    record=dict(raw_C=float((2*value).sqrt().mean()),short=short,joint=joint,
        terminal_projection_applied=not short,output_source='last_loop_uncovered' if short else 'native_final_corrector_then_projection')
    if not short:
        output,chemical=project_chemistry(output,ids,features);record['chemical']=chemical
        if joint:
            output,coordinate=project_coordinates(output,ids,teacher['core_A']);record['coordinate']=coordinate
    record['final_C_is_imposed']=bool(joint and not short)
    record['last_loop']=model._adaptive.last_loop
    write(folder/'OUTPUT_POLICY.json',record)
    clock.phase='endpoint_rescore'
    return output


def execute_variant(sampler,budget_factory,label):
    """Reuse the unchanged sampler execution, correcting its legacy source label."""
    source=inspect.getsource(sampler.execute)
    old='native final corrector prediction returned by sampler; separately preserve state at step 100'
    if source.count(old)!=1:raise ValueError('Sampler annotation changed')
    source=source.replace(old,label)
    scope=dict(sampler.execute.__globals__);scope['Budget']=budget_factory
    exec(compile(source,'endpoint_execute_annotation.py','exec'),scope)
    return scope['execute']


class FactorialBudget:
    def __init__(self,root):self.path=Path(root)/'attempts.json'
    def reserve(self,run_id,stage,count,binding):
        if stage!='formal' or count!=50 or not run_id.startswith('groups/'):
            raise ValueError('Only six fifty-attempt factorial groups are authorized')
        ledger=read(self.path) if self.path.exists() else {};item=dict(stage=POLICY,count=count,binding=binding)
        if run_id in ledger:
            if ledger[run_id]!=item:raise ValueError('Attempt binding changed')
            return
        if sum(r['count'] for r in ledger.values())+count>300:raise ValueError('300-attempt cap; failed attempts remain charged')
        ledger[run_id]=item;write(self.path,ledger)
