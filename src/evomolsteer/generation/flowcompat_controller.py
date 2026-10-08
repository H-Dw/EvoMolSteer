"""One actual FLOWR forward, coordinate VJP and audited experimental controls."""
import argparse
import copy
import json
import sys
from pathlib import Path
import torch
from .endpoint_controller import EndpointCoordinateExtension
from .endpoint_reward import EndpointGeometryReward
from .innovation_reward import InnovationReward
from .coordinate_reward import predictive_flow_increment
from .scalar_guidance import detached
from .flowcompat_control import control_geometry, cosine
from .module_sensitivity import DecoderSensitivity
from ..io import digest, clean


def measured_pullback(predict,reward,current,mask,scale,com,time):
    with torch.enable_grad():
        x=current.detach().clone().requires_grad_(True);pred,cond=predict(x)
        pred=dict(pred)
        if 'affinity' in pred:pred['affinity']={k:detached(v) for k,v in pred['affinity'].items()}
        atoms=pred['atomics'].detach().argmax(-1)
        y=pred['coords']*scale+com[:,None];anchor=y.detach()
        value,detail=reward(y,atoms,mask,time,anchor)
        gx,gy=torch.autograd.grad(value.sum()+x.sum()*0,(x,y))
    if not bool(torch.isfinite(value).all() and torch.isfinite(gx).all() and torch.isfinite(gy).all()):
        raise ValueError('Nonfinite coordinate derivative')
    detail=dict(detail)
    # Convert physical endpoint derivative to native endpoint-coordinate units.
    detail['flowcompat_endpoint_gradient_rms_native']=(gy.square().sum((1,2))/mask.sum(1)).sqrt()*scale
    return detached(pred),detached(cond),gx.detach(),value.detach(),detached(detail),atoms,anchor


class FlowCompatibilityExtension(EndpointCoordinateExtension):
    def configure(self,model,opt,out):
        original=self.program
        self.program=copy.deepcopy(original);self.program['reward_view']='endpoint_pointcloud'
        super().configure(model,opt,out)
        self.program=original
        self.reward=InnovationReward(original,self.reference) if original['reward_view']=='endpoint_innovation' else EndpointGeometryReward(original,self.reference)
        self.previous_gradient=None;self.module_sha=digest(Path(__file__))
        self.sensitivity=DecoderSensitivity(model)

    def describe(self):
        return {**super().describe(),'flowcompat_module_sha256':self.module_sha,
                'gradient_diagnostics':'Endpoint and current-state gradients from one reverse graph; no head derivative',
                'flow_control':self.program.get('flow_control',{}),
                'agent_request_sha256':self.program.get('agent_request_sha256'),
                'control_semantics':'Old-state Euler control added after native SDE; no exact tilted-density claim'}

    def predict(self,curr,pocket,times,cond,equis,invs):
        trace=self.model._lineage;self.before=curr['coords'].detach().clone();self.pocket=pocket;self.cached=None
        if trace.i==0:
            self.path_rms=torch.zeros(len(self.before),device=self.before.device);self.controlled_updates=0
            self.previous_gradient=None
        active=trace.arm!='unguided' and self.reward.active(trace.t,trace.t+trace.dt)
        self.sensitivity.reset()
        condition=detached(cond);calls=[];audit_before=self.preflight_forward_calls
        def forward(x):
            calls.append(1);state=dict(curr);state['coords']=x
            pred,new_cond=self.model._get_predictions(self.model(state,pocket,times,cond_batch=condition,
                pocket_equis=equis,pocket_invs=invs,training=False))
            if 'affinity' in pred:
                def denied(g):raise RuntimeError('Affinity head entered reward gradient')
                for value in pred['affinity'].values():
                    if torch.is_tensor(value) and value.requires_grad:value.register_hook(denied)
                pred['affinity']={k:detached(v) for k,v in pred['affinity'].items()}
            return pred,new_cond
        if active:
            com=torch.stack([torch.as_tensor(v.com) for v in pocket['complex']]).reshape(-1,3).to(self.before)
            pred,new_cond,g,value,detail,atoms,anchor=measured_pullback(forward,self.reward,self.before,curr['mask'].bool(),self.model.coord_scale,com,trace.t)
            if not self.preflight_done and float(g.norm())>1e-7:self.preflight(forward,curr,com,g,atoms,anchor)
            self.cached=(g,value,detail)
        else:
            with torch.no_grad():pred,new_cond=forward(self.before)
        self.audit_calls_this_step=self.preflight_forward_calls-audit_before
        self.production_calls_this_step=len(calls)-self.audit_calls_this_step
        if self.production_calls_this_step!=1:raise RuntimeError('Extra production forward')
        self.endpoint_atoms=pred['atomics'].detach().argmax(-1);self.endpoint_coords=pred['coords'].detach()
        return pred,new_cond

    def after_native(self,curr):
        trace=self.model._lineage;extra={}
        if self.cached is not None:
            g,value,detail=self.cached
            flow=predictive_flow_increment(self.before,self.endpoint_coords,trace.t,trace.dt,self.model.integrator.use_cosine_scheduler)
            adjusted,factor,extra=control_geometry(g,flow,curr['mask'].bool(),trace.t,self.reward.window,
                self.program.get('flow_control',{}),detail['flowcompat_endpoint_gradient_rms_native'])
            extra['flowcompat_gradient_lag_cosine']=cosine(g,self.previous_gradient) if self.previous_gradient is not None else torch.ones(len(g),device=g.device)
            self.previous_gradient=g.detach()
            # No-op configuration avoids additional floating-point arithmetic.
            if self.program.get('flow_control'):
                detail=dict(detail);detail['dose_gate']=detail['dose_gate']*factor
                self.cached=(adjusted,value,detail)
        result=super().after_native(curr)
        # Update the newly written row before the lossless step transaction ends.
        path=trace.path/'guidance_trace.jsonl'
        lines=path.read_text().splitlines();row=json.loads(lines[-1])
        row.update({k:v.detach().cpu().tolist() for k,v in extra.items()})
        row['flowcompat_module_sha256']=self.module_sha
        row['agent_request_sha256']=self.program.get('agent_request_sha256')
        if self.sensitivity.records:
            row['decoder_output_sensitivity']=self.sensitivity.records
        lines[-1]=json.dumps(clean(row),allow_nan=False)
        path.write_text('\n'.join(lines)+'\n')
        return result


def main():
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--flowr-root',required=True);a,remaining=p.parse_known_args()
    root=Path(a.flowr_root).resolve()
    if not (root/'flowr/models/fm_pocket.py').is_file():raise FileNotFoundError(root)
    sys.path.insert(0,str(root));import flowr
    if not Path(flowr.__file__).resolve().is_relative_to(root):raise RuntimeError('Wrong FLOWR import')
    from . import controller
    sys.argv=[sys.argv[0]]+remaining;controller.main(extension=FlowCompatibilityExtension())

if __name__=='__main__':main()
