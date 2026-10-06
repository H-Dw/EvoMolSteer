"""Direct current-state regional control followed by untouched native continuation."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
import torch
from .window_controller import WindowExtension
from .coordinate_reward import remove_rigid_pose_gradient,predictive_flow_increment,calibration_increment
from .coordinate_contrast import make_coordinate_reward
from .local_reward import bounded_local_step
from .multistage_reward import preserve_native_geometry
from ..io import clean,digest,write_json


class CoordinateExtension(WindowExtension):
    def configure(self,model,opt,out):
        if model.inpainting_mode or model.graph_inpainting or model._inpaint_self_condition:raise ValueError('Inpainting unsupported')
        self.model,self.opt,self.out=model,opt,out;model.requires_grad_(False);model._gradient=self
        self.reward=make_coordinate_reward(self.program,self.reference);self.preflight_done=False
        self.code_commit=subprocess.check_output(['git','-C',str(Path(__file__).resolve().parents[3]),'rev-parse','HEAD'],text=True).strip()
        self.checkpoint_hash=digest(opt.checkpoint)
        shutil.copy2(opt.program,out/'reward_program.json');shutil.copy2(opt.reference,out/'reference.json.gz')

    def describe(self):
        return {**super().describe(),'schema_version':'current-coordinate-control-1.0',
            'model_jacobian':'not required: reward directly differentiates the actual native proposal',
            'gradient_target':'actual native proposal coordinates; hard endpoint labels and optional endpoint spatial anchor held fixed',
            'reference_time_alignment':self.reference.get('time_alignment','state time s -> current reference s'),
            'conditional_gradient':'No derivative through endpoint anchor or atom identity; forecast refreshed at each native step',
            'dose_rule':self.program.get('dose_reference','observed_native')+' RMS ratio times bounded residual gate, atom/path caps, native-geometry rejection',
            'initial_update_dose':self.program.get('initial_update_dose','native'),
            'coordinate_representation':self.reference['representation'],
            'native_integrator_parameters':self.model.integrator.hparams}

    def predict(self,curr,pocket,times,cond,equis,invs):
        if self.model._lineage.i==0:self.controlled_updates=0
        pred,new_cond=super().predict(curr,pocket,times,cond,equis,invs)
        self.endpoint_atoms=pred['atomics'].detach().argmax(-1)
        self.endpoint_coords=pred['coords'].detach()
        return pred,new_cond

    def after_native(self,curr):
        from .controller import nparr
        trace=self.model._lineage;scale=self.model.coord_scale;s=trace.t+trace.dt;mask=curr['mask'].bool()
        native=curr['coords']-self.before;actual=torch.zeros_like(native)
        enabled=trace.arm!='unguided' and self.reward.active(trace.t,s)
        row={'step':trace.i,'score_time':trace.t,'state_time':s,'arm':trace.arm,'reward_evaluated':enabled,
             'active':enabled and trace.arm!='gradient_zero','particle_resampled':False}
        trace.arr['native_proposal_coords'].append(nparr(curr['coords']))
        if enabled:
            com=torch.stack([torch.as_tensor(v.com) for v in self.pocket['complex']]).reshape(-1,3).to(native)
            reference_time=trace.t if self.reference.get('control_representation','current')=='proposal' else s
            anchor=self.endpoint_coords*scale+com[:,None] if self.reference.get('spatial_anchor')=='endpoint' else None
            row['reference_time']=reference_time
            with torch.enable_grad():
                x=curr['coords'].detach().clone().requires_grad_(True)
                value,detail=self.reward(x*scale+com[:,None],self.endpoint_atoms,mask,reference_time,anchor)
                g,=torch.autograd.grad(value.sum(),x)
            if not bool(torch.isfinite(g).all()):raise ValueError('Nonfinite coordinate gradient')
            raw_gradient_squared=g.detach().square().sum((1,2))
            if self.program.get('preserve_native_rigid_pose'):
                g=remove_rigid_pose_gradient(g.detach(),x.detach(),mask)
            row.update(raw_gradient_l2_native=raw_gradient_squared.sqrt().cpu().tolist(),
                post_projection_l2_native=g.detach().norm(dim=(1,2)).cpu().tolist(),
                projection_retained_squared_fraction=(g.detach().square().sum((1,2))/raw_gradient_squared.clamp_min(1e-30)).cpu().tolist())
            # Native categorical channels and RNG states are never touched.
            if not self.preflight_done and float(g.norm())>1e-7:
                direction=g/g.norm();analytic=float((g*direction).sum());checks=[]
                with torch.no_grad():
                    for eps in (.001,.003,.01):
                        a,_=self.reward((x+eps*direction)*scale+com[:,None],self.endpoint_atoms,mask,reference_time,anchor)
                        b,_=self.reward((x-eps*direction)*scale+com[:,None],self.endpoint_atoms,mask,reference_time,anchor)
                        num=float((a.sum()-b.sum())/(2*eps));checks.append({'epsilon':eps,'analytic':analytic,'numerical':num,'relative_error':abs(num-analytic)/analytic})
                passed=min(r['relative_error'] for r in checks)<.15 and all(r['numerical']>0 for r in checks)
                write_json(self.out/'coordinate_gradient_preflight.json',{'passed':passed,'state_time':s,'reference_time':reference_time,'anchor_held_fixed':anchor is not None,'checks':checks})
                if not passed:raise ValueError('Coordinate gradient finite difference failed')
                self.preflight_done=True
            c=self.program['constraints'];eta=0. if trace.arm=='gradient_zero' else self.program['native_rms_ratio']
            dose_view=self.program.get('dose_reference','observed_native')
            if dose_view not in ('observed_native','predictive_flow'):raise ValueError('Unknown dose reference')
            # Same endpoint residual as the verified linear FLOWR schedule; no
            # stochastic score drift, initial contraction or categorical gradients.
            cosine=self.model.integrator.use_cosine_scheduler
            flow_delta=predictive_flow_increment(self.before,self.endpoint_coords,trace.t,trace.dt,cosine) if dose_view=='predictive_flow' or not cosine else None
            first_controlled=self.controlled_updates==0
            dose_delta=calibration_increment(native,flow_delta,mask,dose_view,self.program.get('initial_update_dose','native'),first_controlled)
            row.update(first_controlled_update=first_controlled,initial_update_dose=self.program.get('initial_update_dose','native'))
            # Density cancellation is evaluated in float64; the physical update
            # must retain FLOWR's coordinate dtype (including a zero-dose arm).
            amplitude_gate=detail['dose_gate'].to(g)
            proposed,control=bounded_local_step(g,dose_delta,mask,amplitude_gate,eta,scale,c['max_atom_step_A'],c['max_cumulative_rms_A']-self.path_rms)
            row['calibration_rms_A']=control['native_rms_A'].detach().cpu().tolist()
            row.update(dose_reference=dose_view,
                observed_native_rms_A=(native.square().sum((1,2))/mask.sum(1)).sqrt().mul(scale).cpu().tolist(),
                predictive_flow_rms_A=(flow_delta.square().sum((1,2))/mask.sum(1)).sqrt().mul(scale).cpu().tolist() if flow_delta is not None else None)
            self.controlled_updates+=1
            actual,guard=preserve_native_geometry(curr['coords'],proposed,mask,self.pocket['coords'],self.pocket['mask'],scale,c)
            rms=(actual.square().sum((1,2))/mask.sum(1)).sqrt()*scale;self.path_rms+=rms
            row.update({k:v.detach().cpu().tolist() for k,v in {**{k:v for k,v in detail.items() if k!='core_mask'},**control,**guard}.items()})
            outside=(~detail['core_mask'])&mask
            row.update(reward=value.detach().cpu().tolist(),injection_rms_A=rms.cpu().tolist(),cumulative_rms_A=self.path_rms.cpu().tolist(),
                requested_ratio=eta,gradient_norm=g.norm(dim=(1,2)).cpu().tolist(),
                injection_noncore_fraction=(actual.square().sum(-1).mul(outside).sum(1)/actual.square().sum((1,2)).clamp_min(1e-30)).cpu().tolist())
        if bool(actual.count_nonzero()):curr=dict(curr);curr['coords']=curr['coords']+actual
        row.update(injection_l2_A=(actual.norm(dim=(1,2))*scale).cpu().tolist(),max_atom_injection_A=(actual.norm(dim=-1).amax(1)*scale).cpu().tolist())
        if abs(s-self.opt.window)<2e-6:
            np.savez_compressed(trace.path/'window_state.npz',coords=nparr(curr['coords']),atomics=nparr(curr['atomics'].argmax(-1)),
                                bonds=nparr(curr['bonds'].argmax(-1)),mask=nparr(curr['mask']),state_time=s,coord_scale=scale)
        with (trace.path/'guidance_trace.jsonl').open('a') as f:f.write(json.dumps(clean(row),allow_nan=False)+'\n')
        return curr


def main():
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--flowr-root',required=True)
    a,remaining=p.parse_known_args();root=Path(a.flowr_root).resolve()
    if not (root/'flowr/models/fm_pocket.py').is_file():raise FileNotFoundError(root)
    sys.path.insert(0,str(root));import flowr
    if not Path(flowr.__file__).resolve().is_relative_to(root):raise RuntimeError('Wrong FLOWR import')
    from . import controller
    sys.argv=[sys.argv[0]]+remaining;controller.main(extension=CoordinateExtension())
