"""Live FLOWR endpoint Jacobian, bounded regional intervention, native continuation."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
import torch
from .window_controller import WindowExtension
from .local_reward import LocalIntervalReward,bounded_local_step
from .scalar_guidance import detached
from .multistage_reward import preserve_native_geometry
from ..io import digest,clean,write_json


class LocalExtension(WindowExtension):
    def configure(self,model,opt,out):
        if model.inpainting_mode or model.graph_inpainting or model._inpaint_self_condition:raise ValueError('Inpainting unsupported')
        self.model,self.opt,self.out=model,opt,out
        model.requires_grad_(False);model._gradient=self
        self.reward=LocalIntervalReward(self.program,self.reference);self.preflight_done=False
        self.code_commit=subprocess.check_output(['git','-C',str(Path(__file__).resolve().parents[3]),'rev-parse','HEAD'],text=True).strip()
        self.checkpoint_hash=digest(opt.checkpoint)
        shutil.copy2(opt.program,out/'reward_program.json');shutil.copy2(opt.reference,out/'reference.json.gz')

    def describe(self):
        return {**super().describe(),'schema_version':'regional-window-control-1.0',
                'gradient_target':'current native coordinates through live FLOWR endpoint coordinate Jacobian',
                'model_jacobian':'FLOWR.ROOT, parameters frozen, detached categorical labels and self-conditioning',
                'gradient_support':self.program.get('gradient_support','full_pullback'),
                'dose_rule':'eta * native_RMS * excess/sqrt(1+excess^2), then caps and native-geometry guard',
                'terminal_export':self.opt.export_terminal}

    def predict(self,curr,pocket,times,cond,equis,invs):
        trace=self.model._lineage;self.before=curr['coords'].detach().clone();self.pocket=pocket
        if trace.i==0:self.path_rms=torch.zeros(len(self.before),device=self.before.device)
        self.g=None;self.detail=None;self.value=None
        enabled=trace.arm!='unguided' and self.reward.active(trace.t,trace.t+trace.dt)
        def forward(x):
            state=dict(curr);state['coords']=x
            return self.model._get_predictions(self.model(state,pocket,times,cond_batch=cond,pocket_equis=equis,pocket_invs=invs,training=False))
        if not enabled:
            with torch.no_grad():return forward(curr['coords'])
        com=torch.stack([torch.as_tensor(v.com) for v in pocket['complex']]).reshape(-1,3).to(self.before)
        with torch.enable_grad():
            x=curr['coords'].detach().clone().requires_grad_(True)
            pred,new_cond=forward(x)
            atoms=pred['atomics'].detach().argmax(-1)
            value,detail=self.reward(pred['coords']*self.model.coord_scale+com[:,None,:],atoms,curr['mask'],trace.t)
            g,=torch.autograd.grad(value.sum()+x.sum()*0,x)
        if not bool(torch.isfinite(g).all()):raise ValueError('Nonfinite regional gradient')
        if not self.preflight_done and float(g.norm())>1e-7:
            self.preflight(forward,curr,com,atoms,g,value)
            self.preflight_done=True
        self.raw_g=g.detach()
        core=detail['core_mask'].detach()
        support=self.program.get('gradient_support','full_pullback')
        if support=='core_and_neighbors':
            bonds=pred['bonds'].detach().argmax(-1)>0
            core=core|(bonds&core[:,None,:]).any(-1)
            g=g*core[...,None]
        elif support!='full_pullback':raise ValueError('Unknown gradient support')
        self.g=g.detach();self.core=core;self.detail=detached(detail);self.value=value.detach()
        return detached(pred),detached(new_cond)

    def preflight(self,forward,curr,com,atoms,g,value):
        from .controller import rng_state,set_rng
        saved=rng_state();d=g/g.norm();analytic=float((g*d).sum());checks=[]
        try:
            with torch.no_grad():
                for eps in [.001,.003,.01]:
                    vals=[]
                    for sign in [1,-1]:
                        set_rng(saved);pred,_=forward(curr['coords']+sign*eps*d)
                        reward,_=self.reward(pred['coords']*self.model.coord_scale+com[:,None,:],atoms,curr['mask'],self.model._lineage.t)
                        vals.append(float(reward.sum()))
                    numerical=(vals[0]-vals[1])/(2*eps)
                    checks.append({'epsilon':eps,'analytic':analytic,'finite_difference':numerical,'relative_error':abs(numerical-analytic)/max(abs(analytic),1e-12)})
        finally:set_rng(saved)
        passed=analytic>0 and min(v['relative_error'] for v in checks)<.15 and all(v['finite_difference']>0 for v in checks)
        write_json(self.out/'live_jacobian_preflight.json',{'passed':passed,'checks':checks,'categorical_mask':'fixed endpoint hard labels for piecewise derivative check','model':'FLOWR.ROOT','time':self.model._lineage.t})
        if not passed:raise ValueError('Live FLOWR Jacobian finite difference failed')

    def after_native(self,curr):
        from .controller import nparr
        trace=self.model._lineage;scale=self.model.coord_scale;s=trace.t+trace.dt
        native=curr['coords']-self.before;actual=torch.zeros_like(native)
        enabled=self.g is not None;mask=curr['mask'].bool()
        row={'step':trace.i,'score_time':trace.t,'state_time':s,'arm':trace.arm,
             'active':enabled and trace.arm!='gradient_zero','reward_evaluated':enabled,'particle_resampled':False}
        trace.arr['native_proposal_coords'].append(nparr(curr['coords']))
        if enabled:
            c=self.program['constraints'];eta=0. if trace.arm=='gradient_zero' else self.program['native_rms_ratio']
            proposed,control=bounded_local_step(self.g,native,mask,self.detail['dose_gate'],eta,scale,c['max_atom_step_A'],c['max_cumulative_rms_A']-self.path_rms)
            actual,guard=preserve_native_geometry(curr['coords'],proposed,mask,self.pocket['coords'],self.pocket['mask'],scale,c)
            rms=(actual.square().sum((1,2))/mask.sum(1)).sqrt()*scale;self.path_rms+=rms
            noncore=(~self.core)&mask
            detail={k:v for k,v in self.detail.items() if k!='core_mask'}
            row.update({k:v.detach().cpu().tolist() for k,v in {**detail,**control,**guard}.items()})
            row.update(reward=self.value.cpu().tolist(),injection_rms_A=rms.cpu().tolist(),
                cumulative_rms_A=self.path_rms.cpu().tolist(),gradient_norm=self.g.norm(dim=(1,2)).cpu().tolist(),
                gradient_noncore_fraction=(self.raw_g.square().sum(-1).mul(noncore).sum(1)/self.raw_g.square().sum((1,2)).clamp_min(1e-30)).cpu().tolist(),
                injection_noncore_fraction=(actual.square().sum(-1).mul(noncore).sum(1)/actual.square().sum((1,2)).clamp_min(1e-30)).cpu().tolist(),
                core_atoms=self.core.sum(1).cpu().tolist(),requested_ratio=eta)
        if bool(actual.count_nonzero()):curr=dict(curr);curr['coords']=curr['coords']+actual
        row.update(injection_l2_A=(actual.norm(dim=(1,2))*scale).cpu().tolist(),max_atom_injection_A=(actual.norm(dim=-1).amax(1)*scale).cpu().tolist())
        if abs(s-self.opt.window)<2e-6:
            np.savez_compressed(trace.path/'window_state.npz',coords=nparr(curr['coords']),atomics=nparr(curr['atomics'].argmax(-1)),bonds=nparr(curr['bonds'].argmax(-1)),mask=nparr(curr['mask']),state_time=s,coord_scale=scale)
        with (trace.path/'guidance_trace.jsonl').open('a') as f:f.write(json.dumps(clean(row),allow_nan=False)+'\n')
        self.g=None
        return curr


def main():
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--flowr-root',required=True)
    a,remaining=p.parse_known_args();root=Path(a.flowr_root).resolve()
    if not (root/'flowr/models/fm_pocket.py').is_file():raise FileNotFoundError(root)
    sys.path.insert(0,str(root));import flowr
    if not Path(flowr.__file__).resolve().is_relative_to(root):raise RuntimeError('Wrong FLOWR import')
    from . import controller
    sys.argv=[sys.argv[0]]+remaining;controller.main(extension=LocalExtension())
