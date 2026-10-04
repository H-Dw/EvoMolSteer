"""Matched FLOWR runs with explicit native integration plus live reward pullback.

Uses the same recorded upstream loop as the SMC controller. Only its integration-
loop target forward and post-native coordinate injection are replaced. No gradient
arm calls a resampler. Off-target predictions are diagnostics, never objectives.
"""
import argparse
import copy
import hashlib
import inspect
import json
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap
import types

from .instrumentation import instrument_source


def gradient_source(source):
    changed = instrument_source(source)
    start = '            # Run the model on the selected target pocket\n'
    end = '            # Run the model on the selected untarget pocket\n'
    if changed.count(start)!=1 or changed.count(end)!=1: raise RuntimeError('Unsupported target forward')
    first,last = changed.index(start),changed.index(end)
    changed = changed[:first]+(
        '            predicted_target, cond_batch = self._gradient.predict(\n'
        '                curr, pocket_data_target, times, cond,\n'
        '                pocket_equis_target, pocket_invs_target)\n\n')+changed[last:]
    old = '            self._lineage.propose(curr)\n'
    if changed.count(old)!=1: raise RuntimeError('Unsupported proposal site')
    changed = changed.replace(old,'            curr = self._gradient.after_native(curr)\n'+old)
    import ast
    ast.parse(changed)
    return changed


class Extension:
    def add_arguments(self,p):
        p.set_defaults(arms='unguided,single,gradient_region,gradient_region_compact')
        p.add_argument('--input-dataset',required=True)
        p.add_argument('--program',required=True)
        p.add_argument('--catalog',required=True)
        p.add_argument('--strength',type=float,default=.05)
        p.add_argument('--max-atom-step-A',type=float,default=.025)
        p.add_argument('--verify-zero',action='store_true')
        p.add_argument('--live-preflight',action='store_true')

    def prepare(self,opt,out):
        import math
        if opt.verify_passive:raise ValueError('Use --verify-zero for the gradient adapter')
        if not math.isfinite(opt.strength) or opt.strength<0 or not math.isfinite(opt.max_atom_step_A) or opt.max_atom_step_A<=0:
            raise ValueError('Invalid strength/displacement bound')
        if opt.n<1 or opt.batch<1 or opt.n%opt.batch or opt.steps!=100: raise ValueError('Use complete batches and the audited 100-step grid')
        self.arm_terms = {'unguided':[], 'single':[], 'gradient_zero':['region'],
                          'gradient_region':['region'], 'gradient_region_compact':['region','compact']}
        if set(opt.arms.split(','))-set(self.arm_terms): raise ValueError('Unsupported comparison arm')
        if out.exists(): raise FileExistsError('Use a new campaign: '+str(out))
        if opt.window_start!=0 or opt.window!=.5: raise ValueError('This comparison is scoped to [0,.5]')
        inp=Path(opt.input_dataset); inp=inp/'inputs' if (inp/'inputs').is_dir() else inp
        target=Path(opt.root)/'inputs';target.mkdir(parents=True,exist_ok=True)
        from .launcher import INPUT_FILES
        program=json.loads(Path(opt.program).read_text())
        for name in INPUT_FILES:
            src,dst=inp/name,target/name
            if not src.is_file(): raise FileNotFoundError(src)
            expected=program.get('required_input_sha256',{}).get(name)
            if expected is not None and self._sha(src)!=expected:raise ValueError('Reward receptor/frame input mismatch: '+name)
            if dst.exists() and src.read_bytes()!=dst.read_bytes(): raise ValueError('Pocket input mismatch')
            if not dst.exists(): shutil.copy2(src,dst)
        # Seed before model/dataloader construction, making prior generation reproducible.
        import torch,numpy as np,random
        torch.manual_seed(opt.seed);np.random.seed(opt.seed);random.seed(opt.seed)

    def configure(self,model,opt,out):
        import torch
        from .scalar_guidance import RegionalReward
        self.model,self.opt,self.out=model,opt,out
        if model.inpainting_mode or model.graph_inpainting or model._inpaint_self_condition:
            raise ValueError('Masked/inpainting generation requires another adapter')
        model.requires_grad_(False)
        self.reward=RegionalReward(json.loads(Path(opt.program).read_text()),json.loads(Path(opt.catalog).read_text()))
        model._gradient=self
        self.delta=None;self.before=None;self.preflight_done=False
        for name,path in [('reward_program.json',opt.program),('reward_catalog.json',opt.catalog)]:
            shutil.copy2(path,out/name)
        self.code_commit=subprocess.check_output(['git','-C',str(Path(__file__).resolve().parents[3]),'rev-parse','HEAD'],text=True).strip()
        self.checkpoint_hash=self._sha(opt.checkpoint)

    @staticmethod
    def _sha(path):
        h=hashlib.sha256()
        with open(path,'rb') as f:
            for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
        return h.hexdigest()

    def describe(self):
        return {'execution':'native_step + bounded strength*dt*J_endpoint^T grad_world_reward',
                'coordinate_chain':'x_world = coord_scale * endpoint_model(x_native) + target_COM',
                'reward_program_sha256':self._sha(self.opt.program),'reward_catalog_sha256':self._sha(self.opt.catalog),
                'checkpoint_sha256':self.checkpoint_hash,'code_commit':self.code_commit,
                'arm_terms':self.arm_terms,'strength':self.opt.strength,'max_atom_step_A':self.opt.max_atom_step_A,
                'reward_head':'geometry only; no affinity reward','gradient_arms_resample':False,
                'history':'current step derivative; detached self-conditioning; no trajectory backpropagation',
                'difference_from_molexecutor':'adds explicit evidence time support and per-molecule displacement cap',
                'storage':'same per-step verified lossless HDF5 streaming as SMC controller'}

    def instrument(self,model,path):
        path=Path(path);path.mkdir(parents=True,exist_ok=True)
        original=model._generate_selective
        source=textwrap.dedent(inspect.getsource(original))
        ns=dict(original.__func__.__globals__)
        smc=ns['apply_selective_smc_guidance']
        ns['apply_selective_smc_guidance']=lambda **kw:model._lineage.choose(smc,**kw)
        changed=gradient_source(source)
        exec(compile(changed,str(path/'gradient_generate_selective.py'),'exec'),ns)
        model._generate_selective=types.MethodType(ns['_generate_selective'],model)
        (path/'upstream_generate_selective.py').write_text(source)
        (path/'gradient_generate_selective.py').write_text(changed)
        # Independent passive native recorder for a matched weight-zero full-run check.
        ns2=dict(original.__func__.__globals__)
        ns2['apply_selective_smc_guidance']=lambda **kw:model._lineage.choose(smc,**kw)
        exec(compile(instrument_source(source),str(path/'passive_generate_selective.py'),'exec'),ns2)
        self.passive=types.MethodType(ns2['_generate_selective'],model)
        return original

    def make_trace(self,*args,**kw):
        from .controller import Trace
        extension=self
        class GradientTrace(Trace):
            def score(self,pred,onoff):
                arm=self.arm
                try:
                    if arm not in ('single','joint'):self.arm='unguided'
                    super().score(pred,onoff)
                finally:self.arm=arm

            def choose(self,original,**kwargs):
                if self.arm=='single':return super().choose(original,**kwargs)
                return tuple(kwargs[k] for k in ['predicted','prior','current','pocket_data','pocket_equis','pocket_invs','cond_batch'])

            def begin(self,prior,pt,po):
                super().begin(prior,pt,po)
                self.verification_inputs=copy.deepcopy((prior,pt,po)) if extension.opt.verify_zero and self.arm=='gradient_zero' else None

            def finish(self,output):
                super().finish(output)
                if self.verification_inputs is not None:
                    extension.verify_zero(self,output)
        trace=GradientTrace(*args,**kw)
        return trace

    def predict(self,curr,pocket,times,cond,equis,invs):
        import torch
        from .scalar_guidance import live_pullback,bounded_displacement
        trace=self.model._lineage
        arm=trace.arm;terms=self.arm_terms[arm]
        self.before=curr['coords'].detach().clone();self.delta=None
        self.row={'step':trace.i,'score_time':trace.t,'arm':arm,'active':False,'reward_evaluated':False}
        def forward(x):
            state=dict(curr);state['coords']=x
            result=self.model(state,pocket,times,cond_batch=cond,pocket_equis=equis,pocket_invs=invs,training=False)
            return self.model._get_predictions(result)
        if terms and self.reward.active(trace.t,terms):
            com=torch.stack([torch.as_tensor(s.com) for s in pocket['complex']]).reshape(-1,3).to(curr['coords'])
            pred,new_cond,g,value,detail=live_pullback(forward,self.reward,curr['coords'],curr['mask'],self.model.coord_scale,com,trace.t,terms)
            strength=0. if arm=='gradient_zero' else self.opt.strength
            self.delta,factor=bounded_displacement(g,trace.dt,strength,self.model.coord_scale,self.opt.max_atom_step_A)
            self.row.update(active=strength>0,reward_evaluated=True,reward=value.cpu().tolist(),terms=detail,
                            gradient_norm=g.norm(dim=(1,2)).cpu().tolist(),cap_factor=factor.cpu().tolist())
            if self.opt.live_preflight and not self.preflight_done:
                from .controller import rng_state,set_rng
                saved_rng=rng_state()
                try:self.preflight(forward,curr,com,terms,g,value)
                finally:set_rng(saved_rng)
                self.preflight_done=True
            return pred,new_cond
        with torch.no_grad():return forward(curr['coords'])

    def preflight(self,forward,curr,com,terms,g,value):
        import torch
        trace=self.model._lineage
        x=curr['coords'];direction=g/g.norm().clamp_min(1e-30)
        analytic=float((g*direction).sum());checks=[]
        with torch.no_grad():
            for eps in [.001,.003,.01]:
                values=[]
                for sign in [1,-1]:
                    pred,_=forward(x+sign*eps*direction)
                    values.append(float(self.reward(pred['coords']*self.model.coord_scale+com[:,None,:],curr['mask'],trace.t,terms)[0].sum()))
                numeric=(values[0]-values[1])/(2*eps)
                checks.append({'epsilon_native_L2':eps,'analytic':analytic,'finite_difference':numeric,
                               'relative_error':abs(numeric-analytic)/max(abs(analytic),1e-12)})
        good=analytic>1e-8 and min(c['relative_error'] for c in checks)<.1 and all(c['finite_difference']>0 for c in checks)
        result={'status':'passed' if good else 'failed','gradient_target':'current native coordinates through live endpoint',
                'step':trace.i,'arm':trace.arm,'checks':checks,'any_parameter_gradient':any(p.grad is not None for p in self.model.parameters())}
        (self.out/'live_gradient_preflight.json').write_text(json.dumps(result,indent=2))
        if not good:raise RuntimeError('Live pullback finite-difference preflight failed')

    def after_native(self,curr):
        import torch
        trace=self.model._lineage
        native=curr['coords']-self.before
        actual=torch.zeros_like(native) if self.delta is None else self.delta
        if self.delta is not None and bool(self.delta.count_nonzero()):
            curr=dict(curr);curr['coords']=curr['coords']+self.delta
        if not bool(torch.isfinite(curr['coords']).all()):raise ValueError('Nonfinite native proposal')
        scale=self.model.coord_scale
        self.row.update(native_l2_A=(native.norm(dim=(1,2))*scale).cpu().tolist(),
                        injection_l2_A=(actual.norm(dim=(1,2))*scale).cpu().tolist(),
                        injection_max_atom_A=(actual.norm(dim=-1).amax(1)*scale).cpu().tolist())
        with (trace.path/'guidance_trace.jsonl').open('a') as f:f.write(json.dumps(self.row,allow_nan=False)+'\n')
        self.delta=None;self.before=None
        return curr

    def verify_zero(self,trace,output):
        import torch
        from .controller import Trace,rng_state,set_rng
        after=rng_state();prior,pt,po=trace.verification_inputs
        audit=Trace(trace.path/'zero_reference','unguided',trace.seed,trace.batch,trace.scale,save=False,steps=trace.steps)
        self.model._lineage=audit
        try:
            set_rng(trace.initial_rng)
            b=len(prior['coords'])
            baseline=self.passive(prior=prior,pocket_data_target=pt,pocket_data_untarget=po,steps=trace.steps,
                times=[torch.zeros(b,device=prior['coords'].device) for _ in range(3)],apply_guidance=True,
                guidance_window_start=0.,guidance_window_end=.5,coord_noise_level=.2)
            checks={k:bool(torch.equal(output[k],baseline[k])) for k in ['coords','atomics','bonds','charges','mask']}
            (trace.path/'zero_native_equivalence.json').write_text(json.dumps(checks,indent=2))
            if not all(checks.values()):raise AssertionError('Zero-weight native equivalence failed')
        finally:
            self.model._lineage=trace;set_rng(after)
        trace.verification_inputs=None


def main():
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--flowr-root',required=True)
    a,remaining=p.parse_known_args();root=Path(a.flowr_root).resolve()
    if not (root/'flowr/models/fm_pocket.py').is_file():raise FileNotFoundError(root)
    sys.path.insert(0,str(root))
    import flowr
    if not Path(flowr.__file__).resolve().is_relative_to(root):raise RuntimeError('Wrong FLOWR import')
    from . import controller
    sys.argv=[sys.argv[0]]+remaining
    controller.main(extension=Extension())


if __name__=='__main__':main()
