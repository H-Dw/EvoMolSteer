"""Full-horizon FLOWR coordinate guidance using a frozen multistage reward."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
import torch
from .gradient_controller import Extension
from .multistage_reward import MultistageReward, native_relative_step, preserve_native_geometry
from .scalar_guidance import RegionalReward, detached


class MultistageExtension(Extension):
    def add_arguments(self, p):
        super().add_arguments(p)
        p.set_defaults(arms='unguided,single,multistage_full', max_atom_step_A=.025)
        p.add_argument('--native-rms-ratio', type=float, default=.15)
        p.add_argument('--component-audit', action='store_true')

    def prepare(self, opt, out):
        # Reuse input identity, clean destination, seeding and audited-grid checks.
        requested = opt.arms
        try:
            opt.arms = 'unguided'; super().prepare(opt, out)
        finally:
            opt.arms = requested
        self.arm_terms = {a:[] for a in ['unguided','single']}
        self.arm_terms.update({a:['patch'] for a in ['gradient_zero','multistage_r005','multistage_r015',
            'multistage_r030','multistage_full','multistage_window','static_full']})
        self.arm_terms['gradient_legacy'] = ['region','compact']
        if set(requested.split(','))-set(self.arm_terms): raise ValueError('Unsupported arm')
        if not np.isfinite(opt.native_rms_ratio) or opt.native_rms_ratio < 0:
            raise ValueError('Invalid native-step ratio')

    def configure(self, model, opt, out):
        self.model,self.opt,self.out = model,opt,out
        if model.inpainting_mode or model.graph_inpainting or model._inpaint_self_condition:
            raise ValueError('Inpainting needs an explicit editable-mask adapter')
        model.requires_grad_(False)
        self.reward = MultistageReward(json.loads(Path(opt.program).read_text()),json.loads(Path(opt.catalog).read_text()))
        base = Path(__file__).resolve().parents[3]
        legacy = base/'configs/experiments/ck2_single_guidance_v1'
        self.legacy = RegionalReward(json.loads((legacy/'reward_program.json').read_text()),
                                     json.loads((legacy/'reward_catalog.json').read_text()))
        model._gradient = self
        self.delta=None; self.before=None; self.preflight_done=False
        self.code_commit = subprocess.check_output(['git','-C',str(base),'rev-parse','HEAD'],text=True).strip()
        self.checkpoint_hash = self._sha(opt.checkpoint)
        for name,path in [('reward_program.json',opt.program),('reward_catalog.json',opt.catalog)]:
            shutil.copy2(path,out/name)

    def describe(self):
        return {'schema_version':'multistage-control-1.0','code_commit':self.code_commit,
            'checkpoint_sha256':self.checkpoint_hash,
            'reward_program_sha256':self._sha(self.opt.program), 'reward_catalog_sha256':self._sha(self.opt.catalog),
            'gradient_target':'current model coordinates through live endpoint Jacobian; detached current-step atom masks and self-conditioning',
            'execution':'native step + positive scalar preconditioned reward ascent, molecule-wise caps and native-proposal geometry backtracking',
            'control_domain':[0,1], 'evidence_domain':[0,.5], 'late_reference':'freeze .5 reference, experimental continuation',
            'native_rms_ratio':self.opt.native_rms_ratio, 'max_atom_step_A':self.opt.max_atom_step_A,
            'arm_terms':self.arm_terms, 'gradient_arms_resample':False,
            'component_audit':self.opt.component_audit, 'constraints':self.reward.program['constraints'],
            'legacy_arm':'unchanged early-only RegionalReward; strength .05, no native normalization',
            'categorical_updates':'native sampler unchanged; identities can change, no gradient to argmax',
            'storage':'lossless verified step HDF5 with failed/no-offspring candidates retained'}

    def predict(self, curr, pocket, times, cond, equis, invs):
        trace = self.model._lineage
        self.legacy_step = trace.arm == 'gradient_legacy'
        if self.legacy_step:
            original = self.reward
            try:
                self.reward = self.legacy
                return super().predict(curr,pocket,times,cond,equis,invs)
            finally:
                self.reward = original
        self.before = curr['coords'].detach().clone(); self.delta=None; self.gradient=None
        self.pocket = pocket
        if trace.i == 0:
            self.path_rms = torch.zeros(len(self.before),device=self.before.device,dtype=self.before.dtype)
        arm = trace.arm; time = round(trace.t,6)
        static = arm == 'static_full'
        enabled = bool(self.arm_terms[arm]) and (arm != 'multistage_window' or time <= .5)
        self.row = {'step':trace.i,'score_time':trace.t,'arm':arm,'active':False,
                    'reward_evaluated':enabled,'reference_time':.5 if static or time>.5 else time,
                    'continuation_assumption':time>.5 and enabled}
        def forward(x):
            state=dict(curr); state['coords']=x
            out = self.model(state,pocket,times,cond_batch=cond,pocket_equis=equis,pocket_invs=invs,training=False)
            return self.model._get_predictions(out)
        if not enabled:
            with torch.no_grad(): return forward(curr['coords'])
        com = torch.stack([torch.as_tensor(s.com) for s in pocket['complex']]).reshape(-1,3).to(curr['coords'])
        with torch.enable_grad():
            x=curr['coords'].detach().clone().requires_grad_(True)
            predicted,new_cond=forward(x)
            atoms=predicted['atomics'].detach().argmax(-1)
            value,z,valid,detail = self.reward(predicted['coords']*self.model.coord_scale+com[:,None,:],
                                             atoms,curr['mask'],time,static)
            if not bool(torch.isfinite(value).all()): raise ValueError('Nonfinite mixture reward')
            audit = self.opt.component_audit and trace.i in (0,10,25,50,75,99)
            if audit:
                dz, = torch.autograd.grad(value.sum(), z, retain_graph=True)
                components=[]; raw_component_norms=[]
                for k in range(z.shape[-1]):
                    gk, = torch.autograd.grad((z[:,k]*dz[:,k].detach()).sum()+x.sum()*0,x,retain_graph=True)
                    raw_component_norms.append(gk.detach().norm(dim=(1,2)))
                    components.append(gk.detach()*curr['mask'][...,None])
                norms = torch.stack([g.norm(dim=(1,2)) for g in components],-1)
                cosine = (components[0]*components[1]).sum((1,2))/(norms[:,0]*norms[:,1]).clamp_min(1e-30)
                self.row.update(component_gradient_norms=norms.cpu().tolist(),component_gradient_cosine=cosine.cpu().tolist(),
                                raw_component_gradient_norms=torch.stack(raw_component_norms,-1).cpu().tolist())
            gradient, = torch.autograd.grad(value.sum()+x.sum()*0,x)
        if not bool(torch.isfinite(gradient).all()): raise ValueError('Nonfinite live pullback')
        self.gradient=gradient.detach()*curr['mask'][...,None]
        self.mask=curr['mask'].bool()
        ratio = {'multistage_r005':.05,'multistage_r015':.15,'multistage_r030':.3,
                 'gradient_zero':0.}.get(arm,self.opt.native_rms_ratio)
        self.ratio=ratio
        available_value=lambda v:torch.where(valid,v.detach(),torch.full_like(v,float('nan'))).cpu().tolist()
        responsibilities=detail['responsibilities']
        self.row.update(active=ratio>0,native_rms_ratio=ratio,reward=available_value(value),
                        observables=torch.where(valid[:,None],z.detach(),torch.full_like(z,float('nan'))).cpu().tolist(),
                        observable_available=valid.cpu().tolist(),
                        missing_reason=[None if v else 'no_endpoint_N_O_S' for v in valid.cpu().tolist()],
                        raw_gradient_norm=gradient.detach().norm(dim=(1,2)).cpu().tolist(),
                        gradient_norm=self.gradient.norm(dim=(1,2)).cpu().tolist(),
                        nearest_mahalanobis=available_value(detail['nearest_mahalanobis']),
                        responsibility_max=available_value(responsibilities.max(1).values),
                        responsibility_ess=available_value(1/responsibilities.square().sum(1)),
                        responsibility_entropy=available_value(-(responsibilities*responsibilities.clamp_min(1e-30).log()).sum(1)))
        if self.opt.live_preflight and not self.preflight_done:
            self.multistage_preflight(forward,curr,com,atoms,static,value)
            self.preflight_done=True
        return detached(predicted),detached(new_cond)

    def multistage_preflight(self, forward, curr, com, atoms, static, value):
        from .controller import rng_state,set_rng
        trace=self.model._lineage; saved=rng_state()
        g=self.gradient; direction=g/g.norm().clamp_min(1e-30)
        analytic=float((g*direction).sum()); checks=[]
        try:
            with torch.no_grad():
                for eps in [.001,.003,.01]:
                    rewards=[]
                    for sign in (1,-1):
                        pred,_=forward(curr['coords']+sign*eps*direction)
                        # This derivative is conditional on the reference atom mask.
                        r,*_=self.reward(pred['coords']*self.model.coord_scale+com[:,None,:],atoms,curr['mask'],trace.t,static)
                        rewards.append(float(r.sum()))
                    numeric=(rewards[0]-rewards[1])/(2*eps)
                    checks.append({'epsilon_native_L2':eps,'analytic':analytic,'finite_difference':numeric,
                                   'relative_error':abs(numeric-analytic)/max(analytic,1e-12)})
        finally:
            set_rng(saved)
        good=analytic>1e-8 and min(c['relative_error'] for c in checks)<.1 and all(c['finite_difference']>0 for c in checks)
        audit={'status':'passed' if good else 'failed','checks':checks,'fixed_endpoint_atom_mask':True,
               'gradient_target':'current coordinates through live endpoint','any_parameter_gradient':any(p.grad is not None for p in self.model.parameters())}
        (self.out/'live_gradient_preflight.json').write_text(json.dumps(audit,indent=2))
        if not good: raise RuntimeError('Multistage live finite-difference preflight failed')

    def preflight(self,*args,**kwargs):
        # Inherited legacy predict calls this; preserve its original contract.
        return super().preflight(*args,**kwargs)

    def after_native(self,curr):
        if self.legacy_step: return super().after_native(curr)
        trace=self.model._lineage; scale=self.model.coord_scale
        native=curr['coords']-self.before; actual=torch.zeros_like(native)
        if self.gradient is not None:
            c=self.reward.program['constraints']
            proposed,control=native_relative_step(self.gradient,native,self.mask,self.ratio,scale,
                self.opt.max_atom_step_A,c['max_cumulative_rms_A']-self.path_rms)
            if self.ratio>0:
                actual,guard=preserve_native_geometry(curr['coords'],proposed,self.mask,
                    self.pocket['coords'],self.pocket['mask'],scale,c)
            else:
                actual=proposed
                guard={'geometry_accepted':torch.ones(len(actual),dtype=torch.bool,device=actual.device),
                       'backtrack_factor':torch.ones(len(actual),device=actual.device)}
            n=self.mask.sum(1).clamp_min(1)
            actual_rms=torch.sqrt(actual.square().sum((1,2))/n)*scale
            self.path_rms += actual_rms
            conflict=(self.gradient*native).sum((1,2))/(self.gradient.norm(dim=(1,2))*native.norm(dim=(1,2))).clamp_min(1e-30)
            self.row.update({k:v.detach().cpu().tolist() for k,v in {**control,**guard}.items()})
            self.row.update(injection_rms_A=actual_rms.cpu().tolist(),cumulative_injection_rms_A=self.path_rms.cpu().tolist(),
                gradient_native_cosine=conflict.cpu().tolist(),
                applied_native_rms_ratio=(actual_rms/control['native_rms_A'].clamp_min(1e-30)).cpu().tolist())
        if bool(actual.count_nonzero()):
            curr=dict(curr); curr['coords']=curr['coords']+actual
        if not bool(torch.isfinite(curr['coords']).all()): raise ValueError('Nonfinite controlled proposal')
        self.row.update(native_l2_A=(native.norm(dim=(1,2))*scale).cpu().tolist(),
                        injection_l2_A=(actual.norm(dim=(1,2))*scale).cpu().tolist(),
                        injection_max_atom_A=(actual.norm(dim=-1).amax(1)*scale).cpu().tolist())
        from ..io import clean
        with (trace.path/'guidance_trace.jsonl').open('a') as f:
            f.write(json.dumps(clean(self.row),allow_nan=False)+'\n')
        self.gradient=None; self.before=None
        return curr


def main():
    parser=argparse.ArgumentParser(add_help=False); parser.add_argument('--flowr-root',required=True)
    args,remaining=parser.parse_known_args(); root=Path(args.flowr_root).resolve()
    if not (root/'flowr/models/fm_pocket.py').is_file(): raise FileNotFoundError(root)
    sys.path.insert(0,str(root))
    import flowr
    if not Path(flowr.__file__).resolve().is_relative_to(root): raise RuntimeError('Wrong FLOWR import')
    from . import controller
    sys.argv=[sys.argv[0]]+remaining
    controller.main(extension=MultistageExtension())


if __name__=='__main__': main()
