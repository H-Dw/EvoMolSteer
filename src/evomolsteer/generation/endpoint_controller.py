"""One native forward + FLOWR endpoint geometry VJP; no affinity-head gradient.

The derivative is exact at x_t with detached self-conditioning and correspondence.
Its post-native injection is intentionally lagged. A first-order reward estimate
must not be reported as a fresh endpoint reward after the stochastic native step.
"""
import argparse,json,shutil,subprocess,sys
from pathlib import Path
import numpy as np
import torch
from .coordinate_controller import CoordinateExtension
from .coordinate_reward import predictive_flow_increment,calibration_increment,atom_step_cap,remove_rigid_pose_gradient
from .local_reward import bounded_local_step
from .multistage_reward import reject_new_severe_clashes
from .scalar_guidance import detached
from .endpoint_reward import EndpointGeometryReward
from ..io import write_json,clean,digest


def endpoint_pullback(predict,reward,current,mask,scale,com,time):
    """Affinity values are detached; loss depends only on endpoint coordinates."""
    with torch.enable_grad():
        x=current.detach().clone().requires_grad_(True);pred,cond=predict(x)
        if 'affinity' in pred:
            pred=dict(pred);pred['affinity']={k:detached(v) for k,v in pred['affinity'].items()}
        atoms=pred['atomics'].detach().argmax(-1);anchor=pred['coords'].detach()*scale+com[:,None]
        value,detail=reward(pred['coords']*scale+com[:,None],atoms,mask,time,anchor)
        if value.shape!=(len(x),) or not bool(torch.isfinite(value).all()):raise ValueError('Invalid endpoint geometry scalar')
        gradient,=torch.autograd.grad(value.sum()+x.sum()*0,x)
    if not bool(torch.isfinite(gradient).all()):raise ValueError('Nonfinite FLOWR coordinate VJP')
    return detached(pred),detached(cond),gradient.detach(),value.detach(),detached(detail),atoms,anchor


class EndpointCoordinateExtension(CoordinateExtension):
    def configure(self,model,opt,out):
        if model.inpainting_mode or model.graph_inpainting or model._inpaint_self_condition:raise ValueError('Inpainting unsupported')
        self.model,self.opt,self.out=model,opt,out;model.requires_grad_(False);model._gradient=self
        self.reward=EndpointGeometryReward(self.program,self.reference);self.preflight_done=False;self.preflight_forward_calls=0
        self.code_commit=subprocess.check_output(['git','-C',str(Path(__file__).resolve().parents[3]),'rev-parse','HEAD'],text=True).strip()
        self.checkpoint_hash=digest(opt.checkpoint)
        shutil.copy2(opt.program,out/'reward_program.json');shutil.copy2(opt.reference,out/'reference.json.gz')

    def describe(self):
        d=super().describe();d.update(model_jacobian='Actual FLOWR J_endpoint(x_t)^T grad_endpoint_geometry',
            gradient_target='Pre-native x_t; detached self-conditioning/correspondence. Lagged post-native injection',
            conditional_gradient='Hungarian assignments and teacher priors frozen; FLOWR endpoint coordinates differentiated',
            derivative_path='flowr_endpoint_vjp',affinity_head_gradient=False,additional_production_forward_calls_per_step=0,
            dose_rule=self.program.get('dose_reference','predictive_flow')+' RMS calibrated coordinate VJP, atom/path caps and severe-new-clash checks',
            reward_response='Only first-order directional estimate after injection, not fresh endpoint response',
            preflight_extra_forward_calls='One-time numerical validation only, separately logged')
        return d

    def predict(self,curr,pocket,times,cond,equis,invs):
        trace=self.model._lineage
        self.before=curr['coords'].detach().clone();self.pocket=pocket;self.cached=None
        if trace.i==0:
            self.path_rms=torch.zeros(len(self.before),device=self.before.device);self.controlled_updates=0
        active=trace.arm!='unguided' and self.reward.active(trace.t,trace.t+trace.dt)
        condition=detached(cond)
        calls=[];audit_calls_before=self.preflight_forward_calls
        def forward(x):
            calls.append(1)
            state=dict(curr);state['coords']=x
            pred,new_cond=self.model._get_predictions(self.model(state,pocket,times,cond_batch=condition,
                pocket_equis=equis,pocket_invs=invs,training=False))
            # The affinity branch is never part of the scalar or its derivative.
            if 'affinity' in pred:
                def forbidden_head_gradient(gradient):raise RuntimeError('Affinity head entered endpoint reward gradient')
                for value in pred['affinity'].values():
                    if torch.is_tensor(value) and value.requires_grad:value.register_hook(forbidden_head_gradient)
                pred['affinity']={k:detached(v) for k,v in pred['affinity'].items()}
            return pred,new_cond
        if active:
            com=torch.stack([torch.as_tensor(v.com) for v in pocket['complex']]).reshape(-1,3).to(self.before)
            pred,new_cond,g,value,detail,atoms,anchor=endpoint_pullback(forward,self.reward,self.before,curr['mask'].bool(),self.model.coord_scale,com,trace.t)
            if not self.preflight_done and float(g.norm())>1e-7:
                self.preflight(forward,curr,com,g,atoms,anchor)
            self.cached=(g,value,detail)
        else:
            with torch.no_grad():pred,new_cond=forward(self.before)
        self.audit_calls_this_step=self.preflight_forward_calls-audit_calls_before
        self.production_calls_this_step=len(calls)-self.audit_calls_this_step
        if self.production_calls_this_step!=1:raise RuntimeError('Endpoint wrapper made additional production target forwards')
        self.endpoint_atoms=pred['atomics'].detach().argmax(-1);self.endpoint_coords=pred['coords'].detach()
        return pred,new_cond

    def preflight(self,forward,curr,com,g,atoms,anchor):
        from .controller import rng_state,set_rng
        saved=rng_state();checks=[];direction=g/g.norm();analytic=float((g*direction).sum())
        try:
            with torch.no_grad():
                for epsilon in (.003,.01):
                    set_rng(saved);pred_a,_=forward(self.before+epsilon*direction)
                    set_rng(saved);pred_b,_=forward(self.before-epsilon*direction)
                    a,_=self.reward(pred_a['coords']*self.model.coord_scale+com[:,None],atoms,curr['mask'].bool(),self.model._lineage.t,anchor)
                    b,_=self.reward(pred_b['coords']*self.model.coord_scale+com[:,None],atoms,curr['mask'].bool(),self.model._lineage.t,anchor)
                    numerical=float((a.sum()-b.sum())/(2*epsilon));checks.append({'epsilon':epsilon,'analytic':analytic,
                        'numerical':numerical,'relative_error':abs(numerical-analytic)/max(abs(analytic),1e-30)})
                    self.preflight_forward_calls+=2
        finally:set_rng(saved)
        passed=min(r['relative_error'] for r in checks)<.15 and all(r['numerical']>0 for r in checks)
        write_json(self.out/'coordinate_gradient_preflight.json',{'passed':passed,'derivative_path':'flowr_endpoint_vjp',
            'score_time':self.model._lineage.t,'affinity_head_gradient':False,'one_time_extra_forward_calls':self.preflight_forward_calls,
            'production_extra_forward_calls_per_step':0,'frozen_self_condition_and_assignment':True,'checks':checks})
        if not passed:raise ValueError('Real FLOWR coordinate VJP finite difference failed')
        self.preflight_done=True

    def after_native(self,curr):
        from .controller import nparr
        trace=self.model._lineage;scale=self.model.coord_scale;s=trace.t+trace.dt;mask=curr['mask'].bool()
        native=curr['coords']-self.before;actual=torch.zeros_like(native);enabled=self.cached is not None
        row={'step':trace.i,'score_time':trace.t,'state_time':s,'reference_time':trace.t,'arm':trace.arm,
            'reward_evaluated':enabled,'active':enabled and trace.arm!='gradient_zero','particle_resampled':False,
            'derivative_path':'flowr_endpoint_vjp','gradient_evaluation_time':trace.t,
            'production_target_forward_calls':self.production_calls_this_step,'one_time_audit_target_forward_calls':self.audit_calls_this_step,
            'affinity_outputs_detached':True,'affinity_gradient_hook_rejection_enabled':True,
            'reward_response_kind':'First-order endpoint reward at old x_t; no post-native endpoint re-forward'}
        trace.arr['native_proposal_coords'].append(nparr(curr['coords']))
        if enabled:
            g,value,detail=self.cached;raw=g.square().sum((1,2))
            if self.program.get('preserve_native_rigid_pose'):g=remove_rigid_pose_gradient(g,self.before,mask)
            row.update(raw_gradient_l2_native=raw.sqrt().cpu().tolist(),post_projection_l2_native=g.norm(dim=(1,2)).cpu().tolist(),
                projection_retained_squared_fraction=(g.square().sum((1,2))/raw.clamp_min(1e-30)).cpu().tolist())
            c=self.program['constraints'];eta=0. if trace.arm=='gradient_zero' else self.program['native_rms_ratio']
            dose_view=self.program.get('dose_reference','predictive_flow')
            if dose_view not in ('observed_native','predictive_flow'):raise ValueError('Unknown dose calibration')
            cosine=self.model.integrator.use_cosine_scheduler
            flow=predictive_flow_increment(self.before,self.endpoint_coords,trace.t,trace.dt,cosine) if dose_view=='predictive_flow' or not cosine else None
            first=self.controlled_updates==0;calibration=calibration_increment(native,flow,mask,dose_view,self.program.get('initial_update_dose','native'),first)
            cap=atom_step_cap(c,first)
            proposed,control=bounded_local_step(g,calibration,mask,detail['dose_gate'].to(g),eta,scale,cap,c['max_cumulative_rms_A']-self.path_rms)
            actual,guard=reject_new_severe_clashes(curr['coords'],proposed,mask,self.pocket['coords'],self.pocket['mask'],scale,c)
            rms=(actual.square().sum((1,2))/mask.sum(1)).sqrt()*scale;self.path_rms+=rms;self.controlled_updates+=1
            row.update({k:v.detach().cpu().tolist() for k,v in {**{k:v for k,v in detail.items() if k!='core_mask'},**control,**guard}.items()})
            outside=(~detail['core_mask'])&mask
            row.update(reward=value.cpu().tolist(),reward_before=value.cpu().tolist(),
                first_order_reward_change=(g*actual).sum((1,2)).cpu().tolist(),injection_rms_A=rms.cpu().tolist(),
                cumulative_rms_A=self.path_rms.cpu().tolist(),requested_ratio=eta,gradient_norm=g.norm(dim=(1,2)).cpu().tolist(),
                atom_step_cap_A=cap,first_controlled_update=first,initial_update_dose=self.program.get('initial_update_dose','native'),
                dose_reference=dose_view,calibration_rms_A=control['native_rms_A'].cpu().tolist(),
                observed_native_rms_A=(native.square().sum((1,2))/mask.sum(1)).sqrt().mul(scale).cpu().tolist(),
                predictive_flow_rms_A=(flow.square().sum((1,2))/mask.sum(1)).sqrt().mul(scale).cpu().tolist() if flow is not None else None,
                injection_noncore_fraction=(actual.square().sum(-1).mul(outside).sum(1)/actual.square().sum((1,2)).clamp_min(1e-30)).cpu().tolist())
        if bool(actual.count_nonzero()):curr=dict(curr);curr['coords']=curr['coords']+actual
        row.update(injection_l2_A=(actual.norm(dim=(1,2))*scale).cpu().tolist(),max_atom_injection_A=(actual.norm(dim=-1).amax(1)*scale).cpu().tolist())
        if abs(s-self.opt.window)<2e-6:
            np.savez_compressed(trace.path/'window_state.npz',coords=nparr(curr['coords']),atomics=nparr(curr['atomics'].argmax(-1)),
                bonds=nparr(curr['bonds'].argmax(-1)),mask=nparr(curr['mask']),state_time=s,coord_scale=scale)
        with (trace.path/'guidance_trace.jsonl').open('a') as f:f.write(json.dumps(clean(row),allow_nan=False)+'\n')
        self.cached=None
        return curr


def main():
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--flowr-root',required=True);a,remaining=p.parse_known_args()
    root=Path(a.flowr_root).resolve()
    if not (root/'flowr/models/fm_pocket.py').is_file():raise FileNotFoundError(root)
    sys.path.insert(0,str(root));import flowr
    if not Path(flowr.__file__).resolve().is_relative_to(root):raise RuntimeError('Wrong FLOWR import')
    from . import controller
    sys.argv=[sys.argv[0]]+remaining;controller.main(extension=EndpointCoordinateExtension())

if __name__=='__main__':main()
