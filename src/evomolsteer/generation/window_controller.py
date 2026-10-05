"""Strict learned-window gradient inference; particle resampling is prohibited."""
import argparse
import copy
import inspect
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import textwrap
import types
import numpy as np
import torch
from .gradient_controller import Extension, gradient_source
from .window_reference import load_reference
from .window_reward import WindowReward
from .multistage_reward import native_relative_step, preserve_native_geometry
from ..io import digest, write_json, clean


def no_selection_source(source):
    changed=gradient_source(source)
    needle='    assert apply_guidance\n'
    if changed.count(needle)!=1:raise ValueError('Unexpected upstream guidance assertion')
    return changed.replace(needle,'    assert not apply_guidance, "Particle selection is disabled"\n')


class WindowExtension(Extension):
    def add_arguments(self,p):
        p.set_defaults(arms='unguided,gradient',window=None,window_start=None)
        p.add_argument('--input-dataset',required=True)
        p.add_argument('--program',required=True)
        p.add_argument('--reference',required=True)
        p.add_argument('--export-terminal',action='store_true',help='Decode t=1 outputs and export same-state affinity-head predictions for local evaluation')

    def prepare(self,opt,out):
        if out.exists():raise FileExistsError(out)
        if opt.n<1 or opt.batch<1 or opt.n%opt.batch or opt.steps<1:raise ValueError('Complete batches required')
        if set(opt.arms.split(','))-{'unguided','gradient','gradient_zero'}:raise ValueError('No SMC arms allowed')
        self.program=json.loads(Path(opt.program).read_text())
        self.reference=load_reference(opt.reference)
        if digest(opt.reference)!=self.program['reference_sha256']:raise ValueError('Reference hash mismatch')
        a,b=self.reference['window']
        for key,expected in [('window_start',a),('window',b)]:
            value=getattr(opt,key)
            if value is not None and abs(value-expected)>1e-7:raise ValueError('Control/learning support mismatch')
            setattr(opt,key,expected)
        if not all(np.isclose(t*opt.steps,round(t*opt.steps),atol=1e-5) for t in self.reference['times']):
            raise ValueError('Reference and integration grid mismatch')
        for t in np.arange(opt.steps)/opt.steps:
            s=t+1/opt.steps
            if t>=a-1e-6 and s<=b+1e-6 and min(abs(np.asarray(self.reference['times'])-s))>2e-6:
                raise ValueError('Learning reference does not cover every controlled state')
        inp=Path(opt.input_dataset);inp=inp/'inputs' if (inp/'inputs').is_dir() else inp
        target=Path(opt.root)/'inputs';target.mkdir(parents=True,exist_ok=True)
        from .launcher import INPUT_FILES
        for name in INPUT_FILES:
            src,dst=inp/name,target/name
            if digest(src)!=self.reference['required_input_sha256'][name]:raise ValueError('Receptor/frame mismatch')
            if dst.exists() and digest(src)!=digest(dst):raise ValueError('Existing input mismatch')
            if not dst.exists():shutil.copy2(src,dst)
        torch.manual_seed(opt.seed);np.random.seed(opt.seed);random.seed(opt.seed)

    def configure(self,model,opt,out):
        if model.inpainting_mode or model.graph_inpainting or model._inpaint_self_condition:raise ValueError('Inpainting unsupported')
        self.model,self.opt,self.out=model,opt,out
        model.requires_grad_(False);model._gradient=self
        self.reward=WindowReward(self.program,self.reference)
        self.code_commit=subprocess.check_output(['git','-C',str(Path(__file__).resolve().parents[3]),'rev-parse','HEAD'],text=True).strip()
        self.checkpoint_hash=digest(opt.checkpoint)
        shutil.copy2(opt.program,out/'reward_program.json');shutil.copy2(opt.reference,out/'reference.json.gz')

    def generation_kwargs(self):return {'apply_guidance':False}

    def describe(self):
        return {'schema_version':'actual-window-control-1.0','code_commit':self.code_commit,
            'checkpoint_sha256':self.checkpoint_hash,'program_sha256':digest(self.opt.program),
            'reference_sha256':digest(self.opt.reference),'control_domain':self.program['window'],
            'evidence_domain':self.reference['window'],'gradient_target':'actual native proposal coordinates',
            'model_jacobian':'not needed for direct actual-state geometry reward',
            'apply_guidance':False,'particle_resampling':False,
            'native_categorical_sampling':True,'native_sde':True,'native_coord_noise_level':.2,
            'post_window_injection':False,'remote_role':'inference and lossless recording only'}

    def amend_config(self,c):
        c['selection_mode']='Disabled; native categorical sampling only; no particle selection or replication'
        c['full_probability_steps']=[]
        c['runtime_integrator']={'use_sde_simulation':True,'coord_noise_level':.2}

    def instrument(self,model,path):
        path=Path(path);path.mkdir(parents=True,exist_ok=True)
        original=model._generate_selective;source=textwrap.dedent(inspect.getsource(original))
        ns=dict(original.__func__.__globals__)
        def denied(**kwargs):raise RuntimeError('SMC invocation prohibited')
        ns['apply_selective_smc_guidance']=denied
        changed=no_selection_source(source)
        exec(compile(changed,str(path/'window_generate_selective.py'),'exec'),ns)
        model._generate_selective=types.MethodType(ns['_generate_selective'],model)
        (path/'upstream_generate_selective.py').write_text(source)
        (path/'window_generate_selective.py').write_text(changed)
        return original

    def make_trace(self,*args,**kw):
        from .controller import Trace
        class NoSelectionTrace(Trace):
            def choose(self,*args,**kwargs):raise RuntimeError('SMC forbidden')
            def score(self,pred,onoff):
                arm=self.arm
                try:self.arm='unguided';super().score(pred,onoff)
                finally:self.arm=arm
        trace=NoSelectionTrace(*args,**kw);trace.anchor_steps=set()
        return trace

    def predict(self,curr,pocket,times,cond,equis,invs):
        trace=self.model._lineage
        self.before=curr['coords'].detach().clone();self.pocket=pocket
        if trace.i==0:self.path_rms=torch.zeros(len(self.before),device=self.before.device)
        with torch.no_grad():
            return self.model._get_predictions(self.model(curr,pocket,times,cond_batch=cond,
                pocket_equis=equis,pocket_invs=invs,training=False))

    def after_native(self,curr):
        from .controller import nparr
        trace=self.model._lineage;scale=self.model.coord_scale;s=trace.t+trace.dt
        native=curr['coords']-self.before;actual=torch.zeros_like(native)
        enabled=trace.arm!='unguided' and self.reward.active(trace.t,s)
        row={'step':trace.i,'score_time':trace.t,'state_time':s,'active':enabled and trace.arm!='gradient_zero',
             'reward_evaluated':enabled,'arm':trace.arm,'particle_resampled':False}
        # Preserve native proposal separately; proposal_coords later records corrected state.
        trace.arr['native_proposal_coords'].append(nparr(curr['coords']))
        if enabled:
            mask=curr['mask'].bool()
            if not bool(mask.all()):raise ValueError('Fixed active point cloud required')
            com=torch.stack([torch.as_tensor(v.com) for v in self.pocket['complex']]).reshape(-1,3).to(native)
            with torch.enable_grad():
                x=curr['coords'].detach().clone().requires_grad_(True)
                atoms=curr['atomics'].detach().argmax(-1);bonds=curr['bonds'].detach().argmax(-1)
                value,detail=self.reward(x*scale+com[:,None,:],atoms,bonds,s)
                g,=torch.autograd.grad(value.sum(),x)
            if not bool(torch.isfinite(g).all()):raise ValueError('Nonfinite reward gradient')
            ratio=0. if trace.arm=='gradient_zero' else self.reward.ratio(s)
            c=self.program['constraints']
            proposed,control=native_relative_step(g,native,mask,ratio,scale,c['max_atom_step_A'],c['max_cumulative_rms_A']-self.path_rms)
            actual,guard=preserve_native_geometry(curr['coords'],proposed,mask,self.pocket['coords'],self.pocket['mask'],scale,c)
            rms=actual.square().sum((1,2)).div(mask.sum(1)).sqrt()*scale;self.path_rms+=rms
            row.update({k:v.detach().cpu().tolist() for k,v in {**detail,**control,**guard}.items()})
            row.update(reward=value.detach().cpu().tolist(),gradient_norm=g.norm(dim=(1,2)).cpu().tolist(),
                       injection_rms_A=rms.cpu().tolist(),requested_ratio=ratio,cumulative_rms_A=self.path_rms.cpu().tolist())
        if bool(actual.count_nonzero()):curr=dict(curr);curr['coords']=curr['coords']+actual
        row.update(injection_l2_A=(actual.norm(dim=(1,2))*scale).cpu().tolist(),
                   max_atom_injection_A=(actual.norm(dim=-1).amax(1)*scale).cpu().tolist())
        if abs(s-self.opt.window)<2e-6:
            np.savez_compressed(trace.path/'window_state.npz',coords=nparr(curr['coords']),
                atomics=nparr(curr['atomics'].argmax(-1)),bonds=nparr(curr['bonds'].argmax(-1)),
                mask=nparr(curr['mask']),state_time=s,coord_scale=scale)
        with (trace.path/'guidance_trace.jsonl').open('a') as f:f.write(json.dumps(clean(row),allow_nan=False)+'\n')
        return curr

    def final_metrics(self,model,output,trace,path):
        if self.opt.export_terminal:
            from .terminal_export import export_terminal
            return export_terminal(model,output,trace,path)
        rows=[{'arm':trace.arm,'batch':trace.batch,'slot':i,'seed':trace.seed,'build_success':False,
               'build_status':'not_evaluated_remote_inference_only'} for i in range(len(output['coords']))]
        write_json(path/'final_records.json',rows)
        return rows

    def summarize(self,rows):
        result={'n':len(rows),'chemical_build':'not_evaluated','role':'inference_only'}
        if self.opt.export_terminal:
            result.update(chemical_build='decoded',decoded=sum(r['build_success'] for r in rows),scientific_evaluation='deferred_local')
        return result

    def amend_batch_summary(self,summary):
        if self.opt.export_terminal:
            summary['chemical_build']='decoded; scientific evaluation deferred_local'
        else:
            summary['built']=None;summary['chemical_build']='not_evaluated_remote_inference_only'


def main():
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--flowr-root',required=True)
    a,remaining=p.parse_known_args();root=Path(a.flowr_root).resolve()
    if not (root/'flowr/models/fm_pocket.py').is_file():raise FileNotFoundError(root)
    sys.path.insert(0,str(root));import flowr
    if not Path(flowr.__file__).resolve().is_relative_to(root):raise RuntimeError('Wrong FLOWR import')
    from . import controller
    sys.argv=[sys.argv[0]]+remaining;controller.main(extension=WindowExtension())


if __name__=='__main__':main()
