"""Compare actual candidate safety-guard graphs at the frozen first guided step."""
import argparse
import hashlib
import json
import threading
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import h5py
import torch
from agenticgen.uniguide.io import read
from agenticgen.uniguide.native_tools import load_pt,device
from agenticgen.uniguide.prepare import load_model
from agenticgen.uniguide.geometry_shards import initial_shard
from agenticgen.uniguide.geometry_reward import evaluate,KEYS
from agenticgen.uniguide.geometry_batch import CandidateControl


def diff(a,b):return float((a.detach().cpu().double()-b.detach().cpu().double()).abs().max())


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    p.add_argument('--site-probe',action='store_true',help='Isolate each loss guard and a diagnostic healthy-row fast path')
    args=p.parse_args();root=Path(args.root);folder=Path(args.output);folder.mkdir(parents=True,exist_ok=False)
    config=dict(read(root/'GEOMETRY_CONFIG.json'),root=str(folder));subject=read(root/'SUBJECTS.json')[0]
    teacher,condition=subject['teacher'],subject['condition'];base=root/'acceptance'/teacher['target']/'candidate_batch'
    report=dict(timestamp=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        new_candidates=0,new_sampling_steps=0,llm_calls=0,model_forward_calls=0,site_probe=args.site_probe,cases=[])
    model=load_model(config,teacher)
    guard=object.__new__(CandidateControl);guard.initial={'population':20};guard.failed_units={}
    guard.failure_lock=threading.RLock();guard.active_step=10;guard.folder=folder
    try:
        for i,slots in enumerate(([0,1],[6,7])):
            initial=initial_shard(config,teacher['target'],slots);state=load_pt(base/f'separate_{i}/checkpoint_010.pt.gz')
            curr,cond,times=(device(state[k]) for k in ('current','cond','times'))
            pocket=device(initial['pocket_target']);com=initial['target_com'].to('cuda');ids=list(range(i*10,(i+1)*10))
            w=dict(zip(KEYS,(1,0,1) if i==0 else (1,1,1)))
            with torch.no_grad():
                eq,iv=model.gen.get_pocket_encoding(pocket['coords'],pocket['atom_names'],pocket_atom_charges=pocket['charges'].argmax(-1),
                    pocket_bond_types=pocket['bonds'].argmax(-1),pocket_res_types=pocket['res_names'],pocket_atom_mask=pocket['mask'])
            with h5py.File(base/f'separate_{i}/trace.h5') as a,h5py.File(base/'mixed_twenty/trace.h5') as b:
                saved_single=torch.from_numpy(a['input_gradient'][10]);saved_mixed=torch.from_numpy(b['input_gradient'][10,i*10:(i+1)*10])
            baseline={}
            sites={'distance_endpoint_only':'distance endpoint','internal_errors_only':'internal distance errors',
                'interaction_errors_only':'interaction distance errors','total_guard_only':'normalized distance loss'}
            modes=('plain','loss_guards_only',*sites,'healthy_fastpath') if args.site_probe else (
                'plain','full_candidate_guards','loss_guards_only','network_guards_only','endpoint_guard_only')
            for det in ((False,) if args.site_probe else (False,True)):
                torch.use_deterministic_algorithms(det)
                for mode in modes:
                    x=curr['coords'].detach().clone().requires_grad_(True)
                    out=model(dict(curr,coords=x),pocket,times,cond_batch=cond,pocket_equis=eq,pocket_invs=iv,training=False)
                    report['model_forward_calls']+=1
                    if mode in ('full_candidate_guards','network_guards_only'):out=guard.prepare_output(out,ids)
                    pred,_=model._get_predictions(out)
                    endpoint=pred['coords']*model.coord_scale+com[:,None]
                    if mode in ('full_candidate_guards','endpoint_guard_only'):endpoint=guard.check_rows('predicted endpoint',endpoint,ids)
                    check=(lambda name,v:guard.check_rows(name,v,ids)) if mode in ('full_candidate_guards','loss_guards_only') else None
                    if mode in sites:
                        check=lambda name,v:guard.check_rows(name,v,ids) if name==sites[mode] else v
                    elif mode=='healthy_fastpath':
                        def check(name,v):
                            guarded=guard.check_rows(name,v,ids)
                            return v if not any(j//10 in guard.failed_units for j in ids) else guarded
                    value=evaluate(endpoint,condition,curr['mask'].bool(),[w]*10 if mode=='full_candidate_guards' else w,
                        current=curr['coords'].detach()*model.coord_scale+com[:,None],check=check)[0][0]
                    if mode=='full_candidate_guards':value=guard.check_rows('loss',value,ids)
                    upstream=torch.autograd.grad(value.sum(),endpoint,retain_graph=True)[0]
                    gradient=torch.autograd.grad(value.sum(),x)[0]
                    if mode=='plain':baseline[det]=(endpoint.detach(),value.detach(),upstream.detach(),gradient.detach())
                    e,v,u,g=baseline[det]
                    row=dict(unit=i,deterministic_algorithms=det,mode=mode,endpoint_difference=diff(endpoint,e),loss_difference=diff(value,v),
                        upstream_gradient_difference=diff(upstream,u),input_gradient_difference=diff(gradient,g),
                        saved_single_gradient_difference=diff(gradient,saved_single),saved_mixed_gradient_difference=diff(gradient,saved_mixed),
                        failed_units=guard.failed_units)
                    report['cases'].append(row);print(json.dumps(row),flush=True)
                    (folder/'RESULTS.json').write_text(json.dumps(report,ensure_ascii=True,indent=2))
    finally:torch.use_deterministic_algorithms(False)


if __name__=='__main__':main()
