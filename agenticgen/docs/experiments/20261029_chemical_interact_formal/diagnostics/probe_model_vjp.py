"""Replay saved step-10 model inputs, without integration or new candidates."""
import argparse
import hashlib
import json
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
from agenticgen.uniguide.chunking import concurrent_map
from agenticgen.uniguide.random_streams import capture,equal


def sha_tensor(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def difference(a,b):return float((a.double()-b.double()).abs().max())


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();root=Path(args.root);folder=Path(args.output)
    folder.mkdir(parents=True,exist_ok=False)
    config=read(root/'GEOMETRY_CONFIG.json');config=dict(config,root=str(folder))
    subject=read(root/'SUBJECTS.json')[0];teacher,condition=subject['teacher'],subject['condition']
    base=root/'acceptance'/teacher['target']/'candidate_batch'
    report=dict(timestamp=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),teacher=teacher['target'],
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        torch_version=torch.__version__,hip_version=torch.version.hip,model_forward_calls=0,
        new_sampling_steps=0,new_candidates=0,llm_calls=0,cases=[])
    def record(row):
        report['cases'].append(row)
        (folder/'RESULTS.json').write_text(json.dumps(report,ensure_ascii=True,indent=2))
        print(json.dumps(row),flush=True)
    model=load_model(config,teacher);data=[]
    for i,slots in enumerate(([0,1],[6,7])):
        initial=initial_shard(config,teacher['target'],slots);state=load_pt(base/f'separate_{i}/checkpoint_010.pt.gz')
        current,cond,times=(device(state[k]) for k in ('current','cond','times'))
        pocket=device(initial['pocket_target']);com=initial['target_com'].to('cuda')
        with torch.no_grad():
            equis,invs=model.gen.get_pocket_encoding(pocket['coords'],pocket['atom_names'],
                pocket_atom_charges=pocket['charges'].argmax(-1),pocket_bond_types=pocket['bonds'].argmax(-1),
                pocket_res_types=pocket['res_names'],pocket_atom_mask=pocket['mask'])
        with h5py.File(base/f'separate_{i}/trace.h5') as f:
            saved={k:torch.from_numpy(f[k][10]) for k in ('endpoint_A','input_gradient')}
        data.append(dict(current=current,cond=cond,times=times,pocket=pocket,com=com,equis=equis,invs=invs,saved=saved))
    def graph(i):
        d=data[i];x=d['current']['coords'].detach().clone().requires_grad_(True)
        out=model(dict(d['current'],coords=x),d['pocket'],d['times'],cond_batch=d['cond'],
            pocket_equis=d['equis'],pocket_invs=d['invs'],training=False)
        pred,_=model._get_predictions(out)
        return x,pred['coords']*model.coord_scale+d['com'][:,None]
    def objective(endpoint,i):
        d=data[i]
        w=(1,0,1) if i==0 else (1,1,1)
        return evaluate(endpoint,condition,d['current']['mask'].bool(),dict(zip(KEYS,w)),
            current=d['current']['coords'].detach()*model.coord_scale+d['com'][:,None])[0][0]
    def once(i):
        with torch.enable_grad():
            x,endpoint=graph(i);v=objective(endpoint,i)
            grad=torch.autograd.grad(v.sum(),x)[0]
        return endpoint.detach().cpu(),grad.detach().cpu()
    try:
        for deterministic in (False,True):
            torch.use_deterministic_algorithms(deterministic)
            entry=capture()
            try:
                original=[once(i) for i in range(2)];report['model_forward_calls']+=2
                serial=[[once(i) for i in range(2)] for _ in range(3)];report['model_forward_calls']+=6
                parallel=[concurrent_map(once,[0,1],2) for _ in range(3)];report['model_forward_calls']+=6
                record(dict(mode='fresh_forward_and_vjp',deterministic=deterministic,rng_unchanged=equal(entry,capture()),
                    units=[dict(unit=i,saved_endpoint_difference=difference(original[i][0],data[i]['saved']['endpoint_A']),
                        saved_input_gradient_difference=difference(original[i][1],data[i]['saved']['input_gradient']),
                        serial_endpoint_max_difference=max(difference(v[i][0],original[i][0]) for v in serial),
                        parallel_endpoint_max_difference=max(difference(v[i][0],original[i][0]) for v in parallel),
                        serial_gradient_max_difference=max(difference(v[i][1],original[i][1]) for v in serial),
                        parallel_gradient_max_difference=max(difference(v[i][1],original[i][1]) for v in parallel)) for i in range(2)]))
            except Exception as error:record(dict(mode='fresh_forward_and_vjp',deterministic=deterministic,error=str(error)))
            try:
                x,endpoint=graph(0);report['model_forward_calls']+=1
                leaf=endpoint.detach().requires_grad_(True)
                upstream=torch.autograd.grad(objective(leaf,0).sum(),leaf)[0].detach()
                grads=[torch.autograd.grad(endpoint,x,grad_outputs=upstream,retain_graph=True)[0].detach().cpu() for _ in range(12)]
                record(dict(mode='same_graph_same_upstream_vjp',deterministic=deterministic,trials=12,
                    gradient_unique=len({sha_tensor(g) for g in grads}),maximum_repeat_difference=max(difference(g,grads[0]) for g in grads),
                    upstream_gradient_sha256=sha_tensor(upstream)))
                with torch.autograd.profiler.profile(use_cuda=False) as prof:
                    torch.autograd.grad(endpoint,x,grad_outputs=upstream,retain_graph=True)
                record(dict(mode='backward_operator_names',deterministic=deterministic,
                    operations=sorted({e.name for e in prof.function_events if any(t in e.name.lower() for t in ('backward','scatter','index','bmm'))})))
                del x,endpoint,leaf,upstream,grads
            except Exception as error:record(dict(mode='same_graph_same_upstream_vjp',deterministic=deterministic,error=str(error)))
    finally:torch.use_deterministic_algorithms(False)
    report['all_model_parameters_frozen']=all(not p.requires_grad and p.grad is None for p in model.parameters())
    (folder/'RESULTS.json').write_text(json.dumps(report,ensure_ascii=True,indent=2))


if __name__=='__main__':main()
