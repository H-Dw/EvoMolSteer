"""Fixed saved endpoints only: isolate distance-loss gradients; never sample."""
import argparse
import collections
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import h5py
import torch
from agenticgen.uniguide.geometry_reward import evaluate, denominator, KEYS
from agenticgen.uniguide.chunking import concurrent_map


def digest(x):return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();root=Path(args.root);output=Path(args.output)
    if output.exists():raise ValueError('Preserve prior diagnostic results')
    subject=json.loads((root/'SUBJECTS.json').read_text())[0];condition=subject['condition']
    base=root/'acceptance'/subject['teacher']['target']/'candidate_batch'
    data=[]
    for i in range(2):
        with h5py.File(base/f'separate_{i}/trace.h5') as f:
            data.append({k:torch.from_numpy(f[name][10]) for k,name in [('endpoint','endpoint_A'),('current','current_A'),('valid','mask')]})
    torch.set_num_threads(1)
    rows=condition['interaction_pairs'];counts=collections.Counter(p['generated_slot'] for p in rows)
    report=dict(timestamp=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),torch_version=torch.__version__,
        hip_version=torch.version.hip,device=torch.cuda.get_device_name(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        condition_sha256=condition['condition_sha256'],pair_count=len(rows),slot_multiplicity=dict(counts),
        mathematical_duplicate_pairs=len(rows)-len({(p['atom_id'],p['target_atom_id']) for p in rows}),
        new_model_calls=0,new_candidates=0,llm_calls=0,trials=12,cases=[])
    modes=['production_active','production_intra','production_interact','grouped_interact','unique_interact']
    reference={}

    def operation(d,mode):
        x=d['endpoint'].detach().clone().requires_grad_(True)
        if mode.startswith('production_'):
            w={'production_active':(1,0,1),'production_intra':(1,0,0),'production_interact':(0,0,1)}[mode]
            value=evaluate(x,condition,d['valid'].bool(),dict(zip(KEYS,w)),current=d['current'])[0][0]
        else:
            scale=denominator(d['current'],d['valid'],condition,condition['distance_normalization'])
            selected=rows
            if mode=='unique_interact':
                selected=[next(p for p in rows if p['generated_slot']==s) for s in counts]
            parts=[]
            for slot in sorted({p['generated_slot'] for p in selected}):
                group=[p for p in selected if p['generated_slot']==slot]
                target=x.new_tensor([p['target_coordinates_angstrom'] for p in group])
                expected=x.new_tensor([p['reference_distance_angstrom'] for p in group])
                delta=torch.linalg.vector_norm(x[:,slot:slot+1,:]-target,dim=-1)-expected
                parts.append(delta.square().sum(-1))
            value=torch.stack(parts,-1).sum(-1)/(2*len(selected)*scale.square())
        gradient=torch.autograd.grad(value.sum(),x)[0]
        return value.detach().cpu(),gradient.detach().cpu()

    try:
        for device,dtype,det in [('cpu',torch.float64,False),('cuda',torch.float32,False),('cuda',torch.float32,True),('cuda',torch.float64,False)]:
            torch.use_deterministic_algorithms(det)
            tensors=[{k:v.to(device=device,dtype=dtype if k!='valid' else v.dtype) for k,v in d.items()} for d in data]
            for mode in modes:
                record=dict(device=device,dtype=str(dtype),deterministic_algorithms=det,mode=mode)
                try:
                    observed=[operation(tensors[0],mode) for _ in range(12)]
                    first=observed[0];reference[(device,str(dtype),mode)]=first
                    record.update(loss_unique=len({digest(p[0]) for p in observed}),gradient_unique=len({digest(p[1]) for p in observed}),
                        maximum_repeat_gradient_difference=max(float((p[1].double()-first[1].double()).abs().max()) for p in observed),
                        maximum_absolute_gradient=float(first[1].abs().max()),finite=bool(torch.isfinite(first[1]).all()))
                    if device=='cuda':
                        paired=[concurrent_map(lambda i:operation(tensors[i],mode),[0,1],2)[0] for _ in range(12)]
                        record.update(parallel_gradient_unique=len({digest(p[1]) for p in paired}),
                            maximum_parallel_vs_serial_difference=max(float((p[1].double()-first[1].double()).abs().max()) for p in paired))
                    cpu=reference[('cpu',str(torch.float64),mode)][1]
                    record['maximum_gradient_difference_from_cpu_float64']=float((first[1].double()-cpu).abs().max())
                    if mode=='grouped_interact':
                        original=reference[(device,str(dtype),'production_interact')]
                        record.update(equivalent_loss_maximum_difference=float((first[0]-original[0]).abs().max()),
                            equivalent_gradient_maximum_difference=float((first[1]-original[1]).abs().max()))
                except Exception as error:record['error']=str(error)
                report['cases'].append(record)
                print(json.dumps(record),flush=True)
    finally:torch.use_deterministic_algorithms(False)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as f:json.dump(report,f,ensure_ascii=True,indent=2)


if __name__=='__main__':main()
