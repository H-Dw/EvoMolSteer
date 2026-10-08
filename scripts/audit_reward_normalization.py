"""Offline reward-only precision audit; no FLOWR forward or future label access."""
import argparse,copy,gzip,json
from pathlib import Path
import numpy as np
import torch
from evomolsteer.io import read_json,write_json,digest
from evomolsteer.generation.endpoint_reward import EndpointGeometryReward
from evomolsteer.generation.selection_path_reward import SelectionPathReward


def audit(program,reference,output):
    p=read_json(program);r=json.loads(gzip.decompress(Path(reference).read_bytes()));spec=copy.deepcopy(p)
    spec['selection_path']={'legacy_prior_normalization':True};base=EndpointGeometryReward(p,r);new=SelectionPathReward(spec,r)
    rng=np.random.default_rng(42);results=[]
    for f in r['frames']:
        values=np.asarray(f['teacher_endpoint_A'])[:4];x=torch.tensor(values+rng.normal(0,.2,values.shape),dtype=torch.float32,requires_grad=True)
        mask=torch.ones(x.shape[:2],dtype=torch.bool);atoms=torch.zeros_like(mask,dtype=torch.long)
        a,_=base(x,atoms,mask,f['time'],x.detach());b,_=new(x,atoms,mask,f['time'],x.detach())
        ga,=torch.autograd.grad(a.sum(),x);gb,=torch.autograd.grad(b.sum(),x)
        results.append({'time':f['time'],'maximum_scalar_absolute_difference':float((a-b).detach().abs().max()),
          'maximum_gradient_absolute_difference':float((ga-gb).abs().max()),'relative_gradient_l2_difference':float((ga-gb).norm()/ga.norm().clamp_min(1e-12))})
    summary={'program_sha256':digest(program),'reference_sha256':digest(reference),'anchor_source':'Four existing teachers per actual time plus fixed seed-42 Gaussian coordinate jitter; no final labels.',
      'comparison':'R26 torch float32 log_softmax versus a nonempty zero-ESS-floor module using NumPy float64 prior normalization.',
      'results':results,'maximum_relative_gradient_l2_difference':max(v['relative_gradient_l2_difference'] for v in results),
      'limitations':['Empty module delegates R26 and is byte-exact in scalar/gradient tests.',
        'Nonempty zero-floor audit is mathematical near-equivalence only; categorical trajectories can amplify numerical differences.',
        'Full native-model outcome agreement must be assessed by the registered low-floor round, not this reward-only audit.'],
      'extra_FLOWR_forwards':0}
    write_json(output,summary);print({k:v for k,v in summary.items() if k!='results'});return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--program',required=True);p.add_argument('--reference',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();audit(a.program,a.reference,a.output)
