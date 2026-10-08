"""Offline endpoint-scalar response audit; no FLOWR/head inference or new model."""
import argparse,gzip,json
from pathlib import Path
import numpy as np
import torch
from evomolsteer.io import read_json,write_json,digest
from evomolsteer.trajectory_source import open_trajectory
from evomolsteer.generation.endpoint_reward import EndpointGeometryReward
from evomolsteer.generation.dynamic_region_reward import DynamicRegionReward


def audit(dataset,incumbent_program,reference,output,batch=14):
    root=Path(dataset);source=root/'results/main1000_w050';cfg=read_json(source/'config.json')
    base=read_json(incumbent_program);ref=json.loads(gzip.decompress(Path(reference).read_bytes()))
    base['reference_sha256']=digest(reference);rows=[]
    com=np.asarray(read_json(source/f'frame_batch_{batch:03d}.json')['target_com'])[:,None,:]
    with open_trajectory(source/'single'/f'batch_{batch:03d}/trajectory.npz') as z:
        times=np.round(z['score_time'][:,0].astype(float),6);xyz=np.asarray(z['predicted_coords'],float)*cfg['coord_scale']+com
        for time in [ref['times'][0],ref['times'][len(ref['times'])//2],ref['times'][-1]]:
            i=int(np.flatnonzero(times==time)[0]);x=torch.tensor(xyz[i],dtype=torch.float64,requires_grad=True)
            mask=torch.tensor(z['mask'][i]);atoms=torch.zeros(mask.shape,dtype=torch.long)
            b=EndpointGeometryReward(base,ref);v,_=b(x,atoms,mask,time,x.detach());g,=torch.autograd.grad(v.sum(),x)
            for weight in [.05,.15,.40]:
                p={**base,'reward_view':'endpoint_dynamic_region','regional_weight':weight}
                r=DynamicRegionReward(p,ref);value,_=r(x,atoms,mask,time,x.detach());h,=torch.autograd.grad(value.sum(),x)
                cosine=(g*h).sum((1,2))/(g.norm(dim=(1,2))*h.norm(dim=(1,2))).clamp_min(1e-30)
                ratio=(h-g).norm(dim=(1,2))/g.norm(dim=(1,2)).clamp_min(1e-30)
                rows.append({'time':time,'regional_weight':weight,'mean_gradient_cosine_with_incumbent':float(cosine.mean()),
                    'median_added_gradient_ratio':float(ratio.median()),'changed_direction_fraction':float((cosine<.99).double().mean()),
                    'finite_nonzero_gradients':bool(torch.isfinite(h).all() and (h.norm(dim=(1,2))>0).all())})
    result={'schema_version':'dynamic-regional-offline-response-1.0','batch':batch,'reference_sha256':digest(reference),
        'derivative_domain':'Endpoint geometry only; this is not a real FLOWR Jacobian audit or an independent affinity outcome',
        'rows':rows,'production_neural_calls':0};write_json(output,result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['dataset','incumbent-program','reference','output']:p.add_argument('--'+name,required=True)
    p.add_argument('--batch',type=int,default=14);a=p.parse_args();print(json.dumps(audit(a.dataset,a.incumbent_program,a.reference,a.output,a.batch)))
