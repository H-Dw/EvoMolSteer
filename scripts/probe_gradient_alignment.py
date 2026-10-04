"""Local causal-direction diagnostic on trusted, hash-audited FLOWR checkpoints.

No generation and no reward update. Compare the live direction that decreases a
geometric feature with the live affinity-head gradient at the SAME saved state.
"""
import argparse
import gzip
import json
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser();p.add_argument('--flowr-root',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--campaign',required=True);p.add_argument('--arm',default='unguided');p.add_argument('--batch',type=int,default=0)
    p.add_argument('--steps',default='0,10');p.add_argument('--output',required=True);a=p.parse_args()
    sys.path.insert(0,str(Path(a.flowr_root).resolve()))
    import numpy as np
    import torch
    from evomolsteer.generation.controller import load_model
    from evomolsteer.generation.scalar_guidance import RegionalReward
    from evomolsteer.storage.trajectory import TrajectoryPackage
    from evomolsteer.io import read_json,write_json,write_table,digest
    root=Path(a.campaign);cfg=read_json(root/'config.json');directory=root/a.arm/f'batch_{a.batch:03d}'
    options=argparse.Namespace(**cfg['flowr_args']);options.ckpt_path=a.checkpoint
    model,*_=load_model(options);model=model.to('cuda').eval().requires_grad_(False)
    torch.set_float32_matmul_precision('high')
    def device(v):
        if torch.is_tensor(v):return v.to('cuda')
        if isinstance(v,dict):return {k:device(w) for k,w in v.items()}
        if isinstance(v,list):return [device(w) for w in v]
        return v
    # These files were created by this project's own controller and are verified
    # by the generation archive manifest. They contain NumPy RNG objects.
    with gzip.open(directory/'initial_state.pt.gz','rb') as f:
        initial=torch.load(f,map_location='cpu',weights_only=False)
    pocket=device(initial['pocket_target']);com=torch.as_tensor(initial['target_com'],device='cuda',dtype=torch.float32).reshape(-1,3)
    runtime=RegionalReward(read_json(root/'reward_program.json'),read_json(root/'reward_catalog.json'))
    with torch.no_grad():
        equis,invs=model.gen.get_pocket_encoding(pocket['coords'],pocket['atom_names'],
            pocket_atom_charges=pocket['charges'].argmax(-1),pocket_bond_types=pocket['bonds'].argmax(-1),
            pocket_res_types=pocket['res_names'],pocket_atom_mask=pocket['mask'])
    records=[];verification=[]
    with TrajectoryPackage(directory/'trajectory.h5') as trajectory:
        for step in map(int,a.steps.split(',')):
            source=directory/f'restart_step_{step:03d}.pt.gz'
            with gzip.open(source,'rb') as f:state=device(torch.load(f,map_location='cpu',weights_only=False))
            current=state['current'];x=current['coords'].detach().clone().requires_grad_(True)
            def forward(coords):
                row=dict(current);row['coords']=coords
                out=model(row,pocket,state['times'],training=False,cond_batch=state['cond'],pocket_equis=equis,pocket_invs=invs)
                return model._get_predictions(out)[0]
            pred=forward(x);affinity=pred['affinity']['pic50'].reshape(-1)
            expected=trajectory.read('predicted_coords',step);scores=trajectory.read('pic50_on',step)
            coord_error=float(np.abs(pred['coords'].detach().cpu().numpy()-expected).max())
            score_error=float(np.abs(affinity.detach().cpu().numpy()-scores).max())
            if max(coord_error,score_error)>1e-5:raise ValueError('Saved-state forward reconstruction mismatch')
            ga,=torch.autograd.grad(affinity.sum(),x,retain_graph=True)
            mask=current['mask'].bool();ga=ga*mask[...,None]
            for term in runtime.program['terms']:
                values=runtime.observable(term['feature'],pred['coords']*model.coord_scale+com[:,None,:],mask)
                gg,=torch.autograd.grad(-values.sum(),x,retain_graph=True)
                gg=gg*mask[...,None];norm=gg.norm(dim=(1,2));anorm=ga.norm(dim=(1,2));dot=(gg*ga).sum((1,2))
                cos=dot/(norm*anorm).clamp_min(1e-20)
                for slot in range(len(x)):
                    records.append({'step':step,'score_time':float(state['times'][0][slot]),'slot':slot,'term_id':term['id'],
                        'feature_value':float(values[slot].detach()),'above_stopping_target':bool(values[slot]>term['target']),
                        'affinity':float(affinity[slot].detach()),'affinity_grad_norm':float(anorm[slot]),
                        'decreasing_feature_grad_norm':float(norm[slot]),'cosine':float(cos[slot]),
                        'affinity_derivative_per_unit_native_L2':float(dot[slot]/norm[slot].clamp_min(1e-20))})
            verification.append({'step':step,'checkpoint_sha256':digest(source),'max_endpoint_error':coord_error,'max_affinity_error':score_error})
            del pred,ga,x,affinity,values,gg
    out=Path(a.output);write_table(out/'feature_affinity_alignment.csv',records)
    write_json(out/'alignment_provenance.json',{'status':'completed','campaign':str(root),'arm':a.arm,'batch':a.batch,
        'verification':verification,'model_checkpoint_sha256':digest(a.checkpoint),
        'meaning':'Live local feature-decreasing derivative versus live affinity derivative; ungated, unweighted geometry direction.',
        'limits':['Not final affinity efficacy. Discrete mutations and later dynamics are not differentiated.',
                  'Only particles above the stopping target would respond to the one-sided reward, and only during its gate.',
                  't=.1 compactness gate starts at zero; its raw direction is probed to assess the upcoming hypothesis.',
                  'No reward parameters are updated from these diagnostic outcomes.']})
    print('Alignment rows',len(records),'forward reconstruction',verification)


if __name__=='__main__':main()
