"""Native input conversion; identical to established engineering implementation."""
import torch
def nparr(x): return x.detach().cpu().numpy()

def make_inputs(model,batch_t,batch_o,device):
    prior,pt,_,_=batch_t; _,po,_,_=batch_o
    lig=model.builder.extract_ligand_from_complex(prior)
    for k in ['interactions','fragment_mask','fragment_mode']: lig[k]=prior[k]
    def pocket(p):
        d=model.builder.extract_pocket_from_complex(p)
        d['interactions']=p['interactions']; d['complex']=p['complex']
        return {k:v.to(device) if torch.is_tensor(v) else v for k,v in d.items()}
    lig={k:v.to(device) if torch.is_tensor(v) else v for k,v in lig.items()}
    target,off=pocket(pt),pocket(po)
    ct=torch.stack([torch.as_tensor(s.com) for s in target['complex']]).reshape(-1,3).to(device)
    co=torch.stack([torch.as_tensor(s.com) for s in off['complex']]).reshape(-1,3).to(device)
    shift=(co-ct)/model.coord_scale
    off['coords']=off['coords']+shift[:,None,:]*off['mask'][...,None]
    return lig,target,off,{'target_com':nparr(ct).tolist(),'off_com':nparr(co).tolist(),'off_internal_shift':nparr(shift).tolist()}
