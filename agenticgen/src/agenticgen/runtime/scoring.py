"""Unchanged native terminal scoring; affinity is not differentiated."""
import json
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import QED, Descriptors, rdMolDescriptors

def nparr(x): return x.detach().cpu().numpy()

def final_metrics(model,output,trace,path):
    # Preserve upstream scores and supply separately named endpoint-rescore scores.
    # Both rescores see identical decoded categorical state and endpoint coordinates.
    device=trace.pt['coords'].device
    b=output['coords'].shape[0]
    final={k:v.to(device) for k,v in output.items() if torch.is_tensor(v) and k!='affinity'}
    com=torch.stack([torch.as_tensor(s.com) for s in trace.pt['complex']]).reshape(-1,3).to(device)
    final['coords']=(final['coords']-com[:,None,:])/model.coord_scale
    for k in ['atomics','bonds','charges','hybridization']:
        if k in final: final[k]=torch.nn.functional.one_hot(final[k].argmax(-1),final[k].shape[-1]).float()
    times=[torch.full((b,),.9999,device=device) for _ in range(3)]
    with torch.no_grad():
        scores=[]
        for pocket in [trace.pt,trace.po]:
            o=model._predict_affinity(final,final,pocket,times)
            scores.append(nparr(o['affinity']['pic50'].squeeze(-1)))
    mols=model._generate_mols(output)
    assert len(mols)==b
    rows=[]; writer=Chem.SDWriter(str(path/'molecules_all_built.sdf'))
    rawwriter=Chem.SDWriter(str(path/'molecules_raw_decodable.sdf'))
    raw_mols=model._generate_mols(output,sanitise=False)
    for slot,m in enumerate(mols):
        node=f'{trace.arm}_b{trace.batch:03d}_final_{slot:03d}'
        r={'arm':trace.arm,'batch':trace.batch,'seed':trace.seed,'slot':slot,'node_id':node,
           'root_slot':int(trace.roots[slot]),'parent_step':len(trace.events)-1,'parent_slot':int(trace.parents[slot]),
           'pic50_on_upstream':float(output['affinity']['pic50'][slot]),
           'pic50_off_upstream':float(output['affinity']['pic50_untarget'][slot]),
           'pic50_on_rescore':float(scores[0][slot]),'pic50_off_rescore':float(scores[1][slot]),
           'build_success':m is not None,'failure_reason':'','smiles':''}
        r['gap_upstream']=r['pic50_on_upstream']-r['pic50_off_upstream']
        r['gap_rescore']=r['pic50_on_rescore']-r['pic50_off_rescore']
        raw=raw_mols[slot]
        if raw is not None:
            raw.SetProp('node_id',node); rawwriter.write(raw)
        if m is None:
            r['failure_reason']='builder_returned_none'
            if raw is not None:
                try: Chem.SanitizeMol(Chem.Mol(raw))
                except Exception as e: r['failure_reason']=type(e).__name__+': '+str(e)
        else:
            try:
                Chem.SanitizeMol(m); r['smiles']=Chem.MolToSmiles(m)
                r['connected']=len(Chem.GetMolFrags(m))==1
                r['qed']=float(QED.qed(m)); r['mw']=float(Descriptors.MolWt(m)); r['heavy_atoms']=m.GetNumHeavyAtoms()
                r['rings']=int(rdMolDescriptors.CalcNumRings(m))
                xyz=m.GetConformer().GetPositions()
                pc=nparr(trace.pt['coords'][slot][trace.pt['mask'][slot].bool()])*model.coord_scale+nparr(com[slot])
                d=np.linalg.norm(xyz[:,None,:]-pc[None,:,:],axis=-1)
                r['min_protein_distance_A']=float(d.min()); r['pairs_below_1_2A']=int((d<1.2).sum())
            except Exception as e:
                r['failure_reason']=type(e).__name__+': '+str(e)
            for k,v in r.items(): m.SetProp(k,str(v))
            writer.write(m)
        rows.append(r)
    writer.close(); rawwriter.close()
    (path/'final_records.json').write_text(json.dumps(rows,indent=2))
    return rows
