"""Inference-side decoding and affinity-head export; scientific evaluation is local."""
import numpy as np
import torch
from rdkit import Chem
from ..io import write_json


def export_terminal(model,output,trace,path):
    device=trace.pt['coords'].device;b=len(output['coords'])
    final={k:v.to(device) for k,v in output.items() if torch.is_tensor(v) and k!='affinity'}
    com=torch.stack([torch.as_tensor(s.com) for s in trace.pt['complex']]).reshape(-1,3).to(device)
    final['coords']=(final['coords']-com[:,None,:])/model.coord_scale
    for k in ['atomics','bonds','charges','hybridization']:
        if k in final:final[k]=torch.nn.functional.one_hot(final[k].argmax(-1),final[k].shape[-1]).float()
    times=[torch.full((b,),.9999,device=device) for _ in range(3)]
    with torch.no_grad():
        scores=[model._predict_affinity(final,final,p,times)['affinity']['pic50'].detach().cpu().reshape(-1).numpy() for p in [trace.pt,trace.po]]
    mols=model._generate_mols(output);raw=model._generate_mols(output,sanitise=False)
    if len(mols)!=b or len(raw)!=b:raise ValueError('Decoder dropped slots')
    rows=[]
    with Chem.SDWriter(str(path/'molecules_all_built.sdf')) as writer, Chem.SDWriter(str(path/'molecules_raw_decodable.sdf')) as rawwriter:
        for slot,(m,r) in enumerate(zip(mols,raw)):
            node=f'{trace.arm}_b{trace.batch:03d}_final_{slot:03d}'
            row={'arm':trace.arm,'batch':trace.batch,'slot':slot,'seed':trace.seed,'node_id':node,
                 'root_slot':int(trace.roots[slot]),'build_success':m is not None,
                 'decode_status':'built' if m is not None else 'decoder_returned_none',
                 'pic50_on_rescore':float(scores[0][slot]),'pic50_off_rescore':float(scores[1][slot]),
                 'gap_rescore':float(scores[0][slot]-scores[1][slot]),
                 'pic50_on_upstream':float(output['affinity']['pic50'][slot]),
                 'pic50_off_upstream':float(output['affinity']['pic50_untarget'][slot]),
                 'affinity_semantics':'FLOWR head prediction, not measured affinity',
                 'structure_evaluation':'deferred_local'}
            for molecule,sink in [(m,writer),(r,rawwriter)]:
                if molecule is not None:
                    molecule.SetProp('node_id',node);molecule.SetIntProp('slot',slot);sink.write(molecule)
            rows.append(row)
    write_json(path/'final_records.json',rows)
    return rows
