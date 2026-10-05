"""Local historical SMC-to-SMC context, evaluated only after reward freezing.

Uses untouched historical batches as an interpretation reference, never tuning
input. This does not generate or run a new Steer campaign.
"""
import argparse
from pathlib import Path
import numpy as np
from evomolsteer.generation.window_evaluation import state,distances
from evomolsteer.io import write_json


def benchmark(dataset,campaign,time,output):
    root=Path(dataset)/'results'/campaign
    ref=[state(root,'single',i,time) for i in range(14)]
    y,ya,yb=[np.concatenate([r[k] for r in ref]) for k in range(3)]
    rows=[]
    for batch in range(14,20):
        full=state(root,'single',batch,time);x,a,b=[v[:16] for v in full]
        shape,typed,bond,union=distances(x,a,b,y,ya,yb);nearest=shape.argmin(1);ids=np.arange(len(x))
        rows.append({'batch':batch,'n':len(x),'generated_to_reference_A':float(shape.min(1).mean()),
                     'reference_to_generated_A':float(shape.min(0).mean()),
                     'symmetric_shape_A':float((shape.min(1).mean()+shape.min(0).mean())/2),
                     'atom_mismatch_geometry_assignment':float(typed[ids,nearest].mean()),
                     'bonded_union_mismatch':float(union[ids,nearest].mean())})
    result={'scope':'Historical SMC batches14-19, first16 slots per batch vs discovery batches0-13; after reward freeze; no tuning and no new generation',
            'time':time,'batch_results':rows,'means':{k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k not in ['batch','n']},
            'interpretation':'High label mismatch under purely geometric atom correspondence is not, by itself, evidence of chemical invalidity or failure to match a stochastic intermediate distribution.'}
    write_json(output,result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--campaign',required=True)
    p.add_argument('--time',type=float,required=True);p.add_argument('--output',required=True);a=p.parse_args()
    print(benchmark(a.dataset,a.campaign,a.time,a.output)['means'])
