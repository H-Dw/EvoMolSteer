"""Short original-dt replays to diagnose cross-device FP32 arithmetic.

The preview stops through the observer after real historical random draws; it
does not alter the 100-step integrator dt and is never a generation result.
"""
import argparse
import json
from pathlib import Path
import torch

from evomolsteer.io import read_json
from evomolsteer.storage.transactions import atomic_json
from .worker import load_model,attach,make_control
from .native import InitialPool,HistoricalObserver
from .tape import ReplayRandom


class PreviewComplete(Exception):pass


def run(config,work,output,steps=5,group=0):
    if not 1<=steps<=20:raise ValueError('Diagnostic preview must remain small')
    cfg=read_json(config);work=Path(work);model=load_model(cfg,work);attach(model)
    by_id={a['id']:a for a in cfg['attempts']};attempts=[by_id[k] for k in cfg['groups'][group]['ids']]
    reports=[];states=[]
    for precision in ('high','highest','high'):
        torch.set_float32_matmul_precision(precision)
        pool=InitialPool(cfg);prior,pocket,com,scale=pool.load(attempts)
        model._path_pocket_context=pool.encoding(model,attempts)
        replay=ReplayRandom(cfg,work,attempts);model._path_random=replay
        observer=HistoricalObserver(dict(cfg,_coord_scale=scale),work,attempts)
        def observe(i,curr,pred):
            observer(i,curr,pred)
            if i+1==steps:
                states.append(curr['coords'].detach().cpu().clone());raise PreviewComplete()
        model._path_observer=observe
        model._half_guidance=make_control(model,{'kind':'unguided'},None,None,com)
        try:
            with replay.activate():
                model._generate(prior,pocket,steps=100,times=[torch.zeros(50,device='cuda') for _ in range(3)],
                    apply_guidance=False,strategy='linear',solver='euler',save_traj=False,
                    coord_noise_level=cfg['native_options']['coord_noise_level'],final_corr_pred=True)
        except PreviewComplete:pass
        reports.append({'precision':precision,'observer':observer.summary(),'random_steps':len(replay.records)})
        model._half_guidance=None;model._path_random=None;model._path_observer=None;model._path_pocket_context=None
        del pool,observer,replay,prior,pocket;torch.cuda.empty_cache()
    out={'device':torch.cuda.get_device_name(0),'hip':torch.version.hip,'torch':torch.__version__,
         'group':group,'steps':steps,'reports':reports,
         'high_repeat_max_model_unit_difference':float((states[0]-states[2]).abs().max()),
         'precision_change_max_model_unit_difference':float((states[0]-states[1]).abs().max()),
         'claim':'Diagnostic only. Neither changed precision nor a short preview is an approved exact historical replay.'}
    atomic_json(output,out);print(json.dumps(out,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('config','work','output'):p.add_argument('--'+k,required=True)
    p.add_argument('--steps',type=int,default=5);a=p.parse_args();run(a.config,a.work,a.output,a.steps)


if __name__=='__main__':main()
