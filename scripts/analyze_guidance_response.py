"""Compact coordinate-dose/response and matched continuation diagnostics."""
import argparse,gzip,json
from pathlib import Path
import numpy as np
import torch
from evomolsteer.io import read_json,write_json

def analyze(dataset,campaign,output):
    root=Path(dataset)/'results'/campaign;rows=[];pairs=[];cfg=read_json(root/'config.json')
    for path in sorted(root.glob('*/batch_*')):
        trace=[json.loads(v) for v in (path/'guidance_trace.jsonl').read_text().splitlines()]
        active=[r for r in trace if r['active']]
        if active:
            dose=np.array([r['injection_rms_A'] for r in active]);noise=np.array([r['observed_native_rms_A'] for r in active])
            rows.append({'arm':path.parent.name,'batch':int(path.name[6:]),'active_steps':len(active),
                'mean_injection_A':float(dose.mean()),'mean_observed_native_A':float(noise.mean()),
                'mean_injection_over_native':float((dose/noise.clip(1e-30)).mean()),
                'mean_path_A':float(np.mean(active[-1]['cumulative_rms_A'])),
                'reward_gain_per_step':float(np.mean([r['reward_change'] for r in active])) if 'reward_change' in active[0] else None,
                'reward_improved_fraction':float(np.mean(np.array([r['reward_change'] for r in active])>0)) if 'reward_change' in active[0] else None,
                'first_order_reward_gain_per_step':float(np.mean([r['first_order_reward_change'] for r in active])) if 'first_order_reward_change' in active[0] else None,
                'reward_response_is_measured': 'reward_change' in active[0],
                'derivative_path':active[0].get('derivative_path','direct_proposal_geometry'),
                'no_post_window_injection':all(max(r['injection_l2_A'])==0 for r in trace if r['state_time']>cfg['experiment']['window']+1e-6)})
        if path.parent.name!='gradient':continue
        native=root/'unguided'/path.name
        if not native.exists():continue
        with np.load(path/'window_state.npz') as g,np.load(native/'window_state.npz') as n:
            delta=(g['coords']-n['coords'])*float(g['coord_scale']);mid=np.sqrt((delta**2).sum(2).mean(1))
        def final(folder):
            with gzip.open(folder/'final_prediction.pt.gz','rb') as f:return torch.load(f,map_location='cpu',weights_only=False)['coords'].numpy()
        delta=final(path)-final(native);end=np.sqrt((delta**2).sum(2).mean(1))
        scores=lambda folder:np.array([r['pic50_on_rescore'] for r in read_json(folder/'final_records.json')])
        gain=scores(path)-scores(native)
        pairs.append({'batch':int(path.name[6:]),'n':len(gain),'window_paired_RMS_A_mean':float(mid.mean()),
            'final_paired_RMS_A_mean':float(end.mean()),'final_over_window_RMS':float(end.mean()/mid.mean()) if mid.mean()>0 else None,
            'head_pair_delta_mean':float(gain.mean()),'head_pair_delta_abs_mean':float(np.abs(gain).mean()),
            'head_positive_fraction':float((gain>0).mean()),'head_pair_delta_p10':float(np.quantile(gain,.1)),
            'head_pair_delta_p90':float(np.quantile(gain,.9))})
    value={'schema_version':'coordinate-guidance-response-1.0','dose':rows,'matched_continuation':pairs,
        'additional_per_step_affinity_calls':0,'affinity_head_gradient':False,
        'interpretation':'Slot-paired perturbation diagnostics; categorical transitions can amplify differences. RMS ratio alone does not prove damping.'}
    write_json(output,value);return value
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('dataset','campaign','output'):p.add_argument('--'+k,required=True)
    a=p.parse_args();analyze(a.dataset,a.campaign,a.output)
