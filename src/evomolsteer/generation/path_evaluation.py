"""Compact execution and paired terminal-tail evidence for sequential exploration."""
import json,hashlib,shutil
from pathlib import Path
import numpy as np
import pandas as pd
from ..io import read_json,digest,write_json
from ..trajectory_source import open_trajectory

def execution_audit(dataset,campaign):
    root=Path(dataset)/'results'/campaign;cfg=read_json(root/'config.json');p=read_json(root/'reward_program.json')
    if read_json(root/'COMPLETE.json')['status']!='complete' or cfg['experiment']['seed']!=42:
        raise ValueError('Complete seed-42 experiment required')
    a,b=p['window'];rows=[];sources=[];signatures={}
    for arm in cfg['experiment']['arms'].split(','):
        for folder in sorted((root/arm).glob('batch_*')):
            batch=int(folder.name.split('_')[-1]);trace=[json.loads(s) for s in (folder/'guidance_trace.jsonl').read_text().splitlines()]
            if [v['step'] for v in trace]!=list(range(100)):raise ValueError('Full native continuation missing')
            if any(v['particle_resampled'] for v in trace):raise ValueError('Resampling prohibited')
            if any(max(v['injection_l2_A'])>0 for v in trace if v['score_time']<a-2e-6 or v['state_time']>b+2e-6):
                raise ValueError('Injection outside learned support')
            if any(v.get('production_target_forward_calls')!=1 or not v.get('affinity_outputs_detached') for v in trace):
                raise ValueError('One production forward and detached head required')
            with open_trajectory(folder/'trajectory.h5') as tr:
                n=tr['selected_indices'].shape[1]
                if np.asarray(tr['resampled']).any() or not np.array_equal(tr['selected_indices'],np.tile(np.arange(n),(100,1))):
                    raise ValueError('Recorded selection contract violated')
                sig=hashlib.sha256()
                for field in ['current_coords','current_atomics','current_bonds']:sig.update(tr[field][0].tobytes())
                signatures[f'{arm}/{batch}']=sig.hexdigest()
            active=[v for v in trace if v['reward_evaluated']]
            rows.append({'arm':arm,'batch':batch,'n':n,'controlled_steps':len(active),
                'nonzero_steps':sum(max(v['injection_l2_A'])>0 for v in trace),
                'mean_cumulative_rms_A':float(np.mean(active[-1]['cumulative_rms_A'])) if active else 0.,
                'mean_active_injection_rms_A':float(np.mean([v['injection_rms_A'] for v in active])) if active else 0.})
            sources.append({'batch':batch,'arm':arm,'sha256':digest(folder/'trajectory.h5')})
    preflight=read_json(root/'coordinate_gradient_preflight.json')
    if not preflight['passed'] or preflight['affinity_head_gradient']:raise ValueError('Actual FLOWR VJP validation failed')
    for batch in sorted({v['batch'] for v in rows}):
        if len(set(v for k,v in signatures.items() if k.endswith('/'+str(batch))))>1:
            raise ValueError('Paired initial states differ')
    return {'schema_version':'compact-path-execution-1.0','campaign':campaign,'code_commit':cfg['extension']['code_commit'],
        'window':p['window'],'steps':100,'no_particle_resampling':True,'outside_window_injection':False,
        'affinity_head_gradient':False,'additional_production_calls_per_step':0,'initial_state_signatures':signatures,
        'batch_results':rows,'sources':sources,'preflight':preflight}

def summarize_tail(d,threshold):
    if d.valid_connected.dtype!=bool:raise ValueError('Boolean validity required')
    valid=d[d.valid_connected & np.isfinite(d.pic50_on_rescore)]
    elite=valid[valid.pic50_on_rescore>=threshold]
    energy=valid[valid.energy_status.eq('converged')].mmff_relief_per_heavy.dropna()
    score=valid.pic50_on_rescore
    return {'n':len(d),'valid_n':len(valid),'all_mean_pic50':float(d.pic50_on_rescore.mean()),
        'valid_mean_pic50':float(score.mean()),'valid_p95_pic50':float(score.quantile(.95)),
        'valid_max_pic50':float(score.max()),'elite_valid_n':len(elite),'elite_unique_graphs':int(elite.smiles.nunique()),
        'elite_yield':len(elite)/len(d),'pb_fast_rate':float(d.pb_fast_pass.mean()),
        'strain_converged_n':len(energy),'strain_median_per_heavy':float(energy.median()),
        'strain_p90_per_heavy':float(energy.quantile(.9))}

def paired_effect(candidate,control):
    c=candidate[['batch','slot','pic50_on_rescore']].merge(control[['batch','slot','pic50_on_rescore']],on=['batch','slot'],suffixes=('_candidate','_control'),validate='one_to_one')
    if len(c)!=len(candidate) or len(c)!=len(control):raise ValueError('Paired candidate coverage differs')
    delta=c.pic50_on_rescore_candidate-c.pic50_on_rescore_control
    batches=delta.groupby(c.batch).mean().to_numpy()
    rng=np.random.default_rng(42);boot=batches[rng.integers(0,len(batches),(2000,len(batches)))].mean(1)
    return {'paired_mean_pic50':float(delta.mean()),'batch_means':batches.tolist(),
        'batch_bootstrap_CI95':np.quantile(boot,[.025,.975]).tolist() if len(batches)>1 else None,'n_batches':len(batches),
        'limitation':'Exploratory fixed-seed batch panel; adaptive screening is not independent confirmation'}

def retain_round(dataset,campaign,evaluated,output,threshold,baseline=None):
    dataset,evaluated,out=map(Path,(dataset,evaluated,output));root=dataset/'results'/campaign
    if out.exists():raise FileExistsError(out)
    execution=execution_audit(dataset,campaign);write_json(evaluated/'execution_report.json',execution)
    d=pd.read_csv(evaluated/'candidate_metrics.csv');arms=d.arm.unique().tolist()
    summary={'campaign':campaign,'threshold_pic50':threshold,'results':{a:summarize_tail(d[d.arm==a],threshold) for a in arms}}
    if baseline:
        old=pd.read_csv(baseline)
        old=old[old.batch.isin(d.batch.unique())]
        old_execution=read_json(Path(baseline).parent/'execution_report.json')
        for key,value in execution['initial_state_signatures'].items():
            if key not in old_execution['initial_state_signatures'] or value!=old_execution['initial_state_signatures'][key]:
                raise ValueError('Screening and baseline initial states differ')
        for arm in ['gradient','unguided']:
            control=old[old.arm==arm]
            if len(control):summary['versus_'+arm]=paired_effect(d[d.arm=='gradient'],control)
    out.mkdir(parents=True)
    for name in ['candidate_metrics.csv','batch_metrics.csv','terminal_report.json','execution_report.json','posebusters_fast_checks.csv']:
        (out/name).write_bytes((evaluated/name).read_bytes().replace(b'\r\n',b'\n'))
    for name in ['config.json','reward_program.json','COMPLETE.json']:
        target=out/'inference_config'/name;target.parent.mkdir(exist_ok=True)
        target.write_bytes((root/name).read_bytes().replace(b'\r\n',b'\n'))
    write_json(out/'comparison.json',summary)
    files=[{'path':p.relative_to(out).as_posix(),'sha256':digest(p),'bytes':p.stat().st_size} for p in sorted(out.rglob('*')) if p.is_file()]
    retention={'status':'complete','campaign':campaign,'candidate_rows':len(d),'files':files,
        'inference_commit':execution['code_commit'],'raw_trajectory_hashes':execution['sources'],
        'retention':'Scientific reports and provenance only; raw generation can be retired after checksum validation'}
    write_json(out/'retention.json',retention)
    return summary
