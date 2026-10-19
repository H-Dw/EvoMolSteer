"""Verify a collected historical-path extension and write a compact final report.

No model inference, resampling, reward tuning or new random molecular inputs.
Confidence intervals reuse the existing original-source-batch bootstrap.
"""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

import numpy as np

from evomolsteer.io import read_json,digest
from evomolsteer.storage.transactions import atomic_json
from evomolsteer.path_replay.common import groups_for,read_batch,batch_path,condition_program,identity
from evomolsteer.path_replay.extension import load
from evomolsteer.path_replay.analysis import distribution,paired_cluster,write_csv


def verify(repo,cfg,ext,work,base_work):
    c=ext['condition'];program,teacher=condition_program(repo,c)
    source=read_json(work/'methods'/c['id']/'source_audit.json')
    if source['condition']!=c or not source['no_smc'] or source['affinity_head_gradient']:
        raise ValueError('Execution audit differs from the frozen extension')
    gate=read_json(work/'inherited_zero_replay_gate.json')
    if not gate['passed'] or gate['n']!=200 or not gate['discrete_states_equal'] or max(gate['max_intermediate_coordinate_error_A'],gate['max_terminal_coordinate_error_A'])>cfg['coordinate_tolerance_A'] or gate['max_pic50_error']>cfg['affinity_tolerance']:
        raise ValueError('Inherited zero replay gate failed')
    tapes=read_json(work/'tapes/verification.json')
    if not tapes['passed'] or tapes['protocol_sha256']!=cfg['protocol_sha256'] or len(tapes['artifacts'])!=20:
        raise ValueError('Incomplete or incorrect primitive-randomness tapes')
    maps={};roots={}
    for filename,sha in tapes['artifacts'].items():
        if digest(work/'tapes'/filename)!=sha or digest(base_work/'tapes'/filename)!=sha:
            raise ValueError('Extension/base randomness fingerprint differs')
        with np.load(work/'tapes'/filename) as z:
            b=int(filename.split('_')[-1].split('.')[0]);maps[b]=z['before'].copy();roots[b]=z['roots'].copy()
    expected={a['id']:a for a in cfg['attempts']};rows=[];batches=[];minimum_nonzero=50;maximum_injection=0.;sdf_count=0
    window=program['window'];active=[i for i in range(100) if i/100>=window[0]-1e-6 and (i+1)/100<=window[1]+1e-6]
    for group in groups_for(cfg,c):
        receipt=read_batch(work,cfg,c,group)
        if receipt is None or receipt['status']!='complete':raise ValueError('Expected all 20 complete batches')
        if receipt['active_steps']!=active or not receipt['weights_unchanged'] or receipt['parameter_gradients_accumulated']:
            raise ValueError('Unexpected guidance window or weight update')
        folder=batch_path(work,c,group).parent
        with np.load(folder/'states.npz') as z:
            for key in ('terminal_coords_A','half_coords_A'):
                if z[key].shape!=(50,25,3) or not np.isfinite(z[key]).all():raise ValueError('Invalid state coordinates')
        with gzip.open(folder/'guidance.json.gz','rt') as f:g= json.load(f)
        if not g['no_smc'] or g['affinity_head_gradient'] or len(g['steps'])!=100 or len(g['random_steps'])!=100:
            raise ValueError('Missing native integration/randomness coverage')
        if [v['step'] for v in g['random_steps']]!=list(range(100)) or any(v['primitive_calls']!=4 for v in g['random_steps']):
            raise ValueError('Historical random primitives are incomplete')
        if [v['step'] for v in g['steps'] if v['active']]!=active:raise ValueError('Telemetry window differs')
        if [v['id'] for v in g['path_bindings']]!=group['ids']:raise ValueError('Trajectory path ordering differs')
        for result,provenance in zip(receipt['records'],g['path_bindings']):
            a=expected[result['id']]
            for key in ('batch','final_slot','root_slot','path_sha256'):
                if result[key]!=a[key] or provenance[key]!=a[key]:raise ValueError('Input path binding differs')
            if result['rank']!=a['rank'] or result['historical_pic50']!=a['historical_pic50']:raise ValueError('Frozen rank/score differs')
            if int(roots[a['batch']][a['final_slot']])!=a['root_slot'] or identity(maps[a['batch']][:,a['final_slot']].tolist())!=a['path_sha256']:
                raise ValueError('Actual per-step ancestor map differs')
        if not np.isfinite([r['pic50'] for r in receipt['records']]).all():raise ValueError('Missing terminal scores')
        if receipt['scoring_context']['historical_ligands_used']:raise ValueError('Historical ligand was fed to scorer')
        minimum_nonzero=min(minimum_nonzero,min(v['nonzero_molecules'] for v in g['steps'] if v['active']))
        maximum_injection=max(maximum_injection,max(v['injected_rms_A_mean'] for v in g['steps']))
        with gzip.open(folder/'terminal.sdf.gz','rt') as f:decoded=f.read().count('$$$$')
        sdf_count+=decoded;rows+=receipt['records'];batches.append({'group':group['index'],'n':50,'seconds':receipt['seconds'],
            'receipt_sha256':digest(folder/'COMPLETE.json'),'decoded_sdf':decoded})
    if [r['id'] for r in rows]!=[a['id'] for a in cfg['attempts']] or len(set(r['id'] for r in rows))!=1000:
        raise ValueError('Expected exactly 1000 unique original final paths')
    reuse=read_json(work/'reuse_audit.json');state=read_json(work/'queue_status.json')
    if state['status']!='complete' or state['complete']!=1000 or state['failed'] or state['pending']:
        raise ValueError('Queue has not successfully completed')
    audit={'passed':True,'n':1000,'batches':batches,'decoded_sdf':sdf_count,'code_commit':source['code_commit'],
        'base_protocol_sha256':cfg['protocol_sha256'],'extension_sha256':ext['extension_sha256'],
        'program_sha256':c['program_sha256'],'reference_sha256':c['reference_sha256'],'window':window,
        'reward_weights':program['weights'],'local_points':len(program['atom_slots']),'teacher_candidate':teacher['candidate_id'],
        'saved_tensor_policy':source['saved_tensor_policy'],'parameters_requiring_grad':source['parameters_requiring_grad'],
        'minimum_nonzero_molecules_per_active_step':minimum_nonzero,'maximum_mean_injected_rms_A':maximum_injection,
        'steps_per_path':100,'reused_n':reuse['reused_n'],'new_n':1000-reuse['reused_n'],'zero_gate':gate,
        'elapsed_seconds':state['finished_unix']-state['started_unix'],'scoring_failures':0,
        'quality_failures':sum(not r['valid_connected'] for r in rows),'no_smc':True,'affinity_head_gradient':False,
        'program_source_unchanged':True,'historical_noise_maps_verified':1000,
        'energy_status_counts':dict(Counter(r['energy_status'] for r in rows))}
    return rows,audit


def geometry_diagnostic(repo,cfg,ext,work,base_work,rows):
    """Compare actual t=.5 geometry with the same local teacher, on paired controls."""
    from scipy.optimize import linear_sum_assignment
    program,teacher=condition_program(repo,ext['condition']);target=np.asarray(teacher['coords_A'])[program['atom_slots']]
    distances={}
    original=next(c for c in cfg['conditions'] if c['id']=='unguided')
    for label,root,c in [('R11',work,ext['condition']),('unguided',base_work,original)]:
        distances[label]={}
        for group in groups_for(cfg,c):
            receipt=read_batch(root,cfg,c,group)
            if receipt is None or receipt['status']!='complete':continue
            with np.load(batch_path(root,c,group).parent/'states.npz') as z:
                xyz=z['half_coords_A'].copy();mask=z['mask'].copy()
            for i,result in enumerate(receipt['records']):
                points=xyz[i][mask[i]];cost=((points[:,None]-target[None])**2).sum(-1)
                a,b=linear_sum_assignment(cost)
                if len(b)!=len(target):raise ValueError('Not enough ligand points for teacher region')
                distances[label][result['id']]={'rms_A':float(np.sqrt(cost[a,b].mean())),'pic50':result['pic50'],
                    'pb_pass':result.get('pb_pass'),'mmff_relief_per_heavy':result.get('mmff_relief_per_heavy')}
    pairs=[]
    for r in rows:
        key=r['id']
        if key not in distances['unguided']:continue
        a=distances['R11'][key];b=distances['unguided'][key]
        pairs.append({'id':key,'batch':r['batch'],'original_rank':r['rank'],'guided_local_rms_A':a['rms_A'],
            'unguided_local_rms_A':b['rms_A'],'local_rms_delta_A':a['rms_A']-b['rms_A'],'affinity_delta':a['pic50']-b['pic50']})
    result={'n_paired':len(pairs),'snapshot_time':.5,'teacher_time':teacher['time'],'local_points':len(target),
        'metric':'Hungarian RMS over the selected local teacher points in the actual t=.5 state; no rigid alignment',
        'relation_to_reward':'Diagnostic on actual boundary states; low-t reward uses a model bridge prediction, so these are not identical observables.'}
    if pairs:
        result.update(guided_mean_local_rms_A=float(np.mean([p['guided_local_rms_A'] for p in pairs])),
            unguided_mean_local_rms_A=float(np.mean([p['unguided_local_rms_A'] for p in pairs])),
            fraction_geometrically_closer=float(np.mean([p['local_rms_delta_A']<0 for p in pairs])),
            fraction_closer_but_affinity_lower=float(np.mean([p['local_rms_delta_A']<0 and p['affinity_delta']<0 for p in pairs])))
    key=teacher['candidate_id']
    if key in distances['unguided'] and key in distances['R11']:
        result['teacher_path_case']={'id':key,'unguided':distances['unguided'][key],'guided':distances['R11'][key]}
    return pairs,result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('repo','manifest','work','base-work','output'):p.add_argument('--'+key,type=Path,required=True)
    args=p.parse_args();repo=args.repo.resolve();work=args.work.resolve();base=args.base_work.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    cfg,ext,_=load(repo,args.manifest);rows,audit=verify(repo,cfg,ext,work,base)
    summary=read_json(work/'analysis/summary.json');metrics=summary['performance'];comparison=summary['paired']
    if next(r for r in metrics if r['condition']=='NC_R11' and r['cohort']=='all1000')['n_scored']!=1000:raise ValueError('Stale analysis')
    references={a['id']:dict(a,pic50=a['historical_pic50']) for a in cfg['attempts']}
    strata=[]
    for label,lo,hi in [('top05',1,50),('rank05_10',51,100),('rank10_20',101,200),('remaining80',201,1000)]:
        selected=[r for r in rows if lo<=r['rank']<=hi];control=[references[r['id']] for r in selected]
        effect=paired_cluster(selected,control,cfg['bootstrap_replicates'],cfg['bootstrap_seed'])
        strata.append({'stratum':label,'n':len(selected),'historical_mean':float(np.mean([r['pic50'] for r in control])),
            'guided_mean':float(np.mean([r['pic50'] for r in selected])),**effect})
    chosen=sorted(rows,key=lambda r:r['pic50'],reverse=True)
    top=[{k:r.get(k) for k in ('id','rank','historical_pic50','pic50','valid_connected','pb_pass','mmff_relief_per_heavy','smiles')} for r in chosen[:20]]
    exported=[]
    for result in chosen:
        if result['pic50']<=cfg['steer_best'] or not result.get('pb_pass'):continue
        folder=batch_path(work,ext['condition'],cfg['groups'][result['group']]).parent
        with gzip.open(folder/'terminal.sdf.gz','rt') as f:blocks=f.read().split('$$$$')
        matches=[block.lstrip('\r\n') for block in blocks if block.lstrip('\r\n').splitlines() and block.lstrip('\r\n').splitlines()[0]==result['id']]
        if len(matches)!=1:raise ValueError('Best candidate SDF cannot be uniquely located')
        dest=out/'best_candidates'/(result['id']+'.sdf');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(matches[0]+'$$$$\n',encoding='utf-8')
        exported.append({'id':result['id'],'path':dest.relative_to(out).as_posix(),'sha256':digest(dest)})
    qualified_counts={'above_original_best':sum(r['pic50']>cfg['steer_best'] for r in rows),
        'above_original_best_pb':sum(r['pic50']>cfg['steer_best'] and r.get('pb_pass',False) for r in rows)}
    audit.update(qualified_counts,best_candidate_exports=exported);atomic_json(out/'completion_audit.json',audit);write_csv(out/'rank_strata.csv',strata);write_csv(out/'top_candidates.csv',top)
    geometry_pairs,geometry=geometry_diagnostic(repo,cfg,ext,work,base,rows)
    write_csv(out/'geometry_affinity_pairs.csv',geometry_pairs);atomic_json(out/'geometry_diagnostic.json',geometry)
    for name in ('summary.json','performance.csv','paired.csv','report.zh-CN.md'):
        import shutil
        shutil.copy2(work/'analysis'/name,out/('metrics.zh-CN.md' if name=='report.zh-CN.md' else name))
    atomic_json(out/'inputs.json',{'analysis_summary_sha256':digest(work/'analysis/summary.json'),'report_script_sha256':digest(__file__),
        'collection_receipt':read_json(repo/'docs/experiments'/ext['name']/'collection_receipt.json'),
        'manifest_sha256':digest(args.manifest),'base_protocol_file_sha256':ext['base_config_sha256']})
    atomic_json(out/'report_data.json',{'audit':audit,'rank_strata':strata,'top_candidates':top,'performance':metrics,'paired':comparison,'geometry_diagnostic':geometry})
    try:
        import matplotlib
        matplotlib.use('Agg')
        matplotlib.rcParams['svg.hashsalt']='evomolsteer-path-replay'
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(10,3.5),layout='constrained')
        original=np.array([r['historical_pic50'] for r in rows]);guided=np.array([r['pic50'] for r in rows]);bins=np.linspace(min(original.min(),guided.min()),max(original.max(),guided.max()),45)
        axes[0].hist(original,bins,alpha=.5,label='Original Steer');axes[0].hist(guided,bins,alpha=.5,label='R11 replay guidance');axes[0].axvline(cfg['tail_threshold'],color='black',ls=':',label='Frozen high-score threshold')
        axes[0].set(xlabel='Predicted pIC50',ylabel='Historical paths (N=1000)');axes[0].legend(fontsize=7)
        means=np.array([s['mean_delta'] for s in strata]);low=np.array([s['mean_delta_ci_low'] for s in strata]);high=np.array([s['mean_delta_ci_high'] for s in strata])
        axes[1].errorbar(range(4),means,yerr=np.array([means-low,high-means]),fmt='o',capsize=4);axes[1].axhline(0,color='black',lw=.8)
        axes[1].set_xticks(range(4),['Top 5%','5-10%','10-20%','Other 80%']);axes[1].set(ylabel='Paired pIC50 change',title='Original-rank strata; source-batch 95% CIs')
        fig.savefig(out/'score_distribution_and_strata.svg',metadata={'Date':None});fig.savefig(out/'score_distribution_and_strata.png',dpi=160);plt.close(fig)
    except ImportError:pass
    print(json.dumps({'output':str(out),'audit_passed':True,'n':len(rows),'top_candidate':top[0],'rank_strata':strata,'geometry_diagnostic':geometry},indent=2))


if __name__=='__main__':main()
