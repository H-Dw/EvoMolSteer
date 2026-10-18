"""Freeze all original final paths, historical source hashes and common B50 groups."""
import copy
from pathlib import Path

import numpy as np

from evomolsteer.io import read_json,digest
from evomolsteer.storage.transactions import atomic_json
from .common import NAME,FLOWR_COMMIT,FLOWR_FILES,identity,ancestry,program_identity,validate,condition_program


def prepare(repo, output=None):
    repo=Path(repo).resolve()
    old=read_json(repo/'configs/experiments/elite_initial_alignment_20261017/config.json')
    raw=repo/'data/raw/ck2_clk3_lineage_20261003/results/main1000_w050/single'
    records=[]; sources={}; lineage={}
    for b in range(20):
        folder=raw/f'batch_{b:03d}'
        rows=read_json(folder/'final_records.json');records.extend(rows)
        with np.load(folder/'trajectory.npz') as z:
            before,roots=ancestry(z['selected_indices'])
            if before.shape!=(100,50): raise ValueError('Unexpected historical pool dimensions')
            if not np.array_equal(roots,[r['root_slot'] for r in rows]): raise ValueError('Root labels differ from full backtrace')
            lineage[b]=(before,roots)
        names=['initial_state.pt.gz','trajectory.npz','final_records.json','final_prediction.pt.gz']
        names += [f'restart_step_{i:03d}.pt.gz' for i in (0,10,20,30,40,50,75,99)]
        sources[str(b)]={'relative':f'results/main1000_w050/single/batch_{b:03d}',
                         'files':{n:digest(folder/n) for n in names}}
    if len(records)!=1000 or not all(np.isfinite(r['pic50_on_rescore']) for r in records):
        raise ValueError('Ranking requires all 1000 original finite final scores')
    records.sort(key=lambda r:(-r['pic50_on_rescore'],r['batch'],r['slot']))
    ranked=[]
    for rank,r in enumerate(records,1):
        b,s=r['batch'],r['slot'];before,roots=lineage[b]
        ranked.append({'id':f'F_b{b:03d}_s{s:03d}','rank':rank,'batch':b,'final_slot':s,
                       'root_slot':int(roots[s]),'path_sha256':identity(before[:,s].tolist()),
                       'historical_pic50':r['pic50_on_rescore'],
                       'historical_valid_connected':bool(r.get('build_success') and r.get('connected'))})
    # No paths are removed from the N1000 conditions. Packing the common top200
    # first gives every paired subset exactly the same execution neighbours.
    ordered=sorted(ranked[:200],key=lambda r:(r['batch'],r['final_slot']))+sorted(ranked[200:],key=lambda r:(r['batch'],r['final_slot']))
    groups=[{'index':i//50,'ids':[x['id'] for x in ordered[i:i+50]]} for i in range(0,1000,50)]
    chosen={c['id']:copy.deepcopy(c) for c in old['conditions']}
    conditions=[]
    for key,n in [('unguided',200),('M2',200),('NC_R11',200),('ancestor_R02_v3',1000)]:
        c=chosen[key];c['n']=n;conditions.append(c)
    r15=chosen['motif_R15_v2'];r19=copy.deepcopy(r15)
    r19.update(id='motif_R15_B50',n=1000,historical_commit=read_json(repo/'docs/experiments/local_motif20_v2_20261015/R19_v2/results.json')['commit'])
    base='docs/experiments/local_motif20_v2_20261015/R19_v2/candidates/F_b011_s030/'
    r19.update(program=base+'programs/R19_v2.json',reference=base+'teacher.json')
    r19['program_sha256']=digest(repo/r19['program']);r19['reference_sha256']=digest(repo/r19['reference'])
    a,t=condition_program(repo,r15);b,u=condition_program(repo,r19)
    if program_identity(a)!=program_identity(b) or identity(t)!=identity(u): raise ValueError('R15/R19 numerical equivalence failed')
    r15.update(n=1000,alias_of='motif_R15_B50',numeric_program_sha256=program_identity(a))
    r19['numeric_program_sha256']=program_identity(b);conditions += [r19,r15]
    config={k:copy.deepcopy(old[k]) for k in ('checkpoint','checkpoint_sha256','protein','master_seed','steps','native_options','tail_threshold','steer_best')}
    config.update(schema_version='historical-path-replay-1',name=NAME,batch_size=50,
                  flowr_commit=FLOWR_COMMIT,flowr_files=FLOWR_FILES,
                  flowr_repository=old['flowr_root'],flowr_root=old['flowr_root']+'/experiments/evomolsteer_'+NAME+'/flowr_runtime',
                  remote_dataset=old['flowr_root']+'/experiments/ck2_clk3_lineage_20261003',
                  remote_work=old['flowr_root']+'/experiments/evomolsteer_'+NAME,
                  sources=sources,attempts=ordered,groups=groups,conditions=conditions,
                  bootstrap_seed=42,bootstrap_replicates=10000,max_worker_restarts=1,
                  batch_timeout_seconds=1800,startup_timeout_seconds=1800,poll_seconds=3,
                  minimum_free_bytes=256*1024**2,coordinate_tolerance_A=1e-5,affinity_tolerance=1e-5,
                  interpretation='Retrospectively conditioned historical-path replay; not an unconditional native-prior experiment.')
    config['protocol_sha256']=identity(config);validate(config)
    for c in conditions: condition_program(repo,c)
    output=Path(output) if output else repo/'configs/experiments'/NAME/'config.json'
    atomic_json(output,config)
    return output
