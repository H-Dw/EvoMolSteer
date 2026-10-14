"""Freeze paired generator jobs and a bridge to existing M2/native controls."""
import argparse
from pathlib import Path
import pandas as pd
from evomolsteer.io import read_json,write_json,digest


def prepare(repo,programs,output):
    repo=repo.resolve();output=output.resolve();output.mkdir(parents=True,exist_ok=True)
    bundle=read_json(programs);config=bundle['config']
    bank=read_json(repo/'docs/experiments/m2_tools_full_drop_ablation_20261012/shared_control_bank/manifest.json')
    batch=config['development_batches'][0]
    records={}
    for arm in ['gradient','unguided']:
        path=repo/bank['result']['outcomes'][arm]['metrics'];frame=pd.read_csv(path)
        records[arm]={str(batch):{str(int(r.slot)):float(r.pic50_on_rescore) for r in frame[frame.batch.eq(batch)].itertuples()}}
    expected=output/'bridge_expected.json'
    write_json(expected,{'records':records,'maximum_pic50_difference':1e-5,
        'control_manifest_sha256':digest(repo/'docs/experiments/m2_tools_full_drop_ablation_20261012/shared_control_bank/manifest.json'),
        'reason':'Current generator source has changed; verify matched M2/native before reusing fixed outcomes'})
    base_program=bank['program']
    # The bank records the path under either its program field or the canonical
    # immutable program file. Never substitute a newly selected regional reward.
    if not isinstance(base_program,str):base_program='configs/experiments/steer_dependency_v1/D2_without_trajectory_static.json'
    base_reference='configs/experiments/steer_dependency_v1/static.json.gz'
    rows=[{'id':'runtime_bridge','program':base_program,'reference':base_reference,
        'program_sha256':digest(repo/base_program),'reference_sha256':digest(repo/base_reference),
        'window':read_json(repo/base_program)['window'],'arms':'gradient,unguided,gradient_zero',
        'batches':[batch],'bridge_expected':expected.relative_to(repo).as_posix(),'required_bridge':True}]
    rows += [dict(row,arms='gradient',batches=config['development_batches']) for row in bundle['programs']]
    jobs=[]
    for row in rows:
        name='M2REG_'+row['id'].replace('.','p')
        spec={k:row[k] for k in ['program','reference','program_sha256','reference_sha256','window','arms','batches']}
        spec.update(name=name,n=len(row['batches'])*config['batch_size'],batch=config['batch_size'],
            steps=config['steps'],seed=config['master_seed'],checkpoint_sha256=bank['checkpoint_sha256'],
            prior_seed_mode='indexed_batch',minimum_free_bytes=256*1024*1024)
        if row.get('bridge_expected'):spec['bridge_expected']=row['bridge_expected']
        path=output/(name+'.json');write_json(path,spec)
        jobs.append({'name':name,'spec':path.relative_to(repo).as_posix(),'required_bridge':row.get('required_bridge',False)})
    plan={'jobs':jobs,'error_policy':'Skip each failed candidate; runtime bridge failure blocks reuse of frozen controls',
        'control_bank_generated_again':False,'new_candidate_attempts':sum(len(r['batches'])*config['batch_size'] for r in rows[1:]),
        'bridge_attempts':len(rows[0]['batches'])*config['batch_size']*3,'models':['flowr_root_v2.ckpt','deepseek-flash'],
        'agent_choice_sha256':bundle['selection_sha256'],'development_only':True}
    write_json(output/'queue.json',plan)
    return plan


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',default='.',type=Path)
    p.add_argument('--programs',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();print(prepare(a.repo,a.programs,a.output))
