"""Remote inference-only cohort; verify clocks and retire owned dense arrays."""
import argparse
import subprocess
import sys
import time
from pathlib import Path
import numpy as np

from archive_generation import archive
from retire_experiment_outputs import retire
from audit_dependency_execution import audit
from evomolsteer.io import read_json,write_json,digest


def run(repo,flowr,work,spec_path):
    spec=read_json(spec_path)
    if spec['seed']!=42 or spec['steps']!=100 or spec['batch']!=50 or spec['n']!=len(spec['batches'])*spec['batch']:
        raise ValueError('Complete configured matched batches required')
    work=work.resolve();flowr=flowr.resolve();repo=repo.resolve()
    if not work.is_relative_to(flowr/'experiments') or not work.name.startswith('evomolsteer_terminal_regions_'):
        raise ValueError('Explicit owned terminal-region workspace required')
    program,reference=repo/spec['program'],repo/spec['reference']
    if digest(program)!=spec['program_sha256'] or digest(reference)!=spec['reference_sha256']:
        raise ValueError('Frozen program/reference changed')
    checkpoint=flowr/'checkpoints/flowr_root_v2.ckpt'
    if digest(checkpoint)!=spec['checkpoint_sha256']:
        raise ValueError('Unexpected FLOWR checkpoint')
    if '/' in spec['name'] or '\\' in spec['name']:
        raise ValueError('A single cohort basename required')
    work.mkdir(parents=True,exist_ok=True)
    import shutil
    if shutil.disk_usage(work).free<spec['minimum_free_bytes']:
        raise RuntimeError('Insufficient free private workspace before this cohort')
    name=spec['name'];status_path=work/(name+'.status.json')
    if status_path.exists():
        old=read_json(status_path)
        if old.get('status')=='complete' and old.get('spec_sha256')==digest(spec_path):return old
        raise FileExistsError('Existing unsuccessful cohort requires an explicit fresh retry name')
    commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    status={'status':'running','name':name,'code_commit':commit,'spec_sha256':digest(spec_path),'started_unix':time.time()}
    write_json(status_path,status)
    root=work/'generated/results'/name
    try:
        command=[sys.executable,'-u',str(repo/'scripts/generate_dependency_flowr.py'),
            '--flowr-root',str(flowr),'--input-dataset',str(flowr/'experiments/ck2_clk3_lineage_20261003'),
            '--root',str(work/'generated'),'--checkpoint',str(checkpoint),'--program',str(program),
            '--reference',str(reference),'--campaign',name,'--n',str(spec['n']),'--batch',str(spec['batch']),
            '--steps',str(spec['steps']),'--seed',str(spec['seed']),'--arms',spec['arms'],
            '--batch-indices',','.join(map(str,spec['batches'])),'--prior-seed-mode','indexed_batch','--export-terminal']
        write_json(work/(name+'.command.json'),{'command':command,'code_commit':commit})
        source_args=['--input-dataset',str(flowr)]
        for item in read_json(repo/'configs/experiments/flowcompat_scale1000_v1/spec.json')['upstream_sources']:
            source_args += ['--relative-source',item]
        before,after=work/(name+'.before.json'),work/(name+'.after.json')
        subprocess.run([sys.executable,str(repo/'scripts/attest_flowr_sources.py'),*source_args,'--output-record',str(before)],check=True)
        with (work/(name+'.log')).open('wb') as log:
            subprocess.run(command,cwd=repo,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
        subprocess.run([sys.executable,str(repo/'scripts/attest_flowr_sources.py'),*source_args,
            '--reference-record',str(before),'--output-record',str(after)],check=True)
        subprocess.run([sys.executable,str(repo/'scripts/bind_flowr_source_attestation.py'),
            '--dataset',str(work/'generated'),'--campaign',name,'--before-record',str(before),'--after-record',str(after)],check=True)
        # Export removes dense trajectories; execution audit runs on the compact
        # view with exact no-selection arrays and all final slot records.
        transport=archive(work/'generated',name,work/'archives'/(name+'.tar.gz'),evaluation_only=True)
        from evomolsteer.storage.generation_archive import verify_generation_archive
        import tempfile
        with tempfile.TemporaryDirectory(prefix='region-audit-',dir=work) as directory:
            compact=Path(directory)/'dataset'
            verify_generation_archive(work/'archives'/(name+'.tar.gz'),work/'archives'/(name+'.tar.gz.json'),compact)
            execution=audit(compact/'results'/name)
            write_json(work/(name+'.execution_audit.json'),execution)
            expected_count=sum(t>=spec['window'][0]-2e-6 and t+.01<=spec['window'][1]+2e-6 for t in np.arange(100)/100)
            for item in execution['batches']:
                if item['arm']=='gradient' and item['controlled_steps']!=expected_count:
                    raise ValueError('Wrong number of actual controlled steps')
            if spec.get('bridge_expected'):
                expected=read_json(repo/spec['bridge_expected'])
                checks=[]
                for arm in ['gradient','unguided']:
                    for batch in spec['batches']:
                        final=read_json(compact/'results'/name/arm/f'batch_{batch:03d}'/'final_records.json')
                        wanted=expected['records'][arm][str(batch)]
                        errors=[abs(row['pic50_on_rescore']-wanted[str(row['slot'])]) for row in final]
                        checks.append({'arm':arm,'batch':batch,'max_absolute_pic50_difference':max(errors)})
                if max(r['max_absolute_pic50_difference'] for r in checks)>expected['maximum_pic50_difference']:
                    raise ValueError('Current runtime differs from frozen M2/native; old controls cannot be reused')
                write_json(work/(name+'.bridge.json'),{'passed':True,'checks':checks,
                    'zero_native_verified':bool(execution['zero_controls']),'expected_sha256':digest(repo/spec['bridge_expected'])})
        status.update(status='complete',completed_unix=time.time(),transport=transport,
            execution_audit_sha256=digest(work/(name+'.execution_audit.json')),
            particle_selection=False,affinity_head_gradient=False,evaluation='local_after_download')
    except Exception as error:
        status.update(status='failed',stopped_unix=time.time(),error_type=type(error).__name__,error=str(error))
        raise
    finally:
        write_json(status_path,status)
        if root.exists():
            # Preserve this task's complete final outputs or failure traces before
            # deleting only its own dense generation directory.
            if status['status']=='failed':
                import shutil
                destination=work/'failure_diagnostics'/name
                for pattern in ['*/batch_*/guidance_trace.jsonl','*/batch_*/final_records.json','coordinate_gradient_preflight.json']:
                    for path in root.glob(pattern):
                        target=destination/path.relative_to(root);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
            plan=work/(name+'.retire.plan.json')
            write_json(plan,{'allowed_bases':[str(work/'generated/results')],'targets':[str(root)],
                'result_report':str(status_path),'protected':[str(repo),str(flowr/'checkpoints'),
                    str(flowr/'experiments/ck2_clk3_lineage_20261003'),str(work/'archives'),str(work/'failure_diagnostics')]})
            retire(plan,work/(name+'.retire.json'),True)
    print(status)
    return status


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['repo','flowr-root','work','spec']:p.add_argument('--'+name,required=True,type=Path)
    a=p.parse_args();run(a.repo,a.flowr_root,a.work,a.spec)
