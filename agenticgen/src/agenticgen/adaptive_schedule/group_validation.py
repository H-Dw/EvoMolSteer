"""Native acceptance for the fixed-start reduced-population protocol."""
from pathlib import Path
import concurrent.futures
import multiprocessing
import time
from .io import read, write, sha, source_fingerprint, commit


def validate(root):
    import torch
    import h5py
    import numpy as np
    from agenticgen.runtime.locks import lock
    from .group_run import assert_prepared, ensure_r0, initialize_cpu
    from .flowr import load_model, load_pt
    from .wave_native import execute_wave, take
    from .wave_retention import prune_payload, verify_snapshot
    from .contracts import expanded
    root=Path(root);config=assert_prepared(root);n=config['population_size']
    probe=root/'operations'/('fixed_start_acceptance_'+commit()[:8]);probe.mkdir(parents=True,exist_ok=True)
    with lock(root):
        model=load_model(config,read(root/'teachers.json')[0]['target'],root)
        results=[]
        with concurrent.futures.ProcessPoolExecutor(max_workers=config['execution']['quality_workers'],
            mp_context=multiprocessing.get_context('spawn'),initializer=initialize_cpu) as pool:
            for teacher in read(root/'teachers.json'):
                tb=probe/teacher['teacher_id'];tb.mkdir(parents=True,exist_ok=True)
                parent=ensure_r0(model,root,config,teacher,pool)
                verify_snapshot(parent)
                if (tb/'ACCEPTED.json').exists():
                    results.append(read(tb/'ACCEPTED.json'));continue
                repeat=tb/'same_schedule_restart'
                execute_wave(model,config,teacher,[config['initial_ticks']]*(n*4),repeat,
                    root/'contexts'/'population_context.pt.gz',parent=parent,start_step=10)
                expected=take(load_pt(parent/'terminal.pt.gz'),list(range(n))*4,n)
                got=load_pt(repeat/'terminal.pt.gz')
                error=float((got['coords']-expected['coords']).abs().max())
                if not torch.allclose(got['coords'],expected['coords'],atol=1e-5,rtol=1e-5):
                    raise ValueError(f'Fixed-start same-schedule recovery differs: {error}')
                for key in ['atomics','charges','bonds','mask']:
                    if not torch.equal(got[key],expected[key]):raise ValueError('Recovery categorical mismatch: '+key)
                rows=read(parent/'final_records.json')*4;observed=read(repeat/'final_records.json')
                scores=[]
                for a,b in zip(rows,observed):
                    x,y=a.get('pic50_on_rescore'),b.get('pic50_on_rescore')
                    if (x is None)!=(y is None):raise ValueError('Recovery rescore validity differs')
                    if x is not None:scores.append(abs(x-y))
                if scores and max(scores)>1e-4:raise ValueError('Recovery affinity rescore mismatch')
                target_rng=load_pt(parent/'checkpoint_100.pt.gz')['rng_sha256']
                if target_rng!=load_pt(repeat/'checkpoint_100.pt.gz')['rng_sha256']:
                    raise ValueError('Recovery terminal logical RNG differs')
                # Acceptance-only probes, not extra scientific comparison groups.
                ticks=[[3,5,5,5],[7,5,5,5],[5,3,5,5],[5,5,5,7]]
                schedules=[x for x in ticks for _ in range(n)];mixed=tb/'different_schedules'
                execute_wave(model,config,teacher,schedules,mixed,root/'contexts'/'population_context.pt.gz',
                    parent=parent,start_step=10)
                max_move=0
                with h5py.File(mixed/'trace.h5','r') as trace:
                    for step in range(10,100):
                        noise=trace['native_epsilon'][step]
                        if any(not np.array_equal(noise[:n],noise[g*n:(g+1)*n]) for g in range(1,4)):
                            raise ValueError('Common-random groups received different native noise')
                        max_move=max(max_move,float(np.linalg.norm(trace['guidance_delta_A'][step],axis=-1).max()))
                if max_move>.025001:raise ValueError('Guidance displacement cap exceeded')
                for row in read(mixed/'steps.json'):
                    expected_lambda=[expanded(x)[row['step']] for x in schedules]
                    if not np.allclose(row['lambda'],expected_lambda,atol=1e-7):
                        raise ValueError('Physical batch does not apply per-group schedules')
                    if row['affinity_vjp']:raise ValueError('Affinity entered reward gradient')
                record={'teacher':teacher['teacher_id'],'start':10,'unique_starts':n,'physical_rows':4*n,
                    'coordinate_max_abs':error,'score_max_abs':max(scores,default=None),
                    'rng_and_categories_equal':True,'max_atom_guidance_A':max_move,
                    'different_group_schedules':True,'condition_atoms':teacher['core_atoms'],
                    'original_core':teacher['original_core_atoms'],'diagnostic_probes_not_scientific_results':True}
                write(tb/'ACCEPTED.json',record);results.append(record)
                prune_payload(repeat);prune_payload(mixed)
        del model;torch.cuda.empty_cache()
        from .gradient_validation import check
        gradient=check(root)
        write(probe/'results.json',results)
        result={'source_fingerprint':source_fingerprint(),'config_sha256':sha(root/'config.json'),
            'teachers':len(results),'recovery_checks':len(results),'results_sha256':sha(probe/'results.json'),
            'gradient_checks_sha256':sha(root/'validation/gradient_checks.json'),
            'fixed_start':10,'common_randomness':True,'different_hypothesis_schedules':True,
            'affinity_vjp':False,'time':time.time()}
        if not gradient['passed']:raise ValueError('Gradient acceptance incomplete')
        write(root/'ACCEPTANCE_COMPLETE.json',result)
        return result
