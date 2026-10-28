import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import h5py
import numpy as np
from agenticgen.uniguide.native_tools import load_pt

ROOT=Path('/workspace/agenticgen_raw999_limit_acceptance_20261029_511a09b4')
REPO=Path('/workspace/uniguide_audits/raw999-limit-511a09b4')
def read(path): return json.loads(Path(path).read_text())
def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
receipt=read(ROOT/'acceptance/COMPLETE.json')
result=dict(timestamp=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    commit=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip(),
    worktree_status=subprocess.check_output(['git','-C',str(REPO),'status','--porcelain'],text=True),
    acceptance_receipt_sha256=digest(ROOT/'acceptance/COMPLETE.json'),teachers={})
for teacher in receipt['checks']:
    base=ROOT/'acceptance'/teacher/'candidate_batch'
    groups={}
    for group in ('separate_0','separate_1','mixed_twenty','mixed_resumed'):
        folder=base/group
        output=load_pt(folder/'terminal.pt.gz');policy=read(folder/'OUTPUT_POLICY.json')
        with h5py.File(folder/'trace.h5') as h:
            endpoint=h['endpoint_A'][99]
            delta=h['guidance_delta_A'][:].astype(np.float64)
            original=h['unconstrained_guidance_delta_A'][:]
            factor=h['guidance_increment_scale_factor'][:]
        exact=np.array_equal(output['coords'].numpy(),endpoint)
        bound=policy['maximum_single_atom_guidance_increment_angstrom']
        max_applied=float(np.linalg.norm(delta,axis=-1).max())
        prefix_zero=bool((delta[:10]==0).all())
        steps=read(folder/'steps.json')
        assert exact and max_applied<=bound and prefix_zero
        assert policy['windows']==[[10,25],[26,50],[51,75],[76,99]]
        assert policy['last_loop']['step']==99
        assert np.allclose(policy['last_loop']['network_times'],.98901,rtol=0,atol=1e-7)
        assert not policy['extra_terminal_forward'] and not policy['coordinate_projection']
        groups[group]=dict(terminal_equals_last_uncovered_prediction=exact,
            maximum_applied_single_atom_increment_angstrom=max_applied,
            maximum_raw_single_atom_increment_angstrom=float(np.linalg.norm(original,axis=-1).max()),
            minimum_scale_factor=float(factor.min()),guidance_before_step_10_zero=prefix_zero,
            network_times=policy['last_loop']['network_times'],policy=policy,
            output_sha256=digest(folder/'terminal.pt.gz'))
    stress=base/'candidate_local_limit';steps=read(stress/'steps.json')
    flags=np.array([s['guidance_increment_limited'] for s in steps]);factors=np.array([s['guidance_increment_scale_factor'] for s in steps])
    raw=np.array([s['unconstrained_max_atom_increment_A'] for s in steps])
    with h5py.File(stress/'trace.h5') as h:
        applied=float(np.linalg.norm(h['guidance_delta_A'][:].astype(np.float64),axis=-1).max())
    bound=read(stress/'OUTPUT_POLICY.json')['maximum_single_atom_guidance_increment_angstrom']
    assert not flags[:,:10].any() and flags[:,10:].any() and applied<=bound
    result['teachers'][teacher]=dict(full_groups=groups,limiter_stress=dict(
        limited_candidate_count=int(flags.any(0).sum()),limited_candidate_steps=int(flags.sum()),
        unaffected_first_ten_candidates=bool(not flags[:,:10].any()),
        maximum_unconstrained_increment_angstrom=float(raw.max()),
        maximum_applied_increment_angstrom=applied,minimum_scale_factor=float(factors.min())),
        model_checks=receipt['checks'][teacher])
protected=read('/workspace/uniguide_audits/raw999-limit-511a09b4-protected.json')
changed=[path for path,value in protected.items() if digest(path)!=(value if isinstance(value,str) else value['sha256'])]
result['protected_sources']=dict(count=len(protected),changed=changed)
assert not changed
running=[]
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit():continue
    try:args=(proc/'cmdline').read_bytes().decode(errors='replace').replace('\0',' ')
    except (FileNotFoundError,PermissionError,ProcessLookupError):continue
    if 'run_uniguide_geometry.py --config /workspace/uniguide_audits/raw999-limit-511a09b4-config.json' in args and args.startswith('/opt/'):
        running.append(dict(pid=int(proc.name),command=args))
result['active_acceptance_processes']=running
with open('/workspace/agenticgen_uniguide_flowr_20261026/.executor.lock','r') as lock:
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);result['executor_lock_free']=True;fcntl.flock(lock,fcntl.LOCK_UN)
    except BlockingIOError:result['executor_lock_free']=False
assert not running and result['executor_lock_free']
result.update(full_calibration_started=False,formal_experiment_started=False,original_ug05_resumed=False,
    interpretation='Engineering verification only; stress scales are not calibrated multipliers.')
destination=ROOT/'engineering_audit.json'
destination.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(dict(path=str(destination),commit=result['commit'],protected=result['protected_sources'],
    lock_free=result['executor_lock_free'],teachers={k:dict(stress=v['limiter_stress'],
        finite_difference_errors=v['model_checks']['real_model_vjp']['relative_errors'],
        exact_candidate_batch=v['model_checks']['candidate_batch_equivalence']['exact_equivalence'],
        capacity=v['model_checks']['candidate_batch_equivalence']['capacity_probe_success'],
        all_outputs_last_uncovered=all(g['terminal_equals_last_uncovered_prediction'] for g in v['full_groups'].values()))
        for k,v in result['teachers'].items()}),ensure_ascii=True))
