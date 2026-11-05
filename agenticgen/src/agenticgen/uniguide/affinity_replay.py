"""Explicitly audited reuse of completed engineering trajectories, never science."""
from pathlib import Path
import shutil
from .io import read,sha,value_sha,frozen
from .native_tools import load_pt

ROOT_FILES=('GEOMETRY_CONFIG.json','SOURCE_BINDING.json','BASELINE.json','SCALE_SOURCE.json')
TRAJECTORY_FILES=('INPUT.json','initial.pt.gz','initial.pt.sha.json','terminal.pt.gz','terminal.pt.sha.json',
    'trace.h5','steps.json','progress.json','rng_ledger.json','OUTPUT_POLICY.json','CHEMICAL_CONDITION.json',
    'COORDINATE_ASSIGNMENT.json','checkpoint_100.pt.gz','checkpoint_100.pt.sha.json')
REPLAY_FILES=(*ROOT_FILES,*('acceptance/'+name for name in TRAJECTORY_FILES))


def validate_trajectory(folder):
    steps=read(folder/'steps.json')
    if len(steps)!=100 or [s['step'] for s in steps]!=list(range(100)):
        raise ValueError('Engineering acceptance requires all one hundred inference steps')
    for step in steps:
        active=1<=step['step']<=98
        flags=step['chemical_inpainting_applied']
        if len(flags)!=50 or any(flag!=active for flag in flags):
            raise ValueError('Chemical endpoint policy or engineering population changed')
        if active:
            matches=[values for name,values in step.items() if name.startswith('chemical_') and name.endswith('_after_match_fraction')]
            if len(matches)<3 or any(len(v)!=50 or any(x!=1 for x in v) for v in matches):
                raise ValueError('Interaction region was not completely inpainted')
    return steps


def replay(config,destination,teacher,scale_source):
    """Recompute endpoint measurements from an immutable full-run acceptance only."""
    spec=config['engineering_replay'];source=Path(spec['source_attempt'])
    if set(spec)!={'source_attempt','source_commit','files'} or set(spec['files'])!=set(REPLAY_FILES):
        raise ValueError('Engineering replay needs an exact, complete source manifest')
    if source.resolve()==Path(config['root']).resolve() or destination.exists():
        raise ValueError('Engineering replay requires an independent, fresh destination')
    if (source/'attempts.json').exists() or list(source.glob('rounds/*/CLAIM.json')):
        raise ValueError('A scientific attempt cannot be replayed as engineering acceptance')
    for name,digest in spec['files'].items():
        if sha(source/name)!=digest:raise ValueError('Engineering replay source changed: '+name)
    saved=read(source/'GEOMETRY_CONFIG.json')
    normalize=lambda c:{k:v for k,v in c.items() if k not in ('root','engineering_replay')}
    if normalize(saved)!=normalize(config):raise ValueError('Engineering replay generation configuration differs')
    # These generation modules did not change in the device-only scoring repair.
    for name,digest in read(source/'SOURCE_BINDING.json')['implementation'].items():
        if name!='iterative_affinity.py' and sha(Path(__file__).with_name(name))!=digest:
            raise ValueError('Engineering replay generation implementation differs: '+name)
    baseline=read(source/'BASELINE.json')
    if (any(baseline[k]!=teacher[k] for k in ('id','condition','features','source_id'))
            or abs(baseline['pic50']-teacher['pic50'])>1e-5):
        raise ValueError('Engineering replay teacher or endpoint baseline differs')
    if read(source/'SCALE_SOURCE.json')!=scale_source:raise ValueError('Engineering replay fixed-scale provenance differs')
    validate_trajectory(source/'acceptance')
    if read(source/'acceptance/progress.json')['step_completed']!=100:
        raise ValueError('Engineering replay contains an incomplete native trajectory')
    checkpoint=load_pt(source/'acceptance/checkpoint_100.pt.gz')
    if checkpoint['step']!=100:raise ValueError('Engineering replay final checkpoint is incomplete')
    policy=read(source/'acceptance/OUTPUT_POLICY.json')
    if (policy['output_policy']!='raw999' or policy['extra_terminal_forward'] or policy['chemical_projection']
            or policy['last_loop']['step']!=99 or not policy['last_loop']['output_is_uncovered_prediction']):
        raise ValueError('Engineering replay is not the uncovered native raw999 output')
    destination.mkdir(parents=True)
    for name in TRAJECTORY_FILES:shutil.copyfile(source/'acceptance'/name,destination/name)
    for name in TRAJECTORY_FILES:
        if sha(destination/name)!=spec['files']['acceptance/'+name]:raise ValueError('Engineering replay copy changed: '+name)
    initial=load_pt(destination/'initial.pt.gz');output=load_pt(destination/'terminal.pt.gz')
    if initial.get('population')!=50 or len(output['coords'])!=50:raise ValueError('Engineering replay population differs')
    frozen(destination/'ENGINEERING_REPLAY.json',dict(**spec,source_manifest_sha256=value_sha(spec['files']),
        source_trajectory_evaluations=50,new_trajectory_evaluations=0,scientific_candidates=0,
        interpretation='Completed trajectories reused; geometry, chemistry and endpoint scoring revalidated'))
    return initial,output
