"""The observed selection events, rather than arbitrary bins, define scope."""
from pathlib import Path
import numpy as np
import pandas as pd
from .io import read_json
from .trajectory_source import trajectory_paths,open_trajectory

SCOPE = 'actual_resampling_events'
REPRESENTATIONS = {'predicted_endpoint': 'predicted', 'proposal_state': 'proposal'}
FORBIDDEN_COLUMNS = ('descendant', 'terminal', 'outcome', 'success', 'failure')


def stage_edges(times, width):
    start, end = np.round([min(times), max(times)], 6)
    if end <= start or width <= 0:
        raise ValueError('At least two selection times and positive stage width required')
    return np.round(np.r_[start, np.arange(start+width, end-1e-7, width), end], 6).tolist()


def stage_for(times, edges):
    times = np.round(np.asarray(times, float), 6)
    if np.any(times < edges[0]) or np.any(times > edges[-1]):
        raise ValueError('Time outside observed selection window')
    return np.minimum(np.searchsorted(edges, times, side='right')-1, len(edges)-2)


def discover_scope(campaign, cfg):
    """Require matched schedules across analyzed selection arms/batches."""
    schedules,completed = [],{}
    for path in trajectory_paths(campaign):
        arm, batch = path.parent.parent.name, int(path.parent.name.split('_')[1])
        if cfg.get('include_batches') is not None and batch not in cfg['include_batches']:
            continue
        if arm in cfg['selection_arms']+[cfg['background_arm']] and (path.parent/'COMPLETE.json').exists():
            completed.setdefault(batch,set()).add(arm)
        if arm not in cfg['selection_arms'] or not (path.parent/'COMPLETE.json').exists():
            continue
        with open_trajectory(path) as z:
            steps = np.flatnonzero(z['resampled']).tolist()
            times = z['score_time'][steps, 0].astype(float).tolist()
        if not steps:
            raise ValueError(f'No actual selection events: {path}')
        schedules.append((steps, times))
    if not schedules:
        raise ValueError('No completed selection batches')
    required = set(cfg['selection_arms']+[cfg['background_arm']])
    if any(arms!=required for arms in completed.values()):
        raise ValueError('Each analyzed batch requires all selection arms and its matched background')
    steps, times = schedules[0]
    if any(steps!=s or not np.allclose(times,t,rtol=0,atol=1e-7) for s,t in schedules):
        raise ValueError('Analyze different selection schedules as separate campaigns')
    edges = stage_edges(times, cfg['stage_width'])
    assignments = stage_for(times, edges)
    stages = []
    for i, (start,end) in enumerate(zip(edges,edges[1:])):
        selected = np.asarray(times)[assignments==i]
        if len(selected):
            stages.append({'stage': i, 'stage_start': start, 'stage_end': end,
                'right_inclusive': i==len(edges)-2, 'n_events': len(selected),
                'first_score_time': float(selected.min()), 'last_score_time': float(selected.max()),
                'mean_score_time': float(selected.mean())})
    return {'analysis_scope': SCOPE, 'selection_arms': cfg['selection_arms'],
        'background_arm': cfg['background_arm'], 'steps': steps, 'score_times': times,
        'window_start': edges[0], 'window_end': edges[-1], 'stage_edges': edges, 'stages': stages,
        'time_semantics': 'Scope follows score_time; selected proposals can be one integration step later.'}


def validate_frame(df, scope):
    bad = [c for c in df if any(word in c for word in FORBIDDEN_COLUMNS)]
    if bad:
        raise ValueError('Outcome columns forbidden in selection analysis: '+str(bad))
    if df.empty or not df.step.isin(scope['steps']).all():
        raise ValueError('Missing/out-of-scope selection frames')
    selected = df.arm.isin(scope['selection_arms'])
    if not df.loc[selected,'resampled'].all():
        raise ValueError('Selection arm contains a non-selection frame')
    if df.loc[~selected,'resampled'].any() or not (df.loc[~selected,'arm']==scope['background_arm']).all():
        raise ValueError('Invalid matched background')
    expected = dict(zip(scope['steps'],scope['score_times']))
    if not np.allclose(df.score_time,df.step.map(expected),rtol=0,atol=1e-7):
        raise ValueError('Score time disagrees with observed selection event')
    if not np.array_equal(df.stage.to_numpy(),stage_for(df.score_time,scope['stage_edges'])):
        raise ValueError('Stage assignment disagrees with selection scope')
    if {'node_id','representation'}<=set(df) and df.duplicated(['representation','node_id']).any():
        raise ValueError('Duplicated candidate/representation')
    if {'representation','batch','probability','offspring_count'}<=set(df):
        groups = df.groupby(['representation','arm','batch','step'],sort=False)
        sums = groups[['probability','offspring_count']].sum()
        if not np.allclose(sums.probability,1,atol=1e-6) or not np.array_equal(sums.offspring_count,groups.size()):
            raise ValueError('Incomplete candidate population or invalid weights/counts')


def load(analysis, split='discovery', representations=None):
    analysis = Path(analysis)
    cfg = read_json(analysis/'config.json')
    scope = read_json(analysis/'selection_scope.json')
    if cfg.get('analysis_scope')!=SCOPE or scope['analysis_scope']!=SCOPE:
        raise ValueError('Legacy full-trajectory results are not a selection evidence source')
    cache = analysis/'features.parquet'
    if not cache.exists():
        record = analysis/'feature_cache_manifest.json'
        if not record.exists() or read_json(record).get('status')!='retired':
            raise FileNotFoundError('Selection feature cache missing and no retired-cache manifest exists')
        import subprocess,sys
        script = Path(__file__).resolve().parents[2]/'scripts/materialize_feature_cache.py'
        subprocess.run([sys.executable,str(script),'--analysis',str(analysis.resolve())],check=True)
    filters = [('split','==',split),('representation','in',representations or cfg['analysis_representations'])]
    df = pd.read_parquet(cache,filters=filters)
    validate_frame(df,scope)
    return df,cfg,read_json(analysis/'feature_catalog.json'),scope
