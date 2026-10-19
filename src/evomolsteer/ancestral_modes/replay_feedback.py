"""Compact, provenance-bound path-replay feedback for the next real API call."""
import gzip
import json
import csv
from pathlib import Path
import numpy as np

from evomolsteer.io import digest,read_json
from evomolsteer.path_replay.common import groups_for,read_batch,batch_path
from evomolsteer.path_replay.local_research import load
from evomolsteer.path_replay.analysis import distribution
from evomolsteer.path_replay.local_features import measure


def entry(repo,source,program):
    repo=Path(repo);cfg,m,_=load(repo,repo/source['manifest']);work=repo/source['work']
    c=next(c for c in m['conditions'] if c['id']==source['condition'])
    if read_json(repo/c['program'])!=program:raise ValueError('Feedback does not describe the chosen parent program')
    teacher=read_json(repo/c['reference']);records=[];steps=[];geometry=[];sources=[]
    for g in groups_for(cfg,c):
        r=read_batch(work,cfg,c,g)
        if not r or r['status']!='complete':raise ValueError('Parent feedback is incomplete')
        records.extend(r['records']);folder=batch_path(work,c,g).parent
        with gzip.open(folder/'guidance.json.gz','rt') as f:
            active=[s for s in json.load(f)['steps'] if s['active']];steps.extend(active)
        with np.load(folder/'states.npz') as states:
            geometry.extend(measure(states['half_coords_A'],states['terminal_coords_A'],states['mask'],teacher,program,
                assignments=active[-1]['actual_local_assignments']))
        sources.append({'group':g['index'],'receipt_sha256':digest(folder/'COMPLETE.json'),
                        'source_code':read_json(folder.parent/'source_audit.json')['code_commit']})
    keys=('injected_rms_A_mean','predictive_flow_rms_A_mean','assignment_switch_fraction','gradient_native_observed_cosine_mean',
          'direct_support_count_mean','direct_support_fraction_mean','direct_injected_rms_A_mean','outside_injected_rms_A_mean',
          'atom_clip_fraction','point_rms_A','unmatched_input_gradient_rms_fraction_mean')
    statistics={k:float(np.mean([s[k] for s in steps if k in s and s[k] is not None]))
                for k in keys if any(k in s and s[k] is not None for s in steps)}
    from evomolsteer.path_replay.local_research import control_summary
    control_stats=control_summary(steps,c['n']//len(groups_for(cfg,c)))
    control_stats['nonzero_seed_updates']=control_stats.pop('nonzero_updates')
    statistics.update(control_stats)
    snapshots=[{k:s[k] for k in ('step','point_rms_A','injected_rms_A_mean','atom_clip_fraction') if k in s}
               for s in steps[::max(1,len(steps)//5)]]
    shape={k:float(np.mean([r[k] for r in geometry if r.get(k) is not None]))
           for k in geometry[0] if k not in ('generated_slots_at_half','support_size','assignment_source') and any(r.get(k) is not None for r in geometry)}
    controls={}
    observational={}
    table=work/'analysis/local_formation_retention.csv'
    if table.exists():
        with table.open(encoding='utf-8',newline='') as stream:
            for row in csv.DictReader(stream):
                values=observational.setdefault(row['condition'],{})
                for key,value in row.items():
                    if key.startswith(('half_','terminal_','suffix_')) and value:
                        values.setdefault(key,[]).append(float(value))
    for reference in m['conditions']:
        if reference['id']==c['id']:continue
        rows=[]
        for g in groups_for(cfg,reference):
            r=read_batch(work,cfg,reference,g)
            if r:rows.extend(r['records'])
        controls[reference['id']]={'performance':distribution(rows,reference['n'],cfg['tail_threshold']),
            'observational_half_match_geometry':{k:float(np.mean(v)) for k,v in observational.get(reference['id'],{}).items()},
            'interpretation':'Common half-state observational matching, not necessarily this controller actual rewarded support. Read geometry and affinity separately.'}
    return {'program':program,'metrics':distribution(records,c['n'],cfg['tail_threshold']),
        'response':{'active_updates':len(steps),'nonzero_updates':sum(s['nonzero_molecules']>0 for s in steps),
                    'nonzero_seed_updates':sum(s['nonzero_molecules'] for s in steps),
                    'statistics':statistics,'within_window_diagnostics':snapshots},
        'local_outcomes':{'measured_actual_half_and_tracked_suffix':shape,'paired_current_controls':controls,
            'analysis_code_sha256':{'feedback':digest(__file__),
                'local_features':digest(Path(__file__).parents[1]/'path_replay/local_features.py')},
            'scope':m.get('comparison_scope','strict_historical_replay'),
            'source_manifest':source['manifest'],'manifest_sha256':digest(repo/source['manifest']),'batch_receipts':sources,
            'assignment_source':'Actual last-control-forward endpoint reward matching, held for tracked suffix. Terminal rematching is a separate spatial-pattern diagnostic, not slot retention.',
            'teacher_clocks':'Actual half target for half diagnostics; actual final target for terminal diagnostics. Last live point gap is pre-update endpoint comparison and is not the pair/pocket reward value.',
            'limits':'A selected B50 pilot does not establish N1000 enrichment. Support outside input gradients reflect generator coupling, not additional teacher points. Slot migration alone cannot prove loss of the spatial motif or justify a chemical-graph gate.'}}
