"""Audit full-horizon execution and the matched continuation ablation."""
import argparse
import json
from pathlib import Path
import numpy as np
from evomolsteer.io import read_json,write_json
from evomolsteer.trajectory_source import open_trajectory

p=argparse.ArgumentParser();p.add_argument('--campaign',required=True);p.add_argument('--output',required=True)
a=p.parse_args();root=Path(a.campaign);cfg=read_json(root/'config.json')['experiment']
if read_json(root/'COMPLETE.json')['status']!='complete':raise ValueError('Campaign incomplete')
rows=[];prefix=[]
for directory in sorted(root.glob('*/batch_*')):
    path=directory/'guidance_trace.jsonl'
    if not path.exists():continue
    arm=directory.parent.name;batch=int(directory.name.split('_')[1])
    trace=[json.loads(line) for line in path.read_text().splitlines()]
    if len(trace)!=100:raise AssertionError('Missing integration steps')
    evaluation=sum(r['reward_evaluated'] for r in trace)
    if arm in ('multistage_full','static_full','gradient_zero') or arm.startswith('multistage_r'):
        if evaluation!=100:raise AssertionError('Full-horizon reward not evaluated 100 times')
    if arm=='multistage_window' and evaluation!=51:raise AssertionError('Window ablation changed support')
    maximum=max(max(r['injection_max_atom_A']) for r in trace)
    if maximum>cfg['max_atom_step_A']+1e-7:raise AssertionError('Displacement cap violated')
    components=[]
    for r in trace:
        if 'component_gradient_norms' not in r:continue
        n=np.array(r['component_gradient_norms']);c=np.array(r['component_gradient_cosine'])
        implied=np.sqrt(np.maximum(n[:,0]**2+n[:,1]**2+2*n[:,0]*n[:,1]*c,0))
        actual=np.array(r['gradient_norm']);error=np.abs(implied-actual)
        # Strong cancellation makes relative-to-total error misleading.
        scaled=error/np.maximum(n.sum(1),1e-8)
        if scaled.max()>1e-4:raise AssertionError('Component pullbacks do not sum to total gradient')
        components.append({'step':r['step'],'maximum_norm_identity_error':float(error.max()),
                           'maximum_error_relative_to_sum_component_norms':float(scaled.max()),
                           'median_total_over_sum_component_norms':float(np.median(actual/np.maximum(n.sum(1),1e-8)))})
    rows.append({'arm':arm,'batch':batch,'evaluated_steps':evaluation,
        'nonzero_injection_steps':sum(any(v>0 for v in r['injection_max_atom_A']) for r in trace),
        'maximum_atom_injection_A':maximum,'component_pullback_checks':components})
for batch in range(cfg['n']//cfg['batch']):
    full=root/'multistage_full'/f'batch_{batch:03d}'/'trajectory.h5'
    window=root/'multistage_window'/f'batch_{batch:03d}'/'trajectory.h5'
    if not full.exists() or not window.exists():continue
    with open_trajectory(full) as f,open_trajectory(window) as w:
        checks={}
        for name in ['current_coords','current_atomics','current_bonds','current_charges','predicted_coords']:
            checks[name]=bool(np.array_equal(f[name][:52],w[name][:52]))
        checks['proposal_coords_through_step50']=bool(np.array_equal(f['proposal_coords'][:51],w['proposal_coords'][:51]))
        if not all(checks.values()):raise AssertionError('Continuation arms differ before the late intervention')
        prefix.append({'batch':batch,'bitwise_equal_checks':checks,
                       'meaning':'Same states through t=.51 and same proposals through t=.50; divergence can begin at step51 added correction'})
write_json(a.output,{'status':'passed','campaign':str(root),'activity_and_component_audits':rows,
                    'matched_continuation_prefixes':prefix})
print('Verified full-horizon activity and',len(prefix),'matched continuation prefixes')
