"""Evidence-bound rollback plans and compact paired regression diagnostics."""
import copy
from pathlib import Path
import numpy as np
import pandas as pd
from ..io import read_json, digest, write_table
from .prototypes import write_json


def diagnose(evidence, output, parent=3, comparisons=(4, 5)):
    evidence, output = Path(evidence), Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    parent_root = evidence / f'round_{parent:02d}/local'
    window = read_json(parent_root / 'execution_report.json')['window']
    a = pd.read_csv(parent_root / 'coordinate_injection_motion.csv')
    a = a[(a.arm == 'gradient') & (a.time >= window[0]-1e-7) & (a.time < window[1]-1e-7)]
    ca = pd.read_csv(parent_root / 'candidate_metrics.csv')
    ca = ca[ca.arm == 'gradient']
    rows, candidate_rows, sources = [], [], []
    for number in comparisons:
        root = evidence / f'round_{number:02d}/local'
        b = pd.read_csv(root / 'coordinate_injection_motion.csv')
        b = b[(b.arm == 'gradient') & (b.time >= window[0]-1e-7) & (b.time < window[1]-1e-7)]
        x = a.merge(b, on=['arm', 'batch', 'time'], suffixes=('_parent', '_child'), validate='one_to_one')
        if len(x) != len(a) or len(x) != len(b):
            raise ValueError('Unmatched dose grids')
        for batch, g in x.groupby('batch'):
            g = g.sort_values('time');first = g.iloc[0]
            pa, ch = (g[f'mean_total_injection_rms_A_{k}'].to_numpy() for k in ('parent', 'child'))
            ratios = np.divide(pa, ch, out=np.full_like(pa, np.nan), where=ch>0)
            rows.append({'parent_round': parent, 'child_round': number, 'batch': batch,
                         'first_time': float(first.time), 'first_parent_rms_A': pa[0], 'first_child_rms_A': ch[0],
                         'first_parent_child_ratio': ratios[0], 'later_median_parent_child_ratio': np.nanmedian(ratios[1:]),
                         'sum_mean_rms_parent_A': pa.sum(), 'sum_mean_rms_child_A': ch.sum(),
                         'first_fraction_squared_mean_rms_parent': pa[0]**2/(pa@pa),
                         'first_fraction_squared_mean_rms_child': ch[0]**2/(ch@ch)})
        cb = pd.read_csv(root / 'candidate_metrics.csv');cb = cb[cb.arm == 'gradient']
        paired = ca.merge(cb, on=['batch', 'slot', 'seed'], suffixes=('_parent', '_child'), validate='one_to_one')
        if len(paired) != len(ca) or len(paired) != len(cb):
            raise ValueError('Unpaired candidate records')
        for batch, g in paired.groupby('batch'):
            both = g.valid_connected_parent & g.valid_connected_child
            same = both & (g.smiles_parent == g.smiles_child)
            for label, mask in [('all_candidates', np.ones(len(g), bool)), ('both_valid_same_graph', same), ('both_valid_changed_graph', both & ~same)]:
                selected = g[mask];record = {'parent_round': parent, 'child_round': number, 'batch': batch, 'subset': label, 'n': len(selected)}
                for field in ('pic50_on_rescore', 'mmff_relief_per_heavy', 'relax_rms_surround_A'):
                    delta = selected[f'{field}_child']-selected[f'{field}_parent']
                    record[f'{field}_change_mean'] = float(delta.mean()) if delta.notna().any() else None
                    record[f'{field}_paired_n'] = int(delta.notna().sum())
                candidate_rows.append(record)
        sources.extend([parent_root/n for n in ('candidate_metrics.csv','coordinate_injection_motion.csv')] + [root/n for n in ('candidate_metrics.csv','coordinate_injection_motion.csv')])
    write_table(output / 'paired_dose_summary.csv', rows)
    write_table(output / 'paired_graph_summary.csv', candidate_rows)
    write_json(output / 'diagnosis.json', {'parent_round': parent, 'comparisons': list(comparisons), 'window': window,
        'dose_semantics': 'Squared ensemble-mean RMS is not the mean squared particle injection. Replay parent to recover exact dose concentration.',
        'graph_semantics': 'Terminal same/changed graph subsets are post-treatment descriptive comparisons, not causal strata.',
        'independence': 'Two reused seed42 development batches; candidate pairs and time nodes are not independent replicates.',
        'sources': [{'path': str(p.resolve()), 'sha256': digest(p)} for p in dict.fromkeys(sources)]})
    return rows, candidate_rows


def freeze(campaign, parent_program, reference, output, evidence, number, name, changes, reason, kind='single_factor'):
    campaign, parent_program, reference, output, evidence = map(Path, (campaign,parent_program,reference,output,evidence))
    c, p = read_json(campaign), read_json(parent_program)
    if output.exists() or evidence.exists():
        raise FileExistsError('Immutable plan already exists')
    if number != max(r['round'] for r in c['rounds'])+1 or not number <= c['maximum_rounds'] <= 30:
        raise ValueError('Sequential round within remaining budget required')
    parent = next(r for r in c['rounds'] if r['round'] == p['round'])
    if parent['status'] != 'completed' or c['rounds_started'] != c['rounds_completed']:
        raise ValueError('Finish and retain current evaluation before freezing next round')
    if digest(reference) != p['reference_sha256'] or p['seed'] != 42:
        raise ValueError('Parent reference/seed mismatch')
    allowed = {'native_rms_ratio','dose_reference','initial_update_dose','preserve_native_rigid_pose',
               'constraints.max_atom_step_A','mixture_temperature','robust_delta'}
    if set(changes)-allowed or kind not in ('exact_replay','single_factor'):
        raise ValueError('Unsupported rollback factor')
    if (kind == 'exact_replay' and changes) or (kind == 'single_factor' and len(changes) != 1):
        raise ValueError('Replay is unchanged; ablation changes exactly one factor')
    q = copy.deepcopy(p);before = {}
    for field, value in changes.items():
        parts = field.split('.');container = q
        for key in parts[:-1]:container = container[key]
        before[field] = container.get(parts[-1], {'dose_reference':'observed_native','initial_update_dose':'native','preserve_native_rigid_pose':False}.get(field))
        if before[field] == value:raise ValueError('Factor does not change the parent')
        container[parts[-1]] = value
    q['round'] = number
    q['backtracking'] = {'parent_round': p['round'], 'parent_program_sha256': digest(parent_program), 'kind': kind,
                         'changes': changes, 'before': before, 'reason': reason}
    write_json(output, q)
    write_json(evidence, {'schema_version':'coordinate-backtrack-plan-1.0','round':number,'parent_round':p['round'],
        'preserved_commit':'adf9f3670492a666aa109597eefde82d43c0c87f','kind':kind,'changes':changes,'before':before,'reason':reason,
        'program_sha256':digest(output),'reference_sha256':digest(reference),'window':q['window'],
        'validity':'Conditional coordinate imitation; no causal affinity claim. Original Steer remains protected.',
        'fallback':'If a factor regresses, restore the frozen parent and test another identifiable factor; partial regression is not an automatic stop.'})
    c['rounds'].append({'round':number,'campaign':name,'arms':['gradient'],'batches':parent['batches'],'n_per_arm':parent['n_per_arm'],
        'seed':42,'reference_sha256':digest(reference),'reward_view':q['reward_view'],'native_rms_ratio':q['native_rms_ratio'],
        'dose_reference':q.get('dose_reference','observed_native'),'parent_round':p['round'],'kind':kind,'changes':changes,'status':'frozen'})
    c.update(status='backtracking',decision='Resume at user request: diagnose regressions, restore tested parents and continue single-factor exploration.')
    c['backtrack_triggers']=['Intermediate geometry regresses','Final physical/head/validity tradeoff','Reward proxy improves without actual outcome']
    c['stop_conditions']=['30 rounds started','User requests stop','Execution cannot satisfy finite-gradient, no-SMC or data-preservation contract']
    write_json(campaign,c)
    return q
