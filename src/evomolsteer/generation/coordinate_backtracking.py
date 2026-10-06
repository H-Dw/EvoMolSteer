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
               'constraints.max_atom_step_A','constraints.max_pair_distance_change_A','constraints.initial_atom_step_A','mixture_temperature','robust_delta'}
    if set(changes)-allowed or kind not in ('exact_replay','single_factor'):
        raise ValueError('Unsupported rollback factor')
    if (kind == 'exact_replay' and changes) or (kind == 'single_factor' and len(changes) != 1):
        raise ValueError('Replay is unchanged; ablation changes exactly one factor')
    q = copy.deepcopy(p);before = {}
    for field, value in changes.items():
        parts = field.split('.');container = q
        for key in parts[:-1]:container = container[key]
        before[field] = container.get(parts[-1], {'dose_reference':'observed_native','initial_update_dose':'native','preserve_native_rigid_pose':False}.get(field))
        if field=='constraints.initial_atom_step_A' and before[field] is None:before[field]=container['max_atom_step_A']
        if before[field] == value:raise ValueError('Factor does not change the parent')
        container[parts[-1]] = value
    q['round'] = number
    for key in ('native_rms_ratio','mixture_temperature','robust_delta'):
        if not np.isfinite(q[key]) or (q[key]<0 if key=='native_rms_ratio' else q[key]<=0):raise ValueError('Invalid control magnitude')
    if q['native_rms_ratio']>1 or not np.isfinite(q['constraints']['max_atom_step_A']) or q['constraints']['max_atom_step_A']<=0:
        raise ValueError('Invalid dose/atom cap')
    if 'max_pair_distance_change_A' in q['constraints'] and (not np.isfinite(q['constraints']['max_pair_distance_change_A']) or q['constraints']['max_pair_distance_change_A']<=0):
        raise ValueError('Invalid pair-distance cap')
    if 'initial_atom_step_A' in q['constraints'] and not (np.isfinite(q['constraints']['initial_atom_step_A']) and 0<q['constraints']['initial_atom_step_A']<=q['constraints']['max_atom_step_A']):raise ValueError('Initial cap must only tighten the first controlled update')
    if q.get('initial_update_dose','native') not in ('native','cap_to_flow') or q.get('dose_reference','observed_native') not in ('observed_native','predictive_flow'):
        raise ValueError('Invalid dose policy')
    if q.get('initial_update_dose')=='cap_to_flow' and q.get('dose_reference','observed_native')!='observed_native':raise ValueError('Initial cap requires native calibration')
    if 'preserve_native_rigid_pose' in q and not isinstance(q['preserve_native_rigid_pose'],bool):raise ValueError('Pose projection must be boolean')
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


def record(campaign, evidence, number):
    """Record an already preserved evaluation; keep every outcome axis separate."""
    from .terminal_comparison import compare
    campaign,evidence=Path(campaign),Path(evidence)
    c=read_json(campaign);r=next(v for v in c['rounds'] if v['round']==number)
    if c['rounds_started']!=number or c['rounds_completed']!=number-1 or r['status']!='running':raise ValueError('Round counters/state mismatch')
    local=evidence/f'round_{number:02d}/local';baseline=evidence/'round_01/local';destination=local.parent/'comparison'
    if destination.exists():raise FileExistsError('Comparison is immutable')
    nw=read_json(baseline/'window/report.json');aw=read_json(local/'window/report.json')
    native=nw['results']['unguided'];actual=aw['results']['gradient']
    at=read_json(local/'terminal_report.json');nb=read_json(baseline/'terminal_report.json')
    terminal=at['results']['gradient'];nt=nb['results']['unguided']
    execution=read_json(local/'execution_report.json');audit=read_json(local/'coordinate_audit.json')
    if terminal['n']!=r['n_per_arm'] or not execution['no_particle_resampling'] or execution['outside_window_injection'] or not audit['coordinate_preflight']['passed']:
        raise ValueError('Missing candidates or failed execution/gradient check')
    if 'gradient_zero' in r['arms'] and aw['zero_equivalence_passed'] is not True:
        raise ValueError('New architecture requires full zero/native equivalence')
    original=evidence.parent/'ck2_terminal_seed42_20261005/reference/terminal_report.json'
    manifest={'groups':[{'label':'Original Steer','arm':'single','terminal_report':str(original.resolve())},
        {'label':'Native','arm':'unguided','terminal_report':str((baseline/'terminal_report.json').resolve()),'execution_report':str((baseline/'execution_report.json').resolve())},
        {'label':r['campaign'],'arm':'gradient','terminal_report':str((local/'terminal_report.json').resolve()),'execution_report':str((local/'execution_report.json').resolve())}],
        'native':'Native','steer':'Original Steer','tests':[r['campaign']],'round':number,'maximum_rounds':c['maximum_rounds']}
    write_json(local.parent/'comparison_manifest.json',manifest);compare(local.parent/'comparison_manifest.json',destination)
    outcome={'round':number,'parent_round':r['parent_round'],'kind':r['kind'],'changes':r['changes'],
        'shape_improvement_fraction':1-actual['symmetric_shape_A']/native['symmetric_shape_A'],
        'actual_window_shape':actual,'native_shape':native,'terminal':terminal,'dose_audit':audit,
        'all_head_change_vs_native':terminal['all_pic50_on_rescore_mean']-nt['all_pic50_on_rescore_mean'],
        'MMFF_relief_relative_change':terminal['all_mmff_relief_per_heavy_median']/nt['all_mmff_relief_per_heavy_median']-1,
        'surround_RMS_improvement_fraction':1-terminal['all_relax_rms_surround_A_mean']/nt['all_relax_rms_surround_A_mean'],
        'interpretation':'Paired reused development batches; partial regression triggers rollback reasoning, not automatic exploration stop.'}
    outcome['paired_batch_outcomes']=paired_batch_outcomes(aw,nw,at,nb)
    if r['kind']=='exact_replay':
        parent=evidence/f"round_{r['parent_round']:02d}/local"
        pt=read_json(parent/'terminal_report.json')['results']['gradient'];pw=read_json(parent/'window/report.json')['results']['gradient']
        outcome['replay_differences']={f:terminal[f]-pt[f] for f in ('valid_connected','pb_fast_pass','all_pic50_on_rescore_mean','all_mmff_relief_per_heavy_median','all_relax_rms_surround_A_mean')}
        outcome['replay_differences']['symmetric_shape_A']=actual['symmetric_shape_A']-pw['symmetric_shape_A']
        outcome['replay_agreement_1e_7']=all(abs(v)<1e-7 for v in outcome['replay_differences'].values())
    write_json(evidence/f'round_{number:02d}.outcome.json',outcome)
    r['status']='completed';r['inference_commit']=execution['code_commit'];c.update(rounds_completed=number,status='backtracking');write_json(campaign,c)
    return outcome


def paired_batch_outcomes(actual_window,native_window,actual_terminal,native_terminal):
    """Separate batch directions and denominators; never count particles as replicates."""
    def batches(report,arm):
        rows=[v for v in report['batch_results'] if v['arm']==arm]
        result={v['batch']:v for v in rows}
        if len(result)!=len(rows):raise ValueError('Duplicate batch evidence')
        return result
    aw,nw,at,nt=(batches(report,arm) for report,arm in
                 [(actual_window,'gradient'),(native_window,'unguided'),(actual_terminal,'gradient'),(native_terminal,'unguided')])
    if not aw.keys()==nw.keys()==at.keys()==nt.keys():raise ValueError('Paired batch evidence required')
    output=[]
    for batch in sorted(aw):
        if not aw[batch]['n']==nw[batch]['n']==at[batch]['n']==nt[batch]['n']:raise ValueError('Attempted denominators mismatch')
        output.append({'batch':batch,'n_attempted':at[batch]['n'],
            'shape_improvement_fraction':1-aw[batch]['symmetric_shape_A']/nw[batch]['symmetric_shape_A'],
            'all_head_change':at[batch]['all_pic50_on_rescore_mean']-nt[batch]['all_pic50_on_rescore_mean'],
            'MMFF_relief_relative_change':at[batch]['all_mmff_relief_per_heavy_median']/nt[batch]['all_mmff_relief_per_heavy_median']-1,
            'surround_RMS_improvement_fraction':1-at[batch]['all_relax_rms_surround_A_mean']/nt[batch]['all_relax_rms_surround_A_mean'],
            'valid_count_change':at[batch]['valid_connected']-nt[batch]['valid_connected'],
            'PB_count_change':at[batch]['pb_fast_pass']-nt[batch]['pb_fast_pass'],
            'MMFF_n_actual':at[batch]['all_mmff_relief_per_heavy_n'],'MMFF_n_native':nt[batch]['all_mmff_relief_per_heavy_n'],
            'surround_n_actual':at[batch]['all_relax_rms_surround_A_n'],'surround_n_native':nt[batch]['all_relax_rms_surround_A_n']})
    return output


def freeze_contrast(campaign,parent_program,parent_reference,new_reference,output,evidence,number,name,reason,designer_contract=None):
    """Change reward architecture while requiring exact paired selected-moment parity."""
    from .window_reference import load_reference
    campaign,parent_program,parent_reference,new_reference,output,evidence=map(Path,
        (campaign,parent_program,parent_reference,new_reference,output,evidence))
    c=read_json(campaign);p=read_json(parent_program)
    if output.exists() or evidence.exists():raise FileExistsError('Immutable plan already exists')
    if number!=max(r['round'] for r in c['rounds'])+1 or not number<=c['maximum_rounds']<=30:raise ValueError('Sequential round within budget required')
    parent=next(r for r in c['rounds'] if r['round']==p['round'])
    if parent['status']!='completed' or c['rounds_started']!=c['rounds_completed']:raise ValueError('Retain current evaluation first')
    if digest(parent_reference)!=p['reference_sha256'] or p['seed']!=42 or p['reward_view']!='coordinate_mixture':raise ValueError('Verified mixture parent required')
    a,b=load_reference(parent_reference),load_reference(new_reference)
    for key in ('window','times','features','batches','regions','channel','sources','spatial_anchor','control_representation','required_input_sha256'):
        if a[key]!=b[key]:raise ValueError('Contrast representation or source differs')
    if len(a['frames'])!=len(b['frames']) or len(b['frames'])!=len(b['times']):raise ValueError('Complete paired frames required')
    checked=0
    for f,g in zip(a['frames'],b['frames']):
        if f['time']!=g['time']:raise ValueError('Paired frame times required')
        if len(f['modes'])!=len(g['modes']):raise ValueError('Unpaired modes')
        for m,n in zip(f['modes'],g['modes']):
            for key in ('source_batch','center_A','covariance_A2'):
                if m[key]!=n[key]:raise ValueError('Selected moment parity required')
            for key in ('background_center_A','background_covariance_A2','background_support_q90','background_support_q98','paired_gaussian_KL_nats'):
                if key not in n:raise ValueError('Empirical background/support missing')
            checked+=1
    if designer_contract is None:raise ValueError('Designer contrast contract required')
    contract=read_json(designer_contract)
    if contract['agent']!='Designer' or contract['schema_version']!='selection-contrast-designer-contract-1.0':raise ValueError('Wrong Designer contract')
    sources=contract['reference_contract']['sources']
    if sources['active_contrast_reference']['sha256']!=digest(new_reference) or sources['source_selected_reference']['sha256']!=digest(parent_reference):raise ValueError('Designer reference binding differs')
    spec=contract['exploratory_parameters'];defaults={'dose_reference':'observed_native','initial_update_dose':'native','preserve_native_rigid_pose':False}
    if any(p.get(key,defaults.get(key))!=value for key,value in spec['inherit_without_simultaneous_retuning'].items()):raise ValueError('Contrast control must inherit the declared parent')
    q=copy.deepcopy(p);q.update(round=number,reward_view='selection_contrast',contrast_bound_nats=spec['contrast_bound_nats'],reference_sha256=digest(new_reference),designer_sha256=digest(designer_contract))
    changes={'reward_view':'selection_contrast','contrast_bound_nats':1.,'reference_sha256':digest(new_reference)}
    q['backtracking']={'parent_round':p['round'],'parent_program_sha256':digest(parent_program),'kind':'architecture_change','changes':changes,'reason':reason}
    # Validate the executable before mutating the campaign manifest.
    from .coordinate_contrast import make_coordinate_reward
    make_coordinate_reward(q,b)
    write_json(output,q);write_json(evidence,{'round':number,'parent_round':p['round'],'kind':'architecture_change',
        'changes':changes,'reason':reason,'selected_moment_pairs_checked':checked,'selected_moments_exactly_equal':True,
        'program_sha256':digest(output),'old_reference_sha256':digest(parent_reference),'reference_sha256':digest(new_reference),
        'designer_contract_sha256':digest(designer_contract),
        'window':q['window'],'fallback':'Restore parent; weak contrast and unsupported states receive smaller/zero dose, never increase eta to cancel those gates.'})
    c['rounds'].append({'round':number,'campaign':name,'arms':['unguided','gradient_zero','gradient'],'batches':parent['batches'],'n_per_arm':parent['n_per_arm'],
        'seed':42,'reference_sha256':digest(new_reference),'reward_view':q['reward_view'],'native_rms_ratio':q['native_rms_ratio'],
        'dose_reference':q.get('dose_reference','observed_native'),'parent_round':p['round'],'kind':'architecture_change','changes':changes,'status':'frozen'})
    c['status']='backtracking';write_json(campaign,c);return q


def freeze_shape(campaign,parent_program,parent_reference,new_reference,output,evidence,number,name,reason,designer_contract):
    """Freeze a declared tensor representation/availability ablation of a parent."""
    from .window_reference import load_reference
    from .coordinate_shape import CoordinateShapeReward
    campaign,parent_program,parent_reference,new_reference,output,evidence,designer_contract=map(Path,
        (campaign,parent_program,parent_reference,new_reference,output,evidence,designer_contract))
    c,p,contract=read_json(campaign),read_json(parent_program),read_json(designer_contract)
    if output.exists() or evidence.exists():raise FileExistsError('Immutable shape plan exists')
    if number!=max(r['round'] for r in c['rounds'])+1 or not number<=c['maximum_rounds']<=30:
        raise ValueError('Sequential shape round within budget required')
    parent=next(r for r in c['rounds'] if r['round']==p['round'])
    if parent['status']!='completed' or c['rounds_started']!=c['rounds_completed']:
        raise ValueError('Retain preceding result before a new representation test')
    if p['seed']!=42 or digest(parent_reference)!=p['reference_sha256']:
        raise ValueError('Parent seed/reference mismatch')
    a,b=load_reference(parent_reference),load_reference(new_reference)
    same=('window','times','batches','regions','channel','sources','required_input_sha256',
          'spatial_anchor','control_representation','spatial_width_A')
    if any(a[k]!=b[k] for k in same):raise ValueError('Shape parent source/window/region contract differs')
    if contract['schema_version']!='shape-designer-contract-1.0' or contract['agent']!='Designer':
        raise ValueError('Shape Designer contract required')
    if contract['parent_reference_sha256']!=digest(parent_reference) or contract['reference_sha256']!=digest(new_reference):
        raise ValueError('Designer reference binding differs')
    defaults={'dose_reference':'observed_native','initial_update_dose':'native','preserve_native_rigid_pose':False}
    controls=contract['inherit_without_simultaneous_retuning']
    required={'native_rms_ratio','mixture_temperature','robust_delta','dose_reference','initial_update_dose','preserve_native_rigid_pose','constraints'}
    if set(controls)!=required or any(p.get(k,defaults.get(k))!=v for k,v in controls.items()):
        raise ValueError('Shape controls must inherit the declared parent')
    if not contract.get('availability_change_disclosed'):
        raise ValueError('Single-slot support change must be disclosed')
    if len(b['frames'])!=len(a['frames']) or any([m['source_batch'] for m in f['modes']]!=b['batches'] for f in b['frames']):
        raise ValueError('Complete shape batch/time reference required')
    changes={'reward_view':'shape_mixture','reference_sha256':digest(new_reference),
             'observable_family':'six directional central second moments; informative slots>=2'}
    q=copy.deepcopy(p);q.update(round=number,reward_view='shape_mixture',reference_sha256=digest(new_reference),designer_sha256=digest(designer_contract))
    q['backtracking']={'parent_round':p['round'],'parent_program_sha256':digest(parent_program),
        'kind':'representation_ablation','changes':changes,'reason':reason}
    CoordinateShapeReward(q,b)
    write_json(output,q)
    write_json(evidence,{'round':number,'parent_round':p['round'],'kind':'representation_ablation','changes':changes,
        'reason':reason,'program_sha256':digest(output),'old_reference_sha256':digest(parent_reference),
        'reference_sha256':digest(new_reference),'designer_contract_sha256':digest(designer_contract),'window':q['window'],
        'availability_change':'Unlike centroid/spread parent, single NOS slots lack shape direction and receive zero dose; full attempted denominators retained.',
        'fallback':'If only tensor residual improves, restore parent and do not claim physical or affinity benefit.'})
    c['rounds'].append({'round':number,'campaign':name,'arms':['unguided','gradient_zero','gradient'],
        'batches':parent['batches'],'n_per_arm':parent['n_per_arm'],'seed':42,'reference_sha256':digest(new_reference),
        'reward_view':'shape_mixture','native_rms_ratio':q['native_rms_ratio'],'dose_reference':q.get('dose_reference','observed_native'),
        'parent_round':p['round'],'kind':'representation_ablation','changes':changes,'status':'frozen'})
    c['status']='backtracking';write_json(campaign,c);return q


def freeze_conditioning(campaign,parent_program,parent_reference,new_reference,output,evidence,number,name,reason,designer_contract):
    """Preserve geometry moments and all dose controls; declare count stratification."""
    from .window_reference import load_reference
    from .coordinate_conditioning import CoordinateCountConditionedShapeReward
    campaign,parent_program,parent_reference,new_reference,output,evidence,designer_contract=map(Path,
        (campaign,parent_program,parent_reference,new_reference,output,evidence,designer_contract))
    c,p,contract=read_json(campaign),read_json(parent_program),read_json(designer_contract)
    if output.exists() or evidence.exists():raise FileExistsError('Immutable conditioning plan exists')
    if number!=max(r['round'] for r in c['rounds'])+1 or not number<=c['maximum_rounds']<=30:
        raise ValueError('Sequential conditioning round within budget required')
    parent=next(r for r in c['rounds'] if r['round']==p['round'])
    if parent['status']!='completed' or c['rounds_started']!=c['rounds_completed']:
        raise ValueError('Retain preceding evaluation before conditioning')
    if p['seed']!=42 or p['reward_view']!='shape_mixture' or digest(parent_reference)!=p['reference_sha256']:
        raise ValueError('Verified shape parent required')
    a,b=load_reference(parent_reference),load_reference(new_reference)
    if b.get('conditioning_parent_reference_sha256')!=digest(parent_reference) or any(b.get(k)!=v for k,v in a.items()):
        raise ValueError('Unconditional parent geometry or provenance changed')
    if contract['schema_version']!='count-conditioning-designer-contract-1.0' or contract['agent']!='Designer':
        raise ValueError('Conditioning Designer contract required')
    if contract['parent_reference_sha256']!=digest(parent_reference) or contract['reference_sha256']!=digest(new_reference):
        raise ValueError('Designer reference binding differs')
    defaults={'dose_reference':'observed_native','initial_update_dose':'native','preserve_native_rigid_pose':False}
    controls=contract['inherit_without_simultaneous_retuning']
    required={'native_rms_ratio','mixture_temperature','robust_delta','dose_reference','initial_update_dose','preserve_native_rigid_pose','constraints'}
    if set(controls)!=required or any(p.get(k,defaults.get(k))!=v for k,v in controls.items()):
        raise ValueError('Conditioning must inherit all parent dose controls')
    declared=['exact count stratum geometry','empirical selected count-mass mixture prior','zero dose for unsupported counts']
    if contract['declared_changes']!=declared:
        raise ValueError('All three coupled conditioning changes must be disclosed')
    q=copy.deepcopy(p);changes={'reward_view':'count_conditioned_shape','reference_sha256':digest(new_reference),
        'conditional_support':b['conditional_support'],'coupled_changes':declared}
    q.update(round=number,reward_view='count_conditioned_shape',reference_sha256=digest(new_reference),designer_sha256=digest(designer_contract))
    q['backtracking']={'parent_round':p['round'],'parent_program_sha256':digest(parent_program),
        'kind':'conditioning_ablation','changes':changes,'reason':reason}
    CoordinateCountConditionedShapeReward(q,b)
    write_json(output,q)
    write_json(evidence,{'round':number,'parent_round':p['round'],'kind':'conditioning_ablation','changes':changes,
        'reason':reason,'program_sha256':digest(output),'reference_sha256':digest(new_reference),
        'parent_reference_sha256':digest(parent_reference),'designer_contract_sha256':digest(designer_contract),
        'window':q['window'],'parent_geometry_parity':True,
        'limitation':'Conditional coordinate imitation, no discrete force or causal affinity mechanism',
        'fallback':'Restore tested parent when actual window geometry or terminal physics regresses; preserve all attempted denominators.'})
    c['rounds'].append({'round':number,'campaign':name,'arms':['unguided','gradient_zero','gradient'],
        'batches':parent['batches'],'n_per_arm':parent['n_per_arm'],'seed':42,'reference_sha256':digest(new_reference),
        'reward_view':q['reward_view'],'native_rms_ratio':q['native_rms_ratio'],'dose_reference':q.get('dose_reference','observed_native'),
        'parent_round':p['round'],'kind':'conditioning_ablation','changes':changes,'status':'frozen'})
    c['status']='backtracking';write_json(campaign,c);return q
