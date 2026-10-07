"""Bounded evolution of reward programs; never selects generated particles."""
import copy
from pathlib import Path
import numpy as np
import pandas as pd
from ..io import read_json,digest
from .prototypes import write_json
from .terminal_statistics import converged_energy

AXES=('all_head_change_vs_native','shape_improvement_fraction','negative_MMFF_relative_change','surround_RMS_improvement_fraction','negative_MMFF_p90_relative_change')


def validate_initial_pairing(actual,native):
    a,n=actual['initial_state_signatures'],native['initial_state_signatures']
    keys=[k for k in a if k.startswith('gradient/')]
    if not keys or any(n.get(k.replace('gradient/','unguided/',1))!=a[k] for k in keys):raise ValueError('Gradient/native initial states differ')
    return {'passed':True,'batches':sorted(int(k.split('/')[1]) for k in keys),
        'fields':'Initial coordinates, atom/bond/charge states and mask byte signatures',
        'actual_signatures':{k:a[k] for k in keys}}


def pareto_front(outcomes):
    if not outcomes:return []
    vectors=np.asarray([[r.get(k,np.nan) for k in AXES] for r in outcomes],float)
    complete=np.isfinite(vectors).all(axis=1)
    return [r for i,r in enumerate(outcomes) if complete[i] and not any(complete[j] and np.all(vectors[j]>=vectors[i]) and np.any(vectors[j]>vectors[i]) for j in range(len(vectors)) if j!=i)]


def choose(outcomes,mode='balanced'):
    # Selection is over frozen reward programs, not molecular samples. Neither
    # novelty nor reference/native graph identity is an eligibility condition.
    rows=pareto_front(outcomes)
    eligible=[r for r in rows if r['valid_rate_change']>=-.02-1e-12 and r['PB_rate_change']>=-.02-1e-12 and r['energy_coverage_change']>=-.02-1e-12]
    rows=eligible or rows
    if not rows:raise ValueError('No completed program evidence')
    if mode=='affinity':return max(rows,key=lambda r:(r['all_head_change_vs_native'],r['negative_MMFF_p90_relative_change']))
    if mode=='tail':return max(rows,key=lambda r:(r['negative_MMFF_p90_relative_change'],r['all_head_change_vs_native']))
    # First prefer physical non-regression, then head. If none qualifies, retain
    # the closest physical tradeoff instead of silently claiming joint success.
    feasible=[r for r in rows if r['negative_MMFF_relative_change']>=0 and r['surround_RMS_improvement_fraction']>=0 and r['negative_MMFF_p90_relative_change']>=0]
    if feasible:return max(feasible,key=lambda r:(r['all_head_change_vs_native'],r['shape_improvement_fraction']))
    return max(rows,key=lambda r:(min(r['negative_MMFF_relative_change'],r['surround_RMS_improvement_fraction'],r['negative_MMFF_p90_relative_change']),r['all_head_change_vs_native']))


def proposal(number,evidence):
    explicit={
      1:('all','motif_mixture','joint',.1,0.,'Joint spatial motif, native and exact zero-dose controls.'),
      2:('all','motif_mixture','joint',.3,0.,'Dose escalation with the same measured joint geometry.'),
      3:('all','motif_mixture','pair',.1,0.,'Isolate local pair spacing from translation and radial shells.'),
      4:('all','motif_mixture','shell',.1,0.,'Isolate receptor-relative radial shells.'),
      5:('NOS_C','motif_mixture','pair',.1,0.,'Couple polar and surrounding carbon slots through relative distances.'),
      6:('NOS_C','motif_mixture','joint',.1,0.,'Add receptor-frame localization to polar-carbon relative geometry.'),
      7:('all','motif_contrast','joint',.05,0.,'Incremental selected/background density contrast rather than selected means alone.'),
      8:('NOS_C','motif_contrast','pair',.05,0.,'Contrast of polar-carbon spacing; new graphs remain eligible.')}
    if number in explicit:
        channel,view,components,dose,ramp,reason=explicit[number]
        return {'channel':channel,'reward_view':view,'motif_components':components,'native_rms_ratio':dose,'time_ramp_power':ramp,'reason':reason,'parent_round':0}
    if number in (9,10):
        return {'channel':'all' if number==9 else 'NOS_C','reward_view':'motif_mixture',
            'motif_components':'joint' if number==9 else 'pair','native_rms_ratio':.05,'time_ramp_power':0.,
            'target_definition':'boundary_survival','parent_round':0,
            'reason':'Test actual-boundary surviving ancestry distributions instead of immediate selection. Original R1/R2 physical regression motivates this lineage-target ablation; no final t=1 labels or graph locks.'}
    outcomes=[read_json(p) for p in Path(evidence).glob('round_*.outcome.json')]
    discovery=[r for r in outcomes if r['round']<=12]
    if number>13:
        # Validation/heldout results cannot redefine a frozen reward.
        parent=read_json(Path(evidence)/'round_13.plan.json')['parent_round']
        return {'parent_round':parent,'reason':'Replay frozen discovery winner on new matched batches; no further tuning.'}
    modes={11:'balanced',12:'affinity',13:'balanced'}
    # R1-8 historical score-grid scales are not eligible for the final strict
    # window winner. R9/10 already meet the scope; R11/12 retest restored
    # parents with prospective strict-scale references before validation.
    eligible=discovery if number<13 else [r for r in discovery if r['round']>=9]
    parent=choose(eligible,modes[number])
    changes={11:{'dose_factor':.5},12:{'time_ramp_power':1.},13:{}}[number]
    return {'parent_round':parent['round'],**changes,'strict_scale':number in (11,12),
        'reason':f"Restore a measured {modes[number]} Pareto parent; prospective actual-window scale in R11/12 plus the declared dose variation. Final winner restricted to strict-scope R9-12. Not particle resampling."}


def record(campaign,evidence,number):
    from .terminal_comparison import compare
    c=read_json(campaign);r=next(v for v in c['rounds'] if v['round']==number);evidence=Path(evidence)
    local=evidence/f'round_{number:02d}/local';baseline=local if 'unguided' in r['arms'] else evidence/'round_01/local'
    at,nt=(read_json(p/'terminal_report.json') for p in (local,baseline));aw,nw=(read_json(p/'window/report.json') for p in (local,baseline))
    pairing=validate_initial_pairing(aw,nw)
    write_json(local.parent/'initial_pairing.json',pairing)
    actual,native=at['results']['gradient'],nt['results']['unguided'];shape,ns=aw['results']['gradient'],nw['results']['unguided']
    execution=read_json(local/'execution_report.json');audit=read_json(local/'coordinate_audit.json')
    if not execution['no_particle_resampling'] or execution['outside_window_injection'] or not audit['coordinate_preflight']['passed']:raise ValueError('Execution contract failed')
    if 'gradient_zero' in r['arms'] and aw['zero_equivalence_passed'] is not True:raise ValueError('Full zero/native mismatch')
    original=evidence.parent/'ck2_terminal_seed42_20261005/reference/terminal_report.json'
    if c.get('original_steer_report_relative_path'):
        original=Path(__file__).resolve().parents[3]/c['original_steer_report_relative_path']
    groups=[{'label':'Original Steer','arm':'single','terminal_report':str(original.resolve())},
            {'label':'Native','arm':'unguided','terminal_report':str((baseline/'terminal_report.json').resolve()),'execution_report':str((baseline/'execution_report.json').resolve())},
            {'label':r['campaign'],'arm':'gradient','terminal_report':str((local/'terminal_report.json').resolve()),'execution_report':str((local/'execution_report.json').resolve())}]
    manifest={'groups':groups,'native':'Native','steer':'Original Steer','tests':[r['campaign']],'round':number,'maximum_rounds':c['maximum_rounds']}
    write_json(local.parent/'comparison_manifest.json',manifest);compare(local.parent/'comparison_manifest.json',local.parent/'comparison')
    def tail(p,arm):
        d=pd.read_csv(p/'candidate_metrics.csv');v=converged_energy(d[d.arm==arm])
        return float(v.quantile(.9)),int(len(v))
    aq,an=tail(local,'gradient');nq,nn=tail(baseline,'unguided')
    def improvement(actual_value,native_value):
        if actual_value is None or native_value is None or not np.isfinite([actual_value,native_value]).all() or native_value==0:return None
        return 1-actual_value/native_value
    def difference(actual_value,native_value):
        return actual_value-native_value if actual_value is not None and native_value is not None else None
    outcome={'round':number,'parent_round':r['parent_round'],'reason':r['reason'],'split':r['split'],
        'all_head_change_vs_native':actual['all_pic50_on_rescore_mean']-native['all_pic50_on_rescore_mean'],
        'unique_head_change_vs_native':difference(actual['unique_valid_pic50_on_rescore_mean'],native['unique_valid_pic50_on_rescore_mean']),
        'shape_improvement_fraction':1-shape['symmetric_shape_A']/ns['symmetric_shape_A'],
        'negative_MMFF_relative_change':improvement(actual['all_mmff_relief_per_heavy_median'],native['all_mmff_relief_per_heavy_median']),
        'surround_RMS_improvement_fraction':improvement(actual['all_relax_rms_surround_A_mean'],native['all_relax_rms_surround_A_mean']),
        'negative_MMFF_p90_relative_change':improvement(aq,nq),'MMFF_p90':aq,'native_MMFF_p90':nq,
        'valid_rate_change':actual['valid_rate']-native['valid_rate'],'PB_rate_change':actual['pb_fast_pass_rate']-native['pb_fast_pass_rate'],
        'energy_coverage_change':an/actual['n']-nn/native['n'],'terminal':actual,'actual_window_shape':shape,
        'original_steer_decision':read_json(local.parent/'comparison/comparison.json')['decisions'][r['campaign']],
        'interpretation':'Head prediction, not measured affinity; MMFF isolated-ligand relaxation, not receptor energy. Graph changes are free. Candidate rows are not independent replicates.'}
    write_json(evidence/f'round_{number:02d}.outcome.json',outcome)
    r['status']='completed';c.update(rounds_completed=number,status='active');write_json(campaign,c)
    write_json(evidence/'pareto_archive.json',{'axes':list(AXES),'rounds':[v['round'] for v in pareto_front([read_json(p) for p in evidence.glob('round_*.outcome.json') if read_json(p)['split']=='discovery'])],
        'scope':'Reward program archive only; no molecule resampling or graph filtering'})
    return outcome


def summarize(evidence,output):
    evidence=Path(evidence);rows=[read_json(p) for p in sorted(evidence.glob('round_*.outcome.json'))]
    lines=['# Spatial-motif seed42 campaign','',
           f'{len(rows)}/15 rounds retained. Every reported round completed all native 100 steps; gradient acts only inside the learned dynamic window. No SMC. Chemical graphs remain free.','',
           '| Round | Split | Head Δ | Shape improvement | MMFF median improvement | MMFF p90 improvement | Surround RMS improvement |','|---:|---|---:|---:|---:|---:|---:|']
    for r in rows:lines.append(f"|{r['round']}|{r['split']}|{r['all_head_change_vs_native']:.6f}|{r['shape_improvement_fraction']:.3%}|{r['negative_MMFF_relative_change']:.3%}|{r['negative_MMFF_p90_relative_change']:.3%}|{r['surround_RMS_improvement_fraction']:.3%}|")
    lines+=['','All directions remain separate. Head is a shared model prediction; MMFF is isolated-ligand relaxation relief. Historical Steer is not a matched randomized arm.',
            'Rounds 1–12 are adaptive reused discovery screens. Round13 validates the frozen discovery winner; rounds14–15 reuse that same frozen reward on heldout batches. No validation/heldout tuning.',
            'Each round retains candidate failures, coverage, physical tails, frozen formulas/configuration and inference commit; disposable trajectories/archives are retired after verified report retention.']
    Path(output).write_text('\n'.join(lines)+'\n',encoding='utf8')
