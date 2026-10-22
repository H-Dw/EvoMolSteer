"""Continuous condition fidelity with teacher-affinity agreement; no acceptance cutoffs."""
import math
from .group_tools import distribution, score_summary, local_geometry
from .io import eligible


def condition_summary(trials, teacher):
    guided = teacher['core_atoms']; target=[teacher['coords_A'][i] for i in guided]
    errors=[]; eligible_errors=[]; rows=[]; original_core_errors=[]
    for t in trials:
        node=t['nodes']['100']
        geometry=local_geometry(node['local_endpoint_A'],target)
        original=teacher.get('original_core_atoms',guided)
        original_core_errors.append(local_geometry(
            [node['local_endpoint_A'][guided.index(i)] for i in original],
            [teacher['coords_A'][i] for i in original])['rms_A'])
        errors.append(geometry['rms_A'])
        if eligible(t['record']): eligible_errors.append(geometry['rms_A'])
        rows.append({'source_id':t.get('source_id'), 'valid':eligible(t['record']), **geometry})
    score=score_summary([t['record'] for t in trials])
    teacher_score=teacher['source_record']['pic50_on_rescore']
    gap=score['mean']-teacher_score if score['mean'] is not None else None
    return {'attempts':len(trials), 'condition_rmsd_A_all_attempts':distribution(errors),
        'condition_mean_atom_distance_A':distribution([r['mean_distance_A'] for r in rows]),
        'condition_max_atom_distance_A':distribution([r['max_distance_A'] for r in rows]),
        'condition_pseudo_huber_delta1':distribution([r['pseudo_huber_delta1'] for r in rows]),
        'original_core_rmsd_A_all_attempts':distribution(original_core_errors),
        'condition_rmsd_A_valid_only':distribution(eligible_errors),
        'affinity':score, 'teacher_pic50':teacher_score, 'mean_affinity_signed_gap':gap,
        'mean_affinity_absolute_gap':abs(gap) if gap is not None else None,
        'best_minus_teacher':score['max']-teacher_score if score['max'] is not None else None,
        'valid_best_exceeds_teacher':score['max'] > teacher_score if score['max'] is not None else False,
        'top10_percent_count':max(1,math.ceil(len(trials)*.1)),
        'top10_percent_mean':_tail(trials,max(1,math.ceil(len(trials)*.1))),
        'condition_atoms':guided,'original_core':teacher.get('original_core_atoms',guided),
        'thresholds':None, 'individual_condition_errors':rows,
        'interpretation':'No binary success claim. Lower coordinate error is primary; affinity agreement is a separate functional objective. Invalid molecules remain in counts and all-attempt coordinate diagnostics.'}


def _tail(trials,k):
    scores=sorted((t['record']['pic50_on_rescore'] for t in trials if eligible(t['record'])),reverse=True)
    return sum(scores[:k])/k if len(scores)>=k else None


def dominates(a,b):
    """No weighted sum or hidden tolerance: strict continuous Pareto comparison."""
    av=(a['condition_rmsd_A_all_attempts']['mean'],a['mean_affinity_absolute_gap'])
    bv=(b['condition_rmsd_A_all_attempts']['mean'],b['mean_affinity_absolute_gap'])
    return all(x is not None and y is not None and x<=y for x,y in zip(av,bv)) and any(x<y for x,y in zip(av,bv))
