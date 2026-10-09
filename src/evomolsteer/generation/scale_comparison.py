"""Compare a frozen completed-batch panel without implying campaign completion."""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from .path_evaluation import paired_effect, summarize_tail


def validate_candidates(frame: pd.DataFrame) -> None:
    required = {'batch', 'slot', 'pic50_on_rescore', 'valid_connected',
                'pb_fast_pass', 'smiles', 'energy_status', 'mmff_relief_per_heavy'}
    if not required.issubset(frame.columns) or frame.empty:
        raise ValueError('Nonempty evaluated all-attempt candidate table required')
    if frame.duplicated(['batch', 'slot']).any():
        raise ValueError('Duplicate paired batch/slot')
    if not np.isfinite(frame.pic50_on_rescore).all():
        raise ValueError('Every attempted slot must retain its target prediction')
    for field in ('valid_connected', 'pb_fast_pass'):
        if frame[field].dtype != bool:
            raise ValueError(f'Boolean {field} required')


def terminal_summary(frame: pd.DataFrame, threshold: float) -> dict:
    validate_candidates(frame)
    summary = summarize_tail(frame, threshold)
    valid = frame[frame.valid_connected]
    elite = valid[valid.pic50_on_rescore >= threshold]
    summary.update(
        valid_rate=len(valid)/len(frame),
        pb_fast_n=int(frame.pb_fast_pass.sum()),
        unique_valid_graphs=int(valid.smiles.nunique()),
        unique_graph_yield=float(valid.smiles.nunique()/len(frame)),
        energy_status_counts=Counter(frame.energy_status.fillna('not_applicable')),
        elite_pb_fast_n=int(elite.pb_fast_pass.sum()),
        elite_strain_median_per_heavy=float(
            elite[elite.energy_status.eq('converged')].mmff_relief_per_heavy.median()),
    )
    return summary


def paired_summary(candidate: pd.DataFrame, control: pd.DataFrame) -> dict:
    validate_candidates(candidate)
    validate_candidates(control)
    effect = paired_effect(candidate, control)
    delta = candidate[['batch', 'slot', 'pic50_on_rescore']].merge(
        control[['batch', 'slot', 'pic50_on_rescore']],
        on=['batch', 'slot'], suffixes=('_candidate', '_control'), validate='one_to_one')
    delta['difference'] = delta.pic50_on_rescore_candidate-delta.pic50_on_rescore_control
    batch_means = delta.groupby('batch', sort=True).difference.mean()
    effect.update(
        n_paired_slots=len(delta),
        paired_median_pic50=float(delta.difference.median()),
        delta_quantiles=delta.difference.quantile([.1, .25, .5, .75, .9]).to_dict(),
        positive_batch_n=int((batch_means > 0).sum()),
        batch_means_by_id={str(k): float(v) for k, v in batch_means.items()},
        slots_gain_at_least_0_01=int((delta.difference >= .01).sum()),
        slots_loss_at_least_0_01=int((delta.difference <= -.01).sum()),
        limitation='Frozen reward on new batches; completed-batch snapshot of a still-running '
                   'campaign. Bootstrap resamples batches (2000 draws, seed 42), not independent '
                   'molecules. No biological affinity validation.',
    )
    return effect


def compare_completed_panel(r11: pd.DataFrame, r26: pd.DataFrame, native: pd.DataFrame,
                            steer: pd.DataFrame, threshold: float,
                            snapshot_scope: dict, r11_execution: dict,
                            control_execution: dict) -> dict:
    """Require complete per-arm slots and verified initial priors before pairing."""
    for frame in (r11, r26, native, steer):
        validate_candidates(frame)
    batches = sorted(snapshot_scope['included_batches'])
    expected = snapshot_scope['snapshot_n_per_arm']
    if not batches or expected <= 0:
        raise ValueError('Completed snapshot scope required')
    for control in (r26, native):
        if sorted(control.batch.unique()) != batches or len(control) != expected:
            raise ValueError('Control table differs from frozen completed-batch scope')
    candidate = r11[r11.batch.isin(batches)].copy()
    if len(candidate) != expected:
        raise ValueError('R11 paired coverage differs from snapshot')
    if r11_execution['code_commit'] != control_execution['code_commit']:
        raise ValueError('Cross-cohort inference source differs')
    for batch in batches:
        initial = r11_execution['initial_state_signatures'][f'gradient/{batch}']
        for arm in ('gradient', 'unguided'):
            if initial != control_execution['initial_state_signatures'][f'{arm}/{batch}']:
                raise ValueError('Cross-cohort paired initial prior differs')
    per_batch = candidate.groupby('batch').size().to_dict()
    for frame in (r26, native):
        if frame.groupby('batch').size().to_dict() != per_batch:
            raise ValueError('Cross-cohort per-batch coverage differs')
    panel = {'R11': candidate, 'R26': r26, 'unguided': native}
    return {
        'schema_version': 'completed-scale-comparison-1.0',
        'threshold_pic50': float(threshold),
        'snapshot_scope': snapshot_scope,
        'integrity': {'cross_cohort_initial_states_equal': True,
                      'inference_commit': r11_execution['code_commit'],
                      'paired_batches': batches, 'n_per_arm': expected},
        'paired_panel': {name: terminal_summary(frame, threshold)
                         for name, frame in panel.items()},
        'paired_effects': {'R11_vs_unguided': paired_summary(candidate, native),
                           'R11_vs_R26': paired_summary(candidate, r26),
                           'R26_vs_unguided': paired_summary(r26, native)},
        'full_unpaired': {'R11': terminal_summary(r11, threshold),
                          'historical_Steer': terminal_summary(steer, threshold)},
        'reference_limitations': 'Historical Steer has different initial seed streams, extra '
                                 'online selection compute, and design donors used by the reward. '
                                 'Equal final attempt count is not a matched compute-budget test.',
    }
