"""Compact current-rank/final-outcome crossover and descendant summaries."""
from pathlib import Path
import numpy as np
import pandas as pd
from ..io import digest, read_json, write_json


def summarize(labels_path, evidence_path, output):
    out = Path(output)
    if out.exists():
        raise FileExistsError(out)
    labels = pd.read_parquet(labels_path)
    evidence = read_json(evidence_path)
    rows = []
    for (batch, step), frame in labels.groupby(['batch', 'step'], sort=True):
        observed = frame[frame.terminal_mean.notna()]
        if not len(observed):
            continue
        high_online = frame.online_score.quantile(.75)
        high_final = observed.terminal_mean.quantile(.75)
        ordinary_final = observed.terminal_mean.quantile(.5)
        final_high = (observed.terminal_mean >= high_final)&(observed.terminal_mean > ordinary_final+1e-12)
        for group, mask in [
            ('intermediate_high_final_high', (observed.online_score >= high_online)&final_high),
            ('intermediate_high_final_ordinary', (observed.online_score >= high_online)&(observed.terminal_mean <= ordinary_final)),
            ('intermediate_other_final_high', (observed.online_score < high_online)&final_high)]:
            subset = observed[mask]
            rows.append({'batch': batch, 'step': step, 'score_time': frame.score_time.iloc[0], 'group': group,
                'ancestor_n': len(subset), 'known_ancestor_n': len(observed),
                'ancestor_fraction': len(subset)/len(observed), 'observed_descendant_slots': int(subset.observed_n.sum()),
                'mean_final_affinity': float(subset.terminal_mean.mean()) if len(subset) else np.nan,
                'mean_valid_fraction': float(subset.valid_fraction.mean()) if len(subset) else np.nan,
                'mean_tail_fraction': float(subset.tail_fraction.mean()) if len(subset) else np.nan})
    table = pd.DataFrame(rows)
    out.mkdir(parents=True)
    table.to_parquet(out/'crossover_by_batch_event.parquet', compression=None, index=False)
    # Batch averages prevent early/late node count and cloned outcomes from
    # creating spurious independent inferential replicates.
    batch = table.groupby(['batch', 'group'], sort=True).agg(
        ancestor_fraction=('ancestor_fraction', 'mean'), mean_final_affinity=('mean_final_affinity', 'mean'),
        mean_tail_fraction=('mean_tail_fraction', 'mean'), mean_valid_fraction=('mean_valid_fraction', 'mean')).reset_index()
    batch.to_parquet(out/'crossover_by_batch.parquet', compression=None, index=False)
    cross = batch.groupby('group', sort=True).mean(numeric_only=True).drop(columns='batch').reset_index().to_dict('records')
    known = labels[labels.terminal_mean.notna()]
    support = {'known_ancestor_observations': len(known), 'censored_ancestor_observations': int((labels.observed_n == 0).sum()),
        'observed_n_quantiles': known.observed_n.quantile([.25,.5,.75]).tolist(),
        'unique_graph_n_quantiles': known.unique_graph_n.quantile([.25,.5,.75]).tolist(),
        'mean_valid_fraction': float(known.valid_fraction.mean()), 'mean_tail_fraction': float(known.tail_fraction.mean()),
        'semantics': 'Repeated times and related ancestors are descriptive observations, not independent N'}
    evidence['evidence_items'] += [{'id': 'outcome/rank_crossover', 'batch_equal_groups': cross,
        'definitions': 'Within event current score Q75; final-high >= observed Q75 AND > median; final-ordinary <= median. Ties never occupy both classes; donor thresholds only'},
        {'id': 'outcome/descendants', **support}]
    evidence['summary_source_sha256'] = digest(__file__)
    evidence['ancestor_labels_sha256'] = digest(labels_path)
    write_json(out/'evidence.json', evidence)
    write_json(out/'summary.json', {'crossover': cross, 'descendants': support,
        'score_window': evidence['score_window'], 'window': evidence['window'],
        'source_labels_sha256': digest(labels_path), 'source_evidence_sha256': digest(evidence_path)})
    return evidence
