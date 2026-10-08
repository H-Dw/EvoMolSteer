"""Check dynamic quality labels against observed branching without future imputation."""
from pathlib import Path
import numpy as np
import pandas as pd

from ..io import digest, read_json, write_json
from .dynamic_regions import observed_integral


def audit(event_table, mining_manifest, output, seed=42):
    event_table, mining_manifest = Path(event_table), Path(mining_manifest)
    data = pd.read_parquet(event_table) if event_table.suffix == '.parquet' else pd.read_csv(event_table)
    manifest = read_json(mining_manifest)
    grid, batches = np.asarray(manifest['score_grid']), manifest['batches']
    if data[['batch', 'time']].duplicated().any():
        raise ValueError('Duplicate selection events')
    if sorted(data.batch.unique()) != sorted(batches):
        raise ValueError('Batch support differs')
    curves, coverage = [], []
    for batch in batches:
        d = data[data.batch.eq(batch)].sort_values('time')
        if not np.allclose(d.time, grid, atol=1e-8, rtol=0):
            raise ValueError('Node support differs')
        valid = d.identifiable.to_numpy(bool)
        if d.loc[valid, ['positive_offspring_mean', 'negative_offspring_mean']].isna().any().any():
            raise ValueError('Missing observed offspring evidence')
        delta = (d.positive_offspring_mean - d.negative_offspring_mean).to_numpy(copy=True)
        delta[~valid] = np.nan
        curves.append(delta)
        coverage.append(float(valid.mean()))
    effects = observed_integral(np.asarray(curves)[:, :, None], grid).ravel()
    rng = np.random.default_rng(seed)
    boot = effects[rng.integers(0, len(batches), (2000, len(batches)))].mean(1)
    valid = data[data.identifiable]
    delta = valid.positive_offspring_mean - valid.negative_offspring_mean
    result = {'schema_version': 'dynamic-quality-selection-audit-1.0',
              'event_table_sha256': digest(event_table), 'mining_manifest_sha256': digest(mining_manifest),
              'window': manifest['window'], 'score_grid': grid.tolist(), 'batch_count': len(batches),
              'observed_events': len(valid), 'missing_events': len(data) - len(valid),
              'observed_grid_fraction_by_batch': coverage,
              'event_mean_positive_offspring': float(valid.positive_offspring_mean.mean()),
              'event_mean_negative_offspring': float(valid.negative_offspring_mean.mean()),
              'fraction_observed_events_positive_has_more_offspring': float(delta.gt(0).mean()),
              'whole_window_batch_weighted_offspring_difference': float(effects.mean()),
              'batch_bootstrap_CI95': np.quantile(boot, [.025, .975]).tolist(),
              'batch_integrated_offspring_differences': dict(zip(map(str, batches), effects.tolist())),
              'semantics': 'Root-balanced cohort offspring expectations; not survival probability or terminal success.',
              'limitation': 'Online labels and offspring use the same selection mechanism; this audits correspondence, not an independent affinity benefit.'}
    write_json(output, result)
    return result
