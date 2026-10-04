"""Retention of statistical evidence versus rebuildable caches and diagnostics.

Legacy configs retain audit behavior. Neither profile selects features/results
by significance; all batch effects, null results and corrected tests are kept.
"""
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import pyarrow as pa
from .io import digest, read_json, write_json, write_table


@dataclass(frozen=True)
class OutputPolicy:
    profile: str
    retain_feature_cache: bool
    keep_worker_shards: bool
    write_node_details: bool
    write_step_details: bool


def policy(cfg):
    value = cfg.get('output_policy', {})
    allowed = {'profile', 'retain_feature_cache', 'keep_worker_shards', 'write_node_details', 'write_step_details'}
    if set(value) - allowed:
        raise ValueError('Unknown output policy keys: ' + str(set(value) - allowed))
    profile = value.get('profile', 'audit')
    if profile not in ('compact', 'audit'):
        raise ValueError('output_policy.profile must be compact or audit')
    flags = {key: value.get(key, profile == 'audit') for key in allowed - {'profile'}}
    if any(type(v) is not bool for v in flags.values()):
        raise TypeError('Output retention flags must be boolean')
    return OutputPolicy(profile=profile, **flags)


def extraction_representations(cfg):
    if cfg.get('analysis_scope') == 'actual_resampling_events':
        requested = cfg['analysis_representations']
        if not requested or set(requested) - {'predicted_endpoint', 'proposal_state'} or len(set(requested)) != len(requested):
            raise ValueError('Invalid selection analysis representations')
        # Storage profiles never change the scientific scope.
        return list(requested)
    all_reps = ['current_state', 'predicted_endpoint', 'proposal_state']
    if policy(cfg).profile == 'audit' or not cfg.get('analysis_representations'):
        return all_reps
    requested = set(cfg['analysis_representations']) | {'predicted_endpoint'}
    if requested - set(all_reps):
        raise ValueError('Unknown analysis representation')
    return [name for name in all_reps if name in requested]


def parquet_options(cfg, schema):
    if policy(cfg).profile == 'audit':
        return {}
    categorical = [f.name for f in schema if not pa.types.is_floating(f.type)]
    return {'compression': None, 'use_dictionary': categorical}


def write_batch_table(path, rows, cfg):
    path = Path(path)
    if policy(cfg).profile == 'audit':
        return write_table(path, rows)
    path = path.with_suffix('.parquet')
    path.parent.mkdir(parents=True, exist_ok=True)
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    table = pa.Table.from_pandas(df, preserve_index=False)
    import pyarrow.parquet as pq
    pq.write_table(table, path, **parquet_options(cfg, table.schema))
    return df


def validate_output_layout(directory,cfg):
    """Do not silently mix historical verbose files into a compact result."""
    directory = Path(directory)
    p = policy(cfg)
    omitted = []
    if p.profile=='compact':
        omitted += ['batch_effects.csv','batch_effect_rates.csv']
    if not p.write_step_details:
        omitted += ['events.parquet','event_rates.parquet']
    if not p.write_node_details:
        omitted += ['scores.parquet','parent_child_rates.parquet','parent_child_velocity.parquet']
    stale = [name for name in omitted if (directory/name).exists()]
    if stale:
        raise ValueError('Historical outputs conflict with retention policy; use a fresh output directory: '+str(stale))


def register_feature_cache(output, cfg):
    output = Path(output)
    if policy(cfg).profile != 'compact':
        return
    path = output / 'features.parquet'
    write_json(output / 'feature_cache_manifest.json', {
        'schema_version': '1.0', 'status': 'available', 'owned_file': 'features.parquet',
        'sha256': digest(path), 'bytes': path.stat().st_size,
        'representations': extraction_representations(cfg),
        'purpose': 'Rebuildable geometry cache; raw trajectories remain authoritative',
        'rebuild': 'scripts/materialize_feature_cache.py --analysis THIS_RUN',
    })


def retire_feature_cache(output, cfg):
    """Only delete this run's registered, unchanged cache after successful analysis."""
    output = Path(output).resolve()
    p = policy(cfg)
    if p.profile != 'compact' or p.retain_feature_cache:
        return {'retired_bytes': 0}
    manifest_path = output / 'feature_cache_manifest.json'
    manifest = read_json(manifest_path)
    if manifest['status'] == 'retired':
        return {'retired_bytes': 0}
    if manifest['owned_file'] != 'features.parquet':
        raise ValueError('Unexpected cache file')
    original = output / manifest['owned_file']
    path = original.resolve()
    if path.parent != output or original.is_symlink() or digest(path) != manifest['sha256']:
        raise ValueError('Cache integrity/ownership check failed; retained file')
    path.unlink()
    write_json(manifest_path, {**manifest, 'status': 'retired'})
    return {'retired_bytes': manifest['bytes'], 'sha256': manifest['sha256']}


def clean_generated_shards(output, records):
    """Remove exactly inventoried files from a fresh ingestion; never raw data."""
    output = Path(output).resolve()
    shard_root = (output / '_shards').resolve()
    if not shard_root.is_relative_to(output) or shard_root == output:
        raise ValueError('Unsafe shard root')
    files = []
    for relative, expected in records.items():
        original = output / relative
        path = original.resolve()
        if original.is_symlink() or not path.is_relative_to(shard_root) or digest(path) != expected:
            raise ValueError('Shard integrity/ownership check failed: ' + relative)
        files.append(path)
    actual = {p.resolve() for p in shard_root.rglob('*') if p.is_file()}
    if actual != set(files):
        raise ValueError('Unexpected files in shard workspace; retained shards')
    directories = sorted((p for p in shard_root.rglob('*') if p.is_dir()), key=lambda p: len(p.parts), reverse=True)
    if any(p.is_symlink() or not p.resolve().is_relative_to(shard_root) for p in directories):
        raise ValueError('Unsafe shard directory')
    for path in files:
        path.unlink()
    for path in directories:
        path.rmdir()
    shard_root.rmdir()
