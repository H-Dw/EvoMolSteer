"""Observed terminal credit on a sparse Steer genealogy, with explicit censoring.

No learned predictor is introduced. Extinct nodes have unobserved future utility;
duplicate terminal graphs do not multiply evidence. This is a Steer-conditioned
retrospective credit, not the probability of success under native continuation.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from ..io import digest, read_json, write_json
from ..trajectory_source import open_trajectory
from .path_graph import CURRENT, FORMAT, validate_lineage


def terminal_ancestors(selected):
    selected = np.asarray(selected)
    if selected.ndim != 2 or not np.issubdtype(selected.dtype, np.integer):
        raise ValueError("Integer step-by-slot selected indices required")
    steps, n = selected.shape
    if np.any((selected < 0) | (selected >= n)):
        raise ValueError("Invalid parent slot")
    ancestors = np.empty((steps+1, n), dtype=np.int32)
    ancestors[-1] = np.arange(n)
    for step in range(steps-1, -1, -1):
        ancestors[step] = selected[step, ancestors[step+1]]
    return ancestors


def credit_batch(nodes, selected, metrics, threshold):
    n = np.asarray(selected).shape[1]
    m = metrics.sort_values('slot').reset_index(drop=True)
    if len(m) != n or not np.array_equal(m.slot.to_numpy(), np.arange(n)):
        raise ValueError("Exactly one terminal label per final slot required")
    if m.valid_connected.dtype != bool:
        raise ValueError("Validity must be a boolean, not a truthy string")
    score = m.pic50_on_rescore.to_numpy(float)
    valid = m.valid_connected.to_numpy() & np.isfinite(score)
    if m.loc[valid, 'smiles'].isna().any():
        raise ValueError("Valid terminal graphs require canonical identities")
    ancestors = terminal_ancestors(selected)
    rows = []
    for row in nodes[nodes.representation == CURRENT].itertuples(index=False):
        slots = np.flatnonzero(ancestors[row.step] == row.slot)
        observed = len(slots) > 0
        good = slots[valid[slots]]
        # One score per decoded chemical graph; clones remain in n_descendants
        # for genealogy diagnostics, but do not create independent tail evidence.
        unique = m.iloc[good].groupby('smiles', sort=True).pic50_on_rescore.max()
        rows.append((row.node_id, len(slots), len(good), len(unique),
                     int((unique >= threshold).sum()), observed,
                     float(unique.max()) if len(unique) else np.nan,
                     float(unique.mean()) if len(unique) else np.nan,
                     bool(len(unique) and (unique >= threshold).any())))
    return pd.DataFrame(rows, columns=['node_id','n_descendants','n_valid_descendants',
        'n_unique_graphs','n_elite_graphs','future_observed','terminal_max_pic50',
        'terminal_mean_unique_pic50','has_observed_elite'])


def build_terminal_credit(dataset, graph, terminal_metrics, output, quantile=.95):
    root, graph, out = Path(dataset).resolve(), Path(graph), Path(output)
    if out.exists():
        raise FileExistsError("Use a fresh terminal-credit output")
    if not 0 < quantile < 1:
        raise ValueError("Tail quantile must be strictly between zero and one")
    manifest = read_json(graph/'manifest.json')
    if manifest['schema_version'] != FORMAT:
        raise ValueError("Typed path graph required")
    for name, entry in manifest['tables'].items():
        if digest(graph/name) != entry['sha256']:
            raise ValueError("Path graph checksum mismatch")
    nodes = pd.read_parquet(graph/'nodes.parquet')
    metrics = pd.read_csv(terminal_metrics)
    batches = [s['batch'] for s in manifest['sources']]
    m = metrics[metrics.batch.isin(batches)].copy()
    if m.valid_connected.dtype != bool or m.duplicated(['batch','slot']).any():
        raise ValueError("Ambiguous terminal labels")
    valid = m.valid_connected & np.isfinite(m.pic50_on_rescore)
    if not valid.any():
        raise ValueError("No valid discovery terminal scores")
    # Threshold is frozen from the donor panel, never from a tested candidate.
    threshold = float(m.loc[valid, 'pic50_on_rescore'].quantile(quantile))
    tables = []
    for source in manifest['sources']:
        path = root/source['path']
        if digest(path) != source['sha256']:
            raise ValueError("Original trajectory changed")
        with open_trajectory(path) as a:
            _, _, selected, _, _, _ = validate_lineage(a)
            table = credit_batch(nodes[nodes.source_index == source['source_index']],
                                 selected, m[m.batch == source['batch']], threshold)
        tables.append(table)
    credit = pd.concat(tables, ignore_index=True)
    out.mkdir(parents=True)
    credit.to_parquet(out/'credit.parquet', compression=None, index=False)
    joined = nodes[nodes.representation == CURRENT].merge(credit, validate='one_to_one', on='node_id')
    stage = joined.groupby('state_time', sort=True).agg(
        nodes=('node_id','size'), observable_nodes=('future_observed','sum'),
        elite_nodes=('has_observed_elite','sum'), valid_graph_observations=('n_unique_graphs','sum'))
    stage.to_csv(out/'observability.csv', index=True)
    report = {'schema_version':'observed-terminal-path-credit-1.0', 'window':manifest['window'],
        'tail_quantile':quantile, 'threshold_pic50':threshold, 'donor_batches':batches,
        'valid_terminal_count':int(valid.sum()), 'elite_terminal_count':int((valid & (m.pic50_on_rescore >= threshold)).sum()),
        'elite_unique_graphs':int(m.loc[valid & (m.pic50_on_rescore >= threshold),'smiles'].nunique()),
        'source_graph_manifest_sha256':digest(graph/'manifest.json'),
        'terminal_metrics_sha256':digest(terminal_metrics), 'source_code_sha256':digest(__file__),
        'label_clock':1.0, 'label_semantics':'Decoded final molecule rescored by common terminal protocol',
        'future_semantics':'Observed Steer-conditioned descendants; extinction is censored, not a failed affinity label',
        'coordinate_clock':'Current state_time; intermediate head score is not a final label',
        'duplicate_policy':'Maximum score per unique terminal graph at each ancestor, clones counted separately',
        'tables':{p.name:{'bytes':p.stat().st_size,'sha256':digest(p)} for p in out.iterdir() if p.is_file()}}
    write_json(out/'manifest.json', report)
    return report
