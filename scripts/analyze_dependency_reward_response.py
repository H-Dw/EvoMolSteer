"""Compact checks of actual reward response and common terminal geometry.

Forecast observables are recorded by the executed reward, not inferred from an
Agent's explanation. Common terminal fields use original input geometry, with
no generated Steer templates. Export only per-batch aggregates.
"""
import argparse
import json
import gzip
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from rdkit import Chem
from evomolsteer.io import read_json, write_json, digest


def geometry(x, reference):
    protein = np.asarray(reference['protein_points_A'])
    d = np.linalg.norm(x[:, None] - protein[None], axis=-1)
    c = (1 - np.exp(-np.exp(-.5 * ((d - 3.6) / .6)**2).sum(1) / 4)).mean()
    p = (np.maximum(2 - d, 0)**2).sum() / len(x)
    ligand = np.asarray(reference['bound_ligand_A'])
    cost = ((x[:, None] - ligand[None])**2).sum(-1)
    i, j = linear_sum_assignment(cost)
    if len(i) != len(x): raise ValueError('Template atom coverage incomplete')
    return {'template_assignment_rms_A': float(np.sqrt(cost[i, j].mean())),
            'contact_proxy': float(c), 'repulsion_proxy': float(p),
            'nearest_receptor_distance_A': float(d.min(1).mean())}


def run(root, reference):
    if reference.suffix == '.gz':
        with gzip.open(reference, 'rt', encoding='utf-8') as stream: template = json.load(stream)
    else:
        template = read_json(reference)
    if template.get('steer_sources'):
        raise ValueError('Common geometry reference must exclude generated Steer')
    batches = []
    for folder in sorted(root.glob('*/batch_*')):
        rows = [json.loads(x) for x in (folder / 'guidance_trace.jsonl').read_text().splitlines()]
        active = [r for r in rows if r['active']]
        values = [geometry(m.GetConformer().GetPositions(), template)
                  for m in Chem.SDMolSupplier(str(folder / 'molecules_raw_decodable.sdf'), sanitize=False)
                  if m is not None]
        item = {'arm': folder.parent.name, 'batch': int(folder.name.split('_')[-1]),
                'decoded_terminal_n': len(values), 'trace_sha256': digest(folder / 'guidance_trace.jsonl'),
                'SDF_sha256': digest(folder / 'molecules_raw_decodable.sdf')}
        for key in values[0] if values else []:
            item[key + '_mean'] = float(np.mean([v[key] for v in values]))
        if active:
            first_order = np.concatenate([r['first_order_reward_change'] for r in active])
            item['realized_first_order_positive_fraction'] = float((first_order > 0).mean())
            item['realized_first_order_mean'] = float(first_order.mean())
            item['nonzero_control_steps'] = sum(np.any(np.asarray(r['injection_l2_A']) > 0) for r in active)
            # Only the new field has three interpretable observables.
            if np.asarray(active[0]['observables']).shape[-1] == 3:
                for idx, label in enumerate(['forecast_anchor', 'forecast_contact', 'forecast_repulsion']):
                    all_values = np.concatenate([np.asarray(r['observables'])[:, idx] for r in active])
                    item[label + '_first'] = float(np.asarray(active[0]['observables'])[:, idx].mean())
                    item[label + '_last'] = float(np.asarray(active[-1]['observables'])[:, idx].mean())
                    item[label + '_nonzero_fraction'] = float((all_values != 0).mean())
                    item[label + '_min'] = float(all_values.min())
                    item[label + '_max'] = float(all_values.max())
        batches.append(item)
    return {'schema_version': 'dependency-response-1.0', 'reference_sha256': digest(reference),
            'batches': batches, 'limits': 'Terminal proxies cover decoded molecules; forecast progression is observational and includes native flow; first-order ascent is not an exact finite-step reward or affinity improvement'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True); p.add_argument('--reference', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args(); write_json(a.output, run(Path(a.root), Path(a.reference)))
