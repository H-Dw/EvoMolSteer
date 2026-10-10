"""Tiny actual-trajectory response records, retained after structure retirement."""
import hashlib
import json
from pathlib import Path
import numpy as np


def response_records(dataset, campaign, program):
    rows = []
    for folder in sorted((Path(dataset)/'results'/campaign).glob('*/batch_*')):
        trace = [json.loads(s) for s in (folder/'guidance_trace.jsonl').read_text().splitlines()]
        active = [v for v in trace if v['reward_evaluated']]
        first_order = np.array([v['first_order_reward_change'] for v in active], float) if active else np.zeros((0,))
        with np.load(folder/'window_state.npz', allow_pickle=False) as state:
            if not np.isclose(float(state['state_time']), program['window'][1], atol=2e-6):
                raise ValueError('Retained generation snapshot differs from dynamic control boundary')
            coords = state['coords']
            if not state['mask'].all():
                raise ValueError('Fixed active coordinate response support required')
            xyz = coords.astype(float)*float(state['coord_scale'])
            centered = xyz-xyz.mean(1, keepdims=True)
            radius = np.sqrt(np.sum(centered**2, axis=-1).mean(1))
            signature = hashlib.sha256(coords.tobytes()).hexdigest()
            clock = float(state['state_time'])
        rows.append({'arm': folder.parent.name, 'batch': int(folder.name.split('_')[1]),
            'state_time': clock,
            'window_coordinates_sha256': signature, 'window_radius_gyration_mean_A': float(radius.mean()),
            'window_radius_gyration_quartiles_A': np.quantile(radius, [.25,.5,.75]).tolist(),
            'first_order_reward_gain_mean': float(first_order.mean()) if first_order.size else 0.,
            'first_order_positive_fraction': float((first_order > 0).mean()) if first_order.size else 0.,
            'semantics': 'First-order FLOWR VJP response, not a second-forward measured reward improvement'})
    return rows
