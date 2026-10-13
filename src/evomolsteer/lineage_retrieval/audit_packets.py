"""Source evidence for reviewing Selector prose without omitted-history bias."""
from __future__ import annotations

import json
import math
import re
from typing import Any

import numpy as np


CORE_HISTORY = {
    f'{rep}_{suffix}' for rep in ('current', 'forecast') for suffix in (
        'radius_gyration_A', 'centroid_pocket_centroid_A', 'centroid_nearest_pocket_A',
        'centroid_x_A', 'centroid_y_A', 'centroid_z_A', 'nearest_pocket_A',
        'nearest_pocket_A_mean', 'pocket_contacts_4A', 'contact_pairs_4A',
        'ligand_contacts_4A', 'planarity_fraction', 'asphericity',
        'shape_thickness_A', 'residue_contact_footprint', 'atom_count')}
CORE_HISTORY.update({'online_pic50', 'delta_current_Rg', 'delta_forecast_Rg',
    'delta_current_nearest_pocket_A', 'delta_forecast_nearest_pocket_A',
    'delta_current_contact_pairs_4A', 'delta_forecast_contact_pairs_4A',
    'delta_forecast_revision_rms_A'})


def _table(card: dict[str, Any], keys: set[str], cutoff: int) -> dict[str, Any]:
    rows = [s for s in card['history'] if s['step'] <= cutoff]
    available = {k for s in rows for k in s}
    columns = ['step', *sorted(keys & available - {'step'})]
    return {'history_columns': columns,
        'history_rows': [[s.get(k) for k in columns] for s in rows]}


def selector_audit_node(pool: dict[str, Any], mapping: dict[str, str],
                        answer: dict[str, Any], calls: list[dict[str, Any]]) -> dict[str, Any]:
    """Include full relevant past for every named comparison and selected seed.

    The audit receives no private group key, ancestry membership or terminal
    readout. Queries are action evidence, never a reason to omit a claim's
    available historical source from the factual review.
    """
    cutoff = int(pool['step'])
    cards = {mapping_id: next(c for c in pool['cards'] if c['key'] == key)
        for mapping_id, key in mapping.items()}
    selected = {r['candidate_id'] for r in answer['ranking']}
    prose = json.dumps(answer, ensure_ascii=False)
    named = set(re.findall(r'C[0-9a-f]{12}', prose)) & set(cards)
    queried = {c.get('args', {}).get('candidate_id') for c in calls} & set(cards)
    available = {k for c in cards.values() for k in c['now']}
    literal_keys = {k for k in available if k in prose}
    query_keys = {k for c in calls for k in c.get('args', {}).get('features', [])}
    keys = CORE_HISTORY | literal_keys | query_keys | {
        o['feature'] for r in answer['ranking'] for o in r['observations']}
    claims = []
    for row in answer['ranking']:
        card = cards[row['candidate_id']]
        states = {s['step']: s for s in card['history'] if s['step'] <= cutoff}
        claims.append({'candidate_id': row['candidate_id'], 'rank': row['rank'],
            'rationale': row['evidence'], 'counterevidence': row['counterevidence'],
            'observations': [{'claim': o, 'stored_value': states[o['step']][o['feature']]}
                for o in row['observations']], 'current_state': card['now'],
            **_table(card, keys, cutoff)})
    competitors = [{'candidate_id': cid, 'now': cards[cid]['now'],
        **(_table(cards[cid], keys, cutoff) if cid in named else {})}
        for cid in sorted((named | queried) - selected)]
    population = {}
    for key in sorted(available):
        values = [c['now'].get(key) for c in cards.values()]
        values = [float(v) for v in values if not isinstance(v, bool)
            and isinstance(v, (int, float)) and math.isfinite(v)]
        if values and (key.startswith(('current_', 'forecast_', 'delta_')) or key == 'online_pic50'):
            population[key] = {'count': len(values), 'min': min(values), 'max': max(values),
                'P25_P50_P75': np.quantile(values, [.25, .5, .75]).tolist()}
    return {'step': cutoff, 'choices': claims, 'applied_clauses': answer['applied_clauses'],
        'total_available_seed_groups': len(cards), 'compared_candidates': competitors,
        'available_population_current_scalar_ranges': population,
        'observed_broker_actions': calls,
        'semantics': {'population_ranges_weighting': 'one per available visible-equivalent seed group',
            'history_includes_all_past_for_named_claims': True,
            'footprints_are_unordered_sets': True,
            'delta_fields_describe_one_event_not_five_event_window': True}}
