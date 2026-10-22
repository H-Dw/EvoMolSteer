"""Evidence-scoped controller contracts; no target-specific optimization answers."""
import copy
from .contracts import validate_ticks
from .io import value_sha


def enabled(config):
    return config.get('controller', {}).get('protocol') == 'conditional_learning_v1'


def expand_template(template):
    """Exactly one covalent hop from the original PLIP set, never recursive."""
    t = copy.deepcopy(template)
    if 'original_core_atoms' in t:
        raise ValueError('Refuse to expand an already expanded template')
    n = len(t['coords_A']); core = sorted(set(t['core_atoms']))
    bonds = t['bond_labels']
    if not core or any(i < 0 or i >= n for i in core) or len(bonds) != n:
        raise ValueError('Invalid original atom mapping')
    expanded = sorted(set(core) | {j for i in core for j in range(n)
                                  if i != j and (bonds[i][j] > 0 or bonds[j][i] > 0)})
    t.update(original_template_sha256=value_sha(template), original_core_atoms=core,
             original_context_atoms=t.get('context_atoms', []), core_atoms=expanded,
             context_atoms=expanded, guidance_region='core_plus_one_hop',
             guidance_coverage=len(expanded)/n)
    t['teacher_profile'] = {
        'source_template_sha256': t['original_template_sha256'],
        'original_plip_core': core, 'guided_atoms': expanded,
        'one_hop_neighbors': sorted(set(expanded)-set(core)), 'heavy_atom_count': n,
        'coverage': len(expanded)/n, 'large_coverage': len(expanded)/n >= .75,
        'coordinates_A': t['coords_A'], 'atom_labels': t.get('atom_labels'),
        'charge_labels': t.get('charge_labels'), 'bond_labels': bonds,
        'interactions': t.get('interactions'),
        'labels_are': 'Native model vocabulary indices, not atomic numbers or physical charges unless explicitly decoded.',
        'directly_controlled': ['protein-frame coordinates of guided atoms'],
        'indirect_only': ['element identity', 'bonding', 'formal charge', 'affinity', 'protein interactions'],
        'constraint': 'Coordinates alone do not establish chemical or binding equivalence.'}
    return t


def learning_schema(evidence_ids):
    text = {'type': 'string', 'minLength': 1, 'maxLength': 1800}
    props = {k: text for k in ['belief_id', 'prior_judgment', 'intervention', 'observed_result',
                             'revision', 'next_action', 'applicability', 'alternative_explanation']}
    props['status'] = {'enum': ['supported', 'refuted', 'unresolved']}
    props['evidence_ids'] = {'type': 'array', 'minItems': 1, 'maxItems': 8,
                             'uniqueItems': True, 'items': {'enum': sorted(evidence_ids)}}
    return {'type': 'array', 'minItems': 1, 'maxItems': 6, 'items': {
        'type': 'object', 'additionalProperties': False, 'properties': props, 'required': list(props)}}


def verified_large_steps(history, anchor):
    """0.3 only after two comparable single-interval positive edges in one direction."""
    if len(history) < 3 or history[-1]['ticks'] != anchor:
        return {}
    recent = history[-3:]; edges = []
    for a, b in zip(recent, recent[1:]):
        changes = [(i, y-x) for i, (x, y) in enumerate(zip(a['ticks'], b['ticks'])) if x != y]
        if len(changes) != 1 or b.get('condition_error_delta', 0) >= 0 or b.get('affinity_gap_delta', 0) > 0 or b.get('paired_common_valid', 0) < 1:
            return {}
        if b.get('attempts') not in (25,100):
            return {}
        edges.append(changes[0])
    (i, d), (j, e) = edges
    return {i: 1 if d > 0 else -1} if i == j and d*e > 0 else {}


def validate_memory(memories, teacher, version, observed_round):
    for m in memories:
        if (m.get('teacher_id') != teacher or m.get('version') != version
                or m.get('published_after_round', observed_round) >= observed_round):
            raise ValueError('Task knowledge must come from the same teacher/version and a prior published round')


def executed_schedules(data, memories):
    """Known factual executions only; neither pending proposals nor sibling-private data."""
    result = {}
    for chain in data['chains'][:1]:
        for t in chain['trials']:
            if t['id'] != 'parent':
                result[tuple(t['ticks'])] = {'source': t.get('source_id', t['id']), 'ticks': t['ticks']}
    for memory in memories:
        for item in memory.get('executed', []):
            result[tuple(item['ticks'])] = item
    return list(result.values())


def validate_controller_hypotheses(message, anchor, count, evidence_ids, large, executed, policy=None):
    from .search_policy import policy as defaults, validate_probe_quota
    policy = policy or defaults()
    hs = message['hypotheses']
    if len(hs) != count or [h['id'] for h in hs] != [f'H{i+1}' for i in range(count)]:
        raise ValueError('Ordered hypothesis IDs required')
    seen = {tuple(x['ticks']) for x in executed}
    for h in hs:
        ticks = validate_ticks(h['ticks'], int(policy['upper_amplitude']*10))
        if not any(ticks):
            raise ValueError('An all-zero guidance experiment is outside this protocol')
        if tuple(ticks) in seen:
            raise ValueError('Schedule already executed or duplicated: query its cached evidence and propose a new experiment')
        seen.add(tuple(ticks))
        changes = [(i, b-a) for i, (a, b) in enumerate(zip(anchor, ticks)) if a != b]
        if not 1 <= len(changes) <= 2:
            raise ValueError('Change one or two intervals from the explicit anchor')
        mode = h.get('search_mode', 'local')
        if mode not in {'local', 'range_probe'}:
            raise ValueError('Unknown search mode')
        if mode == 'range_probe':
            if policy['upper_amplitude'] <= 1 or not h.get('scale_reason', '').strip():
                raise ValueError('Range probe requires expanded authorization and scale rationale')
        else:
            for i, d in changes:
                if abs(d) > 3 or (abs(d) == 3 and large.get(i) != (1 if d > 0 else -1)):
                    raise ValueError('Step 0.1 or 0.2; 0.3 requires two verified comparable improving edges')
        if h['reference_ticks'] != anchor:
            raise ValueError('Proposal reference_ticks must match its actual action anchor; cite other causal comparisons separately')
        if not set(h['evidence_ids']) <= evidence_ids:
            raise ValueError('Unknown evidence citation')
    # Bootstrap directly selects four. Private two-proposal lists can each offer
    # a probe, but must also supply a local alternative for the council quota.
    validate_probe_quota(hs, active=policy)


def belief_view(memories):
    """Latest revision wins within a belief identity; older evidence remains archived."""
    active = {}
    for m in memories:
        for entry in m.get('learning_update', []):
            active[entry['belief_id']] = dict(entry, published_after_round=m['published_after_round'])
    return list(active.values())
