"""Offline design audit only: no provider calls, sampling, claims or runtime edits.

The reduced view intentionally projects fields; it is NOT a lossless encoding.
The twelve-action shortlist is a design example, not an approved search policy.
"""
import collections
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import sys

import jsonschema
import uniguide_anchor_policy as v2

BASE = Path(__file__).resolve().parents[1]
DEST = BASE / 'docs/designs/uniguide_prompt_v3'
sys.path.insert(0, str(BASE / 'src'))
from agenticgen.uniguide.protocol import lattice_index, validate_action, choose


def canonical(x):
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def write(name, obj):
    (DEST / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
                            encoding='utf-8', newline='\n')


def obj(fields):
    return dict(type='object', properties=fields, required=list(fields), additionalProperties=False)


def text(limit):
    return dict(type='string', minLength=1, maxLength=limit)


def array(items, lo, hi):
    return dict(type='array', items=items, minItems=lo, maxItems=hi, uniqueItems=True)


source = gzip.decompress((DEST / 'UG05_R04_increase.prompt_source.json.gz').read_bytes())
original = json.loads(source)
context = json.loads(original['messages'][1]['content'])
rendered, _, v2_metrics = v2.transform(original['messages'], original['response_schema'],
                                      BASE / 'src/agenticgen/uniguide/skills')
branch = context['branch']
history = context['history']
anchors = {r['id']: r for r in history}
seen = [r['schedule'] for r in history]

# Exhaustively enumerate the existing legal move rules; no LLM arithmetic.
legal = []
for anchor_id, row in anchors.items():
    before = [lattice_index(x) for x in row['schedule']]
    for mode, step in [('fine', 1), ('coarse', 2)]:
        if branch == 'redistribute':
            deltas = [tuple(step if k == i else -step if k == j else 0 for k in range(4))
                      for i in range(4) for j in range(4) if i != j]
        else:
            sign = 1 if branch == 'increase' else -1
            deltas = [tuple(sign * step if i in changed else 0 for i in range(4))
                      for n in (1, 2) for changed in itertools.combinations(range(4), n)]
        for delta in deltas:
            after = [a + b for a, b in zip(before, delta)]
            # Keep unchanged raw scales byte-equivalent to their anchor values.
            schedule = [s if d == 0 else 2 ** (j / 2) for s, d, j in zip(row['schedule'], delta, after)]
            action = dict(branch=branch, anchor_id=anchor_id, mode=mode, schedule=schedule)
            try:
                validate_action(action, anchors, context['domain'], seen)
            except ValueError:
                continue
            legal.append(dict(**action, j=after, delta_j=list(delta)))

# A schedule can have several legal derivations; keep all in the audit sidecar.
priority = [context['default_anchor']['id'], context['comparison_best']['id']]
priority += sorted(set(anchors) - set(priority))
legal.sort(key=lambda r: (priority.index(r['anchor_id']), r['mode'] != 'fine',
                          sum(x != 0 for x in r['delta_j']), r['j']))
by_schedule = {}
for action in legal:
    by_schedule.setdefault(tuple(action['j']), []).append(action)
catalog = []
for j, derivations in by_schedule.items():
    a = derivations[0]
    identity = canonical(dict(anchor_id=a['anchor_id'], branch=branch, mode=a['mode'], j=j))
    catalog.append(dict(action_id='act_' + hashlib.sha256(identity.encode()).hexdigest()[:16],
                        **a, derivations=derivations))
assert len({a['action_id'] for a in catalog}) == len(catalog)

# Illustrative stratification retains both step sizes and different windows.
# A production shortlist needs explicit protocol review; no rollout is performed.
buckets = collections.defaultdict(list)
for action in catalog:
    buckets[(action['mode'], tuple(i for i, d in enumerate(action['delta_j']) if d))].append(action)
shortlist = []
bucket_order = sorted(buckets, key=lambda key:(len(key[1]),key[1],key[0] != 'fine'))
while len(shortlist) < min(12, len(catalog)):
    for key in bucket_order:
        if buckets[key] and len(shortlist) < 12:
            shortlist.append(buckets[key].pop(0))

metric_keys = ['id', 'attempts', 'complete', 'schedule', 'C', 'C_se', 'gap', 'mean_pic50',
               'best', 'built', 'unique_smiles', 'phase', 'error', 'failure_kind']


def row_view(row):
    view = {k: row[k] for k in metric_keys if k in row}
    windows = row.get('window_diagnostics', [])
    view['absolute_increment_by_window'] = [
        {k: w[k] for k in ['steps', 'increment_vector_rms_A', 'max_atom_increment_A']} for w in windows]
    view['window_diagnostics_available'] = bool(windows)
    return view


calibration = [r for rows in context['calibration_knowledge'].values() for r in rows]
calibration_columns = metric_keys + ['max_increment_rms_A', 'rms_window', 'max_atom_increment_A', 'atom_window']
calibration_rows = []
for row in calibration:
    windows = row.get('window_diagnostics', [])
    rms = max(windows, key=lambda w:w['increment_vector_rms_A']) if windows else None
    atom = max(windows, key=lambda w:w['max_atom_increment_A']) if windows else None
    calibration_rows.append([row.get(k) for k in metric_keys] + [
        rms['increment_vector_rms_A'] if rms else None, rms['steps'] if rms else None,
        atom['max_atom_increment_A'] if atom else None, atom['steps'] if atom else None])
best = choose(history)
minimum = min(r['C'] for r in history)
detail_ids = set(priority[:2]) | {r['id'] for r in history[-3:]}
view = dict(
    schema_version='ug-decision-view-v3-draft',
    task={k: context[k] for k in ['version', 'round', 'branch', 'no_future_execution', 'teacher']},
    domain={k: context['domain'][k] for k in ['initial', 'min_index', 'max_index', 'coarse_log2_step', 'fine_log2_step']},
    measured_history=[row_view(r) for r in history],
    calibration_summary=dict(columns=calibration_columns, rows=calibration_rows,
        omitted='Non-extremal window details are in the original evidence; absent values are null, not zero.'),
    selector=dict(minimum_C=minimum, ceiling_C=minimum+.02, representative_id=best['id'],
                  gap=best['gap'], eligible_ids=[r['id'] for r in history if r['C'] <= minimum+.02]),
    window_detail=[dict(evidence_id=r['id'], windows=r['window_diagnostics'])
                   for r in history if r['id'] in detail_ids],
    previous_actions=[dict(source_group=r['source_group'],
        **{k:r['proposal'][k] for k in ['anchor_id', 'prediction', 'falsifier']},
        interpretation='legacy_text_not_machine_graded') for r in context['completed_action_evidence'][-3:]],
    action_catalog=[{k:a[k] for k in ['action_id', 'anchor_id', 'mode', 'j', 'delta_j']} for a in shortlist],
    catalog_scope=dict(kind='illustrative_shortlist_not_approved', legal_derivations=len(legal),
                       unique_legal_schedules=len(catalog), displayed=len(shortlist)),
    slow_quality=dict(scope='not_in_synchronous_view', status='unavailable_not_pass'),
)

evidence_ids = [r['id'] for r in history + calibration]
assert len(set(evidence_ids)) == len(evidence_ids)
refs = array({'$ref':'#/$defs/evidence_id'}, 0, 2)
direction = dict(enum=['lower', 'higher', 'unknown'])
falsifier = dict(enum=['delta_gt_zero', 'delta_lt_zero', 'unresolved'])
option = obj(dict(action_id=dict(enum=[a['action_id'] for a in shortlist]), reason=text(120),
    evidence_ids=refs, prediction=obj(dict(C=direction, gap=direction)),
    falsifier=obj(dict(C=falsifier, gap=falsifier))))
schema = obj(dict(schema_version=dict(const='ug-private-v3-draft'), status=dict(enum=['ok', 'blocked']),
    diagnosis=obj(dict(conclusion=text(180), evidence_ids=refs)), options=array(option, 0, 2), uncertainty=text(160)))
schema['$defs'] = dict(evidence_id=dict(type='string',enum=evidence_ids))
schema['allOf'] = [
    {'if': {'properties': {'status': {'const':'ok'}}}, 'then': {'properties': {'options': {'minItems':2}}}},
    {'if': {'properties': {'status': {'const':'blocked'}}}, 'then': {'properties': {'options': {'maxItems':0}}}}]
schema['$schema'] = 'https://json-schema.org/draft/2020-12/schema'
example = dict(schema_version='ug-private-v3-draft', status='ok',
    diagnosis=dict(conclusion='Schema illustration only; no model inference or new measured outcome.', evidence_ids=[]),
    options=[dict(action_id=a['action_id'], reason='Illustrative legal catalog choice, not an experimental recommendation.',
        evidence_ids=[], prediction=dict(C='unknown', gap='unknown'),
        falsifier=dict(C='unresolved', gap='unresolved')) for a in shortlist[:2]],
    uncertainty='No online pilot has been performed; reasoning-token reduction is unverified.')


def validate_answer(answer):
    jsonschema.validate(answer, schema)
    ids = [x['action_id'] for x in answer['options']]
    assert len(ids) == len(set(ids)), 'Two different option objects cannot reuse the same action'
    mapping = dict(lower='delta_gt_zero', higher='delta_lt_zero', unknown='unresolved')
    for option in answer['options']:
        assert all(option['falsifier'][k] == mapping[option['prediction'][k]] for k in ['C','gap'])
        a = next(a for a in shortlist if a['action_id'] == option['action_id'])
        validate_action(a, anchors, context['domain'], seen)


validate_answer(example)
retrospective_item = obj(dict(source_group=dict(enum=[r['source_group'] for r in context['completed_action_evidence']]),
    C=dict(enum=['supported','refuted','unresolved']), gap=dict(enum=['supported','refuted','unresolved'])))
choices = [obj(dict(branch=dict(const=b),candidate_id=dict(enum=[b+':0',b+':1']),
    reason=text(120),alternative_weakness=text(100),evidence_ids=refs)) for b in ['increase','decrease','redistribute']]
synthesis_schema = obj(dict(schema_version=dict(const='ug-synthesis-v3-draft'),status=dict(enum=['ok','blocked']),
    choices=dict(type='array',prefixItems=choices,items=False,minItems=0,maxItems=3),
    retrospective=array(retrospective_item,0,3),uncertainty=text(160)))
synthesis_schema['$defs']=schema['$defs'];synthesis_schema['$schema']=schema['$schema']
synthesis_schema['allOf']=[
    {'if':{'properties':{'status':{'const':'ok'}}},'then':{'properties':{'choices':{'minItems':3}}}},
    {'if':{'properties':{'status':{'const':'blocked'}}},'then':{'properties':{'choices':{'maxItems':0}}}}]
synthesis_example=dict(schema_version='ug-synthesis-v3-draft',status='blocked',choices=[],retrospective=[],
    uncertainty='Illustration only: this offline private-request audit has no three-branch candidate set.')
final_schema=obj(dict(schema_version=dict(const='ug-final-v3-draft'),representative_id=dict(const=best['id']),
    retrospective=array(retrospective_item,0,3),limitation=text(180)))
final_schema['$schema']=schema['$schema']
final_example=dict(schema_version='ug-final-v3-draft',representative_id=best['id'],retrospective=[],
    limitation='Schema illustration using partial measured history, not UG05 final analysis or completion.')
jsonschema.validate(synthesis_example,synthesis_schema);jsonschema.validate(final_example,final_schema)
negative_checks = {}
for name, mutate in [
    ('reference_label_as_action', lambda x:x['options'][0].update(action_id='r80')),
    ('unknown_action', lambda x:x['options'][0].update(action_id='act_unknown')),
    ('extra_field', lambda x:x.update(schedule=[1,2,3,4])),
    ('too_long_text', lambda x:x['diagnosis'].update(conclusion='x'*181)),
    ('duplicate_action', lambda x:x['options'][1].update(action_id=x['options'][0]['action_id'], reason='Different prose')),
    ('contradictory_falsifier', lambda x:x['options'][0]['falsifier'].update(C='delta_gt_zero')),
    ('missing_option', lambda x:x['options'].pop()),
]:
    bad = json.loads(json.dumps(example));mutate(bad)
    try:validate_answer(bad)
    except (AssertionError,jsonschema.ValidationError):negative_checks[name]=True
    else:raise AssertionError(name)

# Verify every retained field directly against original evidence, including failures.
for source_row, projected in zip(history,view['measured_history']):
    assert all(projected[k] == source_row[k] for k in metric_keys if k in source_row)
    assert len(projected['absolute_increment_by_window']) == len(source_row.get('window_diagnostics', []))
    for original_window, projected_window in zip(source_row.get('window_diagnostics', []),projected['absolute_increment_by_window']):
        assert all(original_window[k] == v for k,v in projected_window.items())
for source_row, projected in zip(calibration,calibration_rows):
    expanded = dict(zip(calibration_columns,projected))
    assert all(expanded[k] == source_row[k] for k in metric_keys if k in source_row)
    windows = source_row.get('window_diagnostics', [])
    for dest_key, source_key in [('max_increment_rms_A','increment_vector_rms_A'),('max_atom_increment_A','max_atom_increment_A')]:
        assert expanded[dest_key] == (max(w[source_key] for w in windows) if windows else None)

fields = {k:len(canonical(v)) for k,v in context.items()}
full_catalog_view = dict(view, action_catalog=[{k:a[k] for k in ['action_id','anchor_id','mode','j','delta_j']} for a in catalog])
skills = '\n\n'.join((DEST/'skills'/name).read_text(encoding='utf-8') for name in ['analysis.md','proposal.md'])
wire_chars = len(skills)+len(canonical(view))+len(v2.format_contract(schema))
old_wire_chars = sum(len(m['content']) for m in rendered)+len(v2.format_contract(v2.transform(
    original['messages'],original['response_schema'],BASE/'src/agenticgen/uniguide/skills')[1]))
audit = dict(design_only=True, production_modified=False, model_calls=0, new_samples=0,
    source_sha256=hashlib.sha256(source).hexdigest(), observed_original_input_tokens=58826,
    field_characters=fields, old_v2=v2_metrics, old_v2_wire_characters=old_wire_chars,
    projected_context_characters=len(canonical(view)), full_catalog_context_characters=len(canonical(full_catalog_view)),
    projected_wire_characters=wire_chars, wire_character_reduction=1-wire_chars/old_wire_chars,
    tokens_not_estimated=True, retained_history_rows=len(history), retained_calibration_rows=len(calibration),
    retained_failed_calibration_rows=sum(not r['complete'] for r in calibration),
    legal_derivations=len(legal),unique_legal_schedules=len(catalog),shortlisted=len(shortlist),
    negative_schema_semantic_checks=negative_checks,
    limitations=['projection is intentionally not lossless; full original retained',
        'twelve-action subset is a search-policy change, not an approved default',
        'character reduction is not token reduction or evidence of reduced reasoning',
        'real inference and frozen production acceptance remain unperformed'])
write('input_projection.example.json', view)
write('action_catalog.audit.json', catalog)
write('proposal.schema.json', schema)
write('proposal.example.json', example)
write('synthesis.schema.json', synthesis_schema)
write('synthesis.example.json', synthesis_example)
write('final.schema.json', final_schema)
write('final.example.json', final_example)
write('offline_audit.json', audit)
print(canonical(audit))
