"""Resume-safe, real-model Scientist/Selector development on frozen CK2 traces.

Only this trusted controller sees oracle membership. Decision workers receive
fresh opaque IDs, clock-bounded observations, and the corresponding hypothesis.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import re
import secrets
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

STEPS = list(range(5, 51, 5))
MODEL = 'gpt-6-luna'
MAIN_MODEL = 'gpt-6.1-sol'
DEFAULT_STUDY = 'docs/experiments/ck2_scientist_selector_20261013'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def object_sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def schema_obj(properties, required=None):
    return {'type': 'object', 'properties': properties, 'required': required or list(properties), 'additionalProperties': False}


def arr(item, minimum=0, maximum=None):
    d = {'type': 'array', 'items': item, 'minItems': minimum}
    if maximum is not None:
        d['maxItems'] = maximum
    return d


TEXT = {'type': 'string'}
INTEGER = {'type': 'integer'}
CLAUSE = schema_obj({'clause_id': TEXT, 'evidence': TEXT})
OBS = schema_obj({'feature': TEXT, 'step': INTEGER, 'direction_or_value': TEXT})
SELECTOR_SCHEMA = schema_obj({
    'step': INTEGER,
    'ranking': arr(schema_obj({'rank': INTEGER, 'candidate_id': TEXT,
        'confidence': {'type': 'number', 'minimum': 0, 'maximum': 1},
        'evidence': TEXT, 'counterevidence': TEXT, 'observations': arr(OBS, 2)}), 5, 5),
    'applied_clauses': arr(CLAUSE, 1),
})
SCIENTIST_SCHEMA = schema_obj({
    'interval_descriptions': arr(schema_obj({'donor_id': TEXT, 'interval': INTEGER,
        'text': TEXT, 'evidence': arr(schema_obj({'step': INTEGER, 'feature': TEXT, 'claim': TEXT}), 1)}), 300, 300),
    'hypotheses': arr(schema_obj({'interval': INTEGER, 'start_step': INTEGER, 'end_step': INTEGER,
        'hypothesis': TEXT, 'support_conditions': arr(TEXT, 1), 'counter_conditions': arr(TEXT, 1),
        'observable_cues': arr(TEXT, 2), 'minority_modes': TEXT}), 10, 10),
    'applied_clauses': arr(CLAUSE, 1),
})
REVISION_SCHEMA = schema_obj({'diagnosis': TEXT, 'mechanism': TEXT, 'skill_text': TEXT,
    'workflow_text': TEXT, 'clause_id': TEXT, 'observable_effect': TEXT,
    'previous_clause_audit': TEXT})
AUDIT_SCHEMA = schema_obj({'passes': {'type': 'boolean'}, 'scientist_assessment': TEXT,
    'scientist_evidence_truth': {'type': 'boolean'}, 'scientist_clause_applied': {'type': 'boolean'},
    'selector_assessments': arr(schema_obj({'step': INTEGER, 'evidence_truth': {'type': 'boolean'},
        'clause_applied': {'type': 'boolean'}, 'evidence': TEXT, 'correction_needed': TEXT}), 10, 10),
    'unsupported_claims': arr(TEXT), 'instruction_effect_assessment': TEXT,
    'failure_analysis': TEXT})

# The same factual quality gate applies to both stages; these corrections never
# contain scores, target membership or future observations and do not change
# the nominal role policy. Every actual corrective prompt is versioned.
QUALITY_GUIDANCE = {
    'scientist': ('Factual quality correction: inspect the actual six states of EACH newly unlocked window. '
        'Write concrete qualitative start/middle/end shape, displacement in the fixed receptor frame, pocket contact footprint and any supported reversal. '
        'A reversal requires a measured change of direction; if it is absent, say so. Compare teacher patterns with the same-time background and distinguish '
        'a shared generation-clock effect from a teacher-specific pattern. Do not copy generic prose across windows instead of describing observed behavior. '
        'A minority mode needs observed examples; conditional possibilities alone do not establish one. Do not infer an attraction, formed molecule, polar-group '
        'orientation, hydrogen bond or improved energy from whole-cloud geometry or raw labels. Use qualitative language and exact single-key, step-span evidence. '
        'Ground the active policy clause in actual observations; if its proposed condition is absent, report the grounded absence rather than inventing a pattern.'),
    'selector': ('Factual quality correction: verify every selected seed and every supporting or opposing claim against its own exact current/past history. '
        'Cite ONE literal feature key and one observed value per observation item; cite both ends of any claimed change. '
        'Do not treat different anonymous IDs as evidence of different geometric modes. Compare actual shape, location, contact footprint and history, and '
        'describe differences in proportion to the observed size of the difference. Validate the newest policy clause by a concrete comparison of observed '
        'competitors or a grounded finding that its conditional trigger is absent. Do not assert binding energy, hydrogen bonds, future survival or final quality. '
        'Use the full allowed population and the same hypothesis. This is factual review; no outcomes or retrieval-score feedback are available.'),
}


SCIENTIST_SKILL = '''---
name: ck2-path-scientist
description: Describe recorded CK2 high-quality candidate growth and propose qualitative interval hypotheses with private provenance.
---

Use the supplied thirty teacher paths and same-time population summaries to describe their growth at ten intervals. Teacher membership was selected retrospectively; this is retrieval, not prospective discovery.

S.BASE.1: Separate current generation clouds from model-predicted endpoint structures. Ground claims in the supplied coordinate features; early crowded point clouds are not physical formed molecules.
S.BASE.2: Describe each interval's beginning, middle, ending, pocket focus, directional change and reversals. Use exact observed ancestry history. A hypothesis for interval i may only describe evidence at or before its endpoint.
S.BASE.3: Compare same-time background geometry; explain majority patterns and plausible minority modes rather than repeating the common generation clock. Do not claim a contrast that is absent in the data.

Public hypotheses contain qualitative macroscopic language only: no donor, candidate, trajectory or branch identifiers; no final graph identity or final scores; no exact continuous geometry numbers. Detailed private descriptions may cite anonymous donor IDs in metadata. Evidence indices remain private to the controller.
'''
SELECTOR_SKILL = '''---
name: ck2-path-selector
description: Rank recorded CK2 intermediate seeds using one qualitative growth hypothesis, available history and current affinity without final outcomes.
---

Select five distinct available seeds by confidence at the supplied cutoff. Inspect the full candidate population; detailed history is available from the restricted broker.

Q.BASE.1: Translate this hypothesis into observable structure and history evidence. Distinguish current geometry, endpoint forecast and temporal revision. Cite measured feature names and their observation steps.
Q.BASE.2: Current and historical on-target affinity are available; weigh them with structural evidence and competing interpretations. They are model forecasts, not known final affinity. Do not reconstruct hidden selection probabilities or final labels.
Q.BASE.3: Examine counterexamples and different geometric modes before allocating five positions. Do not spend all positions on indistinguishable versions of one mode without an evidence-based reason.

Return exactly five ranked IDs with confidence, observed supporting facts, counterevidence, and applied skill clauses. Unknown futures of pruned branches remain unknown. Do not write or execute a replacement numerical selector.
'''
SCIENTIST_WORKFLOW = '''Read all thirty teacher histories and the background summaries. For each of the ten five-event intervals, inspect actual within-interval states rather than only endpoints. Produce one detailed private paragraph per donor and interval, then synthesize an anonymous qualitative hypothesis including minority modes and counterconditions. Link claims to real feature/step evidence. Complete all 300 descriptions and ten hypotheses. Audit each active new clause through a concrete observation or change in the produced descriptions.
'''
SELECTOR_WORKFLOW = '''Read this interval hypothesis and the entire compact candidate pool. Compare forecast shape, current-cloud development, local pocket distribution, signed changes and available affinity. Query full histories for plausible competing modes and counterexamples. Rank five unique seeds by confidence. For each choice cite at least two real feature observations, including history when informative; explain a countercondition. Show concrete application of the newest skill clause. Return JSON only.
'''


def init_versions(study, repo):
    for role, code, skill, workflow in [('scientist', 'S00', SCIENTIST_SKILL, SCIENTIST_WORKFLOW),
                                       ('selector', 'Q00', SELECTOR_SKILL, SELECTOR_WORKFLOW)]:
        folder = study / 'versions' / role / code
        if not folder.exists():
            folder.mkdir(parents=True)
            (folder / 'SKILL.md').write_text(skill, encoding='utf-8')
            (folder / 'workflow.md').write_text(workflow, encoding='utf-8')
            write(folder / 'version.json', {'version': code, 'role': role, 'parent': None,
                'active_clause': 'S.BASE.3' if role == 'scientist' else 'Q.BASE.3',
                'model': MODEL, 'effort': 'medium', 'skill_sha256': digest(folder / 'SKILL.md'),
                'workflow_sha256': digest(folder / 'workflow.md')})
        live = repo / 'skills' / f'ck2-path-{role}'
        live.mkdir(parents=True, exist_ok=True)
        if not (live / 'SKILL.md').exists():
            shutil.copyfile(folder / 'SKILL.md', live / 'SKILL.md')
            shutil.copyfile(folder / 'workflow.md', live / 'workflow.md')


def version(study, role, code):
    p = study / 'versions' / role / code
    return read(p / 'version.json'), (p / 'SKILL.md').read_text(encoding='utf-8'), (p / 'workflow.md').read_text(encoding='utf-8')


def role_prompt(study, role, code, format_note):
    meta, skill, workflow = version(study, role, code)
    return (f'You are the {role.title()} in this recorded CK2 experiment. Use GPT-6 Luna medium.\n'
        f'Version {code}; active new clause {meta["active_clause"]}.\n\n{skill}\n\nWorkflow:\n{workflow}\n\n'
        f'{format_note}\nReturn only JSON matching the supplied schema. Record actual application of {meta["active_clause"]} '
        'in applied_clauses, citing concrete evidence rather than echoing the instruction. No requests to the human.')


def public_hypothesis(h):
    return {k: h[k] for k in ['interval', 'start_step', 'end_step', 'hypothesis',
        'support_conditions', 'counter_conditions', 'observable_cues', 'minority_modes']}


def anonymize(pool, existing_mapping=None, structural_only=False):
    original = list(pool['cards'])
    if existing_mapping:
        mapping = existing_mapping
        reverse = {k: a for a, k in mapping['anonymous_to_key'].items()}
        original = sorted(original, key=lambda c: mapping['order'].index(reverse[c['key']]))
    else:
        random.SystemRandom().shuffle(original)
        reverse = {c['key']: 'C' + secrets.token_hex(6) for c in original}
        mapping = {'anonymous_to_key': {a: k for k, a in reverse.items()},
                   'order': [reverse[c['key']] for c in original]}
    cards = []
    for c in original:
        row = copy.deepcopy(c)
        row.pop('key')
        row['candidate_id'] = reverse[c['key']]
        if structural_only:
            for state in [row['now'], *row['history']]:
                for key in list(state):
                    if any(term in key.lower() for term in ('pic50', 'affinity', 'score', 'probability', 'weight', 'rank')):
                        state.pop(key)
        cards.append(row)
    return {'role': 'selector', 'step': pool['step'], 'time': pool['time'], 'cards': cards,
        'affinity_available': not structural_only}, mapping


def teacher_packet(study):
    teacher = read(study / 'evidence' / 'teacher.json')
    cache = study / 'runtime' / 'population_summaries.json'
    if not cache.exists():
        f = pd.read_parquet(study / 'evidence' / 'features.parquet')
        selected = [k for k in f.select_dtypes(include='number').columns
                    if k.startswith(('current_', 'forecast_', 'delta_')) and not any(s in k for s in ('slot', 'batch', 'root'))]
        rows = []
        for step, group in f.groupby('step'):
            features = {}
            for k in selected:
                v = pd.to_numeric(group[k], errors='coerce').dropna().to_numpy()
                if len(v):
                    features[k] = [round(float(x), 5) for x in np.quantile(v, [.25, .5, .75])]
            rows.append({'step': int(step), 'time': int(step) / 100, 'quantiles_P25_P50_P75': features})
        write(cache, rows)
    return {'role': 'scientist', 'teacher': teacher, 'background': read(cache),
        'clock': {'steps': list(range(51)), 'interval_ends': STEPS,
                  'current_vs_proposal': 'current at s; proposal at s+1, never substitute for current'}}


def clause_audit(answer, expected):
    matches = [r for r in answer.get('applied_clauses', []) if r.get('clause_id') == expected]
    if not matches or any(len(r.get('evidence', '').strip()) < 18 for r in matches):
        raise ValueError(f'New instruction {expected} has no concrete output application')
    return {'clause_id': expected, 'declared_applications': matches,
            'method': 'Prompt binding plus valid source observations; semantic review by next xhigh controller revision',
            'application_claim_not_a_causal_improvement': True}


def validate_scientist(answer, packet, active):
    donors = packet['teacher'].get('donors', [])
    by_id = {r['donor_id']: r for r in donors}
    expected = {(i, d) for i in range(1, 11) for d in by_id}
    desc = answer.get('interval_descriptions', [])
    if len(desc) != 300 or {(r['interval'], r['donor_id']) for r in desc} != expected:
        raise ValueError('Scientist must describe all 30 donor paths in all 10 intervals exactly once')
    for r in desc:
        lo, hi = 5 * (r['interval'] - 1), 5 * r['interval']
        states = {int(v['step']): v for v in by_id[r['donor_id']]['observations']}
        for e in r['evidence']:
            if not lo <= e['step'] <= hi or e['feature'] not in states[e['step']]:
                raise ValueError('Scientist cites nonexistent or out-of-interval evidence')
    hs = sorted(answer['hypotheses'], key=lambda h: h['interval'])
    if [h['interval'] for h in hs] != list(range(1, 11)):
        raise ValueError('Ten distinct hypotheses required')
    for h in hs:
        if (h['start_step'], h['end_step']) != (5 * (h['interval'] - 1), 5 * h['interval']):
            raise ValueError('Hypothesis clock is incorrect')
        text = ' '.join([h['hypothesis'], h['minority_modes'], *h['support_conditions'],
                         *h['counter_conditions'], *h['observable_cues']])
        if re.search(r'(?:(?<![A-Za-z0-9])T\d{1,3}(?![A-Za-z0-9])|single_b\d+|batch[_ ]?\d+|final[_ ]?\d+|(?<![A-Za-z0-9])C[0-9a-f]{8,}(?![A-Za-z0-9])|(?:分支|轨迹|候选|candidate|trajectory|branch)[_ #:-]*\d+)', text, re.I):
            raise ValueError('Public hypothesis contains donor or source identity')
        for token in re.findall(r'\d+\.\d+', text):
            if float(token) not in {n / 100 for n in range(0, 51, 5)}:
                raise ValueError('Public hypothesis contains an exact continuous measurement')
    return clause_audit(answer, active)


def validate_selector(answer, packet, active):
    by_id = {r['candidate_id']: r for r in packet['cards']}
    if answer['step'] != packet['step']:
        raise ValueError('Selector cutoff mismatch')
    if [r['rank'] for r in answer['ranking']] != [1, 2, 3, 4, 5]:
        raise ValueError('Invalid ranking order')
    if len({r['candidate_id'] for r in answer['ranking']}) != 5:
        raise ValueError('Duplicate seeds selected')
    conf = [r['confidence'] for r in answer['ranking']]
    if conf != sorted(conf, reverse=True):
        raise ValueError('Ranking contradicts stated confidence order')
    for r in answer['ranking']:
        if r['candidate_id'] not in by_id:
            raise ValueError(f"Unknown candidate_id {r['candidate_id']!r}; copy the complete ID verbatim from candidate_population and verify with get_candidate")
        states = {int(v['step']): v for v in by_id[r['candidate_id']]['history']}
        for e in r['observations']:
            if e['step'] > packet['step'] or e['step'] not in states or e['feature'] not in states[e['step']]:
                keys = sorted(states.get(e['step'], {}))
                raise ValueError(f"Selector cites unavailable observation: candidate_id={r['candidate_id']}, step={e['step']}, feature={e['feature']!r}. "
                    'Each observations item must contain exactly ONE literal feature key; split joined / or ; feature names into separate items. '
                    'Put one corresponding observed value in direction_or_value. '
                    f'Available feature keys at this step: {keys}')
    return clause_audit(answer, active)


def run_validated(study, job, role, prompt, packet, schema, validate, active, model=MODEL, effort='medium'):
    from .runner import run_job
    expected = object_sha({'prompt': prompt, 'packet': packet, 'schema': schema, 'model': model, 'effort': effort})
    binding_path = job / 'controller_binding.json'
    if binding_path.exists() and read(binding_path)['request_sha256'] != expected:
        raise ValueError(f'Resumed job inputs changed: {job}')
    write(binding_path, {'request_sha256': expected, 'model': model, 'effort': effort,
        'clock': packet.get('step'), 'role': role})
    if (job / 'answer.json').exists() and (job / 'functional_audit.json').exists():
        answer = read(job / 'answer.json')
        validate(answer, packet, active)
        return answer
    prior_attempts = [int(p.name[1:]) for p in (job / 'attempts').glob('A[0-9]*') if p.is_dir()]
    problems = []
    for prior in sorted(prior_attempts, reverse=True)[:3]:
        prior_dir = job / 'attempts' / f'A{prior:02d}'
        if (prior_dir / 'answer.json').exists():
            try:
                validate(read(prior_dir / 'answer.json'), packet, active)
            except Exception as exc:
                problems.append(f'{type(exc).__name__}: {str(exc)[:3500]}')
        elif (prior_dir / 'controller_rejection.json').exists():
            problems.append(read(prior_dir / 'controller_rejection.json')['error'])
    problems = list(dict.fromkeys(problems))
    first_attempt = max(prior_attempts, default=0) + 1
    for attempt in range(first_attempt, first_attempt + 3):
        attempt_dir = job / 'attempts' / f'A{attempt:02d}'
        try:
            extra = ('\nPrevious validation failures to repair without changing available information: ' +
                     json.dumps(problems, ensure_ascii=False)) if problems else ''
            answer = run_job(attempt_dir, role, prompt + extra, packet, schema, model=model, effort=effort)
            audit = validate(answer, packet, active)
            write(job / 'answer.json', answer)
            audit.update({'request_sha256': expected, 'attempt': attempt,
                'prompt_contains_active_clause': active in prompt, 'effort': effort, 'model': model})
            write(job / 'functional_audit.json', audit)
            if (attempt_dir / 'receipt.json').exists():
                shutil.copyfile(attempt_dir / 'receipt.json', job / 'receipt.json')
            return answer
        except Exception as exc:
            problems.append(f'{type(exc).__name__}: {str(exc)[:3500]}')
            write(attempt_dir / 'controller_rejection.json', {'error': problems[-1], 'attempt': attempt})
            print(json.dumps({'job': str(job.relative_to(study)), 'attempt': attempt, 'rejected': problems[-1]}, ensure_ascii=False), flush=True)
    raise RuntimeError(f'No valid {role} result after 3 attempts: ' + problems[-1])


def run_scientist(study, round_dir, code):
    packet = teacher_packet(study)
    meta, _, _ = version(study, 'scientist', code)
    prompt = role_prompt(study, 'scientist', code,
        'Write in Chinese. interval_descriptions requires 300 records: each of 30 donor_id values and intervals1..10. '
        'Each paragraph should explain current-cloud and forecast changes, pocket focus and any reversal, using qualitative language. '
        'Private evidence entries refer to actual donor observation feature keys and steps within that interval. '
        'Ten public hypotheses each include observable cues and counterconditions; identifiers and exact continuous values are forbidden in public prose.')
    recovery = read(round_dir / 'quality_retry.json') if (round_dir / 'quality_retry.json').exists() else {}
    if recovery.get('scientist'):
        prompt += '\n' + QUALITY_GUIDANCE['scientist']
    answer = run_validated(study, round_dir / 'scientist', 'scientist', prompt, packet,
        SCIENTIST_SCHEMA, validate_scientist, meta['active_clause'])
    write(round_dir / 'hypotheses.json', [public_hypothesis(h) for h in sorted(answer['hypotheses'], key=lambda h: h['interval'])])
    return answer


def run_selectors(study, round_dir, code, hypotheses, workers=3, structural_only=False):
    from .data import load_pool
    meta, _, _ = version(study, 'selector', code)
    def one(step):
        job = round_dir / 'selectors' / f't{step:03d}'
        existing = read(job / 'mapping.json') if (job / 'mapping.json').exists() else None
        packet, mapping = anonymize(load_pool(study, step), existing, structural_only)
        packet['hypothesis'] = next(h for h in hypotheses if h['end_step'] == step)
        packet['semantics'] = {'raw_current_cloud_is_not_finished_molecule': True,
            'forecast_is_intermediate_not_final': True, 'final_labels_available': False,
            'history_ends_at_current_preselection_state': True, 'population_scope': 'All original batches, grouped visible-equivalent states'}
        write(job / 'mapping.json', mapping)
        prompt = role_prompt(study, 'selector', code,
            f'Return step={step}. This is an independent selection from the FULL current population, not a continuation of earlier Selector decisions. '
            'No final outcomes are available. Cite observation feature keys and real past/current integer steps. '
            + ('All intermediate affinity is hidden in this control.' if structural_only else 'Current and historical on-target online_pic50 are allowed.'))
        recovery = read(round_dir / 'quality_retry.json') if (round_dir / 'quality_retry.json').exists() else {}
        if step in recovery.get('selector_steps', []):
            prompt += '\n' + QUALITY_GUIDANCE['selector']
        answer = run_validated(study, job, 'selector', prompt, packet, SELECTOR_SCHEMA,
            validate_selector, meta['active_clause'])
        print(json.dumps({'round': round_dir.name, 'selector_step': step, 'valid': True,
                          'condition': 'structure_only' if structural_only else 'affinity_visible'}, ensure_ascii=False), flush=True)
        return answer
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one, s) for s in STEPS]
        for f in as_completed(futures):
            f.result()


def macro_metrics(bundle):
    if not bundle.get('complete'):
        raise ValueError('Incomplete ten-node round')
    return {f'coverage_at_{k}': float(np.mean([r[f'at_{k}']['coverage'] for r in bundle['metrics']])) for k in (1, 3, 5)}


def score_tuple(bundle):
    m = bundle.get('macro') or macro_metrics(bundle)
    return tuple(m[f'coverage_at_{k}'] for k in (5, 3, 1))


def failure_packet(study, previous):
    """Privileged controller feedback, never placed in a Selector context."""
    m = read(previous / 'metrics.json')
    nodes = []
    for result in m['metrics']:
        step = result['step']
        oracle = read(study / 'oracle' / 'pools' / f't{step:03d}.json')
        pool = read(study / 'evidence' / 'pools' / f't{step:03d}.json')
        features = {c['key']: c['now'] for c in pool['cards']}
        selected = result['ranking']
        chosen_keys = {r['key'] for r in selected}
        positive = [g for g in oracle['groups'] if g.get('targets')]
        missed = sorted((g for g in positive if g['key'] not in chosen_keys), key=lambda g: -len(g['targets']))[:4]
        nodes.append({'step': step, 'coverage': {str(k): result[f'at_{k}']['coverage'] for k in (1, 3, 5)},
            'ceiling': result['ideal_weighted_coverage_ceiling'],
            'selected': [{'rank': r['rank'], 'target_count': r['target_count'],
                          'observed_descendants': r.get('observed_descendants'), 'features': features[r['key']]} for r in selected],
            'missed_observed_positive_modes': [{'target_count': len(g['targets']), 'features': features[g['key']]} for g in missed],
            'selector_answer': read(previous / 'selectors' / f't{step:03d}' / 'answer.json'),
            'functional_audit': read(previous / 'selectors' / f't{step:03d}' / 'functional_audit.json')})
    return {'macro': m['macro'], 'nodes': nodes,
        'scientist_hypotheses': read(previous / 'hypotheses.json'),
        'semantic_evidence_audit': read(previous / 'semantic_audit.json') if (previous / 'semantic_audit.json').exists() else None,
        'scientist_description_samples': [row for row in read(
            (study / 'rounds' / read(previous / '_version.json')['scientist_outputs_frozen_from']
             if read(previous / '_version.json').get('scientist_outputs_frozen_from') else previous)
            / 'scientist' / 'answer.json')['interval_descriptions'] if row['donor_id'] in {'T01', 'T02', 'T03'}],
        'previous_version': read(previous / '_version.json'),
        'no_descendants_means': 'censored, not low final affinity',
        'scope': 'This private feedback is for main-agent xhigh only; future decision workers never see it'}


def revise(study, round_dir, role, code, parent, previous):
    from .runner import run_job
    dest = study / 'versions' / role / code
    if (dest / 'version.json').exists():
        return
    base, skill, workflow = version(study, role, parent)
    clause_id = f'{"S" if role == "scientist" else "Q"}.{int(code[1:]):02d}'
    packet = {'role': 'main-agent', 'effort': 'xhigh', 'optimized_role': role,
        'new_version': code, 'parent_version': parent, 'new_clause_id': clause_id,
        'current_skill': skill, 'current_workflow': workflow, 'failure_evidence': failure_packet(study, previous)}
    prompt = ('You are the main-agent for the approved CK2 retrospective Scientist–Selector experiment. Use xhigh. '
        'Review actual previous actions and structure evidence. Audit the previous new clause: was it used, and did the use reflect data? '
        f'Improve ONLY the {role} skill and workflow by one narrowly motivated mechanism. '
        f'Include exactly the new operative clause ID {clause_id} in skill_text and workflow_text, describe an observable effect. '
        'Write complete replacement skill with YAML name/description and complete workflow. Keep useful previous instructions, remove unsupported ones; '
        'do not accumulate generic rules. Do not hardcode donor/candidate/source IDs, exact geometry templates, final scores or oracle identities. '
        'The decision role may not see this oracle feedback. Instructions must be reusable reasoning actions on the allowed current/history evidence. '
        'Scientist prose remains qualitative and interval-specific; Selector must directly inspect and rank available seeds, not program a surrogate. '
        'No invented chemical mechanisms or reconstructed final labels. Hypothesis discrimination first; only then Selector decision logic. '
        'Return JSON only with diagnosis, mechanism, skill_text, workflow_text, clause_id, observable_effect, previous_clause_audit.')
    job = round_dir / 'main_revision'
    answer = run_job(job, 'main-agent', prompt, packet, REVISION_SCHEMA, model=MAIN_MODEL, effort='xhigh')
    if answer['clause_id'] != clause_id or clause_id not in answer['skill_text'] or clause_id not in answer['workflow_text']:
        raise ValueError('Main revision failed prompt-visible clause binding')
    text = answer['skill_text'] + answer['workflow_text']
    if re.search(r'(?:single_b\d+|(?<![A-Za-z0-9])C[0-9a-f]{8,}(?![A-Za-z0-9])|(?<![A-Za-z0-9])T\d{1,3}(?![A-Za-z0-9])|batch[_ ]?\d+)', text, re.I):
        raise ValueError('Revision leaked a specific lineage identity')
    dest.mkdir(parents=True, exist_ok=True)
    (dest / 'SKILL.md').write_text(answer['skill_text'] + '\n', encoding='utf-8')
    (dest / 'workflow.md').write_text(answer['workflow_text'] + '\n', encoding='utf-8')
    write(dest / 'diagnosis.json', {k: answer[k] for k in ['diagnosis', 'mechanism', 'observable_effect', 'previous_clause_audit']})
    write(dest / 'version.json', {'version': code, 'role': role, 'parent': parent,
        'active_clause': clause_id, 'optimized_from_round': previous.name, 'controller_model': MAIN_MODEL,
        'controller_effort': 'xhigh', 'model': MODEL, 'effort': 'medium',
        'skill_sha256': digest(dest / 'SKILL.md'), 'workflow_sha256': digest(dest / 'workflow.md'),
        'revision_response_sha256': digest(job / 'answer.json')})


def commit_versions(repo, study, label):
    scope = str((study / 'versions').relative_to(repo)).replace('\\', '/')
    subprocess.run(['git', 'add', '--', scope], cwd=repo, check=True, capture_output=True)
    diff = subprocess.run(['git', 'diff', '--cached', '--quiet', '--', scope], cwd=repo)
    if diff.returncode == 1:
        result = subprocess.run(['git', 'commit', '--only', '-m', f'CK2 retrieval {label}: immutable role instruction versions', '--', scope],
                                cwd=repo, text=True, capture_output=True, check=True)
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()


def seal_metrics(study, folder, meta):
    from .evaluate import evaluate_round
    m = evaluate_round(study, folder)
    m['macro'] = macro_metrics(m)
    m['version'] = meta
    write(folder / 'metrics.json', m)
    return m


def semantic_audit(study, folder, meta, enforce=True):
    """The xhigh main role checks real observations, not mere clause echoes."""
    from .runner import run_job
    recovery = read(folder / 'quality_retry.json') if (folder / 'quality_retry.json').exists() else {}
    target = folder / 'main_semantic_audit'
    if recovery:
        target = target / f"T{recovery['trial'] + 1:02d}"
    if (target / 'answer.json').exists():
        answer = read(target / 'answer.json')
    else:
        nodes = []
        for step in STEPS:
            job = folder / 'selectors' / f't{step:03d}'
            answer = read(job / 'answer.json')
            mapping = read(job / 'mapping.json')['anonymous_to_key']
            cards = {c['key']: c for c in read(study / 'evidence' / 'pools' / f't{step:03d}.json')['cards']}
            claims = []
            for row in answer['ranking']:
                card = cards[mapping[row['candidate_id']]]
                states = {v['step']: v for v in card['history']}
                history_keys = sorted({e['feature'] for e in row['observations']})
                claims.append({'rank': row['rank'], 'rationale': row['evidence'], 'counterevidence': row['counterevidence'],
                    'observations': [{'claim': e, 'stored_value': states[e['step']][e['feature']]} for e in row['observations']],
                    'current_state': card['now'], 'history_columns': ['step', *history_keys],
                    'history_rows': [[s['step'], *[s.get(k) for k in history_keys]] for s in card['history']]})
            nodes.append({'step': step, 'choices': claims, 'applied_clauses': answer['applied_clauses']})
        scientist_source = folder
        if meta.get('scientist_outputs_frozen_from'):
            scientist_source = study / 'rounds' / meta['scientist_outputs_frozen_from']
        scientist = read(scientist_source / 'scientist' / 'answer.json')
        teacher = {d['donor_id']: {r['step']: r for r in d['observations']}
                   for d in read(study / 'evidence' / 'teacher.json')['donors']}
        descriptions = []
        for row in scientist['interval_descriptions']:
            keys = sorted({e['feature'] for e in row['evidence']} | {
                f'{prefix}_{suffix}' for prefix in ('current', 'forecast') for suffix in (
                    'radius_gyration_A', 'centroid_x_A', 'centroid_y_A', 'centroid_z_A',
                    'nearest_pocket_A', 'pocket_contacts_4A', 'contact_pairs_4A',
                    'planarity_fraction', 'residue_contact_footprint')})
            low, high = 5 * (row['interval'] - 1), 5 * row['interval']
            descriptions.append({'interval': row['interval'], 'text': row['text'],
                'facts': [{'claim': e, 'stored_value': teacher[row['donor_id']][e['step']][e['feature']]}
                          for e in row['evidence']], 'window_columns': ['step', *keys],
                'window_rows': [[s, *[teacher[row['donor_id']][s].get(k) for k in keys]] for s in range(low, high + 1)]})
        scmeta, scskill, scworkflow = version(study, 'scientist', meta['scientist_version'])
        qmeta, qskill, qworkflow = version(study, 'selector', meta['selector_version'])
        packet = {'role': 'main-agent', 'scientist_skill': scskill, 'scientist_workflow': scworkflow,
            'scientist_active_clause': scmeta['active_clause'], 'scientist_descriptions': descriptions,
            'scientist_clause_evidence': scientist['applied_clauses'], 'hypotheses': read(folder / 'hypotheses.json'),
            'same_time_population_quantiles': read(study / 'runtime' / 'population_summaries.json')[:51],
            'selector_skill': qskill, 'selector_workflow': qworkflow,
            'selector_active_clause': qmeta['active_clause'], 'selector_nodes': nodes,
            'evaluation': read(folder / 'metrics.json')['macro']}
        prompt = ('You are the xhigh main-agent performing a functional and scientific evidence audit. '
            'Check the submitted natural-language claims against actual stored observations. '
            'Check specifically that the newest active clauses affected observable comparisons, evidence use or conclusions; '
            'matching an ID or repeating an instruction is insufficient. A conditional clause may be satisfied by a grounded '
            'assessment that the condition is absent; do not force an untrue pattern. '
            'Current-cloud squeezing or raw bond labels do not establish formed molecules, energies, H-bonds or causal affinity effects. '
            'Principal-axis signs are arbitrary; planarity_fraction lower means thinner, and geometric contacts are distance proxies. '
            'Do not penalize an unsuccessful retrieval solely because oracle coverage is low. Review evidence truth and action, '
            'not hidden chain of thought. Return ten selector_assessments plus a Scientist assessment, actual unsupported claims '
            'and concrete failure analysis. passes=true only when the operative new instructions have observable grounded application '
            'and no material false claims require correction. Return schema JSON only; no external tools.')
        answer = run_job(target, 'main-agent', prompt, packet, AUDIT_SCHEMA, model=MAIN_MODEL, effort='xhigh')
    write(folder / 'semantic_audit.json', answer)
    if enforce and not audit_passes(answer):
        raise ValueError('xhigh main-agent semantic/application audit requires a workflow correction; artifacts retained')
    return answer


def audit_passes(answer):
    assessments = answer.get('selector_assessments', [])
    return bool(answer.get('passes') and len(assessments) == 10
        and {r['step'] for r in assessments} == set(STEPS)
        and all(r['evidence_truth'] and r['clause_applied'] for r in assessments)
        and answer.get('scientist_evidence_truth', True) and answer.get('scientist_clause_applied', True))


def audit_and_repair(study, folder, meta, workers):
    """Retry factual failures without showing oracle feedback to either worker."""
    from .repairs import archive_round_trial, semantic_repair_targets
    for _ in range(4):
        audit = semantic_audit(study, folder, meta, enforce=False)
        if audit_passes(audit):
            return audit
        targets = semantic_repair_targets(audit)
        if not targets['scientist'] and not targets['selector_steps']:
            # Legacy prose-only audits may not localize their Scientist failure.
            targets['scientist'] = True
        if targets['scientist'] and meta.get('scientist_outputs_frozen_from'):
            raise ValueError('Frozen Scientist evidence conflicts with its previous passing audit; preserve it for main-agent review')
        repeat_scientist = targets['scientist']
        failed_steps = set(targets['selector_steps'])
        previous = read(folder / 'quality_retry.json') if (folder / 'quality_retry.json').exists() else {}
        trial = int(previous.get('trial', 0)) + 1
        if trial > 3:
            raise ValueError('Three factual correction trials failed the xhigh evidence audit; main-agent review required')
        previous_targets = previous.get('selector_steps', [])
        targets['selector_steps'] = sorted(set(targets['selector_steps']) | set(previous_targets))
        targets['scientist'] = targets['scientist'] or bool(previous.get('scientist'))
        archived = archive_round_trial(study, folder, 'xhigh factual evidence/application audit rejected this trial')
        recovery = {**targets, 'trial': trial, 'archived_trial': archived.name,
            'audit_sha256': digest(archived / 'semantic_audit.json'),
            'kind': 'factual correction; nominal optimization policy and information boundary unchanged'}
        write(folder / 'quality_retry.json', recovery)
        write(folder / '_version.json', meta)
        for role, code in [('scientist', meta['scientist_version']), ('selector', meta['selector_version'])]:
            applicable = targets['scientist'] if role == 'scientist' else bool(targets['selector_steps'])
            if not applicable:
                continue
            _, skill, workflow = version(study, role, code)
            record = study / 'versions' / role / code / 'repairs' / f'{folder.name}_T{trial:02d}'
            record.mkdir(parents=True, exist_ok=True)
            (record / 'SKILL.md').write_text(skill, encoding='utf-8')
            (record / 'workflow.md').write_text(workflow + '\n' + QUALITY_GUIDANCE[role], encoding='utf-8')
            write(record / 'repair.json', {**recovery, 'policy_version': code,
                'correction_guidance_sha256': object_sha(QUALITY_GUIDANCE[role])})
        commit_versions(Path(__file__).resolve().parents[3], study, f'{folder.name} quality T{trial:02d}')
        # Restore unaffected accepted records. All rejected originals remain in
        # the archive; all model attempts and opaque mappings remain in place.
        if not repeat_scientist:
            for p in (archived / 'scientist').glob('*'):
                if p.is_file():
                    shutil.copyfile(p, folder / 'scientist' / p.name)
            shutil.copyfile(archived / 'hypotheses.json', folder / 'hypotheses.json')
            for step in STEPS:
                if step not in failed_steps:
                    for p in (archived / 'selectors' / f't{step:03d}').glob('*'):
                        if p.is_file():
                            shutil.copyfile(p, folder / 'selectors' / f't{step:03d}' / p.name)
        else:
            run_scientist(study, folder, meta['scientist_version'])
        run_selectors(study, folder, meta['selector_version'], read(folder / 'hypotheses.json'), workers)
        seal_metrics(study, folder, meta)
        print(json.dumps({'round': folder.name, 'quality_correction_trial': trial,
            'scientist_repeated': repeat_scientist, 'selector_steps': sorted(failed_steps)}, ensure_ascii=False), flush=True)
    raise RuntimeError('Factual correction loop ended without an accepted audit')


def run(study, repo, workspace, through=50, workers=3):
    from .data import prepare
    from .evaluate import evaluate_baselines
    from .report import report
    from .runner import preflight
    study.mkdir(parents=True, exist_ok=True)
    status = read(study / 'run_status.json') if (study / 'run_status.json').exists() else {}
    def state(**updates):
        status.update(updates, updated_unix=time.time(), main_model=MAIN_MODEL, main_effort='xhigh',
                      subagent_model=MODEL, subagent_effort='medium', complete=False)
        write(study / 'run_status.json', status)
        print(json.dumps({k: status.get(k) for k in ['status', 'phase', 'round', 'best_round', 'error']}, ensure_ascii=False), flush=True)
    try:
        state(status='running', phase='prepare', error=None)
        prepare(study, workspace)
        init_versions(study, repo)
        state(phase='isolation_preflight')
        preflight(study / 'diagnostics' / 'preflight' / f'P{time.time_ns()}')
        if not (study / 'baselines.json').exists():
            evaluate_baselines(study)
        best_science = 'R00'; best_total = 'R00'
        for number in range(through + 1):
            folder = study / 'rounds' / f'R{number:02d}'
            folder.mkdir(parents=True, exist_ok=True)
            state(phase='scientist' if number <= 25 else 'selector', round=number, best_round=best_total)
            if number == 0:
                scode, qcode = 'S00', 'Q00'
            elif number <= 25:
                incumbent = read(study / 'rounds' / best_science / '_version.json')
                scode, qcode = f'S{number:02d}', 'Q00'
                revise(study, folder, 'scientist', scode, incumbent['scientist_version'], study / 'rounds' / f'R{number-1:02d}')
            else:
                frozen = read(study / 'rounds' / best_science / '_version.json')
                incumbent = read(study / 'rounds' / best_total / '_version.json')
                scode, qcode = frozen['scientist_version'], f'Q{number-25:02d}'
                revise(study, folder, 'selector', qcode, incumbent['selector_version'], study / 'rounds' / f'R{number-1:02d}')
            meta = {'round': number, 'scientist_version': scode, 'selector_version': qcode,
                    'scientist_outputs_frozen_from': best_science if number > 25 else None,
                    'main_effort': 'xhigh', 'subagent_effort': 'medium'}
            write(folder / '_version.json', meta)
            if not (folder / 'metrics.json').exists() or not read(folder / 'metrics.json').get('complete'):
                if number <= 25:
                    run_scientist(study, folder, scode)
                else:
                    source = study / 'rounds' / best_science
                    shutil.copyfile(source / 'hypotheses.json', folder / 'hypotheses.json')
                    write(folder / 'scientist_frozen.json', {'source_round': best_science,
                        'source_hypotheses_sha256': digest(source / 'hypotheses.json'), 'scientist_version': scode})
                run_selectors(study, folder, qcode, read(folder / 'hypotheses.json'), workers)
                seal_metrics(study, folder, meta)
            audit_and_repair(study, folder, meta, workers)
            m = read(folder / 'metrics.json')
            if number <= 25 and score_tuple(m) > score_tuple(read(study / 'rounds' / best_science / 'metrics.json')):
                best_science = folder.name
            if score_tuple(m) > score_tuple(read(study / 'rounds' / best_total / 'metrics.json')):
                best_total = folder.name
            git_sha = commit_versions(repo, study, folder.name)
            write(folder / 'git_version_receipt.json', {'commit': git_sha, 'scientist': scode, 'selector': qcode})
            state(phase='round_complete', best_round=best_total, best_scientist_round=best_science,
                  completed_rounds=number + 1, last_macro=m['macro'])
            report(study)
        if through < 50:
            state(status='partial', phase='requested_round_boundary')
            report(study)
            return
        state(phase='structure_only_controls')
        for label in dict.fromkeys(['R00', best_science, best_total]):
            source = study / 'rounds' / label
            meta = read(source / '_version.json')
            control = study / 'controls' / 'structure_only' / label
            control.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / 'hypotheses.json', control / 'hypotheses.json')
            run_selectors(study, control, meta['selector_version'], read(control / 'hypotheses.json'), workers, True)
            seal_metrics(study, control, meta)
        state(phase='independent_confirmations')
        # The original sealed selection is repeat1. Two fresh contexts and ID
        # permutations give three runs for baseline and best, in both conditions.
        for label in dict.fromkeys(['R00', best_total]):
            source = study / 'rounds' / label
            meta = read(source / '_version.json')
            for structural in (False, True):
                for repeat in (2, 3):
                    condition = 'structure_only' if structural else 'affinity_visible'
                    confirm = study / 'confirmations' / label / condition / f'repeat{repeat}'
                    confirm.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source / 'hypotheses.json', confirm / 'hypotheses.json')
                    run_selectors(study, confirm, meta['selector_version'], read(confirm / 'hypotheses.json'), workers, structural)
                    seal_metrics(study, confirm, meta)
        for role, code in [('scientist', read(study / 'rounds' / best_total / '_version.json')['scientist_version']),
                           ('selector', read(study / 'rounds' / best_total / '_version.json')['selector_version'])]:
            live = repo / 'skills' / f'ck2-path-{role}'
            source = study / 'versions' / role / code
            shutil.copyfile(source / 'SKILL.md', live / 'SKILL.md')
            shutil.copyfile(source / 'workflow.md', live / 'workflow.md')
        status.update(status='complete', phase='complete', complete=True, best_round=best_total,
                      best_scientist_round=best_science, completed_rounds=51, updated_unix=time.time())
        write(study / 'run_status.json', status)
        report(study)
        print(json.dumps({'status': 'complete', 'best_round': best_total}, ensure_ascii=False), flush=True)
    except BaseException as exc:
        state(status='failed', error=f'{type(exc).__name__}: {exc}')
        try:
            report(study)
        except Exception:
            pass
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', default=DEFAULT_STUDY)
    parser.add_argument('--through-round', type=int, default=50)
    parser.add_argument('--workers', type=int, default=3, choices=[1, 2, 3])
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    study = Path(args.study)
    study = study if study.is_absolute() else repo / study
    if args.prepare_only:
        from .data import prepare
        prepare(study, repo)
        init_versions(study, repo)
    else:
        run(study, repo, repo, args.through_round, args.workers)


if __name__ == '__main__':
    main()
