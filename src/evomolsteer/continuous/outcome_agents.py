"""Identical hash-bound API/subagent transport for decoded-outcome trials."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import httpx

from ..io import digest, read_json, write_json

ROOT = Path(__file__).resolve().parents[3]
VERSION = 'terminal-outcome-agent-1.0'
UPDATES = {'native_rms_ratio': (.05, .8), 'teacher_score_beta': (0., 6.),
    'teacher_neighbors': (1, 28), 'teacher_endpoint_temperature_A2': (.5, 12.),
    'time_ramp_power': (0., 2.), 'branch_mixture.virtual_mass': (0., .1),
    'branch_mixture.region_weight_mix': (0., 1.)}


def outcome_capabilities():
    from ..generation.formula_registry import formula_registry
    caps = formula_registry(ROOT)
    caps['families'] = {'endpoint_pointcloud': caps['families']['endpoint_pointcloud'],
        'endpoint_branch_mixture': {
            'formula': 'Original endpoint-pointcloud modes retain mass 1-alpha; eligible natural branch modes add mass alpha at Y_m+min(raw_branch_RMS_A,0.2)*direction_unit. Regional residual weight=1+region_mix*confidence*(recorded_atom_weight-1). Same robust curvature and logsumexp.',
            'support': 'Actual distinct immediate parents with a common grandparent; missing directions are exact baseline, no chemical graph restriction',
            'source': 'branch_mixture_reward.py'}}
    caps['source_sha256']['branch_mixture_reward.py'] = digest(ROOT/'src/evomolsteer/generation/branch_mixture_reward.py')
    return caps


def run_tool(plan_path, receipt):
    plan = read_json(plan_path)
    specs = {'terminal_outcome': ('mine_terminal_outcomes.py', 'terminal_outcome.py'),
             'outcome_summary': ('summarize_outcome_labels.py', 'outcome_summary.py'),
             'outcome_feedback': ('summarize_outcome_feedback.py', 'outcome_feedback.py'),
             'outcome_alias_credit': ('pool_outcome_alias_credit.py', 'outcome_alias_credit.py'),
             'outcome_matched_reference': ('build_outcome_matched_reference.py', 'outcome_matched_reference.py'),
             'outcome_conditioned_geometry': ('analyze_outcome_conditioned_geometry.py', 'outcome_conditioned_geometry.py')}
    if plan.get('tool_id') not in specs or set(plan) != {'tool_id', 'arguments', 'input_files', 'output_files'}:
        raise ValueError('Registered outcome calculation plan required')
    flags = {'--dataset', '--campaign', '--metrics', '--baseline', '--output', '--score-start', '--score-end',
             '--mode', '--budget', '--threshold', '--tail-weight', '--branch-mode', '--score-tolerance', '--shrinkage'}
    if plan['tool_id'] == 'outcome_summary':
        flags = {'--labels', '--evidence', '--output'}
    elif plan['tool_id'] == 'outcome_feedback':
        flags = {'--evidence', '--reports', '--output'}
    elif plan['tool_id'] == 'outcome_conditioned_geometry':
        flags = {'--dataset', '--campaign', '--labels', '--evidence', '--output', '--score-tolerance'}
    elif plan['tool_id'] == 'outcome_alias_credit':
        flags = {'--dataset', '--campaign', '--labels', '--metrics', '--evidence', '--output', '--tail-weight'}
    elif plan['tool_id'] == 'outcome_matched_reference':
        flags = {'--dataset', '--campaign', '--labels', '--metrics', '--evidence', '--output', '--score-tolerance'}
    args = plan['arguments']
    if len(args) % 2 or set(args[::2])-flags or len(set(args[::2])) != len(args[::2]):
        raise ValueError('Literal registered argument pairs required')
    files = {str(Path(v).resolve()): digest(v) for v in plan['input_files']}
    if any(Path(v).exists() for v in plan['output_files']):
        raise FileExistsError('Immutable tool output already exists')
    script, module = specs[plan['tool_id']]
    command = [sys.executable, str(ROOT/'scripts'/script), *args]
    p = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding='utf-8', errors='replace')
    outputs = {str(Path(v).resolve()): digest(v) for v in plan['output_files'] if Path(v).is_file()}
    result = {'schema_version': VERSION, 'tool_id': plan['tool_id'], 'plan_sha256': digest(plan_path),
        'command': command, 'returncode': p.returncode, 'input_files': files, 'output_files': outputs,
        'inputs_unchanged': all(digest(v) == sha for v, sha in files.items()),
        'source_sha256': digest(ROOT/'src/evomolsteer/continuous'/module),
        'stdout_tail': p.stdout[-1000:], 'stderr_tail': p.stderr[-3000:]}
    write_json(receipt, result)
    if p.returncode or not result['inputs_unchanged'] or len(outputs) != len(plan['output_files']):
        raise ValueError('Outcome tool failed; inspect retained receipt')
    return result


def export_request(role, evidence, registry, receipt, output, analyst=None):
    if role not in ('Analyst', 'Designer'):
        raise ValueError('Registered role required')
    packet, reg, r = read_json(evidence), read_json(registry), read_json(receipt)
    if r['returncode'] or not r['inputs_unchanged'] or str(Path(evidence).resolve()) not in r['output_files']:
        raise ValueError('Executed calculation receipt required')
    verify_tool_artifacts(r)
    instruction_files = [ROOT/f'skills/{role.lower()}/SKILL.md', ROOT/'skills/terminal-outcome/SKILL.md']
    text = '\n\n'.join(p.read_text(encoding='utf-8').strip() for p in instruction_files)
    text += '\n\nTASK:\n'+reg['task']+'\n'
    text += '\nREGISTERED FORMULAS:\n'+json.dumps(outcome_capabilities(), ensure_ascii=False, sort_keys=True)+'\n'
    request = {'schema_version': VERSION, 'role': role, 'window': packet['window'],
        'score_window': packet['score_window'], 'label_clock': 1., 'system_instruction': text,
        'instruction_sha256': hashlib.sha256(text.encode()).hexdigest(), 'evidence_sha256': digest(evidence),
        'bindings': {'evidence': str(Path(evidence).resolve()), 'registry': str(Path(registry).resolve()),
            'registry_sha256': digest(registry), 'receipt': str(Path(receipt).resolve()), 'receipt_sha256': digest(receipt),
            'skills': {str(p.resolve()): digest(p) for p in instruction_files}},
        'response_contract': {'common_required': ['schema_version', 'agent', 'request_sha256', 'instruction_sha256',
            'evidence_sha256', 'tool_receipt_sha256', 'score_window', 'window', 'evidence_ids', 'rationale', 'counterevidence'],
            'Analyst_required': ['label_source', 'findings', 'extensions_ready'],
            'Designer_required': ['analyst_response_sha256', 'base_program_id', 'updates'],
            'label_source': 'decoded_final', 'parameter_limits': UPDATES,
            'one_change_per_trial': True}}
    if role == 'Designer':
        if not analyst:
            raise ValueError('Actual Analyst response required')
        validate_response(Path(analyst).parent/'Analyst.request.json', analyst)
        request['bindings'].update(analyst=str(Path(analyst).resolve()), analyst_response_sha256=digest(analyst))
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    path = out/(role+'.request.json')
    if path.exists():
        raise FileExistsError(path)
    write_json(path, request)
    (out/(role+'.instructions.md')).write_text(text, encoding='utf-8')
    return path


def verify_tool_artifacts(receipt):
    """A receipt is evidence only while its immutable inputs and outputs match."""
    if receipt.get('returncode') != 0 or receipt.get('inputs_unchanged') is not True:
        raise ValueError('Successful immutable calculation required')
    for key in ('input_files', 'output_files'):
        if not receipt.get(key):
            raise ValueError('Calculation artifact inventory missing')
        for path, sha in receipt[key].items():
            if not Path(path).is_file() or digest(path) != sha:
                raise ValueError('Executed calculation artifact changed: '+path)


def validate_response(request_path, response_path):
    req, r = read_json(request_path), read_json(response_path)
    b = req['bindings']
    for key in ('evidence', 'registry', 'receipt'):
        expected = req['evidence_sha256'] if key == 'evidence' else b[key+'_sha256']
        if digest(b[key]) != expected:
            raise ValueError('Bound calculation artifact changed')
    if any(digest(p) != sha for p, sha in b['skills'].items()):
        raise ValueError('Bound literal Skill changed')
    verify_tool_artifacts(read_json(b['receipt']))
    required = req['response_contract']['common_required']+req['response_contract'][req['role']+'_required']
    if set(required)-set(r):
        raise ValueError('Required role response fields missing')
    expected = {'schema_version': VERSION, 'agent': req['role'], 'request_sha256': digest(request_path),
        'instruction_sha256': req['instruction_sha256'], 'evidence_sha256': req['evidence_sha256'],
        'tool_receipt_sha256': b['receipt_sha256'], 'score_window': req['score_window'], 'window': req['window']}
    if any(r.get(k) != v for k, v in expected.items()):
        raise ValueError('Response does not bind actual instructions/calculation/window')
    packet = read_json(b['evidence'])
    ids = {e['id'] for e in packet['evidence_items']}
    if not r.get('evidence_ids') or set(r['evidence_ids'])-ids or not r.get('rationale') or not r.get('counterevidence'):
        raise ValueError('Bound evidence and counterevidence required')
    if req['role'] == 'Analyst':
        if r.get('label_source') != 'decoded_final' or not r.get('findings') or not isinstance(r.get('extensions_ready'), bool):
            raise ValueError('Final-label response required')
        if packet['mode'] == 'instantaneous':
            raise ValueError('An intermediate control cannot masquerade as decoded-outcome learning')
    else:
        if r.get('analyst_response_sha256') != b['analyst_response_sha256'] or digest(b['analyst']) != b['analyst_response_sha256']:
            raise ValueError('Designer is not bound to actual Analyst')
        updates = r.get('updates', {})
        allowed = read_json(b['registry']).get('allowed_updates', list(UPDATES))
        if set(updates)-set(allowed):
            raise ValueError('Update is outside this serial module trial')
        if len(updates) > 1 or set(updates)-set(UPDATES):
            raise ValueError('At most one registered parameter axis per trial')
        for key, value in updates.items():
            lo, hi = UPDATES[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not lo <= value <= hi:
                raise ValueError('Parameter outside finite bounded registry')
            if key == 'teacher_neighbors' and int(value) != value:
                raise ValueError('Teacher neighbor count must be integral')
    return r


def compile_program(request, response, output, number):
    r = validate_response(request, response)
    req = read_json(request)
    registry = read_json(req['bindings']['registry'])
    if r['agent'] != 'Designer' or r['base_program_id'] not in registry['base_programs'] or not 1 <= number <= 15:
        raise ValueError('Registered base and fifteen-round budget required')
    base = registry['base_programs'][r['base_program_id']]
    p = read_json(base['path'])
    if digest(base['path']) != base['sha256']:
        raise ValueError('Base checkpoint changed')
    provenance = {key: p[key] for key in ('program_id', 'agent_request_sha256', 'skill_sha256',
        'endpoint_designer_response_sha256', 'derivation') if key in p}
    for key in ['flowcompat_binding', 'agent_binding', 'flowcompat_supplemental_binding',
                'skill_sha256', 'skill_behavior_audit_sha256', 'endpoint_designer_response_sha256',
                'geometry_review_sha256', 'endpoint_compiled_baseline_sha256', 'derivation', 'control_origin']:
        p.pop(key, None)
    for dotted, value in r['updates'].items():
        target = p
        keys = dotted.split('.')
        for key in keys[:-1]:
            target = target.setdefault(key, {})
        target[keys[-1]] = value
    evidence = read_json(req['bindings']['evidence'])
    ref = registry['reference']
    if digest(ref['path']) != ref['sha256'] or ref['sha256'] != evidence['reference_sha256']:
        raise ValueError('Designer reference is not the executed final-outcome library')
    p.update(program_id=f'terminal_outcome15_round{number:02d}', round=number, seed=42,
        window=evidence['window'], score_window=evidence['score_window'], reference_sha256=ref['sha256'],
        target_definition='decoded_final_observed_ancestor_distribution', agent_request_sha256=digest(request),
        derivation={'source': 'Executed terminal-outcome tools and validated Luna Analyst/Designer responses',
                    'rationale': r['rationale'], 'counterevidence': r['counterevidence'],
                    'parent_provenance': provenance, 'no_private_reasoning_transcript': True},
        generation_interface='flowcompat_v2', outcome_binding={'schema_version': VERSION,
            'base_program_sha256': base['sha256'],
            'analyst_response_sha256': req['bindings']['analyst_response_sha256'], 'designer_response_sha256': digest(response),
            'request_sha256': digest(request), 'instruction_sha256': req['instruction_sha256'],
            'receipt_sha256': req['bindings']['receipt_sha256'], 'evidence_sha256': req['evidence_sha256'],
            'reference_sha256': ref['sha256'], 'label_source': 'decoded_final', 'updates': r['updates']})
    write_json(output, p)
    return p


def api_payload(request_path):
    """Expose the same literal programs and response hashes as file-aware agents."""
    req = read_json(request_path)
    b = req['bindings']
    verify_tool_artifacts(read_json(b['receipt']))
    reg = read_json(b['registry'])
    programs = {}
    for key, record in reg['base_programs'].items():
        if digest(record['path']) != record['sha256']:
            raise ValueError('API parent program changed')
        programs[key] = read_json(record['path'])
    payload = {'request_sha256': digest(request_path), 'instruction_sha256': req['instruction_sha256'],
        'evidence_sha256': req['evidence_sha256'], 'tool_receipt_sha256': b['receipt_sha256'],
        'window': req['window'], 'score_window': req['score_window'], 'response_contract': req['response_contract'],
        'evidence': read_json(b['evidence']), 'registry': reg, 'base_programs_actual': programs,
        'capabilities': outcome_capabilities()}
    if req['role'] == 'Designer':
        payload['analyst'] = read_json(b['analyst'])
        payload['analyst_response_sha256'] = b['analyst_response_sha256']
    return payload


def call_api(request_path, output):
    """Optional external API transport; current experiments use subagents."""
    req = read_json(request_path)
    payload = api_payload(request_path)
    with httpx.Client(timeout=180) as client:
        res = client.post(os.environ['EVOMOLSTEER_LLM_BASE_URL'].rstrip('/')+'/chat/completions',
            headers={'Authorization': 'Bearer '+os.environ['EVOMOLSTEER_LLM_API_KEY']},
            json={'model': os.environ['EVOMOLSTEER_LLM_MODEL'], 'messages': [
                {'role': 'system', 'content': req['system_instruction']},
                {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}],
                'response_format': {'type': 'json_object'}})
        res.raise_for_status()
        value = json.loads(res.json()['choices'][0]['message']['content'])
    write_json(output, value)
    validate_response(request_path, output)
