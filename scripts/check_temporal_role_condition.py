"""Read-only completion audit for a temporal condition; no API or generation."""
import argparse
import hashlib
from pathlib import Path

from evomolsteer.io import digest, read_json, write_json
from evomolsteer.temporal_guidance.evidence import analyst_finish, designer_finish, temporal_sources
from evomolsteer.temporal_guidance.evidence_protocol import PROTOCOL, TemporalEvidenceProtocol
from evomolsteer.temporal_guidance.policy import program_signature
from evomolsteer.temporal_guidance.workflow import TemporalEvidenceTools, compile_recipe, sources


def audit(repo, config, condition):
    config = read_json(repo / config)
    plan = read_json(repo / config['plan_config'])
    root = repo / plan['output']
    state = read_json(root / 'run_status.json')
    task = state['jobs']['roles_' + condition]
    if task['status'] != 'complete':
        raise ValueError('Condition has not completed all its role chains and measurement')
    selection = read_json(root / 'selections' / (condition + '.json'))
    if len(selection['rows']) != plan['chains']:
        raise ValueError('Incomplete independent role chain coverage')
    rows, files = [], {}
    for chain in selection['rows']:
        folder = root / 'calls' / condition / f"chain_{chain['chain']:02d}"
        tools = TemporalEvidenceTools(repo / plan['evidence'],
                                      repo / plan['stage_profiles'] if plan.get('stage_profiles') else None)
        provenance = read_json(folder / 'Analyst/references.json')
        tools.import_references(provenance)
        wire = read_json(folder / 'Analyst/wire_result.json')
        analyst_guard = TemporalEvidenceProtocol('Analyst').validate if plan.get('evidence_protocol') == PROTOCOL else analyst_finish
        if analyst_guard(wire, tools):
            raise ValueError('Analyst scientific delivery audit failed')
        source_map = {f'H{i+1:03d}': sorted(temporal_sources(tools, h['evidence_refs']))
                      for i, h in enumerate(wire['hypotheses'])}
        tools.import_references(read_json(folder / 'Designer/references.json'))
        decision = read_json(folder / 'Designer/wire_result.json')
        designer_guard = TemporalEvidenceProtocol('Designer', source_map).validate if plan.get('evidence_protocol') == PROTOCOL else \
                         lambda r, t: designer_finish(r, t, source_map)
        if designer_guard(decision, tools):
            raise ValueError('Designer scientific delivery audit failed')
        tools.verify_sources()
        api_calls, feedback = 0, 0
        for role in ['Analyst', 'Designer']:
            path = folder / role
            binding = read_json(path / 'binding.json')
            completed = read_json(path / 'completed.json')
            if completed['backend'] != 'real_api' or digest(path / 'binding.json') != completed['binding_sha256']:
                raise ValueError('Unbound real API completion')
            if hashlib.sha256((path / 'instructions.md').read_text(encoding='utf-8').encode()).hexdigest() != binding['instruction_sha256']:
                raise ValueError('Changed delivered instructions')
            for name, sha in read_json(folder / (role + '.skills.json')).items():
                if digest(repo / name) != sha:
                    raise ValueError('Changed Skill source: ' + name)
            for index in range(completed['steps']):
                receipt = read_json(path / f'step_{index:02d}.receipt.json')
                if receipt['returned_model'] != 'deepseek-flash' or receipt['backend'] != 'real_api':
                    raise ValueError('Wrong API model/backend')
                for target, sha in [(path / f'step_{index:02d}.json', receipt['response_sha256']),
                                    (path / f'step_{index:02d}.request.json', receipt['request_sha256']),
                                    (path / receipt['raw_response_path'], receipt['raw_response_sha256'])]:
                    if digest(target) != sha:
                        raise ValueError('Changed API artifact: ' + str(target))
                api_calls += 1
            feedback += len(list(path.glob('*.finish_rejected.json')))
            files[(path / 'completed.json').relative_to(repo).as_posix()] = digest(path / 'completed.json')
        program = read_json(repo / chain['program'])
        expected = sources(repo, plan)['base_program'] if chain['recipe'] == 'retain_M2' else compile_recipe(repo, plan, chain['recipe'])[0]
        if program_signature(program) != chain['signature'] or program_signature(expected) != chain['signature']:
            raise ValueError('Compiled reward differs from the recorded choice')
        rows.append({'chain': chain['chain'], 'recipe': chain['recipe'], 'API_action_calls': api_calls,
                     'rejected_then_repaired_finishes': feedback, 'hypothesis_temporal_sources': source_map,
                     'designer_curves_cited': sorted(temporal_sources(tools, decision['evidence_refs'])),
                     'actual_curve_sources_read': sorted(temporal_sources(tools, provenance['references'])),
                     'compiled_signature': chain['signature']})
    collection = read_json(repo / config['reuse_audit'])
    return {'condition': condition, 'status': 'complete', 'chains': rows, 'selected': selection['selected'],
            'API_action_calls': sum(r['API_action_calls'] for r in rows),
            'rejected_then_repaired_finishes': sum(r['rejected_then_repaired_finishes'] for r in rows),
            'measurement': task['result']['measurement'], 'frozen_control_summary': collection['controls'],
            'new_FLOWR_runs_for_same_signature': 0, 'remaining_queue_background': True, 'source_sha256': files,
            'scope': 'Source delivery, compilation and matched existing result reuse; not proof of improved affinity'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--condition', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    repo = Path(__file__).resolve().parents[1]
    result = audit(repo, args.config, args.condition)
    write_json(repo / args.output, result)
    print({k: result[k] for k in ['condition', 'status', 'API_action_calls', 'rejected_then_repaired_finishes']})
