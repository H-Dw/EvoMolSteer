"""Versioned deployment of compact role Skills through the real API workflow.

This is a deployment verification, not another module-ablation treatment.
The historical study and its role outputs are immutable inputs.
"""
import copy
import shutil
from collections import Counter
from pathlib import Path

from evomolsteer.ablation.agents import operative_incumbent, validate_evidence_citations
from evomolsteer.ablation.compiler import compile_program
from evomolsteer.continuous.m2_skill_api import BASE, effective_signature
from evomolsteer.io import digest, read_json, write_json
from .ablation_adapter import registered_choices
from .engine import run_agent
from .sources import EvidenceTools


def prepare(repo, config):
    repo = Path(repo).resolve()
    source, study = repo / config['source_study'], repo / config['study']
    protocol = copy.deepcopy(read_json(source / 'activation.json')['protocol'])
    activation = {'schema_version': 'compact-role-verification-1.0',
        'source_study': config['source_study'], 'source_activation_sha256': digest(source / 'activation.json'),
        'protocol': protocol, 'screen_batches': config['batches'],
        'skills': {role: {'path': path, 'sha256': digest(repo / path)}
                   for role, path in config['skills'].items()}}
    for name in ['packet.json', 'index.json', 'continuous_functions.json',
                 'observable_definitions.json', 'regional_salience.json', 'temporal_reference.json.gz']:
        dest = study / 'evidence' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and digest(dest) != digest(source / 'evidence' / name):
            raise ValueError('Changed frozen deployment evidence: ' + name)
        if not dest.exists():
            shutil.copyfile(source / 'evidence' / name, dest)
    activation['evidence'] = {p.name: digest(p) for p in (study / 'evidence').iterdir()}
    if (study / 'activation.json').exists() and read_json(study / 'activation.json') != activation:
        raise ValueError('Changed compact verification activation')
    write_json(study / 'activation.json', activation)
    return activation


def run_roles(repo, config):
    repo = Path(repo).resolve()
    activation = prepare(repo, config)
    study, source = repo / config['study'], repo / config['source_study']
    base = read_json(repo / BASE)
    # Research outcomes do not enter role inputs. The factual execution contract
    # is identical to the completed ablation; only the research Skills change.
    contract = (repo / config['contract']).read_text(encoding='utf-8')
    interface = (repo / 'skills/agent-json-interface/SKILL.md').read_text(encoding='utf-8')
    rows = []
    for chain in range(config['chains']):
        folder = study / 'calls' / f'chain_{chain:02d}'
        analyst, references = None, None
        for role in ['Analyst', 'Designer']:
            skill = repo / config['skills'][role]
            if digest(skill) != activation['skills'][role]['sha256']:
                raise ValueError('Skill changed during real API verification')
            tools = EvidenceTools(study / 'evidence')
            if references is not None:
                tools.import_references(references)
            instructions = skill.read_text(encoding='utf-8') + '\n\n' + contract + '\n\n' + interface
            task = {'role': role, 'objective': tools.packet['objective'],
                'task_wording': ('Interpret the evidence for the coordinate-design task.' if chain < 3 else
                                 'Use the supplied observations to answer the coordinate reward task.'),
                'maximum_edits': 1, 'incumbent': operative_incumbent(base),
                'fixed_parameters': activation['protocol']['temperature_policy']['parameters'],
                'support': tools.packet['clocks'],
                'fixed_controller': {k: base[k] for k in ['native_rms_ratio', 'time_ramp_power']},
                'formula_registry': {
                    'pointcloud': 'R=tau*logsumexp(log_pi-rho_delta(q)/tau); q=detached-Hungarian endpoint MSD; rho_delta=delta^2*(sqrt(1+q/delta^2)-1); delta defaults to 1 A.',
                    'prior': 'Detached nearest-k static teachers, scores beta=0, supplied prior temperature; k is a count.',
                    'regional': 'Gaussian landmark attention with positive background, same endpoint scalar.',
                    'temporal_region': 'Supplied validated contrast changes attention, not an absolute coordinate target.',
                    'temporal_teacher': '(1-blend)*R_static + blend*R_observed_teacher(time); supplied observed-node coordinates only.',
                    'gradient': 'Real FLOWR endpoint VJP; no affinity derivative or extra production forward. Controller and temperatures fixed.',
                    'inactive': 'Legacy robust_delta, geometry_block_weights and motif metadata are not operative reward terms.'}}
            choices = registered_choices(activation['protocol']['lanes']['G']['allowed'], tools, analyst) if role == 'Designer' else None
            response = run_agent(role, tools, instructions, task, folder / role,
                choices=choices, handoff=analyst, max_steps=config['max_steps'],
                max_output_tokens=config['max_output_tokens'])
            validate_evidence_citations(response, set(tools.index['ids']) | set(tools.packet) | {'observable_definitions'})
            write_json(folder / (role + '.json'), response)
            (folder / (role + '.instructions.md')).write_text(instructions, encoding='utf-8')
            completion = read_json(folder / role / 'completed.json')
            if completion['backend'] != 'real_api':
                raise ValueError('Deployment requires actual API receipts')
            observations = [read_json(p) for p in sorted((folder / role).glob('step_*.observation.json'))]
            write_json(folder / (role + '.delivery.json'), {
                'skill_sha256': digest(skill), 'combined_instruction_sha256': digest(folder / (role + '.instructions.md')),
                'backend': completion['backend'], 'steps': completion['steps'],
                'successful_tool_observations': sum(o['ok'] for o in observations),
                'failed_tool_observations': sum(not o['ok'] for o in observations),
                'curve_sources_read': [v['source_id'] for v in tools.delivered.values() if v['kind'] == 'curve'],
                'findings': len(response.get('findings', [])), 'hypotheses': len(response.get('hypotheses', [])),
                'source_references_verified': True,
                'claim_scope': 'Delivery, extraction and citations verified; semantic scientific effectiveness is assessed separately.'})
            if role == 'Analyst':
                analyst, references = response, tools.provenance()
        path, reference, signature, receipt = compile_program(repo, study, response, base, folder=folder)
        rows.append({'chain': chain, 'decision': response['decision'], 'signature': signature,
            'program': path.relative_to(repo).as_posix(), 'reference': reference.relative_to(repo).as_posix(),
            'compiler_receipt': receipt})
    # Choose once, without looking at new generation quality or the old reward
    # ranking. Modal numerical program; ties prefer the supplied incumbent.
    counts = Counter(r['signature'] for r in rows)
    tied = sorted(k for k, n in counts.items() if n == max(counts.values()))
    base_signature = effective_signature(base)
    winner = base_signature if base_signature in tied else tied[0]
    chosen = next(r for r in rows if r['signature'] == winner)
    selection = {'schema_version': 'compact-selection-1.0', 'rows': rows, 'counts': dict(counts),
        'selected': chosen, 'selection_policy': 'modal numerical program, incumbent tie-break; no outcome selection',
        'equals_M2': winner == base_signature, 'fresh_role_chains': config['chains'],
        'note': '本次 Agent 输出未改变执行函数' if winner == base_signature else 'Agent selected a numerically changed program',
        'inference': 'One explicitly requested repeat candidate cohort; controls reused; identical RNG is not additional independent evidence.'}
    write_json(study / 'role_selection.json', selection)
    return selection
