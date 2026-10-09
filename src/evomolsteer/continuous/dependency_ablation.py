"""Information-bounded Agent packets and declarative reward compilation."""
import copy
import gzip
import json
from pathlib import Path
import numpy as np
from rdkit import Chem
from ..io import read_json, write_json, digest

BLOCKS = {
    'lineage': ('Analyst', 'When descendants are supplied, compare immediate selection with later survival and score credit. Quantify ancestral diversity and missingness before interpreting future-credit contrasts.'),
    'temporal': ('Analyst', 'Inspect empirical node contrasts alongside whole-window effects. Check sign changes, support and fitted-versus-observed disagreement before proposing a stage-dependent rule.'),
    'elite_geometry': ('Designer', 'Consider whether a multimodal coordinate objective preserves real conformations better than independently averaged feature targets. Keep alternatives distinct when averaging would create unsupported geometry.'),
    'history_matching': ('Designer', 'Use parent-child increments only when their coordinate meaning, support and uncertainty are supplied. An increment-matching term is a separate hypothesis; compare it with a design without history.')}

CONDITIONS = [('I0_full', [], 'full')] + [
    ('I' + str(i + 1) + '_without_' + key, [key], 'full') for i, key in enumerate(BLOCKS)] + [
    ('I5_without_all', list(BLOCKS), 'full'),
    ('D1_without_scores', list(BLOCKS), 'unweighted'),
    ('D2_without_trajectory', list(BLOCKS), 'static'),
    ('D3_without_steer', list(BLOCKS), 'structure'),
    ('D4_pocket_only', list(BLOCKS), 'pocket')]

def save_gzip(path, data):
    Path(path).write_bytes(gzip.compress(json.dumps(data, separators=(',', ':'), allow_nan=False).encode(), mtime=0))

def prepare(repo, input_path, config, docs):
    repo, config, docs = Path(repo), Path(config), Path(docs)
    if (docs / 'protocol.json').exists():
        raise FileExistsError('Frozen study already exists; use explicit new config/report outputs')
    config.mkdir(parents=True, exist_ok=True); docs.mkdir(parents=True, exist_ok=True)
    original = repo / 'configs/experiments/skill_ablation_v1/incumbent.json'
    refpath = repo / 'configs/experiments/skill_ablation_v1/endpoint_reference.json.gz'
    base = read_json(original)
    reference = json.loads(gzip.decompress(refpath.read_bytes()))
    references = {'full': refpath}
    for name in ['unweighted', 'static']:
        data = copy.deepcopy(reference)
        for frame in data['frames']:
            if name == 'static':
                for key in list(frame):
                    if key.startswith('teacher_'):
                        frame[key] = copy.deepcopy(data['frames'][-1][key])
            frame['teacher_scores'] = [0.0] * len(frame['teacher_endpoint_A'])
            for key in ['high_scaled', 'low_scaled', 'variance_scaled', 'score_gap_mean']:
                frame.pop(key, None)
        data['allowed_reward_views'] = ['endpoint_pointcloud']
        data['dependency_ablation'] = name
        data['label_semantics'] = 'Recorded affinities removed; equal score prior'
        path = config / (name + '.json.gz'); save_gzip(path, data); references[name] = path
    inp = Path(input_path)
    mol = Chem.RemoveHs(Chem.SDMolSupplier(str(inp / '3PE1_ligand_aligned.sdf'))[0])
    ligand = mol.GetConformer().GetPositions()
    points = []
    for line in (inp / '3PE1_protein_aligned.pdb').read_text().splitlines():
        if line.startswith(('ATOM  ', 'HETATM')) and line[76:78].strip() != 'H':
            point = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
            if np.linalg.norm(np.asarray(point) - ligand.mean(0)) <= 12:
                points.append(point)
    structural = {'schema_version': 'structure-field-reference-1.0', 'window': base['window'],
        'times': reference['times'], 'control_representation': 'proposal',
        'representation': 'FLOWR predicted endpoint world A', 'steer_sources': [],
        'required_input_sha256': reference['required_input_sha256'],
        'protein_points_A': points, 'bound_ligand_A': ligand.tolist(),
        'support_origin': 'Declared matched execution protocol, not observed selection support',
        'source_policy': 'Original receptor and bound ligand only; no generated Steer data'}
    for name in ['structure', 'pocket']:
        data = copy.deepcopy(structural)
        if name == 'pocket':
            data.pop('bound_ligand_A')
            data['source_policy'] = 'Receptor crop only; crop location inherits original FLOWR input'
        path = config / (name + '.json.gz'); save_gzip(path, data); references[name] = path
    old_evidence = read_json(repo / 'docs/experiments/skill_ablation_20261007/study_v1/evidence.json')
    # Deterministic, disclosed compression: strongest 8 and weakest 2 per table.
    entries = []
    for kind in ['endpoint_effects', 'increment_effects', 'regional_effects', 'survival_effects']:
        items = [x for x in old_evidence['evidence'] if x['kind'] == kind]
        ranked = sorted(items, key=lambda x: (x['data'].get('q', 1), x['evidence_id']))
        entries += ranked[:8] + ranked[-2:]
    support = [{'evidence_id': 'structure:geometry', 'kind': 'structure_geometry', 'data': {
        'protein_heavy_atoms_in_crop': len(points), 'known_ligand_heavy_atoms': len(ligand),
        'ligand_centroid_A': ligand.mean(0).tolist(),
        'ligand_covariance_eigenvalues_A2': np.linalg.eigvalsh(np.cov(ligand.T, bias=True)).tolist(),
        'origin': 'Original common receptor/bound ligand, not generated Steer',
        'no_affinity_labels': True}}]
    from .structure_spatial_evidence import spatial_evidence
    structural_support = support + spatial_evidence(inp / '3PE1_protein_aligned.pdb', ligand)
    out = []
    for identity, removed, level in CONDITIONS:
        folder = docs / identity; folder.mkdir(exist_ok=True)
        registry = {}
        allowed = ['full', 'unweighted', 'static', 'structure', 'pocket'] if level == 'full' else (
            ['unweighted', 'static', 'structure', 'pocket'] if level == 'unweighted' else
            ['static', 'structure', 'pocket'] if level == 'static' else
            ['structure', 'pocket'] if level == 'structure' else ['pocket'])
        for family in allowed:
            program = copy.deepcopy(base)
            # Historical provenance is not an execution parameter and must not
            # appear in teacher-free files or Agent packets.
            for key in list(program):
                if key.endswith('_sha256') and key != 'reference_sha256' or key == 'derivation':
                    program.pop(key)
            program.update(program_id=identity + '_' + family, reference_sha256=digest(references[family]))
            if family in ['unweighted', 'static']: program['teacher_score_beta'] = 0
            if family in ['structure', 'pocket']:
                for key in ['teacher_neighbors', 'teacher_score_beta', 'teacher_endpoint_temperature_A2',
                            'geometry_block_weights', 'target_definition', 'robust_delta']:
                    program.pop(key, None)
                program.update(reward_view='endpoint_structure_field',
                    ligand_anchor_weight=1.0 if family == 'structure' else 0.0,
                    contact_weight=0.0 if family == 'structure' else 1.0,
                    repulsion_weight=2.0, reference_origin='original_structure_only')
            p = config / (identity + '_' + family + '.json'); write_json(p, program)
            registry[family] = {'program': p.relative_to(repo).as_posix(),
                'reference': references[family].relative_to(repo).as_posix(),
                'historical_dependency': {'full': 'Stage-specific elite coordinates and affinity weights',
                    'unweighted': 'Stage-specific selected coordinates; scores zero',
                    'static': 'Final selected teacher clouds only; no score or trajectory',
                    'structure': 'No generated Steer; original bound ligand and receptor',
                    'pocket': 'No generated Steer or bound ligand reward; receptor crop'}[family]}
        packet = {'condition': identity, 'information_level': level, 'removed_instruction_blocks': removed,
            'objective': 'Improve final FLOWR predicted target pIC50; strain secondary; allow graph transitions',
            'execution': {'window': base['window'], 'window_origin': 'R26 matched protocol',
                'steps': 100, 'seed': 42, 'native_rms_ratio': .33, 'derivative_path': 'FLOWR endpoint VJP',
                'affinity_head_gradient': False, 'particle_selection': False},
            'compression': 'Strongest 8 and weakest 2 q-values per supplied table; exploratory selected evidence, not a new complete enrichment analysis',
            'evidence': entries + support if level == 'full' else structural_support,
            'formula_registry': registry,
            'formula_notes': {'full': 'R26 robust multimodal pointcloud', 'unweighted': 'R26 with score beta=0',
                'static': 'All times use last elite pointclouds, beta=0',
                'structure': '-wL*(sqrt(1+Hungarian_MSE)-1)+wC*saturated_3.6A_contact-wR*sum(relu(2A-d)^2)/N',
                'pocket': 'Same field without ligand attraction; untyped geometric proxy, not a physical affinity energy'},
            'allowed_updates': {'native_rms_ratio': [.1, .2, .33], 'time_ramp_power': [0, 1, 2],
                'ligand_anchor_weight': [0.5, 1, 2], 'contact_weight': [0, .1, .25, .5, 1],
                'repulsion_weight': [.5, 1, 2, 4]}}
        if level == 'pocket':
            packet['evidence'][0] = copy.deepcopy(support[0])
            for key in ['known_ligand_heavy_atoms', 'ligand_centroid_A', 'ligand_covariance_eigenvalues_A2']:
                packet['evidence'][0]['data'].pop(key)
            packet['evidence'] = [x for x in packet['evidence'] if x['kind'] != 'original_ligand_geometry']
        write_json(folder / 'evidence.json', packet)
        for role in ['Analyst', 'Designer']:
            text = (repo / ('skills/steer-dependency-' + role.lower() + '/SKILL.md')).read_text()
            active = [value for key, (owner, value) in BLOCKS.items() if owner == role and key not in removed]
            (folder / (role + '.instructions.md')).write_text(text + '\n\n' + '\n\n'.join(active) + '\n')
        out.append({'condition': identity, 'information_level': level, 'removed': removed})
    write_json(docs / 'protocol.json', {'schema_version': 'steer-dependency-study-1.0',
        'frozen_R26_sha256': digest(original), 'reference_sha256': digest(refpath), 'conditions': out,
        'instruction_block_origin': 'Literal excerpts from skill_ablation_20261007/study_v1/treatment_modules.json',
        'screen': {'batches': [75, 76], 'n_per_arm': 100, 'master_seed': 42},
        'confirmation': {'batches': [77, 78, 79, 80, 81, 82], 'n_per_arm': 300,
            'rule': 'Confirm strongest teacher-free screen candidate plus matched R26/native; report screening selection bias separately'},
        'equivalence': 'Identical numerical programs share measured inference; no fictitious independent trials',
        'limits': 'One independent fresh-context Analyst and Designer per condition; no claim of LLM response variance from a single call'})

SCHEMAS = {
    'Analyst': {'type': 'object', 'required': ['role', 'condition', 'findings', 'directions', 'limitations'],
        'additionalProperties': False, 'properties': {'role': {'const': 'Analyst'}, 'condition': {'type': 'string'},
            'findings': {'type': 'array', 'items': {'type': 'object', 'required': ['evidence_id', 'finding'],
                'additionalProperties': False, 'properties': {'evidence_id': {'type': 'string'}, 'finding': {'type': 'string'}}}},
            'directions': {'type': 'array', 'items': {'type': 'string'}}, 'limitations': {'type': 'array', 'items': {'type': 'string'}}}},
    'Designer': {'type': 'object', 'required': ['role', 'condition', 'family', 'updates', 'rationale', 'failure_modes'],
        'additionalProperties': False, 'properties': {'role': {'const': 'Designer'}, 'condition': {'type': 'string'},
            'family': {'type': 'string'}, 'updates': {'type': 'object'}, 'rationale': {'type': 'string'},
            'failure_modes': {'type': 'array', 'items': {'type': 'string'}}}}}

def declared_model(folder):
    for directory in (Path(folder), *Path(folder).parents):
        protocol = directory / 'protocol.json'
        if protocol.is_file():
            model = read_json(protocol).get('agent_model')
            if model: return model
    return None


def export(folder, role, requested_model=None):
    folder = Path(folder); packet = read_json(folder / 'evidence.json')
    model = declared_model(folder)
    if model and requested_model and model != requested_model:
        raise ValueError('Requested model conflicts with the study protocol')
    requested_model = requested_model or model
    for family, entry in packet['formula_registry'].items():
        entry['supported_update_keys'] = (list(packet['allowed_updates']) if family == 'structure' else
            [k for k in packet['allowed_updates'] if k != 'ligand_anchor_weight'] if family == 'pocket' else
            ['native_rms_ratio', 'time_ramp_power'])
    payload = {'evidence': packet}
    if role == 'Designer': payload['Analyst'] = read_json(folder / 'Analyst.response.json')
    instruction = (folder / (role + '.instructions.md')).read_text()
    request = {'schema_version': 'steer-dependency-agent-1.0', 'role': role,
        'condition': packet['condition'], 'response_schema': SCHEMAS[role],
        'messages': [{'role': 'system', 'content': instruction + '\nReturn only JSON matching ' + json.dumps(SCHEMAS[role])},
            {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}],
        'evidence_sha256': digest(folder / 'evidence.json'),
        'instruction_sha256': digest(folder / (role + '.instructions.md'))}
    if requested_model is not None: request['requested_model'] = requested_model
    write_json(folder / (role + '.request.json'), request)
    return request

def validate(folder, role):
    import jsonschema
    folder = Path(folder); response = read_json(folder / (role + '.response.json'))
    request = read_json(folder / (role + '.request.json')); packet = read_json(folder / 'evidence.json')
    if declared_model(folder) and request.get('requested_model') != declared_model(folder):
        raise ValueError('Agent request does not pin the study model')
    jsonschema.validate(response, SCHEMAS[role])
    if request['evidence_sha256'] != digest(folder / 'evidence.json') or response['condition'] != packet['condition']:
        raise ValueError('Request/evidence/condition changed')
    if role == 'Analyst':
        known = {x['evidence_id'] for x in packet['evidence']}
        if any(x['evidence_id'] not in known for x in response['findings']):
            raise ValueError('Agent cited unavailable evidence')
    else:
        if response['family'] == 'defer': return None
        if response['family'] not in packet['formula_registry']:
            raise ValueError('Reward requires unavailable information')
        for key, value in response['updates'].items():
            if key not in packet['allowed_updates'] or value not in packet['allowed_updates'][key]:
                raise ValueError('Unregistered parameter update')
        if response['family'] not in ['structure', 'pocket'] and any(k.endswith('_weight') for k in response['updates']):
            raise ValueError('Inactive component weights are not permitted')
        if response['family'] == 'pocket' and 'ligand_anchor_weight' in response['updates']:
            raise ValueError('Bound ligand is absent')
    write_json(folder / (role + '.validation.json'), {'valid': True, 'response_sha256': digest(folder / (role + '.response.json')),
        'request_sha256': digest(folder / (role + '.request.json')), 'available_evidence_ids_verified': True})
    return response

def compile_program(repo, folder, output_dir=None):
    repo, folder = Path(repo), Path(folder)
    validate(folder, 'Analyst'); response = validate(folder, 'Designer')
    if response is None: return None
    packet = read_json(folder / 'evidence.json'); entry = packet['formula_registry'][response['family']]
    program = read_json(repo / entry['program']); program.update(response['updates'])
    program['program_id'] = packet['condition']
    program['agent_provenance'] = {'Analyst_sha256': digest(folder / 'Analyst.response.json'),
        'Designer_sha256': digest(folder / 'Designer.response.json'), 'evidence_sha256': digest(folder / 'evidence.json')}
    identity = packet['condition']
    if folder.parent.name == identity and folder.name.startswith('replicate_'):
        identity += '__' + folder.name
        program['program_id'] = identity
    output = (repo / output_dir if output_dir is not None else (repo / entry['program']).parent) / (identity + '.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and read_json(output) != program:
        raise FileExistsError('Compiled program is frozen; create a new replicate or condition')
    write_json(output, program)
    numerical = {k: v for k, v in program.items() if k not in ['program_id', 'agent_provenance']}
    import hashlib
    fingerprint = hashlib.sha256(json.dumps(numerical, sort_keys=True).encode()).hexdigest()
    write_json(folder / 'compiled.json', {'program': output.relative_to(repo).as_posix(),
        'reference': entry['reference'], 'family': response['family'], 'numerical_fingerprint': fingerprint,
        'reference_sha256': program['reference_sha256'], 'information_level': packet['information_level']})
    return program
