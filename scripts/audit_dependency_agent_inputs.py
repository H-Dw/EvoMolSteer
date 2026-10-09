"""Verify exact instruction removal, evidence boundaries and compiler provenance.

Delivery and execution checks do not measure a model's hidden attention, nor
identify causal instruction effects from a single stochastic response.
"""
import argparse
import gzip
import json
from pathlib import Path
from evomolsteer.continuous.dependency_ablation import BLOCKS, validate
from evomolsteer.io import read_json, write_json, digest


def keys(value):
    if isinstance(value, dict):
        return set(value) | set().union(*(keys(v) for v in value.values()))
    if isinstance(value, list): return set().union(*(keys(v) for v in value))
    return set()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', required=True); p.add_argument('--output', required=True)
    a = p.parse_args(); study = Path(a.study); repo = Path(__file__).resolve().parents[1]
    checked = []
    for entry in read_json(study / 'protocol.json')['conditions']:
        folder = study / entry['condition']; packet = read_json(folder / 'evidence.json')
        for role in ['Analyst', 'Designer']:
            request = read_json(folder / (role + '.request.json'))
            instruction = (folder / (role + '.instructions.md')).read_text(encoding='utf-8')
            if request['instruction_sha256'] != digest(folder / (role + '.instructions.md')):
                raise ValueError('Instruction delivery hash changed')
            if not request['messages'][0]['content'].startswith(instruction):
                raise ValueError('Skill instructions absent from actual request')
            for block, (owner, sentence) in BLOCKS.items():
                if owner == role and (sentence in instruction) != (block not in entry['removed']):
                    raise ValueError('Instruction ablation mismatch')
            validate(folder, role)
        if entry['information_level'] in ['structure', 'pocket']:
            forbidden = {'p','q','high_low_integrated_z','survival_high_low_integrated_z',
                         'teacher_endpoint_A','teacher_scores','success_fail_rate_integrated_z'}
            if forbidden & keys(packet['evidence']):
                raise ValueError('Historical statistical/teacher data leaked')
        compiled = read_json(folder / 'compiled.json'); program = read_json(repo / compiled['program'])
        expected = {'Analyst_sha256': digest(folder/'Analyst.response.json'),
                    'Designer_sha256': digest(folder/'Designer.response.json'),
                    'evidence_sha256': digest(folder/'evidence.json')}
        if program['agent_provenance'] != expected: raise ValueError('Compiler ignored Agent outputs')
        if compiled['family'] in ['structure','pocket']:
            with gzip.open(repo / compiled['reference'],'rt',encoding='utf-8') as stream: reference = json.load(stream)
            if reference['steer_sources'] != []: raise ValueError('Historical coordinate leakage')
            if compiled['family'] == 'pocket' and 'bound_ligand_A' in reference:
                raise ValueError('Pocket reward unexpectedly contains a ligand template')
        checked.append({'condition':entry['condition'],'instruction_delivery_verified':True,
            'literal_block_removal_verified':True,'available_evidence_citations_verified':True,
            'response_used_by_compiler':True,'information_level':entry['information_level'],
            'family':compiled['family'],'evidence_sha256':digest(folder/'evidence.json')})
    write_json(a.output,{'passed':True,'conditions':checked,
        'limits':'Verifies delivered instructions and executed choices, not hidden attention or instruction causality. Teacher-free reward still inherits the matched R26 window/controller and original FLOWR input site.'})
