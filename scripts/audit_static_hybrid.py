"""Audit prepared hybrid endpoint limits against existing reward implementations.

No model inference or affinity prediction is performed. Small perturbed point
clouds exercise the conditional derivatives, with detached correspondence.
"""
import argparse
import gzip
import json
from pathlib import Path
import numpy as np
import torch
from evomolsteer.io import read_json, write_json, digest
from evomolsteer.generation.endpoint_reward import EndpointGeometryReward
from evomolsteer.generation.static_hybrid_reward import StaticHybridReward
from evomolsteer.generation.structure_field_reward import StructureFieldReward
from evomolsteer.generation.window_reference import reference_node_time


def load(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        return json.load(stream)


def audit(program_path, hybrid_path, teacher_program_path, teacher_path, structure_path, output):
    program, hybrid = read_json(program_path), load(hybrid_path)
    old_program, old_teacher, structure = read_json(teacher_program_path), load(teacher_path), load(structure_path)
    if digest(hybrid_path) != program['reference_sha256'] or digest(teacher_path) != old_program['reference_sha256']:
        raise ValueError('Frozen reference mismatch')
    if old_program['teacher_score_beta'] != 0 or old_teacher.get('dependency_ablation') != 'static':
        raise ValueError('Score-unweighted static teacher control required')
    a, b = hybrid['window']; time = hybrid['times'][len(hybrid['times']) // 2]
    clocks = [reference_node_time(hybrid, t, t + .01) for t in np.arange(100) / 100
              if t >= a - 1e-6 and t + .01 <= b + 1e-6]
    if any(min(abs(np.asarray(hybrid['times']) - t)) > 2e-6 for t in clocks):
        raise ValueError('Incomplete controlled clock coverage')
    rng = np.random.default_rng(42)
    clouds = np.asarray([hybrid['bound_ligand_A']] + hybrid['teacher_bank_A'][:8])
    clouds = clouds + rng.normal(0, .05, clouds.shape)
    cases = []
    for dtype in [torch.float64, torch.float32]:
        x = torch.tensor(clouds, dtype=dtype, requires_grad=True)
        anchor = x.detach() + torch.tensor(rng.normal(0, .03, clouds.shape), dtype=dtype)
        mask = torch.ones(x.shape[:2], dtype=torch.bool)
        for rho, name in [(0., 'M2_static_teacher'), (1., 'unit_original_anchor')]:
            p = {**program, 'original_pose_prior_mass': rho, 'contact_weight': 0, 'repulsion_weight': 0}
            new_reward = StaticHybridReward(p, hybrid)
            if rho == 0:
                old_reward = EndpointGeometryReward(old_program, old_teacher)
            else:
                old_reward = StructureFieldReward({**p, 'ligand_anchor_weight': 1}, structure)
            left, _ = new_reward(x, None, mask, time, anchor)
            right, _ = old_reward(x, None, mask, time, anchor)
            gl, = torch.autograd.grad(left.sum(), x, retain_graph=True)
            gr, = torch.autograd.grad(right.sum(), x, retain_graph=True)
            tolerance = 1e-5 if dtype == torch.float32 else 1e-10
            if not torch.allclose(left, right, atol=tolerance, rtol=tolerance) or not torch.allclose(gl, gr, atol=tolerance, rtol=tolerance):
                raise AssertionError('Endpoint limit mismatch: ' + name)
            cases.append({'limit': name, 'dtype': str(dtype), 'samples': len(x),
                          'reward_max_abs_error': float((left - right).abs().max().detach()),
                          'gradient_max_abs_error': float((gl - gr).abs().max()),
                          'gradient_relative_l2_error': float((gl - gr).norm() / gr.norm().clamp_min(1e-30))})
    result = {'schema_version': 'static-hybrid-mathematical-audit-1.0', 'passed': True,
              'cases': cases, 'controlled_steps': len(clocks), 'execution_window': [a, b],
              'teacher_source_clock': hybrid['teacher_bank_source_time'],
              'source_sha256': {str(p): digest(p) for p in
                                [program_path, hybrid_path, teacher_program_path, teacher_path, structure_path]},
              'limits': 'Conditional coordinate derivatives only; no actual FLOWR Jacobian, inference quality or new affinity gain validated. The original-only limit matches I3 direction only while its repulsion term is inactive.'}
    write_json(output, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ['program', 'hybrid-reference', 'teacher-program', 'teacher-reference', 'structure-reference', 'output']:
        parser.add_argument('--' + flag, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(Path(args.program), Path(args.hybrid_reference), Path(args.teacher_program),
                           Path(args.teacher_reference), Path(args.structure_reference), Path(args.output))))
