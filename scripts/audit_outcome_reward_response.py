"""Local calculator: verify final-label instructions alter executable gradients.

This is a reward-space audit on frozen native endpoint anchors. Actual FLOWR
Jacobian and trajectory support are independently checked by generation traces.
"""
import argparse
import gzip
import json
from pathlib import Path
import numpy as np
import torch
from evomolsteer.io import digest, read_json, write_json
from evomolsteer.generation.endpoint_reward import EndpointGeometryReward
from evomolsteer.generation.branch_mixture_reward import BranchMixtureReward


def objective(program, reference):
    factory = BranchMixtureReward if program['reward_view'] == 'endpoint_branch_mixture' else EndpointGeometryReward
    return factory(program, reference)


def audit(program, reference, control_program, control_reference, output):
    p, c = read_json(program), read_json(control_program)
    ref, old = [json.loads(gzip.decompress(Path(v).read_bytes())) for v in (reference, control_reference)]
    if p['window'] != c['window'] or p['score_window'] != c['score_window']:
        raise ValueError('Clock-matched reward comparison required')
    if p['reference_sha256'] != digest(reference) or c['reference_sha256'] != digest(control_reference):
        raise ValueError('Compiled references differ')
    reward, baseline = objective(p, ref), objective(c, old)
    rows = []
    for frame in old['frames']:
        t = float(frame['time'])
        if not p['score_window'][0]-2e-6 <= t <= p['score_window'][1]+2e-6:
            continue
        anchor = torch.tensor(frame['teacher_endpoint_A'][:2], dtype=torch.float64)
        gradients = []
        values = []
        for function in (reward, baseline):
            x = anchor.clone().requires_grad_()
            value, _ = function(x, torch.zeros(x.shape[:2], dtype=torch.long),
                                torch.ones(x.shape[:2], dtype=torch.bool), t, anchor=anchor)
            gradient = torch.autograd.grad(value.sum(), x)[0].detach().numpy()
            gradients.append(gradient)
            values.append(value.detach().numpy())
        new, native = gradients
        den = np.sqrt(np.sum(new**2)*np.sum(native**2))
        rows.append({'score_time': t, 'gradient_delta_RMS': float(np.sqrt(np.mean((new-native)**2))),
                     'gradient_cosine': float(np.sum(new*native)/den) if den > 0 else None,
                     'reward_mean_difference': float(np.mean(values[0]-values[1]))})
    changed = sum(v['gradient_delta_RMS'] > 1e-10 for v in rows)
    result = {'stage': 'reward_objective_only', 'events': rows, 'changed_event_count': changed,
              'source_sha256': digest(__file__), 'program_sha256': digest(program),
              'reference_sha256': digest(reference), 'control_program_sha256': digest(control_program),
              'control_reference_sha256': digest(control_reference),
              'limitations': 'Frozen endpoint reward gradients; actual model VJP response requires prospective execution audit'}
    if not changed:
        raise ValueError('Instruction change does not alter any tested reward gradient')
    write_json(output, result)
    print({'changed_events': changed, 'total_events': len(rows)})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for key in ('program', 'reference', 'control-program', 'control-reference', 'output'):
        p.add_argument('--'+key, required=True)
    a = p.parse_args()
    audit(a.program, a.reference, a.control_program, a.control_reference, a.output)
