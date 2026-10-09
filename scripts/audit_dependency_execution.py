"""Verify actual controlled clocks, no SMC, dose and zero-control equality."""
import argparse
import json
from pathlib import Path
import numpy as np
from rdkit import Chem
from evomolsteer.io import read_json, write_json, digest

def audit(root):
    root = Path(root); program = read_json(root / 'reward_program.json')
    a, b = program['window']; report = []
    for path in sorted(root.glob('*/batch_*')):
        if not path.is_dir(): continue
        rows = [json.loads(line) for line in (path / 'guidance_trace.jsonl').read_text().splitlines()]
        if len(rows) != 100: raise ValueError('Incomplete inference')
        with np.load(path / 'execution_state.npz', allow_pickle=False) as state:
            if state['resampled'].any() or not np.array_equal(state['selected_indices'], np.tile(np.arange(50), (100, 1))):
                raise ValueError('Particle replication detected')
        arm = path.parent.name
        eligible = [r['score_time'] >= a-1e-6 and r['state_time'] <= b+1e-6 for r in rows]
        expected = [e and arm == 'gradient' for e in eligible]
        if [r['active'] for r in rows] != expected: raise ValueError('Control window mismatch')
        if [r['reward_evaluated'] for r in rows] != [e and arm != 'unguided' for e in eligible]:
            raise ValueError('Reward clock mismatch')
        if any(r['production_target_forward_calls'] != 1 or r['particle_resampled'] for r in rows):
            raise ValueError('Additional target forward or selection detected')
        injections = [np.asarray(r['injection_l2_A']) for r in rows]
        if any(np.any(x != 0) for r, x in zip(rows, injections) if not r['active'] or arm == 'gradient_zero'):
            raise ValueError('Inactive/zero control injected displacement')
        active = [r for r in rows if r['active']]
        report.append({'arm': arm, 'batch': int(path.name.split('_')[-1]), 'steps': len(rows),
            'controlled_steps': len(active), 'post_window_nonzero_injections': 0,
            'reward_evaluated_steps': sum(r['reward_evaluated'] for r in rows),
            'preflight_extra_forwards': sum(r['one_time_audit_target_forward_calls'] for r in rows),
            'mean_active_injection_rms_A': float(np.mean([np.mean(r.get('injection_rms_A', [0])) for r in active])) if active else 0,
            'max_atom_injection_A': float(max(np.max(r['max_atom_injection_A']) for r in rows)),
            'trace_sha256': digest(path / 'guidance_trace.jsonl')})
    zero = []
    if (root / 'gradient_zero').is_dir():
        for native in sorted((root / 'unguided').glob('batch_*')):
            other = root / 'gradient_zero' / native.name
            with np.load(native / 'window_state.npz') as left, np.load(other / 'window_state.npz') as right:
                checks = {k: left[k].dtype == right[k].dtype and left[k].tobytes() == right[k].tobytes() for k in left.files}
            fields = ['pic50_on_rescore', 'pic50_off_rescore', 'pic50_on_upstream', 'pic50_off_upstream', 'build_success']
            scores = all(all(l[k] == r[k] for k in fields) for l, r in zip(read_json(native / 'final_records.json'), read_json(other / 'final_records.json')))
            def clouds(path):
                return {m.GetIntProp('slot'): m.GetConformer().GetPositions() for m in Chem.SDMolSupplier(str(path), sanitize=False) if m is not None}
            left, right = clouds(native / 'molecules_raw_decodable.sdf'), clouds(other / 'molecules_raw_decodable.sdf')
            coordinates = set(left) == set(right) and all(np.array_equal(left[k], right[k]) for k in left)
            zero.append({'batch': int(native.name.split('_')[-1]), 'window_tensor_byte_equal': checks,
                'terminal_scores_equal': scores, 'terminal_coordinates_equal_at_SDF_precision': coordinates})
            if not all(checks.values()) or not scores or not coordinates: raise ValueError('Zero-dose native mismatch')
    return {'passed': True, 'batches': report, 'zero_controls': zero,
        'zero_check_limit': 't=window_end tensors byte-exact; final scores exact, final coordinates at exported SDF precision; full final tensor omitted by evaluation transport'}

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--root', required=True); p.add_argument('--output', required=True)
    a = p.parse_args(); write_json(a.output, audit(a.root))
