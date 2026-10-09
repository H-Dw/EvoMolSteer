"""Compact, reusable diagnostics of realized coordinate guidance directions.

Reads existing traces; makes no FLOWR or affinity calls. Adjacent-step gradient
cosines exclude the initial synthetic value. Different objectives' gradient
norms or scalar reward values are not treated as directly comparable quality.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from evomolsteer.io import read_json, write_json, digest

FIELDS = ['flowcompat_gradient_lag_cosine', 'flowcompat_native_cosine_before',
          'injection_rms_A', 'cap_factor', 'backtrack_factor',
          'flowcompat_jacobian_gain']


def summarize(root, arm):
    results = []
    for file in sorted((root / arm).glob('batch_*/guidance_trace.jsonl')):
        rows = [json.loads(x) for x in file.read_text(encoding='utf-8').splitlines()]
        active = [r for r in rows if r['active']]
        if not active:
            raise ValueError('Controlled trace required')
        out = {'batch': int(file.parent.name.split('_')[-1]),
               'active_steps': len(active), 'trace_sha256': digest(file)}
        for key in FIELDS:
            selected = active[1:] if key == 'flowcompat_gradient_lag_cosine' else active
            values = np.concatenate([np.atleast_1d(r[key]) for r in selected])
            if not np.isfinite(values).all():
                raise ValueError('Nonfinite diagnostic: ' + key)
            out[key] = {'mean': float(values.mean()), 'median': float(np.median(values)),
                        'negative_fraction': float((values < 0).mean()),
                        'below_one_fraction': float((values < 1).mean())}
        out['cumulative_rms_A_last_mean'] = float(np.mean(active[-1]['cumulative_rms_A']))
        out['inactive_nonzero_steps'] = sum(
            np.any(np.asarray(r['injection_l2_A']) != 0) for r in rows if not r['active'])
        results.append(out)
    if not results:
        raise ValueError('No batches: ' + str(root))
    return {'program_sha256': digest(root / 'reward_program.json'), 'batches': results}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True); p.add_argument('--output', required=True)
    a = p.parse_args(); plan = read_json(a.manifest)
    write_json(a.output, {'cohorts': {r['name']: summarize(Path(r['root']), r['arm'])
                                    for r in plan['cohorts']},
                         'manifest_sha256': digest(a.manifest),
                         'limits': 'Observed directional stability and realized dose; neither instruction causality nor proof of affinity mechanism. Negative native cosine can be useful when native generation is suboptimal.'})
