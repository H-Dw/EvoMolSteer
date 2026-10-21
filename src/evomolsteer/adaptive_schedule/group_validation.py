"""Real native acceptance before formal R0; all probes stay outside scientific counts."""
from pathlib import Path
import time

from .io import read, write, sha, source_fingerprint, commit


def validate(root):
    import torch
    import h5py
    from .runner import lock
    from .group_run import assert_prepared
    from .flowr import load_model, load_pt, save_pt, tensor_digest
    from .wave_native import execute_wave, take
    from .wave_retention import prune_payload
    from .grid import CHECKPOINT_STEPS
    root = Path(root); config = assert_prepared(root)
    probe = root/'operations'/('native_acceptance_'+commit()[:8])
    probe.mkdir(parents=True, exist_ok=True)
    with lock(root):
        model = load_model(config, read(root/'teachers.json')[0]['target'], root)
        results = []
        for teacher in read(root/'teachers.json'):
            tb = probe/teacher['teacher_id']
            if (tb/'ACCEPTED.json').exists():
                results.extend(read(tb/'ACCEPTED.json')['results'])
                continue
            folder = tb/'full'
            ticks = [[4, 5, 5, 5], [6, 5, 5, 5], [5, 4, 5, 5], [5, 6, 5, 5]]
            schedules = [t for t in ticks for _ in range(100)]
            execute_wave(model, config, teacher, schedules, folder, probe/'context400.pt.gz')
            original = load_pt(folder/'terminal.pt.gz')
            selected = [(i%4)*100+i for i in range(100)]
            parent = tb/'mixed_parent'; parent.mkdir(parents=True, exist_ok=True)
            for step in CHECKPOINT_STEPS:
                state = load_pt(folder/f'checkpoint_{step:03d}.pt.gz')
                state['rows'] = 100
                for key in ['current', 'cond', 'times', 'previous_assignment']:
                    state[key] = take(state[key], selected, 400)
                save_pt(parent/f'checkpoint_{step:03d}.pt.gz', state)
            write(parent/'COMPLETE.json', {'diagnostic_only': True, 'indices': selected})
            mixed_schedules = [ticks[i%4] for i in range(100)]*4
            expected = take(original, selected*4, 400)
            with h5py.File(folder/'trace.h5', 'r') as trace:
                for step in range(100):
                    eps = trace['native_epsilon'][step]
                    if any(not (eps[:100] == eps[g*100:(g+1)*100]).all() for g in range(1, 4)):
                        raise ValueError('Common Gaussian stream differs across hypotheses')
                if trace['guidance_delta_A'][:].reshape(100, 400, -1).shape[1] != 400:
                    raise ValueError('Guidance layout differs')
            for start in [10, 26, 51, 76]:
                resumed = tb/f'resume_{start:03d}'
                execute_wave(model, config, teacher, mixed_schedules, resumed,
                             probe/'context400.pt.gz', parent=parent, start_step=start)
                got = load_pt(resumed/'terminal.pt.gz')
                coord_error = float((got['coords']-expected['coords']).abs().max())
                if not torch.allclose(got['coords'], expected['coords'], atol=1e-5, rtol=1e-5):
                    raise ValueError(f'Mixed-parent restart coordinate mismatch: {coord_error}')
                for key in ['atomics', 'charges', 'bonds', 'mask']:
                    if not torch.equal(got[key], expected[key]):
                        raise ValueError('Mixed-parent categorical/mask mismatch: '+key)
                end = load_pt(resumed/'checkpoint_100.pt.gz')
                if end['rng_sha256'] != load_pt(folder/'checkpoint_100.pt.gz')['rng_sha256']:
                    raise ValueError('Mixed-parent terminal RNG mismatch')
                rows = read(folder/'final_records.json')
                observed = read(resumed/'final_records.json')
                scores = [abs(rows[i].get('pic50_on_rescore')-r.get('pic50_on_rescore'))
                          for i, r in zip(selected*4, observed)
                          if rows[i].get('pic50_on_rescore') is not None and r.get('pic50_on_rescore') is not None]
                if scores and max(scores) > 1e-4:
                    raise ValueError('Mixed-parent rescore mismatch')
                results.append({'teacher': teacher['teacher_id'], 'start': start, 'coordinate_max_abs': coord_error,
                                'score_max_abs': max(scores, default=None), 'rng_and_categories_equal': True})
                write(probe/'results.json', results)
            # Keep documentary evidence, not diagnostics as extra scientific samples.
            for path in [folder, parent]+[tb/f'resume_{s:03d}' for s in [10,26,51,76]]:
                prune_payload(path)
            write(tb/'ACCEPTED.json', {'results': [r for r in results if r['teacher'] == teacher['teacher_id']]})
        result = {'source_fingerprint': source_fingerprint(), 'config_sha256': sha(root/'config.json'),
                  'teachers': 2, 'recovery_checks': len(results), 'results_sha256': sha(probe/'results.json'),
                  'common_randomness': True, 'different_hypothesis_schedules': True,
                  'affinity_vjp': False, 'time': time.time()}
        write(root/'ACCEPTANCE_COMPLETE.json', result)
        return result
