"""Recheck retained bytes, frozen reward identity and actual runtime contracts.

Run from the repository root with its Python environment. This reads only
compact local evidence; it never executes a model or computes another score.
"""
import json
from pathlib import Path
from evomolsteer.io import read_json, digest, write_json
from evomolsteer.generation.coordinate_exploration import validate_retention

ROOT = Path(__file__).resolve().parents[3]
EVIDENCE = Path(__file__).resolve().parent
CONFIG = ROOT/'configs/experiments/ck2_affinity_geometry30_v1'

def verify():
    c = read_json(CONFIG/'campaign.json')
    assert c['rounds_started'] == c['rounds_completed'] == c['maximum_rounds'] == 30
    assert [r['round'] for r in c['rounds']] == list(range(1,31))
    frozen = read_json(EVIDENCE/'frozen_effective_program.json')
    rows, zero = [], []
    for entry in c['rounds']:
        n = entry['round']
        local = EVIDENCE/f'round_{n:02d}/local'
        retention_sha = validate_retention(local, entry['n_per_arm']*len(entry['arms']))
        retained = read_json(local/'retention.json')
        p = CONFIG/f'backtrack_round{n:02d}.json'
        program = read_json(p)
        assert digest(p) == digest(local/'inference_config/reward_program.json')
        cfg = read_json(local/'inference_config/config.json')
        ext = cfg['extension']
        execution = read_json(local/'execution_report.json')
        audit = read_json(local/'coordinate_audit.json')
        assert cfg['experiment']['steps'] == execution['integration_steps'] == 100
        assert execution['seed'] == 42 and execution['no_particle_resampling'] is True
        assert execution['outside_window_injection'] is False
        assert ext['particle_resampling'] is False and ext['post_window_injection'] is False
        assert ext['control_domain'] == ext['evidence_domain'] == program['window']
        assert ext['program_sha256'] == digest(p)
        assert ext['reference_sha256'] == program['reference_sha256'] == audit['reference_sha256']
        assert read_json(EVIDENCE/f'round_{n:02d}/initial_pairing.json')['passed'] is True
        if n >= 5:
            assert ext['affinity_head_gradient'] is False
            assert ext['additional_production_forward_calls_per_step'] == 0
            assert audit['coordinate_preflight']['passed'] is True
            for b in audit['batch_results']:
                assert b['production_target_forward_calls'] == 100
                assert b['affinity_head_gradient'] is False and b['affinity_outputs_detached'] is True
                assert b['no_injection_after_window'] is True
        for b in audit['batch_results']:
            if b['arm'] == 'gradient':
                assert b['active_steps'] == b['nonzero_injection_steps'] == 50
        if n >= 27:
            body = {k:v for k,v in program.items() if k not in frozen['excluded_metadata_fields']}
            assert body == frozen['effective_program']
        if 'gradient_zero' in entry['arms']:
            for batch in entry['batches']:
                h = retained['raw_trajectory_hashes']
                a = next(v for v in h if f'unguided\\batch_{batch:03d}' in v['path'])
                z = next(v for v in h if f'gradient_zero\\batch_{batch:03d}' in v['path'])
                assert a['sha256'] == z['sha256']
                zero.append({'round':n,'batch':batch,'full100_zero_native_equal':True,'sha256':a['sha256']})
        deleted = [ROOT/'data/generated'/entry['campaign'], ROOT/'results'/f'affinity_round{n}',
            ROOT/'data/archives'/(entry['campaign']+'.tar.gz'),
            ROOT/'data/archives'/(entry['campaign']+'.tar.gz.json')]
        assert not any(p.exists() for p in deleted), deleted
        rows.append({'round':n,'retention_sha256':retention_sha,'program_sha256':digest(p),
            'inference_commit':retained['inference_commit'], 'local_generated_outputs_absent':True,
            'instrumented_production_call_counts':n >= 5})
    value = {'passed':True,'completed':30,'discovery_rounds':26,'frozen_rounds':[27,28,29,30],
        'all_retained_bytes_verified':True,'all_runtime_windows_match_learning_window':True,
        'frozen_reward_identity_verified':True,'zero_dose_certificates':zero,'rounds':rows,
        'source_code_sha256':digest(__file__),
        'limits':'Rounds1-4 used direct coordinate reward; per-forward counters were added from round5. SHA equality refers to retained pre-deletion full100 trajectory hashes.'}
    write_json(EVIDENCE/'completion_audit.json',value)
    print(json.dumps({'passed':True,'rounds':30,'full100_zero_certificates':len(zero)}))
    return value

if __name__ == '__main__':
    verify()
