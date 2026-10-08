import json
from pathlib import Path

import pandas as pd
import pytest

from evomolsteer.io import digest, write_json
from evomolsteer.generation.dynamic_campaign_report import build_report, paired_valid_summary


def fixture_campaign(tmp_path):
    reports, configs = tmp_path / 'reports', tmp_path / 'configs'
    reports.mkdir(); configs.mkdir()
    rounds = []
    for n in range(1, 11):
        folder = reports / f'round{n:02d}'; folder.mkdir(); (folder / 'inference_config').mkdir()
        batches = [36] if n < 8 else [39, 40] if n == 10 else [37, 38]
        arms = ['gradient', 'unguided'] if n in [1, 8, 10] else ['gradient']
        rows = [{'arm': arm, 'batch': batch, 'slot': slot,
                 'pic50_on_rescore': 7. + (1. if arm == 'gradient' else 0.) + (.1 if n in [9, 10] else 0.),
                 'valid_connected': slot == 0, 'pb_fast_pass': slot == 0, 'smiles': 'CC',
                 'energy_status': 'converged', 'mmff_relief_per_heavy': .4}
                for arm in arms for batch in batches for slot in range(2)]
        pd.DataFrame(rows).to_csv(folder / 'candidate_metrics.csv', index=False)
        signatures = {f'{arm}/{batch}': f'start_{batch}' for arm in arms for batch in batches}
        execution = {'steps': 100, 'no_particle_resampling': True, 'outside_window_injection': False,
                     'affinity_head_gradient': False, 'additional_production_calls_per_step': 0,
                     'preflight': {'passed': True}, 'window': [.1, .6], 'code_commit': 'fixture',
                     'initial_state_signatures': signatures}
        write_json(folder / 'execution_report.json', execution)
        program = {'round': n, 'program_id': f'fixture{n}', 'window': [.1, .6],
                   'reward_view': 'endpoint_pointcloud' if n in [1, 8] else 'endpoint_dynamic_region',
                   'reference_sha256': 'reference', 'regional_weight': 0. if n in [1, 8] else .05}
        write_json(configs / f'round{n:02d}.json', program)
        write_json(folder / 'inference_config/reward_program.json', program)
        write_json(folder / 'inference_config/config.json', {'experiment': {'seed': 42},
                   'extension': {'code_commit': 'fixture', 'checkpoint_sha256': 'checkpoint',
                                 'reference_sha256': 'reference'}})
        files = [{'path': p.relative_to(folder).as_posix(), 'sha256': digest(p)}
                 for p in folder.rglob('*') if p.is_file()]
        write_json(folder / 'retention.json', {'status': 'complete', 'campaign': f'dynamic_contrast_r{n:02d}', 'files': files})
        write_json(reports / f'round{n:02d}_plan.json', {'parent': 1, 'reason': 'fixture', 'change': {},
                   'batches': ','.join(map(str, batches)), 'arms': ','.join(arms),
                   'n_per_arm': 2 * len(batches), 'program_sha256': digest(configs / f'round{n:02d}.json')})
        rounds.append({'round': n, 'status': 'complete'})
    write_json(configs / 'campaign.json', {'maximum_rounds': 10, 'rounds_completed': 10, 'rounds': rounds})
    write_json(reports / 'protocol.json', {'tail_threshold_pic50': 8.05})
    write_json(reports / 'frozen_validation.json', {'candidate_round': 2,
               'program_sha256': digest(configs / 'round02.json')})
    return reports, configs


def test_complete_report_keeps_incumbent_and_fresh_panels_separate(tmp_path):
    reports, configs = fixture_campaign(tmp_path)
    result = build_report(reports, configs, tmp_path / 'out')
    assert result['attempted_records'] == 36
    assert result['selected_vs_incumbent_panel_a']['paired_mean_pic50'] == pytest.approx(.1)
    assert result['selected_vs_native_pooled']['n_batches'] == 4
    assert result['selected_vs_native_pooled']['paired_mean_pic50'] == pytest.approx(1.05)
    assert result['selected']['elite_pb_fast_n'] == 4
    assert result['selected']['elite_pb_fast_yield'] == .5
    assert result['selected_vs_incumbent_panel_a']['valid_intersection']['common_valid_n'] == 2
    assert len(pd.read_csv(tmp_path / 'out/rounds.csv')) == 10


def test_freeze_and_retained_checksums_reject_silent_validation_tuning(tmp_path):
    reports, configs = fixture_campaign(tmp_path)
    file = configs / 'round02.json'; program = json.loads(file.read_text()); program['regional_weight'] = 9.
    write_json(file, program)
    with pytest.raises(ValueError, match='Frozen'): build_report(reports, configs, tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


def test_no_report_before_full_sequence(tmp_path):
    reports, configs = fixture_campaign(tmp_path)
    file = configs / 'campaign.json'; cfg = json.loads(file.read_text()); cfg['rounds_completed'] = 9
    write_json(file, cfg)
    with pytest.raises(ValueError, match='Every registered'): build_report(reports, configs, tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


def test_common_valid_is_descriptive_and_preserves_failed_denominators():
    a = pd.DataFrame({'batch': [1, 1], 'slot': [0, 1], 'pic50_on_rescore': [9., 8.], 'valid_connected': [True, False]})
    b = a.copy(); b['valid_connected'] = [True, True]; b['pic50_on_rescore'] = [8., 7.]
    result = paired_valid_summary(a, b)
    assert result['common_valid_n'] == 1 and result['common_valid_mean_difference_pic50'] == 1
    assert result['control_valid_only_n'] == 1
