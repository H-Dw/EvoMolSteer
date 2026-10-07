import json
from pathlib import Path

import pandas as pd
import pytest

from evomolsteer.generation.path_campaign_report import (
    build_campaign_report, confirmation_pair, qualified_summary, verified_retention, verify_initial_pair,
)
from evomolsteer.io import digest, write_json


def data(score=8.5):
    return pd.DataFrame({'batch': [32, 32, 33, 33], 'slot': [0, 1, 0, 1],
                         'pic50_on_rescore': [score] * 4, 'valid_connected': [True, False, True, True],
                         'smiles': ['CC', 'CN', 'CC', 'CO'], 'pb_fast_pass': [True, False, True, False],
                         'energy_status': ['converged'] * 4, 'mmff_relief_per_heavy': [.2, .3, .4, .5]})


def test_quality_tail_reports_failed_attempt_denominator_without_graph_gate():
    summary = qualified_summary(data(), 8.2)
    assert summary['elite_valid_n'] == 3 and summary['elite_unique_graphs'] == 2
    assert summary['elite_pb_fast_n'] == 2 and summary['elite_pb_fast_yield'] == .5
    assert summary['elite_pb_fast_unique_graphs'] == 1
    assert summary['elite_pb_strain_median_per_heavy'] == pytest.approx(.3)


def test_confirmation_uses_batch_pairing_and_explicit_limitations():
    effect = confirmation_pair(data(8.5), data(7.5))
    assert effect['paired_mean_pic50'] == 1 and effect['batch_bootstrap_CI95'] == [1., 1.]
    assert [r['batch'] for r in effect['batch_details']] == [32, 33]
    assert 'Independent frozen' in effect['limitation']


def test_confirmation_requires_same_real_initial_states():
    left = {'initial_state_signatures': {'gradient/32': 'abc'}}
    right = {'initial_state_signatures': {'unguided/32': 'abc'}}
    verify_initial_pair(left, right)
    right['initial_state_signatures']['unguided/32'] = 'different'
    with pytest.raises(ValueError, match='initial states'): verify_initial_pair(left, right)


def test_retention_checksum_and_premature_summary_guard(tmp_path):
    score = tmp_path / 'scores.csv';score.write_text('score\n8.5\n')
    write_json(tmp_path / 'retention.json', {'status': 'complete', 'files': [{'path': score.name, 'sha256': digest(score)}]})
    verified_retention(tmp_path)
    score.write_text('score\n9.0\n')
    with pytest.raises(ValueError, match='checksum'): verified_retention(tmp_path)
    write_json(tmp_path / 'campaign.json', {'rounds_completed': 17, 'maximum_rounds': 20, 'rounds': []})
    with pytest.raises(ValueError, match='Finish every'): build_campaign_report(tmp_path, tmp_path / 'campaign.json', tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


def test_complete_synthesis_uses_repaired_round_and_separate_frozen_panels(tmp_path):
    reports = tmp_path / 'reports';reports.mkdir()
    rounds = []
    for number in range(1, 21):
        item = {'round': number, 'module': 'fixture', 'kind': 'local_data_module' if number < 4 else 'inference', 'status': 'complete'}
        if number >= 4:
            if number == 8:item['repaired_retention_report'] = 'round08_replay/retention.json'
            folder = reports / ('round08_replay' if number == 8 else f'round{number:02d}')
            folder.mkdir();(folder / 'inference_config').mkdir()
            gradient = data();gradient['arm'] = 'gradient'
            if number == 20:gradient['batch'] += 2
            rows = gradient
            if number in [18, 20]:
                native = gradient.copy();native['arm'] = 'unguided';native['pic50_on_rescore'] -= 1
                rows = pd.concat([gradient, native])
            rows.to_csv(folder / 'candidate_metrics.csv', index=False)
            signatures = {f'{arm}/{batch}': f'initial_{batch}' for arm in rows.arm.unique() for batch in rows.batch.unique()}
            write_json(folder / 'execution_report.json', {'initial_state_signatures': signatures, 'code_commit': 'fixture'})
            write_json(folder / 'inference_config/reward_program.json',
                       {'round': number, 'reward_view': 'endpoint_pointcloud', 'window': [.2, .6]})
            files = [{'path': p.relative_to(folder).as_posix(), 'sha256': digest(p)} for p in folder.rglob('*') if p.is_file()]
            write_json(folder / 'retention.json', {'status': 'complete', 'files': files})
        rounds.append(item)
    write_json(reports / 'round02_credit_manifest.json', {'threshold_pic50': 8.2})
    write_json(reports / 'frozen_validation.json', {'winner_round': 5})
    config = tmp_path / 'campaign.json'
    write_json(config, {'maximum_rounds': 20, 'rounds_completed': 20, 'master_seed': 42, 'rounds': rounds})
    result = build_campaign_report(reports, config, tmp_path / 'out')
    assert result['selected_vs_native_pooled']['n_batches'] == 4
    assert result['selected_vs_native_pooled']['paired_mean_pic50'] == 1
    assert result['selected_vs_old_panel_a']['paired_mean_pic50'] == 0
    provenance = json.loads((tmp_path / 'out/input_provenance.json').read_text())
    assert provenance['inputs'][4]['retention'] == 'round08_replay/retention.json'
