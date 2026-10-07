import importlib.util
import json
from pathlib import Path
import sys

import pytest

from evomolsteer.io import digest, write_json

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from finalize_path_campaign import finalize


def fixture(tmp_path):
    repo = tmp_path / 'EvoMolSteer';repo.mkdir()
    work = tmp_path / 'flowr_root/experiments/campaign';work.mkdir(parents=True)
    original = tmp_path / 'flowr_root/experiments/ck2_clk3_lineage_20261003'
    original.mkdir();(original / 'keep').write_text('original')
    report = repo / 'docs/experiments/elite_path20_20261007/round20';report.mkdir(parents=True)
    (report / 'scores.csv').write_text('score\n8\n')
    write_json(report / 'retention.json', {'status': 'complete', 'campaign': 'elite_path_r20',
               'files': [{'path': 'scores.csv', 'sha256': digest(report / 'scores.csv')}]})
    write_json(repo / 'configs/experiments/elite_path20_v1/campaign.json',
               {'maximum_rounds': 20, 'rounds_completed': 20,
                'rounds': [{'round': n, 'status': 'complete'} for n in range(1, 21)]})
    write_json(work / 'active.json', {'round': 20, 'campaign': 'elite_path_r20', 'status': 'launched'})
    write_json(work / 'round_ready.json', {'status': 'active'})
    (work / 'round20.exit').write_text('0\n')
    (work / 'generated').mkdir();(work / 'generated/disposable').write_text('generated')
    return repo, work, original


def test_final_retirement_preserves_originals_and_reports(tmp_path):
    repo, work, original = fixture(tmp_path)
    assert finalize(repo, work)['status'] == 'inventory' and (work / 'generated').exists()
    result = finalize(repo, work, apply=True)
    assert result['status'] == 'deleted' and not (work / 'generated').exists()
    assert (original / 'keep').read_text() == 'original'
    assert json.loads((work / 'active.json').read_text())['status'] == 'complete'


def test_final_retirement_refuses_running_inference(tmp_path):
    repo, work, _ = fixture(tmp_path);(work / 'round20.exit').write_text('1\n')
    with pytest.raises(ValueError, match='not complete'): finalize(repo, work, apply=True)
    assert (work / 'generated').exists()
