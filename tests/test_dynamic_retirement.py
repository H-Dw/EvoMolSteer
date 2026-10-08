import sys
from pathlib import Path

import pytest

from evomolsteer.io import digest, read_json, write_json

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from finalize_dynamic_campaign import finalize


def fixture(tmp_path):
    repo = tmp_path / 'MolSteer/EvoMolSteer'; repo.mkdir(parents=True)
    flowr = repo.parent / 'flowr_root'
    original = flowr / 'experiments/ck2_clk3_lineage_20261003'; original.mkdir(parents=True)
    (original / 'protected').write_text('original seeds')
    work = flowr / 'experiments/new_contrast'; work.mkdir()
    (work / 'generated').mkdir(); (work / 'generated/payload').write_text('retire this')
    write_json(work / 'active.json', {'round': 10, 'campaign': 'dynamic_contrast_r10'})
    (work / 'round10.exit').write_text('0')
    write_json(work / 'round_ready.json', {'status': 'active'})
    write_json(repo / 'configs/experiments/dynamic_contrast10_v1/campaign.json',
               {'maximum_rounds': 10, 'rounds_completed': 10,
                'rounds': [{'round': n, 'status': 'complete'} for n in range(1, 11)]})
    for n in range(1, 11):
        f = repo / f'docs/experiments/dynamic_contrast10_20261008/round{n:02d}/score.json'
        write_json(f, {'score': 8.})
        write_json(f.parent / 'retention.json', {'status': 'complete', 'campaign': f'dynamic_contrast_r{n:02d}',
                   'files': [{'path': f.name, 'sha256': digest(f)}]})
    return repo, work, original


def test_final_retirement_preserves_original_and_reports(tmp_path):
    repo, work, original = fixture(tmp_path)
    audit = finalize(repo, work, apply=True)
    assert audit['status'] == 'deleted' and not (work / 'generated').exists()
    assert (original / 'protected').read_text() == 'original seeds'
    assert Path(audit['surviving_report']).exists()
    assert read_json(work / 'active.json')['generated_outputs_retired']


def test_final_retirement_requires_every_retained_report(tmp_path):
    repo, work, original = fixture(tmp_path)
    f = repo / 'docs/experiments/dynamic_contrast10_20261008/round01/score.json'
    write_json(f, {'score': 99.})
    with pytest.raises(ValueError): finalize(repo, work, apply=True)
    assert (work / 'generated/payload').exists()
    assert (original / 'protected').exists()


def test_dry_run_and_protected_original_abort(tmp_path):
    repo, work, original = fixture(tmp_path)
    assert finalize(repo, work)['status'] == 'inventory'
    assert (work / 'generated/payload').exists()
    with pytest.raises((ValueError, FileNotFoundError)): finalize(repo, original, apply=True)
    assert (original / 'protected').exists()
