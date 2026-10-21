import contextlib
import importlib.util
from pathlib import Path

import pytest

from evomolsteer.adaptive_schedule.io import read, write, sha
from evomolsteer.adaptive_schedule.group_run import SKILLS


@pytest.fixture
def repair_fixture(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1]/'scripts/repair_plip_council_bootstrap.py'
    spec = importlib.util.spec_from_file_location('bootstrap_repair_test', script)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    root, repo = tmp_path/'results', tmp_path/'repo'
    old = {'../deepseek_api.py': 'old-api', 'group_agent.py': 'old-agent', 'flowr.py': 'native'}
    new = dict(old, **{'../deepseek_api.py': 'new-api', 'group_agent.py': 'new-agent'})
    monkeypatch.setattr(module, 'source_fingerprint', lambda: new)
    monkeypatch.setattr(module, 'commit', lambda: 'new-commit')
    monkeypatch.setattr(module, 'repository', lambda: repo)
    monkeypatch.setattr(module, 'lock', lambda root: contextlib.nullcontext())
    monkeypatch.setattr(module.subprocess, 'check_output', lambda *a, **k: b'synthetic reviewed diff')
    write(root/'config.json', {'unchanged': True})
    write(root/'teachers.json', [{'teacher_id': 't1'}, {'teacher_id': 't2'}])
    write(root/'historical_references.json', {'fixed': True})
    write(root/'baselines/unguided.json', {'historical': True})
    write(root/'PREPARED.json', {'source_fingerprint': old,
        'config_sha256': sha(root/'config.json'), 'teachers_sha256': sha(root/'teachers.json')})
    write(root/'ACCEPTANCE_COMPLETE.json', {'source_fingerprint': old, 'recovery_checks': 8})
    skills = {}
    for role, relative in SKILLS.items():
        for p in [repo/relative, root/f'versions/v1/{role}_SKILL.md']:
            p.parent.mkdir(parents=True, exist_ok=True); p.write_text('frozen skill', encoding='utf-8')
        skills[role] = sha(repo/relative)
    write(root/'versions/v1/binding.json', {'source_fingerprint': old, 'skill_hashes': skills})
    write(root/'versions/v1/deployment.json', {'commit': 'old-commit'})
    for teacher in ['t1', 't2']:
        r0 = root/'shared_r0'/teacher
        write(r0/'COMPLETE.json', {'attempts': 100})
        write(r0/'quality_records.json', {'fixed': teacher})
        write(r0/'QUALITY_COMPLETE.json', {'attempts': 100, 'records_sha256': sha(r0/'quality_records.json')})
    write(root/'versions/v1/t1/bootstrap/old.request.json', {'preserve': True})
    log = root/'tests.log'; log.write_text('synthetic passing tests')
    receipt = root/'tests.json'
    write(receipt, {'exit_code': 0, 'commit': 'new-commit', 'log': str(log), 'log_sha256': sha(log)})
    return module, root, receipt, new


def test_repair_preserves_r0_and_archives_actual_bootstrap_inputs(repair_fixture):
    module, root, receipt, _ = repair_fixture
    before = module.inventory(root)
    request = root/'versions/v1/t1/bootstrap/old.request.json'
    request_hash = sha(request)
    assert module.repair(root, receipt)['eligible'] and request.exists()
    result = module.repair(root, receipt, apply=True)
    assert module.inventory(root) == before and result['protected_hashes_equal']
    archived = list((request.parent/'saved_attempts').glob('*/old.request.json'))
    assert len(archived) == 1 and sha(archived[0]) == request_hash
    assert not request.exists()
    assert read(Path(result['operation'])/'journal.json')['phase'] == 'complete'


def test_repair_cannot_adopt_a_changed_native_sampler(repair_fixture):
    module, root, receipt, new = repair_fixture
    new['flowr.py'] = 'different'
    with pytest.raises(ValueError, match='exactly the two'):
        module.repair(root, receipt, apply=True)
    assert not (root/'operations').exists()


def test_repair_cannot_reset_already_started_hypotheses(repair_fixture):
    module, root, receipt, _ = repair_fixture
    write(root/'versions/v1/pipeline.json', {'started': True})
    with pytest.raises(ValueError, match='before R1'):
        module.repair(root, receipt, apply=True)
