"""Reject failed, skipped or empty remote test evidence before protocol mutation."""
import importlib.util
from pathlib import Path
import pytest

script = Path(__file__).parents[2]/'scripts/start_v8_strategy.py'
spec = importlib.util.spec_from_file_location('strategy_start', script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize('kind', ['failures', 'errors', 'skipped', 'empty'])
def test_incomplete_remote_test_evidence_rejected(tmp_path, kind):
    p = tmp_path/'tests.xml'
    p.write_text('<testsuites><testsuite tests="'+('0' if kind=='empty' else '3')+'" '+
                 ('' if kind=='empty' else kind+'="1"')+'/></testsuites>')
    with pytest.raises(ValueError):
        module.passing_tests(p)


def test_successful_remote_test_evidence(tmp_path):
    p = tmp_path/'tests.xml'
    p.write_text('<testsuites><testsuite tests="3" failures="0" errors="0" skipped="0"/></testsuites>')
    module.passing_tests(p)
