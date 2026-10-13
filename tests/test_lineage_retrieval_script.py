from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "run_ck2_scientist_selector.py"
spec = importlib.util.spec_from_file_location("ck2_run_script", SCRIPT)
runner = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(runner)


def test_lock_rejects_concurrent_process_and_releases_after_exit(tmp_path):
    study = tmp_path / "study"
    code = (
        "import runpy,sys,time; "
        "m=runpy.run_path(sys.argv[1]); "
        "ctx=m['exclusive_study_lock'](sys.argv[2]); ctx.__enter__(); "
        "print('LOCKED',flush=True); sys.stdin.readline(); ctx.__exit__(None,None,None)"
    )
    child = subprocess.Popen([sys.executable, "-c", code, str(SCRIPT), str(study)],
                             cwd=REPO, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True)
    try:
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "LOCKED"
        with pytest.raises(RuntimeError, match="already locked"):
            with runner.exclusive_study_lock(study):
                pytest.fail("second process unexpectedly acquired the study lock")
    finally:
        if child.stdin:
            child.stdin.write("release\n")
            child.stdin.flush()
        child.wait(timeout=10)
    with runner.exclusive_study_lock(study):
        assert (study / "runtime" / ".run.lock").exists()


def test_existing_zero_byte_lock_file_is_reusable_and_never_truncated(tmp_path):
    study = tmp_path / "study"
    lock = study / "runtime" / ".run.lock"
    lock.parent.mkdir(parents=True)
    lock.write_bytes(b"")
    with runner.exclusive_study_lock(study):
        assert lock.stat().st_size >= 1
    before = lock.read_bytes()
    with runner.exclusive_study_lock(study):
        assert lock.stat().st_size == len(before)
    assert lock.read_bytes() == before


@pytest.mark.parametrize("args, expected", [
    ([], REPO / "docs" / "experiments" / "ck2_scientist_selector_20261013"),
    (("--study", "relative-study"), REPO / "relative-study"),
    (("--study=relative-study" ,), REPO / "relative-study"),
])
def test_study_path_matches_orchestrator_resolution(args, expected):
    assert runner._study_path(list(args)) == expected.resolve()
