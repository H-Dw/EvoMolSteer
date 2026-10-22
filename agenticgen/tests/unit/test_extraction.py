import ast
import json
import os
from pathlib import Path
import subprocess
import sys

import agenticgen
from agenticgen.runtime.paths import skill, host_config, repository


def test_package_has_no_old_project_imports():
    root = Path(agenticgen.__file__).parent
    for path in root.rglob('*.py'):
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or '').startswith('evomolsteer'), path
            elif isinstance(node, ast.Import):
                assert all(not alias.name.startswith('evomolsteer') for alias in node.names), path


def test_native_scoring_and_reward_functions_are_mechanically_preserved():
    repo = repository()
    for original, migrated, names in [
        ('src/evomolsteer/generation/controller.py', 'runtime/scoring.py', {'final_metrics', 'nparr'}),
        ('src/evomolsteer/adaptive_schedule/vendor/reward.py', 'adaptive_schedule/vendor/reward.py', None),
        ('src/evomolsteer/generation/rng.py', 'runtime/rng.py', None),
    ]:
        def functions(path):
            return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(path.read_text(encoding='utf-8')).body
                    if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and (names is None or n.name in names)}
        assert functions(repo/original) == functions(Path(agenticgen.__file__).parent/migrated)


def test_entry_and_skills_work_from_unrelated_directory(tmp_path):
    package_parent = Path(agenticgen.__file__).parent.parent
    code = f"import sys; sys.path.insert(0, {str(package_parent)!r}); from agenticgen.runtime.paths import skill; import agenticgen.cli; assert skill('plip-group-analyst').is_file(); assert not any(k.startswith('evomolsteer') for k in sys.modules)"
    subprocess.run([sys.executable, '-I', '-c', code], cwd=tmp_path, check=True)


def test_host_relative_paths_follow_config_not_working_directory(tmp_path):
    config=tmp_path/'host.json'; config.write_text(json.dumps({'flowr':'native','history':'data'}))
    assert host_config(config)['flowr'] == str(tmp_path/'native')
    assert skill('plip-group-synthesizer').is_file()
    assert host_config()['flowr'] == '/opt/MolSteer/flowr_root'
