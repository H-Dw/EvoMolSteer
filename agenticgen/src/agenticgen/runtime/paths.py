"""Installed resources, deployment checkout and host paths have distinct identities."""
import json
import os
from importlib.resources import files
from pathlib import Path, PurePosixPath, PureWindowsPath


def resource(*parts):
    return Path(str(files('agenticgen').joinpath('resources', *parts)))


def skill(name):
    return resource('skills', name, 'SKILL.md')


def repository():
    explicit = os.environ.get('AGENTICGEN_REPOSITORY')
    candidates = [Path(explicit).resolve()] if explicit else list(Path(__file__).resolve().parents)
    for candidate in candidates:
        if (candidate / '.git').exists():
            return candidate
    raise RuntimeError('Review/deployment requires AGENTICGEN_REPOSITORY pointing to its exact Git checkout')


def host_config(path=None):
    path = Path(path or os.environ.get('AGENTICGEN_HOST_CONFIG') or resource('config', 'scnet11762.json')).resolve()
    data = json.loads(path.read_text(encoding='utf-8'))
    for key in ('flowr', 'history', 'examples', 'python', 'activation', 'result_root', 'previous_protocol_root'):
        if key in data:
            value = data[key]
            if not (PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute()):
                data[key] = str((path.parent/value).resolve())
    return data
