"""Canonical role instructions and explicit, independently ablatable advice."""
from pathlib import Path
import hashlib
import re

PROJECT = Path(__file__).resolve().parents[2]


def modules(role, root=PROJECT):
    path = Path(root)/'skills'/role.lower()/'references/guidance_modules.md'
    if not path.exists():return {}
    body = path.read_text(encoding='utf-8')
    chunks = re.split(r'^## Module: ([a-z_]+)\s*$', body, flags=re.MULTILINE)
    return {chunks[i]: chunks[i+1].strip() for i in range(1, len(chunks), 2)}


def render(role, selected=(), root=PROJECT):
    if role not in ('Analyst', 'Designer'):
        raise ValueError('Unknown role')
    core = (Path(root)/'skills'/role.lower()/'SKILL.md').read_text(encoding='utf-8')
    advice = modules(role, root)
    if len(set(selected)) != len(selected) or set(selected)-set(advice):
        raise ValueError('Unknown or duplicate guidance module')
    text = core.rstrip()
    for name in selected:
        text += '\n\n## Guidance: '+name+'\n\n'+advice[name]
    return text+'\n'


def sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()
