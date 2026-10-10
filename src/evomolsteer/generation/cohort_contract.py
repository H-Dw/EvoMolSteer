"""Portable, frozen program/reference contracts for inference job submission."""
from pathlib import Path
from ..io import digest, read_json


def artifact_path(repo, value):
    relative = Path(str(value).replace('\\', '/'))
    target = (repo / relative).resolve()
    if relative.is_absolute() or ':' in str(relative) or not target.is_relative_to(repo.resolve()):
        raise ValueError('Program/reference must remain inside the pinned repository')
    return target


def validate_artifacts(repo, spec):
    program = artifact_path(repo, spec['program'])
    reference = artifact_path(repo, spec['reference'])
    program_hash, reference_hash = digest(program), digest(reference)
    if program_hash != spec['program_sha256'] or reference_hash != spec['reference_sha256']:
        raise ValueError('Frozen program/reference mismatch')
    if read_json(program)['reference_sha256'] != reference_hash:
        raise ValueError('Job reference differs from the reward program reference')
    return program, reference
