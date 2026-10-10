"""Prepare a distinct retry from compiled reference metadata; preserve failed job."""
import argparse
import json
from pathlib import Path
from evomolsteer.io import digest, read_json, write_json
from evomolsteer.generation.cohort_contract import artifact_path, validate_artifacts


def prepare(repo, spec_path, compiled_path, output, name):
    spec, compiled = read_json(spec_path), read_json(compiled_path)
    if compiled['program'].replace('\\', '/') != spec['program'].replace('\\', '/'):
        raise ValueError('Compiled metadata belongs to another program')
    reference = artifact_path(repo, compiled['reference'])
    if digest(reference) != compiled['reference_sha256']:
        raise ValueError('Compiled reference content changed')
    if output.exists() or name == spec['name']:
        raise ValueError('Distinct retry name and fresh output required')
    retry = {**spec, 'name': name, 'reference': compiled['reference'],
             'reference_sha256': compiled['reference_sha256'],
             'retry_of': spec['name'], 'failed_spec_sha256': digest(spec_path),
             'compiled_metadata_sha256': digest(compiled_path)}
    validate_artifacts(repo, retry)
    write_json(output, retry)
    return retry


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ['repo', 'spec', 'compiled', 'output', 'name']:
        parser.add_argument('--' + flag, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(Path(args.repo), Path(args.spec), Path(args.compiled), Path(args.output), args.name)))
