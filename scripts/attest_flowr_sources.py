"""Record declared upstream source bytes before/after inference, without models.

Explicit input dataset, output record, and relative source list. This finite
scope attestation does not claim to freeze every dependency or runtime setting.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

from evomolsteer.io import digest, read_json, write_json


def attest(input_dataset, output_record, relative_sources, reference_record=None):
    root, output = Path(input_dataset).resolve(), Path(output_record).resolve()
    if not root.is_dir() or output.is_relative_to(root):
        raise ValueError('An existing input dataset and separate output record are required')
    if output.exists():
        raise FileExistsError('Attestation records cannot be overwritten')
    if not relative_sources or len(set(relative_sources)) != len(relative_sources):
        raise ValueError('Declare a unique nonempty relative source list')
    files = {}
    for name in relative_sources:
        relative = Path(name)
        target = (root / relative).resolve()
        if relative.is_absolute() or not target.is_relative_to(root) or not target.is_file():
            raise ValueError('Each declared source must resolve inside the input dataset')
        files[relative.as_posix()] = {'sha256': digest(target), 'bytes': target.stat().st_size}
    record = {'schema_version': 'flowr-upstream-source-attestation-1.0',
              'input_dataset': str(root), 'files': files,
              'captured_UTC': datetime.now(timezone.utc).isoformat(),
              'script_sha256': digest(__file__), 'argv': sys.argv,
              'scope': 'Only declared upstream source files; no model forward, checkpoint read, or all-dependency freeze'}
    if reference_record is not None:
        prior = read_json(reference_record)
        if (prior.get('schema_version') != record['schema_version'] or
                prior.get('input_dataset') != str(root) or prior.get('files') != files):
            raise ValueError('Declared upstream source changed during inference')
        record.update(reference_sha256=digest(reference_record), source_unchanged=True)
    write_json(output, record)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dataset', required=True)
    parser.add_argument('--output-record', required=True)
    parser.add_argument('--relative-source', action='append', required=True)
    parser.add_argument('--reference-record')
    args = parser.parse_args()
    result = attest(args.input_dataset, args.output_record, args.relative_source, args.reference_record)
    print({'files': len(result['files']), 'source_unchanged': result.get('source_unchanged')})
