"""Bind actual prospective comparisons into the next Analyst calculation packet."""
from pathlib import Path
from ..io import digest, read_json, write_json


def summarize(evidence, reports, output):
    out = Path(output)
    if out.exists():
        raise FileExistsError(out)
    packet = read_json(evidence)
    records = []
    for path in reports:
        report = read_json(path)
        if report.get('status') != 'complete' or report.get('seed') != 42:
            raise ValueError('Completed registered seed-42 feedback required')
        records.append({'round': report['round'], 'source_sha256': digest(path),
            'results': report['results'], 'paired_comparisons': report['paired_comparisons'],
            'label_response': report['label_response'],
            'historical_steer_unpaired': report['historical_steer_unpaired'],
            'interpretation': report['interpretation']})
    if len({r['round'] for r in records}) != len(records):
        raise ValueError('Round feedback repeated')
    packet['evidence_items'].append({'id': 'outcome/prospective', 'completed_rounds': sorted(records, key=lambda r:r['round']),
        'semantics': 'Actual full native continuation; adaptive development panels are not independent confirmation'})
    packet['feedback_source_sha256'] = digest(__file__)
    write_json(out/'evidence.json', packet)
    return packet
