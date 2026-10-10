"""Read one evidence family without mixing inherited or conditional effects."""
import math
from pathlib import Path

from ..io import digest, read_json, write_json

PREFIXES = {'geometry': 'outcome/geometry/', 'matched_geometry': 'outcome/matched/'}


def summarize(evidence, scope, output):
    prefix = PREFIXES[scope]
    packet = read_json(evidence)
    rows = [v for v in packet['evidence_items'] if v['id'].startswith(prefix)]
    if not rows or len({v['id'] for v in rows}) != len(rows):
        raise ValueError('A nonempty, uniquely identified feature scope is required')
    for row in rows:
        for key in ('p', 'q'):
            value = row.get(key)
            if value is not None and (not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError('Finite bounded p/q values or explicit missingness required')
    qs = [v['q'] for v in rows if v.get('q') is not None]
    result = {'schema_version': 'outcome-evidence-scope-1.0', 'scope': scope,
        'prefix': prefix, 'feature_count': len(rows), 'minimum_q': min(qs) if qs else None,
        'corrected_significant_n': sum(q < .05 for q in qs),
        'features': rows, 'evidence_sha256': digest(evidence), 'source_sha256': digest(__file__),
        'limitation': 'Selected evidence family only; association and temporal derivatives do not establish a causal coordinate force'}
    if scope == 'matched_geometry':
        support = [v for v in packet['evidence_items'] if v['id'] == 'outcome/matched_support']
        if len(support) != 1:
            raise ValueError('Matching denominator/support record required')
        result['support'] = support[0]
    if Path(output).exists():
        raise FileExistsError(output)
    write_json(output, result)
    return result
