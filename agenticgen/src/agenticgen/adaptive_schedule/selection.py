"""Anchored condition bands, separate functional ranking and auditable regressions.

The tolerance is an ordering convention, never a reward dead zone or a claim of
biological equivalence. All comparisons retain their unrounded observations.
"""
import math

TOLERANCE_A = .02
SENSITIVITY_A = (.01, .05)


def _finite(x):
    return isinstance(x, (float, int)) and not isinstance(x, bool) and math.isfinite(x)


def in_band(value, minimum, tolerance=TOLERANCE_A):
    # Boundary guard only for binary floating representation, not a second tolerance.
    d = value - minimum
    return d <= tolerance or math.isclose(d, tolerance, rel_tol=0, abs_tol=1e-12)


def choose(rows, tolerance=TOLERANCE_A):
    """Input records: id, condition, functional_gap, best. No rolling tie chain."""
    if not _finite(tolerance) or tolerance < 0:
        raise ValueError('Nonnegative finite ranking tolerance required')
    rows = list(rows)
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Ranking identities must be unique')
    missing = [r['id'] for r in rows if not all(_finite(r.get(k)) for k in ('condition', 'functional_gap', 'best'))]
    if not rows or missing:
        return {'status': 'incomplete', 'missing': missing, 'selected': None, 'tolerance_A': tolerance}
    minimum = min(r['condition'] for r in rows)
    band = [r for r in rows if in_band(r['condition'], minimum, tolerance)]
    chosen = min(band, key=lambda r: (r['functional_gap'], r['condition'], -r['best'], r['id']))
    return {'status': 'complete', 'selected': chosen, 'minimum_condition_A': minimum,
            'band_ids': sorted(r['id'] for r in band), 'tolerance_A': tolerance,
            'interpretation': 'Band anchored to this entire pool minimum; raw errors remain primary observations.'}


def adjacent(previous, current, tolerance=TOLERANCE_A):
    if not all(_finite(r.get(k)) for r in (previous, current) for k in ('condition', 'functional_gap')):
        return {'status': 'incomplete', 'regression': None}
    dc = current['condition'] - previous['condition']
    df = current['functional_gap'] - previous['functional_gap']
    within = in_band(abs(dc), 0., tolerance)
    regression = df > 0 if within else dc > 0
    return {'status': 'complete', 'condition_delta_A': dc, 'functional_gap_delta': df,
            'within_condition_ranking_band': within, 'regression': regression,
            'criterion': 'functional_gap' if within else 'condition', 'tolerance_A': tolerance}


def rollback_advice(versions):
    """Completed versions in order; reset after a completed rollback version only."""
    ordered = sorted(versions, key=lambda r: int(r['id'][1:]))
    if ordered and ordered[0]['id'] != 'v6':
        return {'status':'incomplete','reason':'Historical comparison must include the first new-protocol version V6'}
    streak = 0; comparisons = []
    for i, row in enumerate(ordered):
        if i:
            if int(row['id'][1:]) != int(ordered[i-1]['id'][1:]) + 1:
                return {'status': 'incomplete', 'reason': 'Nonconsecutive completed versions'}
            pair = adjacent(ordered[i-1], row)
            comparisons.append(dict(pair, previous=ordered[i-1]['id'], current=row['id']))
            if pair['status'] != 'complete':
                return {'status': 'incomplete', 'comparisons': comparisons}
            streak = streak + 1 if pair['regression'] else 0
        if row.get('rollback_completed'):
            streak = 0
    best = choose(ordered)
    return {'status': best['status'], 'consecutive_regressions': streak, 'required': streak >= 2,
            'restore_version': best['selected']['id'] if best['selected'] else None,
            'historical_best': best, 'comparisons': comparisons}
