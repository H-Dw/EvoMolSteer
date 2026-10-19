"""Deterministic reading limits for exploratory local-coordinate contrasts."""
import math


def contrast_reading(evidence):
    """Expose what each interval supports without changing frozen mining output.

    These are unadjusted, descriptive intervals. A positive local association
    does not establish information beyond the global-cloud comparator, a
    causal affinity mechanism, or successful transfer to an endpoint reward.
    """
    contrast = evidence.get('terminal_contrast', {})
    readings = {}
    for feature, row in contrast.get('features', {}).items():
        interval = row.get('cluster_interval_95')
        if interval is None:
            status = 'unavailable'
        else:
            if (len(interval) != 2 or not all(math.isfinite(v) for v in interval)
                    or interval[0] > interval[1]):
                raise ValueError('Invalid exploratory contrast interval: ' + feature)
            lo, hi = interval
            status = 'positive' if lo > 0 else 'negative' if hi < 0 else 'includes_zero'
        readings[feature] = {'interval_status': status,
            'positive_error_contrast_supported': feature.endswith('error_A') and status == 'positive'}
    increment = readings.get('beyond_global_error_A', {}).get('positive_error_contrast_supported', False)
    return {'features': readings,
        'positive_beyond_global_contrast_supported': increment,
        'beyond_global_claim': ('Positive descriptive increment at the unadjusted interval level; not causal or held-out evidence.'
            if increment else 'Positive increment beyond global similarity is not established by the supplied interval.'),
        'limits': 'Local positive intervals cannot replace the beyond-global interval. None of these intervals establishes final affinity gain under guidance.'}
