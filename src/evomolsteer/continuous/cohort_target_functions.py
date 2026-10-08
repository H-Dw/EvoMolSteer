"""Whole-window functions for cohort targets, distinct from contrast functions.

Use compact observed batch moments. Do not fabricate boundary observations,
particle coordinates or a spatial force from a temporal derivative.
"""
import gzip
import json
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import legder, legval

from ..io import digest, write_json
from .dynamic_regions import masked_curve_fit


def fit_targets(reference, output, maximum_degree=3):
    reference = Path(reference)
    ref = json.loads(gzip.decompress(reference.read_bytes()))
    times = np.asarray(ref['times'], float)
    ids = ref['dynamic_cohort']['selected_feature_indices']
    batches = sorted({b for f in ref['frames'] for b in f['dynamic_region']['observed_batches']})
    positions = {b: i for i, b in enumerate(batches)}
    values = np.full((2, len(batches), len(times), len(ids)), np.nan)
    for j, frame in enumerate(ref['frames']):
        if not np.isclose(frame['time'], times[j], atol=1e-8, rtol=0):
            raise ValueError('Cohort moments and node clocks differ')
        d = frame['dynamic_region']
        for role, name in enumerate(['positive_centers', 'negative_centers']):
            if len(d[name]) != len(d['observed_batches']):
                raise ValueError('Observed batch moment coverage differs')
            for b, row in zip(d['observed_batches'], d[name]):
                values[role, positions[b], j] = np.asarray(row)[ids]
    functions = {}
    for role, name in enumerate(['positive', 'negative']):
        functions[name] = {}
        for k, feature_id in enumerate(ids):
            curves = values[role, :, :, k]
            fit = masked_curve_fit(times, curves, max_degree=maximum_degree)
            u = np.array([-1., 1.]); coefficients = fit['legendre_coefficients']
            fitted = legval(u, coefficients)
            rate = legval(u, legder(coefficients)) * 2 / (times[-1]-times[0])
            fit.update(feature_index=feature_id,
                       feature_unit='Angstrom' if ref['features'][feature_id].endswith('softmin') else 'dimensionless shell density',
                       observed_start_mean=float(np.nanmean(curves[:, 0])),
                       observed_end_mean=float(np.nanmean(curves[:, -1])),
                       observed_end_minus_start=float(np.nanmean(curves[:, -1])-np.nanmean(curves[:, 0])),
                       fitted_start=float(fitted[0]), fitted_end=float(fitted[1]),
                       fitted_rate_start=float(rate[0]), fitted_rate_end=float(rate[1]),
                       derivative_semantics='Temporal derivative of a cohort field target, not a coordinate gradient')
            functions[name][ref['features'][feature_id]] = fit
    result = {'schema_version': 'cohort-target-function-1.0', 'reference_sha256': digest(reference),
              'learned_support': ref['window'], 'observed_score_support': [float(times[0]), float(times[-1])],
              'independent_batches': batches, 'selected_feature_indices': ids, 'functions': functions,
              'semantics': 'Descriptive global target curves; not used to tune the frozen inference reward or extrapolate beyond observed nodes.'}
    write_json(output, result)
    return result
