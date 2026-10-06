import argparse
from evomolsteer.generation.coordinate_conditioning import build

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Dataset -> empirical composition strata and compact support diagnostics')
    for name in ('dataset', 'campaign', 'reference', 'output', 'audit-output'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--min-group-size', type=int, default=3)
    p.add_argument('--min-batches', type=int, default=2)
    p.add_argument('--min-prior-ess', type=float, default=1.5)
    p.add_argument('--max-loo-rms', type=float, default=1.0)
    a = p.parse_args()
    result = build(a.dataset, a.campaign, a.reference, a.output, a.audit_output, a.min_group_size, a.min_batches, a.min_prior_ess, a.max_loo_rms)
    print({k: result[k] for k in ('mean_native_support_fraction', 'n_supported_time_combinations', 'bytes_reference', 'geometry_moment_max_difference')})
