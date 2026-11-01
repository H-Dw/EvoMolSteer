"""Bounded local finite-difference validation; no sampling or relaxed tolerances."""
import math

EPSILONS=(.01,.005,.0025,.00125)
TOLERANCE=.05
CONVERGENCE_TOLERANCE=.025


def verify(evaluate, derivative, reference_identity):
    """Require two adjacent local stencils to agree with AD and with each other.

    A coarse stencil estimates a finite secant, not an infinitesimal derivative.
    Keep every probe, and reject single lucky matches, branch changes and
    nonfinite values. Refinement is bounded and cannot alter molecular state.
    """
    if not math.isfinite(derivative) or derivative<=1e-8:
        raise ValueError('A finite positive input-gradient norm is required')
    report=dict(policy='local-step-convergence-v1',tolerance=TOLERANCE,
                convergence_tolerance=CONVERGENCE_TOLERANCE,
                declared_epsilons=list(EPSILONS),probes=[],passed=False)
    for epsilon in EPSILONS:
        positive,negative=evaluate(epsilon),evaluate(-epsilon)
        row=dict(epsilon=epsilon,positive_value=positive['value'],negative_value=negative['value'],
                 local_identity=(positive['identity']==reference_identity==negative['identity']))
        report['probes'].append(row)
        if not all(math.isfinite(v['value']) for v in (positive,negative)):
            report['failure_reason']='nonfinite_perturbed_objective'
            return report
        estimate=(positive['value']-negative['value'])/(2*epsilon)
        if not math.isfinite(estimate):
            report['failure_reason']='nonfinite_numerical_derivative'
            return report
        row.update(estimate=estimate,relative_error=abs(estimate-derivative)/derivative)
        if len(report['probes'])<2:continue
        prior=report['probes'][-2]
        row['adjacent_estimate_difference']=abs(estimate-prior['estimate'])/derivative
        if (row['local_identity'] and prior['local_identity']
            and row['relative_error']<=TOLERANCE and prior['relative_error']<=TOLERANCE
            and row['adjacent_estimate_difference']<=CONVERGENCE_TOLERANCE):
            report.update(passed=True,accepted_epsilons=[prior['epsilon'],epsilon])
            return report
    report['failure_reason']='no_two_adjacent_converged_local_stencils'
    return report
