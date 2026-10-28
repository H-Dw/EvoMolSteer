# Confirmed structure-design decision policy

Policy identifier: `condition-preservation-design-v1`. Source: the human's four answers to `guidance-decisions-20261028`, received 2026-10-28. The interview and verbatim answers are recorded in `agenticgen/docs/designs/20261028_guidance_decision_interview/interview.json` in this repository. This policy covers the current molecular structure-design workflow, not a universal optimization preference.

## Human-confirmed priorities

- Prioritize improvement of distances among conditioned atoms. The human's stated reason is that their final relative spatial arrangement determines whether the conditioning information is retained. Neighborhood deterioration is acceptable only within an explicitly established limit.
- If compatible completed experiments repeatedly fail to produce meaningful improvement in the conditioned-atom distance loss, give greater attention to distances between the conditioned region and the other ligand atoms. Retain achieved condition preservation. The human distinguished structure design from structure prediction: the generative model may optimize and rearrange the surrounding ligand.
- Follow supported local benefits; after unsuccessful tests narrow the intervention, and move to another measured anchor when informative local actions are exhausted.
- For extreme displacement or structural failure, diagnose the cause and prioritize weaker implicated intervals or a more robust allocation within the existing budget before pursuing average gains. Preserve failure records and do not implicitly replace samples.

## Operational interpretation

Use the supplied controlled-atom pair distance root mean square deviation as evidence of conditioned internal geometry, with available per-pair and tail diagnostics. It is not proof of chemistry, chirality or absolute pose. Do not substitute a lower aggregate error for preserved conditioned geometry.

Evaluate neighborhood adaptation against its declared geometric and chemical requirements. A larger distance error against one reference conformer can reflect allowed redesign; it is neither automatic damage nor automatic improvement. When only reference-distance error is measured, record what it measures and what remains unassessed. Do not silently redefine the reference, replace its metric, change the loss or remove a constraint.

The secondary phase changes which legal weight experiments are informative. It does not authorize modifying loss components, moving conditioned atoms out of their accepted geometry, adding samples or changing calibrated multipliers. Risk diagnosis takes precedence over chasing an average gain in either phase.

## Search-policy amendment

The later human answer recorded in [confirmed-search-policy.md](confirmed-search-policy.md) permits temporary exploratory deterioration and specifies recovery after two consecutive deteriorating rounds. Condition preservation remains the priority for accepting an incumbent, not a demand that every experiment succeed. Existing hard constraints and risk diagnosis still apply. The recovery trigger is an externally supplied policy parameter, distinct from total iteration count and plateau detection. Do not reinstate the old option label's prohibition on expected deterioration.

## Unresolved parameters and implementation boundary

No numerical neighborhood-deterioration limit, primary-geometry protection tolerance, plateau evidence span or minimum meaningful improvement was supplied. No particular universal threshold or teacher-specific threshold formula was endorsed. Keep these external and unresolved; apply each teacher's existing calibration and comparison conditions without pooling them. The qualitative policy is scoped to this workflow; do not infer new numerical permission from it.

Until a plateau criterion is supplied, describe suspected stagnation with actual distinct configuration contrasts and uncertainty rather than certifying a phase transition. Until acceptance limits are supplied, distinguish a directional preference from a configuration judged admissible. Deterministic duplicates, API failures and a single unsuccessful experiment do not establish a scientific plateau.

At adoption, inspect the controller's actual ranking and loss. The inspected internal-distance controller ranks all constrained pairs together and measures neighborhood deviation from fixed reference distances. These differ from a conditional-geometry-first ranking and a design-aware neighborhood acceptability test. Prompt changes record and guide the preference but do not implement those numerical changes. Report the gap and retain the frozen protocol until corresponding implementation and validation exist; this policy is not authorization to resume paused experiments.

## Counterexamples for the supervising agent

- A candidate with lower controlled-atom pair distance root mean square deviation but somewhat higher neighborhood reference-distance error is directionally preferred only subject to the supplied neighborhood bound and risk evidence. Missing bounds do not justify saying it passed.
- On an established primary plateau, a neighborhood improvement with a material loss of conditioned geometry violates the protection principle; the secondary objective cannot erase it.
- A changed neighborhood with valid supplied geometry diagnostics and preserved conditioned geometry must not be rejected solely because it differs from the teacher conformer. Missing diagnostics also prevent declaring it valid.
- A tail failure despite a favorable mean triggers diagnosis; it must not be averaged away or reused as evidence that a larger step is safe.
