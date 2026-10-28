---
name: guidance-experiment-optimizer
description: Use measured molecular generation history to propose or select legal guidance-weight experiments, test geometric hypotheses, and update interval allocation. Use when a controller supplies experiment evidence and an action contract; execution and iteration budgets remain external.
metadata:
  version: "1.5.0"
---

# Guidance experiment optimizer

<!-- runtime:shared:start -->
## Objective and authority

Apply `condition-preservation-design-v1` and `history-adaptive-search-v2` to molecular structure design. Prioritize final relative geometry among conditioned atoms, measured by **controlled-atom pair distance root mean square deviation**. Among comparable evaluated configurations, favor its overall mean improvement within explicitly supplied deterioration limits; seed consistency is diagnostic, not a separate ranking objective. Use full metric names while preserving exact schema fields and identifiers.

Assess **controlled-to-neighborhood distance root mean square deviation** using its actual reference or allowed-distance definition. The generative model can redesign the surrounding ligand. A change from a teacher conformer's neighborhood is not itself damage; distinguish it from measured bond distortion, clashes, disconnection or loss of conditioned geometry. Appropriate separation does not mean shrinking every distance or reconstructing the entire conformer. Preserve mappings, metric definitions and bonded/nonbonded distinctions; missing measurements remain unknown.

Choose legal experiments as an outer planner; the executor computes gradients, trajectories and complete weight arrays. Preserve the model, bound loss, current-state vector-Jacobian product, integration, inpainting/output policy, teacher-specific calibration, random streams and budget. Distinguish a recorded aggregate-metric winner from the condition-preservation preference if the controller ranks differently. Instructions do not change numerical ranking or authorize execution. Total iterations remain external.

## Evidence-to-action cycle

1. **Reconcile history.** Learn from all supplied compatible completed results, including earlier iterations and branches, failures and counterexamples. The preferred working context is a global coverage overview, key contrasts around the current anchor, and counterexamples/risk records with source identifiers. Keep the full ledger outside that context. Report coverage; do not call selected evidence complete or assume an omitted configuration is untested. Retrieve a needed source through the supplied mechanism or flag the missing evidence.
2. **Update the hypothesis.** Match original predictions and falsifiers to executed results; mark each supported, contradicted or unresolved. Keep the lesson scoped to teacher, anchor and affected intervals. Never rewrite a prediction after seeing its outcome.
3. **Choose the experiment.** Favor measured joint-configuration performance when interval responses conflict; decomposition of every interval effect is not a prerequisite. State the expected joint effect without claiming an isolated causal mechanism. A targeted contrast remains useful when it changes the next decision or diagnoses a risk.
4. **Register value and prediction.** Name the measured anchor, relevant evidence, purpose (expected benefit or useful discrimination), expected metric directions and an observable falsifier. An informative experiment may temporarily worsen metrics within supplied constraints; disclose the tradeoff. This permission does not replace the retained best checkpoint with a worse result.
5. **Close the loop.** Check proposal versus executed configuration, compare matched populations and update the allocation lesson. Equivalent configurations on the same random streams reuse one measurement; a new name or parent does not create new evidence.

## Decision precedence

Apply legality, compatibility and evidence validity first. Extreme displacement or structural failure requires diagnosis and a supported weaker or more robust allocation before mean gains, even if the mean improved. Keep failures; do not add retries, samples or invented numerical limits.

When the external controller reports the confirmed consecutive-regression trigger, restore the best compatible checkpoint as the strategy basis and propose a new, untested allocation strategy informed by subsequent failures. This recovery takes precedence over ordinary local continuation. Keep the full history, outcomes, budget, frozen calibration and random-stream contract; do not rerun the best configuration or restore an old ledger. Count comparable completed round outcomes supplied by the controller, not individual bad candidates, request failures, missing results, reuse or abstention. If the round comparison or best-checkpoint identity is absent, expose the gap rather than inventing a trigger or performing a destructive reset.

Otherwise follow supported local gains, narrow unsuccessful interventions with smaller legal steps or fewer changed intervals, and move when useful local options are exhausted. Prefer another compatible measured anchor that tests a different allocation hypothesis while considering conditioned geometry and risk, rather than choosing only the next numerical score. At the minimum legal step, change the contrast or anchor instead of inventing fractions or repeating a measured configuration.

If compatible distinct evidence establishes a primary plateau under the externally supplied criterion, emphasize neighborhood relationships while protecting attained conditioned geometry. This changes search focus, not loss weights or the priority of the condition. Missing plateau criteria or deterioration bounds remain unresolved; do not certify an admissible transition from an example.

Recovery never overrides the latest confirmed human policy, scientific compatibility or action validators with an older checkpoint's instructions. On an established primary plateau, compare mean neighborhood gains within primary-geometry protection limits; do not require a new primary-metric decrease for every secondary-focus improvement.

## Search allocation and interface limits

Retain `measured-search-policy-v1`: where external staging is supported, screen more distinct meaningful configurations, then selectively confirm. Smaller-population evidence prioritizes confirmation; only matching required populations can replace a fully evaluated best. Stage counts, promotion and budget remain external. In a fixed-population workflow, choose distinct useful proposals within that contract.

Retain `branch-evidence-abstention-v1`: if a branch has neither supported benefit nor useful discrimination, prefer abstaining with the missing-evidence reason while supported branches continue. Unmeasured alone does not mean uninformative. Use only an explicitly supported abstention representation. A fixed-count interface cannot enforce this preference: disclose the gap without fabricating value, candidates or skip fields. Do not use retrospective mode as an execution bypass.

Continue the fixed evaluation panel as chosen by the human. Limit conclusions to that panel; repeated search is not independent validation or demonstrated generalization. Do not reserve new validation samples by default. Report supplied seed variation and tails, but do not infer seed outcomes from aggregate means or silently substitute a worst-case objective.

## Concise decision record

Return the exact role schema: decisive evidence identifiers, updated hypothesis, legal intervention, predicted observations and falsifier within existing fields. Use at most two decisive comparisons in the explanation, not as a limit on evidence inspected. Avoid copied history, duplicated role analysis, sampler derivations and step-by-step deliberation. A short visible answer does not guarantee bounded internal reasoning. Confirmed human choices govern within scope; examples and recommendations are not permission. Report a material missing rule to the supervisor rather than inventing thresholds or unsupported operations.
<!-- runtime:shared:end -->

## Load the relevant guidance

- For every decision, read [history-learning.md](references/history-learning.md) and [control-diagnostics.md](references/control-diagnostics.md).
- For a private proposal or private retrospective, also read [proposal.md](references/proposal.md).
- For selection or selector retrospective, instead read [synthesis.md](references/synthesis.md).
- For source provenance and limits of the borrowed methods, read [sources.md](references/sources.md) when maintaining or reviewing this skill. Source literature is not required input for every experiment call.
- When the human asks to teach decision preferences, the supervising agent uses [preference-elicitation.md](references/preference-elicitation.md). The inference agent consumes the confirmed policy supplied by its controller; it does not assume access to a human-question tool.
- For the accepted answers, scope and unresolved numerical parameters behind `condition-preservation-design-v1`, read [confirmed-design-policy.md](references/confirmed-design-policy.md). The essential decision rules are included in the shared runtime section above.
- When reviewing search-resource preferences, read [confirmed-search-policy.md](references/confirmed-search-policy.md). It consolidates all confirmed search decisions, the externally configured recovery threshold and executor capabilities; detailed interviews are not inference input.
- When auditing abnormal decision branches, read [decision-boundaries.md](references/decision-boundaries.md). It records rule precedence, conflict resolutions and required interface changes.

## Repository integration

`scripts/render_runtime.py --output-dir <skills-directory>` renders the shared section plus history/control guidance into `analysis.md`, and the corresponding role reference into `proposal.md` or `synthesis.md`. It writes `BUNDLE.json` with source and output hashes. Run the same command with `--check` to detect drift without writing. The existing prompt router then loads shared analysis plus only the requested role. This makes the referenced knowledge available even to an application model without filesystem tools.

This package defines instructions, not new metrics, an experiment database or a replacement optimizer. Do not use rendering as permission to overwrite a frozen run. Current controller limitations and validation evidence are documented in the repository's version delivery note.
