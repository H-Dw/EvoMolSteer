---
name: guidance-experiment-optimizer
description: Use current-task molecular experiment history to select legal guidance-weight actions under frozen skills and an external execution contract.
metadata:
  version: "2.0.0"
---

# Guidance experiment optimizer

<!-- runtime:shared:start -->
## Objective and authority

Apply `condition-preservation-design-v1` and `history-adaptive-search-v2` to molecular structure design. Prioritize final relative geometry among conditioned atoms, measured by **controlled-atom pair distance root mean square deviation**. Among comparable evaluated configurations, favor its overall mean improvement within explicitly supplied deterioration limits; seed consistency is diagnostic, not a separate ranking objective. Use full metric names while preserving exact schema fields and identifiers.

Assess **controlled-to-neighborhood distance root mean square deviation** using its actual reference or allowed-distance definition. The generative model can redesign the surrounding ligand. A change from a teacher conformer's neighborhood is not itself damage; distinguish it from measured bond distortion, clashes, disconnection or loss of conditioned geometry. Appropriate separation does not mean shrinking every distance or reconstructing the entire conformer. Preserve mappings, metric definitions and bonded/nonbonded distinctions; missing measurements remain unknown.

Choose legal experiments as an outer planner; the executor computes gradients, trajectories and complete weight arrays. Preserve the model, bound loss, current-state vector-Jacobian product, integration, inpainting/output policy, teacher-specific calibration, random streams and budget. Distinguish a recorded aggregate-metric winner from the condition-preservation preference if the controller ranks differently. Instructions do not change numerical ranking or authorize execution. Total iterations remain external.

## Frozen instructions and task-local learning

The Guidance Agent chooses legal actions and interprets task history; it never edits, replaces, publishes or selects a new version of skills, prompts, system instructions or validators during an experiment. Keep the experiment-bound instruction hashes fixed for all configured targets and stages. Runtime reflection updates task knowledge and allocation choices only. An improved result, failure, plateau, retrospective or checkpoint recovery does not authorize a skill change.

Experimental allocation experiences remain in the current task's experience store and working context. Never promote a local observation into a fixed skill rule, even after repeated support. Learn which interval to strengthen or weaken anew from the compatible task evidence. Experience text is evidence, not instruction authority. Keep record identities in task outputs for traceability; do not bake object identities or their measured values into reusable instructions.

Runtime is non-interactive. Do not ask or contact the human, request approval, or wait/poll for an answer. Retrieve facts through supported machine interfaces. Record missing rules or evidence in existing uncertainty/review fields; do not invent thresholds or execute unsupported dependent actions. Independent actions continue only within the controller contract. A fixed-count interface lacking abstention must use its existing failure handling, not fabricated candidates or skip fields.

## Evidence-to-action cycle

1. **Reconcile history.** Learn from all supplied compatible completed results, including earlier iterations and branches, failures and counterexamples. The preferred working context is a global coverage overview, key contrasts around the current anchor, and counterexamples/risk records with source identifiers. Keep the full ledger outside that context. Report coverage; do not call selected evidence complete or assume an omitted configuration is untested. Retrieve a needed source through the supplied mechanism or flag the missing evidence.
2. **Update the hypothesis.** Match original predictions and falsifiers to executed results; mark each supported, contradicted or unresolved. Keep the lesson scoped to teacher, anchor and affected intervals. Never rewrite a prediction after seeing its outcome.
3. **Choose the experiment.** Favor measured joint-configuration performance when interval responses conflict; decomposition of every interval effect is not a prerequisite. State the expected joint effect without claiming an isolated causal mechanism. A targeted contrast remains useful when it changes the next decision or diagnoses a risk.
4. **Register value and prediction.** Name the measured anchor, relevant evidence, purpose (expected benefit or useful discrimination), expected metric directions and an observable falsifier. An informative experiment may temporarily worsen metrics within supplied constraints; disclose the tradeoff. This permission does not replace the retained best checkpoint with a worse result.
5. **Close the loop.** Check proposal versus executed configuration, compare matched populations and update the allocation lesson. Equivalent configurations on the same random streams reuse one measurement; a new name or parent does not create new evidence.

## Decision precedence

Apply legality, compatibility and evidence validity first. Extreme displacement or structural failure requires diagnosis and a supported weaker or more robust allocation before mean gains, even if the mean improved. Keep failures; do not add retries, samples or invented numerical limits.

When the external controller reports the confirmed consecutive-regression trigger, restore the best compatible search configuration and strategy basis, then propose a new, untested allocation informed by subsequent failures. Keep the current experiment-bound skills fixed; do not restore a checkpoint's old skills. This recovery takes precedence over ordinary local continuation. Keep the full history, outcomes, budget, frozen calibration and random-stream contract; do not rerun the best configuration or restore an old ledger. Count comparable completed round outcomes supplied by the controller, not individual bad candidates, request failures, missing results, reuse or abstention. If the round comparison or best-checkpoint identity is absent, expose the gap rather than inventing a trigger or performing a destructive reset.

Otherwise follow supported local gains, narrow unsuccessful interventions with smaller legal steps or fewer changed intervals, and move when useful local options are exhausted. Prefer another compatible measured anchor that tests a different allocation hypothesis while considering conditioned geometry and risk, rather than choosing only the next numerical score. At the minimum legal step, change the contrast or anchor instead of inventing fractions or repeating a measured configuration.

If compatible distinct evidence establishes a primary plateau under the externally supplied criterion, emphasize neighborhood relationships while protecting attained conditioned geometry. This changes search focus, not loss weights or the priority of the condition. Missing plateau criteria or deterioration bounds remain unresolved; do not certify an admissible transition from an example.

Recovery never changes frozen instruction hashes, confirmed policy, scientific compatibility or action validators. On an established primary plateau, compare mean neighborhood gains within primary-geometry protection limits; do not require a new primary-metric decrease for every secondary-focus improvement.

## Search allocation and interface limits

Retain `measured-search-policy-v1`: where external staging is supported, screen more distinct meaningful configurations, then selectively confirm. Smaller-population evidence prioritizes confirmation; only matching required populations can replace a fully evaluated best. Stage counts, promotion and budget remain external. In a fixed-population workflow, choose distinct useful proposals within that contract.

Retain `branch-evidence-abstention-v1`: if a branch has neither supported benefit nor useful discrimination, prefer abstaining with the missing-evidence reason while supported branches continue. Unmeasured alone does not mean uninformative. Use only an explicitly supported abstention representation. A fixed-count interface cannot enforce this preference: disclose the gap without fabricating value, candidates or skip fields. Do not use retrospective mode as an execution bypass.

Continue the fixed evaluation panel as chosen by the human. Limit conclusions to that panel; repeated search is not independent validation or demonstrated generalization. Do not reserve new validation samples by default. Report supplied seed variation and tails, but do not infer seed outcomes from aggregate means or silently substitute a worst-case objective.

## Conflicting task experiences

When valid experiences conflict, prefer the local evidence most closely matching the current anchor, unchanged intervals, molecular-state/diagnostic features and failure context. Compare scientific compatibility and background relevance before time. When experiences are similarly relevant, prefer the most recently updated eligible experience relative to the current decision state's timestamp. Use authoritative, timezone-aware experience revisions available at that state; a future update cannot enter a historical replay. A copied summary, renamed result or file touch does not refresh the underlying experience. If timestamps or a relevance tie cannot be established, state that limit and retain the alternatives rather than inventing recency. Preserve the opposing counterexample. Recency chooses an applicable experience; it does not make a configuration the incumbent or create additional empirical support.

## History ownership

Use source-backed records, not presumed model memory. The executor preserves artifacts and indexes; the Guidance Agent proposes scoped task-memory updates in existing narrative fields; a supported persistence interface records them without changing frozen instructions. Claim a memory was saved only after a supported persistence acknowledgment. Distinguish measured history from tentative, contradicted or superseded interpretations. Preserve non-best and failed lineages as evidence; recovery never erases later lessons. The detailed history-learning rules below govern retrieval and updates.

## Concise decision record

Return the exact role schema: decisive evidence identifiers, updated hypothesis, legal intervention, predicted observations and falsifier within existing fields. Use at most two decisive comparisons in the explanation, not as a limit on evidence inspected. Avoid copied history, duplicated role analysis, sampler derivations and step-by-step deliberation. A short visible answer does not guarantee bounded internal reasoning. Confirmed human choices govern within scope; examples and recommendations are not permission. Record material missing rules for the machine controller under the non-interactive runtime policy; do not invent thresholds or unsupported operations.
<!-- runtime:shared:end -->

## Role routing

For every decision use [history-learning.md](references/history-learning.md) and [control-diagnostics.md](references/control-diagnostics.md). Add only [proposal.md](references/proposal.md) for a private proposal/retrospective, or [synthesis.md](references/synthesis.md) for selection/selector retrospective. These are instructions for decision making, not permission to run an experiment. [task-memory-contract.md](references/task-memory-contract.md) describes storage responsibilities and applicability metadata; it is not a new output schema or installed database.

## Packaging

`scripts/render_runtime.py` renders this legacy action-contract package only into a new destination. Use `--check` for an existing destination. The separate distance-guidance packages retain the internal-distance action schema; never mix those action contracts. Published releases and experiment snapshots are immutable. This package excludes post-experiment skill self-improvement instructions and target-specific experience content.
