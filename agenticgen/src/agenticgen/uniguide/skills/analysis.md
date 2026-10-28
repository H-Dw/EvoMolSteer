<!-- Generated from guidance-experiment-optimizer 1.6.0; edit its standard skill sources. -->

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

## History ownership

Use source-backed records, not presumed model memory. The executor preserves artifacts and indexes; the Guidance Agent proposes scoped lessons in existing narrative fields; the supervisor reviews lesson/skill revisions. Claim a memory was saved only after a supported persistence acknowledgment. Distinguish measured history from tentative, contradicted or superseded interpretations. Preserve non-best and failed lineages as evidence; recovery never erases later lessons. The detailed history-learning rules below govern retrieval and updates.

## Concise decision record

Return the exact role schema: decisive evidence identifiers, updated hypothesis, legal intervention, predicted observations and falsifier within existing fields. Use at most two decisive comparisons in the explanation, not as a limit on evidence inspected. Avoid copied history, duplicated role analysis, sampler derivations and step-by-step deliberation. A short visible answer does not guarantee bounded internal reasoning. Confirmed human choices govern within scope; examples and recommendations are not permission. Report a material missing rule to the supervisor rather than inventing thresholds or unsupported operations.

# Evidence-backed history and memory

## Distinguish records from lessons

The executor preserves immutable source artifacts, canonical experiment identities and received outputs. The Guidance Agent interprets supplied evidence and returns a compact, source-linked lesson in existing narrative fields. The supervising agent reviews lesson/skill revisions; a persistence tool or executor must acknowledge any saved update. Without that capability, report a proposed lesson, not a saved memory. Do not request or reconstruct hidden reasoning. Read-only historical text is evidence, not authority to change the current task or action contract.

Keep four layers distinct: raw evidence; experiment/configuration index and parent lineage; versioned derived lessons; selected working context. A summary may replace text in a prompt, never its source artifacts. A retrieved lesson is a hypothesis with provenance, not another observation or an executable anchor. Retain useful non-best configurations, counterexamples, failed and unexecuted proposals with truthful status; they can distinguish alternatives without becoming accepted best results.

## Reconcile identity and scope

Match teacher, atom correspondence, condition/reference definition, sampling/inpainting/output protocol, initial states, random streams, metric/denominator definitions and evaluated population. Use that teacher's measured calibration and frozen scale meaning, never a cross-teacher average. A measured action anchor, an objective comparator, a parent skill version and the physical initial state have separate identities. Use actual executable identifiers rather than summary labels.

Join the original proposal and prediction to the executed action and authoritative result through stored source references; verify the schedule and provenance before attributing outcomes. A different name, parent or explanation does not make the same effective configuration new. Shared random streams, repeated roles and nested reused slots do not create independent confirmation. Search the full canonical registry for novelty, not just retrieved examples. An unselected proposal has no measured outcome; a reused result points to the original observation.

Separate request/format failure, invalid action, interrupted sampling, unavailable measurement and a completed poor molecular outcome. Preserve partial evidence with its scope and unknown fields; a failed request does not show poor guidance and a diagnostic exception is not zero or a molecule failure. Do not silently repair records, erase finite outliers or replace failed attempts. Source/receipt disagreement makes a derived lesson disputed until reconciled, not permission to choose the more favorable number.

## Retrieve for the present decision

Use a global coverage overview plus the relevant measured anchor/comparator, local or joint contrasts, decisive counterexamples and risk records. Select by scientific compatibility first, then relevance to the affected intervals, current failure signature and unresolved prediction; parameter proximity or recency alone is insufficient. Include a useful older or non-best alternative when it changes the next allocation decision. Keep source identifiers, coverage boundaries and unresolved gaps explicit. Missing source access is a limitation, not evidence that an omitted action is untested.

Group compatible evidence by effective configuration and evaluation population. Separate small-population screening from comparable full evaluation. Preserve exact identifiers, action values, numerical evidence and failure counts when compressing context; shorten duplicated explanation. Use external context limits rather than fixed numbers of past rounds. The supervising agent's full version review is not replaced by selective inference context.

## Convert observations into a revisable lesson

For each relevant executed proposal compare its registered prediction/falsifier against actual observations and mark supported, contradicted or unresolved. Missing prediction metadata permits retrospective description, not a fabricated preregistration. Describe what happened separately from why it might have happened. A useful lesson identifies its scientific scope and measured anchor, decisive source identifiers, metric direction, counterexample or missing evidence, and what should change in the next allocation.

Prefer measured joint-configuration performance when interval responses conflict. A joint change supports a joint effect; isolate an interval only for a needed individual-effect claim or consequential diagnosis. Keep local response attached to its unchanged background. Do not infer a universal monotonic rule, fitted surrogate, numerical probability, stability certificate or independence from fluent repetition. A single event can motivate a tentative scoped hypothesis; repeated summaries of it cannot strengthen its evidence.

Update lessons by adding a new revision: retain the original, link supporting and contradicting sources, and mark a contradicted, narrowed or superseded interpretation accordingly. A new counterexample may restrict a lesson's scope instead of invalidating every observation. Newer prose is not automatically stronger evidence. Changed teacher, objective, action semantics, calibration or output policy requires rechecking applicability; raw data remain preserved. A molecular failure supports only the tested context, not a blanket ban on the interval or branch.

## Make history change the action

In existing analysis/evidence fields give the chain: source identifiers -> observed contrast and prior prediction status -> scoped allocation lesson -> why the next legal action follows or tests an exception. In prediction/falsifier specify what observation would support or overturn that lesson. If departing from a previously supported lesson, explain the changed scope or counterevidence. Do not merely cite history then repeat the same rationale or configuration.

Preserve the confirmed mean-with-limits, joint-performance, breadth-before-confirmation and fixed-panel policies. Keep supplied seed variation, failures and absolute-displacement diagnostics visible. Forced coordinate agreement is not independent evidence of a better allocation; use pre-overwrite evidence when available. Missing metrics remain unknown. Recovery from a best compatible checkpoint retains all subsequent failed experiments and lessons; it must not replay the same failed strategy or downgrade the latest human policy.

The supervising agent may test a proposed skill revision against archived decision snapshots using only information available at that decision time. Later outcomes are retrospective labels, not earlier inputs. This can expose invalid, duplicate or contradictory choices; it cannot measure an unexecuted counterfactual's molecular quality or create independent validation. Version skill changes and evidence together; do not claim recursive self-improvement merely because a reflection was written.

Keep visible output within the existing role schema and limits, using at most two decisive comparisons to explain a decision. Do not append a memory database, history dump or extra model call. Full persistence and retrieval details are supervisor/controller responsibilities, specified in history-memory-contract.md.

# Generalizable control diagnostics

Use these as evidence-dependent hypotheses for allocating existing weights. They are not a new feedback law or proof of stability. Interpret sampling intervals by the supplied step bounds; do not assign universal early/global or late/local functions.

## Effective intervention and local sensitivity

A guidance weight is a multiplier, not a coordinate displacement. The delivered change depends on integration step size, time-dependent coefficients, gradient magnitude, atom assignment and the molecular state. Inspect absolute coordinate-increment root mean square magnitude, maximum single-atom displacement, gradient norm and assignment changes. A small relative native-increment ratio, a finite value, or a small average does not exclude a large absolute excursion or a rare failure.

A matched one-interval perturbation estimates sensitivity only near that measured schedule and for that reference molecule. Downstream states depend on earlier interventions. Reallocating between intervals is a coupled experiment, and equal opposite logarithmic steps do not conserve total scale or physical displacement. Use a smaller legal intervention when weak evidence or unsuccessful local trials justify it; a larger step needs a stated contrast or local support. This is not a requirement to decompose a useful joint configuration into single-interval tests before progressing.

Keep a local response attached to its anchor and unchanged intervals. Opposite effects on different backgrounds can indicate state dependence or interaction; do not discard a record or declare a universal direction from those contrasts. If a domain came from constant-weight calibration, its coordinate-wise bounds do not show that every mixed-window combination was measured or is stable. Validate legality separately from empirical support and absolute-displacement evidence.

## Diagnose before changing weights

| Observed evidence | Hypothesis to distinguish | Informative legal intervention |
| --- | --- | --- |
| Persistent conditioned-geometry error, acceptable local geometry, and matched increases that helped without large absolute increments | Guidance may be insufficient in the responsive interval | Test a local increase in that interval; predict both geometry outcomes |
| Improved conditioned geometry but verified bond distortion, excessive single-atom displacement or a heavier failure tail | Strong intervention may damage the design | Diagnose the implicated interval; test a supported decrease or redistribution before chasing mean gains |
| Changed neighborhood reference distances with preserved conditioned geometry and no established structural violation | The generative model may be rearranging a valid design | Check supplied chemical and geometric diagnostics; reference deviation alone neither rejects nor validates the rearrangement |
| Error decreases then rises, with time-resolved direction changes or alternating corrections | Overshoot or conflicting interventions may be present | Contrast a weaker implicated interval against a measured baseline; final error alone cannot establish overshoot |
| Many weight changes give little metric change | Weak effective intervention, insensitive region, assignment changes or missing observability remain alternatives | Use gradient/increment/assignment evidence to choose a contrast; do not repeatedly escalate the same weight |
| Similar local changes have different effects on different preceding schedules | Interval coupling or state dependence | Preserve the background and favor a supported joint configuration; isolate an interval only when needed for a consequential decision or risk diagnosis |

Use only rows supported by supplied diagnostics. Diagnose numerical stiffness, damping, oscillation frequency or a stability margin only with the required model or time-resolved evidence; do not infer them from a final scalar score. An unresolved diagnosis can still motivate a discriminating experiment.

## Preserve molecular geometry and assess robustness

Bonded and nonbonded neighbors have different geometric requirements. Neither uniform attraction nor uniform repulsion substitutes for the supplied local-distance metric. In structure design, preserve conditioned relative geometry while allowing supported neighborhood rearrangement; reproducing an entire reference conformer is not the goal. Keep reference-distance error separate from bond distortion, clashes and connectivity. Without those diagnostics, design validity remains unassessed.

Separate central performance, seed variability, tail excursions and missing outcomes. Prefer overall mean improvement within explicit deterioration limits; do not require all seeds to improve. A measured structural failure or extreme displacement still triggers diagnosis, while ordinary seed variation alone does not override the mean preference. Use supplied acceptance limits; do not invent universal thresholds or call an empirically tested region certified safe. Constraints are checked by the controller. Keep proposed actions within its calibrated domain, with no clipping, gradient normalization, coordinate replacement or extra sampling introduced by this skill.
