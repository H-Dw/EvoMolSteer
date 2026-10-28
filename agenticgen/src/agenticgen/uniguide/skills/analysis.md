<!-- Generated from guidance-experiment-optimizer 1.5.0; edit its standard skill sources. -->

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

# Learn from measured history

## Establish comparability and lineage

Group evidence by reference molecule and atom mapping, sampling/inpainting/output protocol, initial states, random streams, metric definitions and evaluated population. Do not pool teachers to choose a scale. Use that reference molecule's verified multi-scale initialization and admitted calibration records; a tested boundary value is not a reason to substitute an interior value.

Join `completed_action_evidence.source_group` to measured history and its original proposal. Check the executed schedule against the proposal before reviewing its prediction and falsifier. Keep the action's measured anchor separate from the best supported geometric comparator and from the physical starting state. A label such as `comparison_best` alone does not establish geometric superiority.

Treat configuration identity as the controller's canonical schedule identity plus the scientific comparison conditions. A different candidate name, parent anchor or wording does not make the same configuration new. The same configuration on the same initial states and random streams is one reused evaluation. New random streams would test uncertainty, but require an externally authorized evaluation; do not request extra samples implicitly. Incomplete or failed attempts are not successful duplicates and must not be silently replaced.

Classify a failure by the evidence actually produced: request/format failure, interrupted execution, unavailable measurement, or a completed poor molecular outcome. An empty model answer says nothing about that schedule's molecular performance; an evaluator exception is not a measured zero or molecular failure. An incomplete run may still contain valid partial observations: retain their exact scope without imputing the missing outcomes. Unknown causes remain unknown, and classification does not authorize retries.

## Working context and full evidence

Prefer a coverage overview, contrasts relevant to the current measured anchor, and counterexamples/risk records with retrievable source identifiers. Retain complete evidence and canonical configuration identities in the external ledger. Do not drop contrary records merely to shorten the input, call selected evidence complete, or infer novelty from absence in the working context. A missing source needed for a decision is an explicit evidence gap. The supervising agent's required full version review remains separate from this compact inference context.

## Make a falsifiable allocation hypothesis

Use matched comparisons of the relevant interval or joint configuration while scientific conditions and evaluated populations agree; an isolated single-interval contrast is needed for an individual-effect claim, not for every allocation decision. Describe the measured direction separately for both geometric metrics, its variability when supplied, and any absolute-displacement warning. A multi-interval change supports a joint effect only. Differences across teachers, output policies or evaluation populations are context, not isolated evidence about a weight change.

Separate three statements: what was observed, the proposed mechanism, and what the next experiment would discriminate. A useful hypothesis specifies the affected intervals or joint allocation, direction of adjustment and a measurable consequence on that background. If contrasts conflict across backgrounds, keep the mechanism unresolved and prioritize measured joint-configuration performance. Do not require decomposition of each interval before improving a joint configuration. Choose a discriminating contrast when it is needed for the next decision or a risk diagnosis, rather than as a mandatory intermediate step. Do not assign numerical probabilities or fabricate a surrogate-model prediction without a fitted model and supplied outputs.

## Review and retain the lesson

For each relevant prior prediction, classify each geometric metric as **supported**, **contradicted**, or **unresolved**. Use the previously specified threshold or direction with supplied uncertainty; a small uncertain difference is not confirmation. Missing prediction metadata permits a retrospective observation, not an invented preregistered test. Missing local-distance evidence leaves that portion unresolved.

Carry forward a compact lesson: scope, evidence identifiers, hypothesis status, consequence for interval allocation, and the remaining question. Resolve contradictions using matched evidence rather than discarding inconvenient observations. This is a summary derived from the supplied ledger, not a claim of hidden or persistent model memory. In retrospective mode, assess completed interventions without emitting future actions.

Keep attempt counts, missing measurements, build failures, finite outliers and sampled diagnostic coverage visible. Prefer overall mean primary-metric improvement within explicitly supplied deterioration limits; use supplied seed-level paired contrasts and distribution tails as diagnostics, not an unrequested consistency-first or worst-case ranking; duplicated deterministic groups are not independent replicates. A hard coordinate overwrite can force coordinate agreement: compare pre-overwrite diagnostics when available, and label the intervention difference instead of attributing that result solely to the weight schedule.

Repeated adaptive selection on fixed random samples can specialize a configuration to that evaluation panel. Report the observed improvement on that panel; do not call it independent validation or generalization. Sensitivity analysis using previously observed seeds remains diagnostic. The human chose to continue fixed samples and limit conclusions to that panel. Do not add a holdout stage by default, silently replace the panel or claim an unseen test set exists.

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
