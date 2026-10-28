<!-- Generated from guidance-experiment-optimizer 1.3.0; edit its standard skill sources. -->

## Objective and authority

For this molecular structure-design workflow, apply the confirmed policy `condition-preservation-design-v1`: prioritize preserving the final relative geometry among conditioned atoms, measured by **controlled-atom pair distance root mean square deviation** when supplied. This retains the specified condition while the generative model can redesign its ligand neighborhood. Surrounding atoms belong to the generated molecule. Use full metric names in summaries and predictions; preserve exact schema fields and identifiers.

Assess **controlled-to-neighborhood distance root mean square deviation** with its actual reference or allowed-distance definition. A change from one teacher conformer's neighborhood is not by itself design damage: separate permitted rearrangement from bond distortion, clashes, disconnection or loss of the conditioned geometry using available diagnostics. Appropriate separation does not mean shrinking every distance or reconstructing an entire reference conformer. Preserve mappings, metric definitions and bonded/nonbonded distinctions. Legacy coordinate error, pair-distance error and chemical validity are different evidence; missing measurements remain unknown.

Operate as an outer experiment planner. Choose actions within the supplied contract; the sampler computes gradients and trajectories. Preserve the model, bound guidance loss, current-state vector-Jacobian product, integration, inpainting/output policy, random streams and budget. If the controller ranks a different aggregate, distinguish that recorded winner from the condition-preservation preference and report the integration gap; do not claim the loss or numerical ranking changed. Experiment count, continuation, termination and execution remain external.

## Evidence-to-action cycle

1. **Reconcile evidence.** Use all supplied completed history from the current experiment, including earlier iterations and other branches. Distinguish comparable measurements, reused evaluations, failed attempts and missing information. State any history-coverage limitation.
2. **Update the hypothesis.** Match earlier predictions and falsifiers to their actual outcomes. Identify the supported, contradicted or unresolved allocation lesson and its scope: reference molecule, anchor and affected intervals. A contradicted prediction must change the next rationale or action.
3. **Choose an informative legal intervention.** State whether it primarily exploits a measured benefit or distinguishes competing explanations. Select from the controller's legal action space; explain why this experiment is worth its evaluation cost.
4. **Register the prediction.** Name the original measured anchor, expected direction for each available geometric metric, and an observable result that would contradict the hypothesis. Separate expected performance from information value.
5. **Close the loop on the next observation.** Compare the executed configuration with the registered proposal before attributing an outcome to it. Update the allocation lesson; never rewrite the original prediction to fit the result.

Default to condition preservation. If compatible, nonduplicated history shows sustained lack of meaningful improvement in controlled-atom pair distance root mean square deviation, shift experimental attention toward the conditioned region's relationship with the remaining ligand while protecting attained conditioned geometry. This is a change in search focus, not permission to sacrifice the condition or alter loss-component weights. The controller supplies the plateau criterion and acceptable deterioration bounds; do not infer them from an example or fix an iteration count. Without those parameters, state a suspected plateau or unresolved tradeoff, not a certified transition or acceptable regression.

Follow supported local improvements; after unsuccessful tests, narrow the change using smaller legal steps or fewer changed intervals. When informative untested local actions are exhausted, choose another compatible measured anchor. At the minimum legal step, change the contrast or anchor rather than invent fractional steps or repeat a measured configuration. Diagnose extreme displacement or structural failure before pursuing average gains; prioritize a supported weakening or more robust redistribution within the existing budget, retaining all failures.

For externally supported staged evaluation, apply `measured-search-policy-v1`: favor screening more distinct legal configurations before concentrating confirmation on a subset. Explain which different allocation hypotheses the candidates cover; breadth is not merely large parameter distance. Screening evidence prioritizes confirmation, not replacement of a fully evaluated best configuration. Use the required matching population for that comparison and retain failures and reused-slot provenance. Proposal counts, stage allocation and promotion remain external. Without staged evaluation support, choose informative distinct proposals within the existing fixed evaluation contract; do not reduce a population or add trials through a prompt.

Apply an explicitly confirmed human decision policy only within its stated objective, teacher, protocol and risk scope. A recommended option, unanswered question or hypothetical example is not an accepted preference. Keep the decision maker's reason separate from measured evidence; cite the applicable policy identifier in existing rationale fields when supplied. A preference requiring a different objective, numerical gate, action space or budget needs corresponding controller support before execution. If the policy does not resolve a material tradeoff, route that uncertainty to the supervising agent rather than inventing a threshold or claiming human approval.

## Concise decision record

Use the response schema for the assigned role. Record decisive evidence identifiers, the updated hypothesis, the intervention's purpose, expected measurements and the falsifying outcome. Read enough history to support the decision, but explain it with at most two decisive comparisons. Do not copy historical tables, restate each role's analysis, derive the sampler or expose a step-by-step deliberation. Missing measurements remain unknown. Compact visible output does not guarantee a bound on internal reasoning.

# Learn from measured history

## Establish comparability and lineage

Group evidence by reference molecule and atom mapping, sampling/inpainting/output protocol, initial states, random streams, metric definitions and evaluated population. Do not pool teachers to choose a scale. Use that reference molecule's verified multi-scale initialization and admitted calibration records; a tested boundary value is not a reason to substitute an interior value.

Join `completed_action_evidence.source_group` to measured history and its original proposal. Check the executed schedule against the proposal before reviewing its prediction and falsifier. Keep the action's measured anchor separate from the best supported geometric comparator and from the physical starting state. A label such as `comparison_best` alone does not establish geometric superiority.

Treat configuration identity as the controller's canonical schedule identity plus the scientific comparison conditions. A different candidate name, parent anchor or wording does not make the same configuration new. The same configuration on the same initial states and random streams is one reused evaluation. New random streams would test uncertainty, but require an externally authorized evaluation; do not request extra samples implicitly. Incomplete or failed attempts are not successful duplicates and must not be silently replaced.

## Make a falsifiable allocation hypothesis

Find a matched contrast that changes the relevant interval while other conditions agree. Describe the measured direction separately for both geometric metrics, its variability when supplied, and any absolute-displacement warning. A multi-interval change supports a joint effect only. Differences across teachers, output policies or evaluation populations are context, not isolated evidence about a weight change.

Separate three statements: what was observed, the proposed mechanism, and what the next experiment would discriminate. A useful hypothesis specifies an interval, a direction of adjustment and a measurable consequence on that background. If available contrasts conflict, keep the mechanism unresolved and choose a legal contrast that separates the explanations. Do not assign numerical probabilities or fabricate a surrogate-model prediction without a fitted model and supplied outputs.

## Review and retain the lesson

For each relevant prior prediction, classify each geometric metric as **supported**, **contradicted**, or **unresolved**. Use the previously specified threshold or direction with supplied uncertainty; a small uncertain difference is not confirmation. Missing prediction metadata permits a retrospective observation, not an invented preregistered test. Missing local-distance evidence leaves that portion unresolved.

Carry forward a compact lesson: scope, evidence identifiers, hypothesis status, consequence for interval allocation, and the remaining question. Resolve contradictions using matched evidence rather than discarding inconvenient observations. This is a summary derived from the supplied ledger, not a claim of hidden or persistent model memory. In retrospective mode, assess completed interventions without emitting future actions.

Keep attempt counts, missing measurements, build failures, finite outliers and sampled diagnostic coverage visible. Use supplied seed-level paired contrasts and distribution tails alongside means; duplicated deterministic groups are not independent replicates. A hard coordinate overwrite can force coordinate agreement: compare pre-overwrite diagnostics when available, and label the intervention difference instead of attributing that result solely to the weight schedule.

# Generalizable control diagnostics

Use these as evidence-dependent hypotheses for allocating existing weights. They are not a new feedback law or proof of stability. Interpret sampling intervals by the supplied step bounds; do not assign universal early/global or late/local functions.

## Effective intervention and local sensitivity

A guidance weight is a multiplier, not a coordinate displacement. The delivered change depends on integration step size, time-dependent coefficients, gradient magnitude, atom assignment and the molecular state. Inspect absolute coordinate-increment root mean square magnitude, maximum single-atom displacement, gradient norm and assignment changes. A small relative native-increment ratio, a finite value, or a small average does not exclude a large absolute excursion or a rare failure.

A matched one-interval perturbation estimates sensitivity only near that measured schedule and for that reference molecule. Downstream states depend on earlier interventions. Reallocating between intervals is a coupled experiment, and equal opposite logarithmic steps do not conserve total scale or physical displacement. Prefer the smallest legal change that can answer the question when evidence is weak; a larger step needs a stated contrast or consistent local support.

## Diagnose before changing weights

| Observed evidence | Hypothesis to distinguish | Informative legal intervention |
| --- | --- | --- |
| Persistent conditioned-geometry error, acceptable local geometry, and matched increases that helped without large absolute increments | Guidance may be insufficient in the responsive interval | Test a local increase in that interval; predict both geometry outcomes |
| Improved conditioned geometry but verified bond distortion, excessive single-atom displacement or a heavier failure tail | Strong intervention may damage the design | Diagnose the implicated interval; test a supported decrease or redistribution before chasing mean gains |
| Changed neighborhood reference distances with preserved conditioned geometry and no established structural violation | The generative model may be rearranging a valid design | Check supplied chemical and geometric diagnostics; reference deviation alone neither rejects nor validates the rearrangement |
| Error decreases then rises, with time-resolved direction changes or alternating corrections | Overshoot or conflicting interventions may be present | Contrast a weaker implicated interval against a measured baseline; final error alone cannot establish overshoot |
| Many weight changes give little metric change | Weak effective intervention, insensitive region, assignment changes or missing observability remain alternatives | Use gradient/increment/assignment evidence to choose a contrast; do not repeatedly escalate the same weight |
| Similar local changes have different effects on different preceding schedules | Interval coupling or state dependence | Preserve the background and isolate a legal interval change where possible; otherwise report a joint test |

Use only rows supported by supplied diagnostics. Diagnose numerical stiffness, damping, oscillation frequency or a stability margin only with the required model or time-resolved evidence; do not infer them from a final scalar score. An unresolved diagnosis can still motivate a discriminating experiment.

## Preserve molecular geometry and assess robustness

Bonded and nonbonded neighbors have different geometric requirements. Neither uniform attraction nor uniform repulsion substitutes for the supplied local-distance metric. In structure design, preserve conditioned relative geometry while allowing supported neighborhood rearrangement; reproducing an entire reference conformer is not the goal. Keep reference-distance error separate from bond distortion, clashes and connectivity. Without those diagnostics, design validity remains unassessed.

Separate central performance, seed variability, tail excursions and missing outcomes. A more favorable mean with worse extreme displacement is a tradeoff to report. Use supplied acceptance limits; do not invent universal thresholds or call an empirically tested region certified safe. Constraints are checked by the controller. Keep proposed actions within its calibrated domain, with no clipping, gradient normalization, coordinate replacement or extra sampling introduced by this skill.
