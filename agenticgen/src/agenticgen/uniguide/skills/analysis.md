<!-- Generated from guidance-experiment-optimizer 1.1.0; edit its standard skill sources. -->

## Objective and authority

Improve **controlled-atom coordinate root mean square deviation** and **controlled-to-surrounding-atom distance deviation** through evidence-based allocation of guidance weights across the supplied sampling intervals. Surrounding atoms belong to the generated molecule; exclude protein, receptor, pocket and other environment atoms. Use full metric names in reasoning summaries and predictions. Preserve exact machine field names and identifiers required by the response schema.

The first metric compares matched controlled atoms with their reference coordinates. The second compares supplied atom-pair distances with reference distances or allowed intervals. Preserve the supplied atom mapping, neighborhood, bonded/nonbonded distinction, tolerances and aggregation. Appropriate distances do not mean making all distances smaller. When the second metric or its definition is missing, mark it unavailable; coordinate agreement does not substitute for local-distance evidence.

Operate as an outer experiment planner. Choose weight schedules; the unchanged sampler computes gradients and trajectories. Preserve the model, coordinate loss, current-state vector-Jacobian product, integration, inpainting/output policy, random streams and budget. Do not change the scientific objective or turn diagnostic observations into an invented controller. Experiment count, continuation, termination and execution are external parameters. Learning from history grants no authority to launch, repeat or extend experiments.

## Evidence-to-action cycle

1. **Reconcile evidence.** Use all supplied completed history from the current experiment, including earlier iterations and other branches. Distinguish comparable measurements, reused evaluations, failed attempts and missing information. State any history-coverage limitation.
2. **Update the hypothesis.** Match earlier predictions and falsifiers to their actual outcomes. Identify the supported, contradicted or unresolved allocation lesson and its scope: reference molecule, anchor and affected intervals. A contradicted prediction must change the next rationale or action.
3. **Choose an informative legal intervention.** State whether it primarily exploits a measured benefit or distinguishes competing explanations. Select from the controller's legal action space; explain why this experiment is worth its evaluation cost.
4. **Register the prediction.** Name the original measured anchor, expected direction for each available geometric metric, and an observable result that would contradict the hypothesis. Separate expected performance from information value.
5. **Close the loop on the next observation.** Compare the executed configuration with the registered proposal before attributing an outcome to it. Update the allocation lesson; never rewrite the original prediction to fit the result.

Distinguish a coordinate improvement from an acceptable geometric tradeoff. Apply only externally supplied tolerances and selection priorities; do not invent a weighted score, universal distance threshold or stability guarantee. Prefer non-dominated measured alternatives when both metrics are available. If alternatives trade off and the controller supplies no preference, expose the tradeoff rather than claiming a unique geometric optimum.

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
| Persistent coordinate error, well-behaved local distances, and matched increases that helped without large absolute increments | Guidance may be insufficient in the responsive interval | Test a local increase in that interval; predict both geometry outcomes |
| Better coordinate agreement but worse local-distance deviation, excessive single-atom displacement or a heavier failure tail | Strong intervention may sacrifice local geometry | Test a decrease in the implicated interval or an evidence-supported redistribution |
| Error decreases then rises, with time-resolved direction changes or alternating corrections | Overshoot or conflicting interventions may be present | Contrast a weaker implicated interval against a measured baseline; final error alone cannot establish overshoot |
| Many weight changes give little metric change | Weak effective intervention, insensitive region, assignment changes or missing observability remain alternatives | Use gradient/increment/assignment evidence to choose a contrast; do not repeatedly escalate the same weight |
| Similar local changes have different effects on different preceding schedules | Interval coupling or state dependence | Preserve the background and isolate a legal interval change where possible; otherwise report a joint test |

Use only rows supported by supplied diagnostics. Diagnose numerical stiffness, damping, oscillation frequency or a stability margin only with the required model or time-resolved evidence; do not infer them from a final scalar score. An unresolved diagnosis can still motivate a discriminating experiment.

## Preserve molecular geometry and assess robustness

Bonded and nonbonded neighbors have different reference-distance requirements. Neither uniform attraction nor uniform repulsion is a valid substitute for the supplied local-distance metric. Preserve atom correspondence when interpreting whether positional improvement damaged the surrounding molecular structure.

Separate central performance, seed variability, tail excursions and missing outcomes. A more favorable mean with worse extreme displacement is a tradeoff to report. Use supplied acceptance limits; do not invent universal thresholds or call an empirically tested region certified safe. Constraints are checked by the controller. Keep proposed actions within its calibrated domain, with no clipping, gradient normalization, coordinate replacement or extra sampling introduced by this skill.
