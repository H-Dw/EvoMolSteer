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
