---
name: plip-schedule-optimizer
description: Diagnose a FLOWR PLIP-local trajectory and propose a small change to four amplitude intervals, prioritizing the teacher population mean before its best score.
---

The primary objective is the mean final `pic50_on_rescore` across the teacher's 100 original candidate chains; the secondary objective is the best pIC50 in that population. Your action changes only this chain, with the same ten-update budget as every other chain. Use the supplied frozen previous-round population evidence and this chain's full history to seek reliable improvements that contribute to the mean. Do not concentrate effort only on an already exceptional maximum or trade a plausible mean decline for a hoped-for extreme. When expected mean effects are comparable, prefer the action with a better-supported Best effect. Give qualitative expected_mean_effect and expected_best_effect with uncertainty; do not invent a numerical population forecast from one molecule.

All five batches finish a round before the next round starts. Every decision in a round receives the same population summary, including on resume. A candidate with a low score remains part of the objective and keeps its budget. Failures remain in the denominator of validity reporting; improving a conditional mean by dropping failures is not success. Keep each chain's historical best, and distinguish that selected population from the latest unselected proposals. Recovering the teacher is a milestone, not a reason to stop before ten updates. Neither a predicted score nor its improvement proves experimental binding affinity.

## Fixed mechanics

The host runs FLOWR from prior at t=0 to molecule at t=1, with 100 native updates. It keeps the PLIP core, endpoint pseudo-Huber reward, geometric Hungarian matching, native SDE and categorical sampling fixed. You may change only FOUR amplitude ticks, in order: steps 10..25, 26..50, 51..75, 76..90, all inclusive. These correspond to time labels 0.10–0.25, 0.26–0.50, 0.51–0.75, 0.76–0.90. The three diagnostic control nodes are 0.25, 0.50 and 0.75. The host also retains exact interval-entry states 10, 26, 51 and 76; checkpoint k is BEFORE update k. The interval lengths are 16, 25, 25 and 15 updates. Steps 0..9 and 91..99 have zero guidance. Tick 5 means lambda=0.5; integer ticks 0..10 are allowed. Editing an interval edits every update inside it. Restart occurs at the first edited interval's entry, not by rounding it to a tenth. Do not propose a different reward, region, seed, precision, direction, feature intervention, or integration grid.

Usually change one interval by one tick (0.1); at most two intervals may change, preferably adjacent. Two ticks in one interval are allowed only when the supplied `allowed_double_tick_directions_by_parent` explicitly permits that interval and sign. Treat this as a local trust region, not a reason to approach the maximum. Lowering guidance is a valid experiment.

## Read evidence, diagnose, then act

Use the retained TOP1 and the complete analysis documents of every previous attempt, including failures. Only `restartable_parent_ids` are physically available; in TOP1 mode this is the current incumbent alone. Set `parent_id` to that incumbent. Nonbest structures and raw trajectories have been removed after evaluation; their numerical checkpoint diagnostics, schedules, prompts, hypotheses, outcomes and quality reports remain valid evidence. Do not request deleted structures or infer that absence means a zero metric. Never repeat an already evaluated full schedule.

Reinterpret failed actions relative to the current incumbent's schedule. When the best stays unchanged, test a different sign or interval using documentary counterexamples. A historical failure is evidence for reasoning, not an available restart parent. TOP1 is per original candidate chain; other slots' winners cannot replace this chain. If no eligible molecule exists yet, the supplied initial fallback remains the restart state and is not evidence of a valid success.

Diagnose primarily from t=0.5..1.0: endpoint affinity trends, local coordinate errors, matching changes, categorical uncertainty, and the difference between an intermediate endpoint and the final decoded/rescored molecule. Intermediate `affinity_head_at_t` is not the final `pic50_on_rescore`; do not treat their numerical differences as calibrated affinity gains.

Prefer testing the first two intervals (steps10..25 and26..50), where intervention can alter the later path. Diagnose with the 0.5–1.0 checkpoints, including the post-boundary entry states. A late loss rise does not prove that the late interval caused it or that early guidance should increase. Cite the evidence linking the observed late behavior to the proposed intervention. Intervals3/4 remain available when evidence supports them. Compare actual displacement totals and clipping because the four intervals have unequal lengths.

Check actual update magnitudes before changing nominal lambda:

- Falling local loss with falling affinity can mean overconstraining a favorable alternative; consider reducing or moving guidance.
- Persistent clipping means larger lambda may produce no additional physical displacement.
- Negative gradient/native-drift cosine indicates competing directions, not by itself a bad update.
- Assignment switches or high type uncertainty weaken local structural diagnoses.
- A vanishing gradient cannot be repaired by assuming the maximum allowed coefficient is sufficient.

These are hypotheses, not universal molecular laws. Image-generation interval-guidance and APG findings motivate timing/strength diagnostics but do not establish that warmup, early suppression, or projection is optimal here. No projection is available.

## Learn from each trial

Compare the result with its parent and the incumbent. State which earlier hypothesis was supported, contradicted, or remains unresolved. If an increase failed, first consider undoing it, testing the opposite sign, or moving the intervention; do not blindly increase again. Separate geometry response, affinity response, clipping response, and validity. Missing values are missing evidence, never zero.

Your compact memory update must retain the observed action, outcome, applicable interval and a limitation or counterexample. Preserve failed directions. Do not extrapolate a rule from one candidate to all teachers.

## Output

Return only the supplied JSON schema: parent_id, four integer ticks, evidence_ids, concise diagnosis, hypothesis, expected_change, expected_mean_effect, expected_best_effect, falsifier, memory_update. Cite actual supplied evidence IDs, including the population summary when discussing its statistics. Describe an observable outcome that would falsify the hypothesis. Do not invent checkpoint values, hidden evaluations, or evidence IDs.

Keep reasoning concise and decision-focused. All structural coordinates are in the fixed protein frame in Angstrom; no ligand alignment is performed.
