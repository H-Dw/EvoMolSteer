---
name: plip-schedule-optimizer
description: Diagnose a FLOWR PLIP-local coordinate-guidance trajectory and propose a small, evidence-based change to its eight temporal amplitude intervals.
---

Optimize final `pic50_on_rescore` for this single candidate. Recovering the teacher is a milestone, not a reason to stop before the ten-update budget. Keep the historical best while conducting informative trials. Neither the teacher score nor the best historical score proves experimental binding affinity.

## Fixed mechanics

The host runs FLOWR from prior at t=0 to molecule at t=1, with 100 steps. It keeps the PLIP core, endpoint pseudo-Huber coordinate reward, geometric Hungarian matching, native SDE and categorical sampling fixed. You may change only eight amplitude ticks: entries correspond to [0.1,0.2), ..., [0.8,0.9). Tick 5 means lambda=0.5. Valid ticks are integers 0..10. A changed interval applies to all ten native integration steps in that interval. The host restores the earliest affected checkpoint. Do not propose a different reward, region, seed, precision, force direction, atom type intervention, or integration grid.

Usually change one interval by one tick (0.1); at most two intervals may change, preferably adjacent. Two ticks in one interval are allowed only when the supplied `allowed_double_tick_directions_by_parent` explicitly permits that interval and sign. Treat this as a local trust region, not a reason to approach the maximum. Lowering guidance is a valid experiment.

## Read evidence, diagnose, then act

Use the current best and the latest failed or successful attempt. Read every previous schedule/outcome before selecting an action. Prefer the current best as parent. A nonbest parent needs a concrete, falsifiable exploration hypothesis. Never repeat an already evaluated full schedule.

Diagnose primarily from t=0.5..1.0: endpoint affinity trends, local coordinate errors, matching changes, categorical uncertainty, and the difference between an intermediate endpoint and the final decoded/rescored molecule. Intermediate `affinity_head_at_t` is not the final `pic50_on_rescore`; do not treat their numerical differences as calibrated affinity gains.

Prefer controlling t=0.1..0.5, where an intervention can alter the later path. A late loss rise does not prove the late interval caused it, nor prove early guidance should be stronger. Cite the evidence that links the observed late failure to the proposed early intervention. Later intervals remain available when evidence supports testing them.

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

Return only the supplied JSON schema: parent_id, eight integer ticks, evidence_ids, concise diagnosis, hypothesis, expected_change, falsifier, memory_update. Cite actual supplied evidence IDs. Describe an observable outcome that would falsify the hypothesis. Do not invent checkpoint values, hidden evaluations, or evidence IDs.

Keep reasoning concise and decision-focused. All structural coordinates are in the fixed protein frame in Angstrom; no ligand alignment is performed.
