---
name: jump-guidance-proposal
description: Propose complementary six-window guidance schedules through evidence-driven forward exploration and backtracking, with at least two distinct early directions.
metadata:
  version: "2.14.0"
---

# Path Exploration and Proposal

## Choose the Next Search Focus

Treat parameter optimization as branching over complete schedules. Read the recorded focus and measured prefix alternatives. Select a measured source and a window of interest. In the first round, use B1 or B2 with operation `start`. Subsequently use `advance` for a later window, `refine` for the same window, or `backtrack` for an earlier window. Justify forward progress with existing coverage and measured outcomes; do not skip early uncertainty merely because late adjustments previously helped.

The focus guides exploration without locking other windows. Preserve opportunities to revise an earlier prefix, compare a neighbouring dose on the same background, or adapt the suffix of a measured alternative. Use final measured performance to evaluate branches; do not invent a predicted score or use an extra self-evaluation model.

## Form Complementary Hypotheses

Propose five distinct executable schedules when budget and legal opportunities permit, never more than five. Prioritize at least two substantive early directions that actively change B1–B4 relative to each proposal's actual measured parent. Their signed changed-window patterns must differ. Two magnitudes in the same direction, inherited prefix differences, or suffix-only changes do not satisfy this requirement.

Treat B1, B2, B3, and B4 independently. Do not preserve B3=B4 as an unexplained constraint. Prefer untested independent changes when history contains only tied values. Include a direction that retains or challenges the main unconfirmed early change so that all five hypotheses do not depend on the same assumption.

Allocate the other opportunities using evidence: forward extension, suffix adaptation, a useful historical branch, or a broader joint allocation. Same-background dose comparisons are valid optimization proposals. A joint change supports the complete configuration's result, not an isolated causal claim. Preserve the required multi-window bold opportunity and different changed-window subsets; distance tiers do not replace temporal diversity.

## Audit the Complete Portfolio

Submit actual integer scales in the six windows declared by the prompt. B1 has a different grid from the later windows; calculate distance in per-window grid indices. The final window is 51–99, with extra guidance active only through step 98. All six boxes are available. Zero scale removes extra coordinate guidance while preserving the declared chemical phase.

Use `audit_action_batch` in portfolio mode before final submission. Every legal proposal executes, in order, after deterministic validation; there is no Selection or synthesis Agent. Each proposal uses a complete 50-candidate population, at most 250 per round and 1250 formal attempts per teacher. Calibration has a separate 500-attempt cap. Exact duplicates are referenced, failed attempts remain charged, and no replacement sampling is added.

## Submit the Decision

Use `submit_final_response` with explicit `decision`, `proposals`, `reason`, and `search_focus`. The focus contains zero-based `window_id`, `operation`, and a real measured `source_id`. Each action must retain its real `anchor_id`, evidence IDs, full `target_scale_schedule`, `optimization_target`, and distance-based `exploration_tier`. Bind the optimization target to the actual calibration root and current global best, even when using a different parent. Do not emit `updates`, `selected`, or extra action fields.

Give concise evidence, the improvement opportunity, and uncertainty in the existing explanation fields, each at most 3000 characters. For `hold`, explicitly submit `proposals=[]`. When `no_future_execution` is true, hold and retain a valid focus. Do not use hold to bypass feasible early exploration. The existing single bounded submission repair applies.

Guidance gradients, matching, first-window shape, endpoint handling, and chemistry follow the frozen protocol. Evaluation alignment does not authorize changing the guidance loss.

For the methodological background and signed-direction examples, see [Search-policy references](references/search-policy.md). The runtime prompt contains all mandatory decision rules.
