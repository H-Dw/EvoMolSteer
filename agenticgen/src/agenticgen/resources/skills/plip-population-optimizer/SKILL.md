---
name: plip-population-optimizer
description: Analyze one teacher's complete 100-candidate population with registered numerical tools and propose distinct shared guidance schedules. Use for group hypothesis planning; do not launch sampling or change the reward.
---

Optimize the unselected group's mean rescored pIC50 first, Best second. Each
hypothesis applies one common four-interval schedule to all 100 candidates.
Return exactly the requested number of distinct hypotheses; it is selected from
two-teacher GPU capacity evidence, not a number for you to increase.

Use the supplied application-level JSON tool protocol. A `kind=tools` response
requests registered computations; the host returns numerical evidence. A
`kind=hypotheses` response ends the planning session. Do not invent shell code,
file paths, coordinate calculations or tool outputs. Integers 0..10 encode
amplitudes 0.0..1.0; the reward, matching, step window and displacement cap remain
fixed. Return only the schema supplied for the current turn.

Start with `history_summary` and paired latest/incumbent evidence already supplied.
Before finalizing, request late-state geometry (step 50, 75, 90 or 100), interval
diagnostics and subgroup response as needed to distinguish these explanations:

- Weak early movement leaves a mismatch that persists late: modestly increase an
  early interval and predict which score quartiles should improve.
- Good local geometry accompanies worse affinity: test weaker or differently
  timed guidance; template similarity is a proxy, not the affinity objective.
- Clipping or matching instability dominates: changing nominal amplitude may
  barely change displacement. Check the numerical displacement and clip rate.
- Some subgroups benefit while others degrade: a shared schedule is a compromise.
  Use frozen prior-score quartiles; do not exclude low-scoring or failed chains.

Late (0.5–1.0) checkpoints support diagnosis; early intervals are useful control
targets. Neither preference proves a strategy is universally best. Inspect the
recorded failed directions before repeating them. `candidate_history` can inspect
identified divergent or representative cases; the main decision must still use
all 100. Request a small subset of details instead of loading every coordinate
into the prompt. All historical analysis remains available in the evidence file.

Propose contrasting, testable directions, rather than paraphrases of one schedule.
Normally change one interval by one tick, at most two intervals per hypothesis.
Two-tick changes require the host's verified direction allowance from consecutive
group-mean improvement. Different exploration degrees can mean one versus two
changed intervals; do not interpret this as permission for large amplitude jumps.
All hypotheses share the same pre-round parent population, so never condition a
hypothesis on another hypothesis's not-yet-observed result.

For each hypothesis cite supplied evidence IDs, explain expected Mean contribution
and Best potential separately, and state an observable falsifier. Predictions are
hypotheses, not numerical results. Intermediate affinity is diagnostic and must
not be equated with final rescored pIC50.

Interpret validity conversions alongside common-valid paired changes. Keep the
unselected 100-candidate results of each hypothesis distinct from per-chain TOP1
selection across hypotheses and rounds. More searches can improve TOP1 even with
no improvement in the underlying proposal distribution. Report the search budget.
