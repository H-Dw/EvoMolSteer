# Experiment 3: five rounds of twenty candidates

The user replaced the proposed additional 250-candidate rerun with five rounds
of twenty new candidates. Run only 6pyy, retain experiment 1's fixed best scales
`[50,100,40,60,80,200]`, and do not call an Agent or recalibrate scales.

`uniguide_iterative_affinity_20.json` selects twenty candidates per round, split
into five independent seed groups of four. Native prior draws and random calls
retain their original twenty-row shape; unused prior/padding rows are not
candidates. Network calls retain the frozen ten-row physical batch shape.
Scientific seed-block offset 7 avoids all previously consumed scientific groups.

The runtime supports the explicit twenty/fifty population choices, propagating
the actual count through prior shapes, pocket/COM audits, candidate identities,
endpoint scoring, teacher tensor selection, interrupted/failed reservations and
final accounting. Legacy fifty-candidate configurations and release files are
retained. Complete original experiment-1 measurements still determine the fixed
scale source; the new population changes only experiment 3.

The changed logical batch requires a fresh, complete twenty-candidate engineering
run before science. All hundred inference steps, chemistry at 1–98 and uncovered
native endpoints are checked. Its endpoint scores are independently compared
against experiment 1's scoring entry point. A fifty-row replay cannot stand in
for this twenty-row execution check. These twenty engineering trajectories are
registered separately from the 100 scientific candidates.

Promotion rules remain unchanged: finish inference, use condition-only proper
Kabsch RMSD strictly below 0.5 angstrom, rank finite built/connected candidates
by frozen endpoint pIC50, assess at most ten strict teacher improvements with PB
and PLIP, and apply the highest usable teacher next round. Interaction-region
inpainting, loss-checked promoted graph serialization, fixed protein/pocket and
the decoded t=.9999 affinity definition are retained.

The independent output is `jobs/05_arm3_6pyy/attempt_008`, coordinated under
`revisions/arm3_twenty_sample`. Attempt 007 and its 150 reservations stay intact.
The new scientific cap is 100; the teacher's cumulative cap remains 250 and the
campaign's maximum used/reserved science becomes 6,500/12,500. User authorization
binds this exact population, commit and configuration. No candidates are refilled
and no rounds are appended after failures. Other tasks are not restarted.

Verification covers both populations, four slots per seed, preserved native RNG
advancement, full count-aware acceptance, failures and idempotent recovery. The
twenty-candidate promotion fixture selects the last tensor row and checks that
only one teacher row is stored. Real twenty-row GPU acceptance is performed by
the deployed job before scientific reservations.
