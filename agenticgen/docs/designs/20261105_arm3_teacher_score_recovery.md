# Experiment 3/6pyy: initial teacher score recovery

The interaction-region attempt stopped before generating engineering or scientific
candidates: its synthetic teacher endpoint used a floating-point atom mask.
FLOWR sums this mask and indexes an atom-count embedding, which requires integer
indices. Build the scoring-only full-ligand endpoint with a `torch.long` mask.
Keep decoded chemistry and coordinates unchanged. This does not change the
interaction-region inpainting scope or introduce affinity gradients.

The regression covers one, four and 29 atoms through the actual PyTorch embedding
operation and the padded ten-row endpoint rescorer. It also checks input and RNG
preservation. The existing five-round failure/replay test now exercises the mask
index contract at the baseline instead of completely bypassing it.

The first recovery (`attempt_003`, commit `cd955853`) passed 19 remote CPU tests
and scored the initial teacher on the real GPU (pIC50 5.81679630279541). Before
the first inference step, it exposed a second error: a bare `constant` string
was supplied instead of the frozen first-window shape descriptor. Use the shared
shape constructor for fixed-scale agent-free iteration too. The regression now
expands the actual binding through `GeometryControl`, including native endpoints.

The second recovery (`attempt_004`, commit `c820e17d`) passed 20 remote tests
and reached the first native forward. Upstream `GeometricMolBatch.mask` constructs
a floating padding mask, unlike the frozen native initial priors. Normalize this
fresh-prior mask to `torch.long` too. It stopped before completing any inference
step: `steps.json` and `rng_ledger.json` are empty and science reservations are
zero. The new regression exercises this batch construction and the embedding
consumer while checking that fresh-prior generation restores the caller RNG.

The third recovery (`attempt_005`, commit `f9ecaa8e`) passed 21 remote tests and
then exposed an additional initial-input adapter error at the first forward.
`GeometricMolBatch.bonds` is a sparse edge list (50 x 289 x 3 for this teacher),
whereas the native model expects dense bond probabilities (50 x 17 x 17 x 5).
Use its native `.adjacency` property, preserving soft probabilities. Validate all
core tensor shapes, categorical vocabulary widths and dtypes against frozen native
priors before inference; the teacher atom count may change. Preserve ligand
residue metadata, resizing only uniform labels when the atom count changes.
The regression includes both the sparse/dense distinction and a real upstream
`GeometricMolBatch` fixture in the remote environment. No inference step or science
candidate completed in this failed engineering attempt; it remains archived.

Run `deploy_arm3_score_recovery.py` only after committing and publishing the fix.
It verifies the preserved first-forward bond-format failure, zero scientific
reservations and no completed engineering step, the unchanged experiment-1
fixed scale source, and the campaign budget.
It creates a separate frozen checkout and revision plan, and runs only experiment
3/6pyy in `jobs/05_arm3_6pyy/attempt_006`. It preserves attempts 001 through 005 and does
not restart experiments 1/2 or the dependency-blocked 3w54 chain.

The new coordinator runs CPU tests, waits for the shared GPU lock, then runs the
existing full engineering acceptance and five rounds of 50 scientific candidates.
Scale remains `[50,100,40,60,80,200]` from experiment 1/6pyy's R10/A4, with no Agent
calls or scale search. Round seeds and all scientific eligibility and promotion
rules remain as frozen in the interaction-region protocol. Failed attempts are
recorded without automatic resampling. The new plan caps science at 250 candidates;
the prior campaign has used 6,250 and its total cap remains 12,500.

Recovery state lives under
`/workspace/agenticgen_stage_affinity_campaign_20261104/revisions/arm3_native_prior`.
`REVISION_FINISHED.all_success` describes this single recovered task. It must not
be reported as success of the whole six-task campaign; the other failures remain.
Background inspection should follow the new launch identity and this attempt,
and stop after its sampling and promotion diagnostics are terminal.
