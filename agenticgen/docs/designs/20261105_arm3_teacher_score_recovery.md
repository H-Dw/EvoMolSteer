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

Run `deploy_arm3_score_recovery.py` only after committing and publishing the fix.
It verifies the prior failure, zero scientific reservations and no engineering
input, the unchanged experiment-1 fixed scale source, and the campaign budget.
It creates a separate frozen checkout and revision plan, and runs only experiment
3/6pyy in `jobs/05_arm3_6pyy/attempt_003`. It preserves attempts 001/002 and does
not restart experiments 1/2 or the dependency-blocked 3w54 chain.

The new coordinator runs CPU tests, waits for the shared GPU lock, then runs the
existing full engineering acceptance and five rounds of 50 scientific candidates.
Scale remains `[50,100,40,60,80,200]` from experiment 1/6pyy's R10/A4, with no Agent
calls or scale search. Round seeds and all scientific eligibility and promotion
rules remain as frozen in the interaction-region protocol. Failed attempts are
recorded without automatic resampling. The new plan caps science at 250 candidates;
the prior campaign has used 6,250 and its total cap remains 12,500.

Recovery state lives under
`/workspace/agenticgen_stage_affinity_campaign_20261104/revisions/arm3_teacher_score_dtype`.
`REVISION_FINISHED.all_success` describes this single recovered task. It must not
be reported as success of the whole six-task campaign; the other failures remain.
Background inspection should follow the new launch identity and this attempt,
and stop after its sampling and promotion diagnostics are terminal.
