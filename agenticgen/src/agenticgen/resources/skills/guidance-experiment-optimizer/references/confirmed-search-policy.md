# Confirmed adaptive search policy

Current policy: `history-adaptive-search-v2`, skills version 1.6.0. It supplements `condition-preservation-design-v1` and retains `measured-search-policy-v1` and `branch-evidence-abstention-v1`. The latest direct human answer is recorded in `agenticgen/docs/designs/20261028_guidance_policy_reconciliation/decisions.json`. Earlier interview records preserve offered options and original evidence snapshots.

## Confirmed decisions

| Decision | Accepted instruction | Scope and limit |
| --- | --- | --- |
| Temporary deterioration, answer 1B with explicit amendment | Allow worse exploratory results; after two consecutive deteriorating rounds, recover from the best compatible checkpoint and propose a new strategy | The explicit amendment overrides option B's former prohibition. Exploration is not incumbent replacement. Keep all history and already evaluated candidates |
| Replacement anchor, 2A | Prefer a measured anchor testing a different allocation hypothesis, considering conditioned geometry and risk | Do not choose solely by next-best score or claim a mechanism has been proven |
| Working context, 3A | Global coverage overview + relevant anchor contrasts + counterexamples and risk records; trace sources as needed | Preserve the full ledger; disclose selected coverage. Full supervising-GPT version review remains required |
| Background-dependent interval response, 4C | Prioritize measured performance of joint configurations | No requirement to decompose every interval first; do not infer individual causality from a joint change |
| Seed tradeoff, 5B | Prefer overall mean improvement with externally explicit deterioration limits | Primary conditioned geometry keeps priority. Seed consistency is not an extra veto; measured structural violations are not averaged away |
| Validation, 6B | Keep the fixed sample panel and restrict conclusions to its optimization results | No independent-validation or generalization claim; do not allocate new holdout samples by default |
| Earlier resource preference | Screen more distinct configurations, then concentrate confirmation on selected candidates when staging is supported | Unequal populations do not establish a final winner; quotas, stages and budget remain external |
| Earlier branch preference | Abstain a branch lacking both supported benefit and useful discrimination; keep supported branches eligible | Requires explicit proposal/selection/dispatch/accounting support; do not manufacture a skip field |

## Recovery contract for the supervising agent

The human supplied `consecutive_regression_limit = 2`. This is a recovery trigger, not the total iteration count. Store and pass it through an external, version-bound decision policy; the concise inference instructions refer to the supplied trigger. [confirmed-search-policy.json](confirmed-search-policy.json) records the decision and integration state; it is not an installed run configuration or a claim of automatic consumption.

Before enabling recovery, bind the round unit, round representative, comparison reference, meaningful-deterioration criterion, treatment of unavailable/intervening outcomes, and compatible best-checkpoint identity. These numerical/comparison semantics have not been selected. The inference agent must not guess them from candidate order or from an all-time best, whose stored score cannot worsen. The two-round rule does not mean two bad candidates or two rounds without improvement. A plateau and a measured regression are different states. Missing/request-failed/reused/abstained outcomes are not measured regressions; their counter transition needs the external contract. Do not pool teachers or bridge incompatible protocols.

Recovery restores the search/skill strategy basis of the best compatible checkpoint, then develops a distinct untested hypothesis from both its successful evidence and the later failures. Its older instructions never override the latest confirmed human policy or current compatible action validators. Reuse saved results. Never roll back the evidence ledger, repeat the best molecular generation, reset attempts, change frozen calibration, restore physical initial states, or overwrite a frozen manifest. If the best checkpoint cannot itself provide a legal novel action, retain it as the recovery reference and select a compatible measured anchor supporting the new hypothesis. A missing compatible checkpoint or unsupported interface requires a machine-controller error/review record and withholding the dependent unsupported operation. The supervising GPT must not contact the human or wait for a preference answer during runtime; confirmed preferences remain direct rules.

## Remaining external definitions

No neighborhood-deterioration tolerance, primary-protection tolerance, plateau criterion, mean aggregation/missing-outcome rule, screening quota or shortlist size is invented here. Preserve actual measured denominators and failure counts; a favorable mean over surviving molecules does not erase failures. The scope of averaging must be declared before judging a winner. This update changes instructions and preference provenance, not the loss, runtime ranking, sampling protocol or paused-task state.

The earlier primary-source assessments remain applicable as methodological analogies: BOHB separates configuration choice and evaluation allocation; control tuning motivates experiments that resolve uncertainty; AIDE/AlphaEvolve motivate measured candidate lineage; local-search and risk work distinguish background effects and mean versus tail objectives. The exact decisions above are the human's choices, not algorithms or guarantees imported from those papers. See [sources.md](sources.md) and [decision-boundaries.md](decision-boundaries.md) for source limits.
