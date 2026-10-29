# Completed-experiment gate

Use the experiment scope fixed before execution: target identifiers, required phases, evaluation populations, output policies, stopping rules, completion criteria and quality coverage. Enumerate every declared target and reconcile required work with actual receipts. Neither directory existence, process exit, an intermediate review marker nor one completed target proves the experiment complete.

An explicitly terminal completed experiment can contain recorded failed molecular attempts when its original accounting permits them; do not require replacing those samples. A target stopped before its required work, a still-running target, missing final receipts or unresolved required postprocessing prevents post-experiment RSI. Preserve failed/paused evidence and produce an incomplete-scope record; do not relabel it, shrink scope, retry, resume or ask the human to pass the gate. Optional diagnostics declared optional by the original protocol stay optional, but their absence remains visible.

Record the scope/manifest hash, per-target completion evidence, population/seed/slot identity, failure and reuse accounting, required diagnostics, retained artifacts and run/skill hashes. Source paths and object identifiers belong in this private-to-the-run update log, not reusable instructions. These field groups describe an external review artifact, not new Guidance Agent JSON fields.

Use all targets in the review, reporting each separately where science or calibration differs. Compare original prediction with executed action before interpreting the result. Preserve counterexamples, unknown diagnostics, finite extremes, invalid proposals and unchosen actions. No outcome is invented for an unexecuted candidate. A base statistic is not quality completion; sampled quality coverage cannot be extrapolated to an untested population.

The completion gate is a skill requirement, not an installed completion validator. Audit actual controller support before claiming automatic enforcement. In particular, an older per-version accept-review hook must not force instruction mutation inside the experiment.
