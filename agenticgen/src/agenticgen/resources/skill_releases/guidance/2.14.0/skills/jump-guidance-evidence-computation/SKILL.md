---
name: jump-guidance-evidence-computation
description: Verify condition-aligned population means, compare measured complete schedules, and audit every proposed guidance direction without sampling.
metadata:
  version: "2.14.0"
---

# Evidence Computation

## Measured Performance

Use `summarize_population` to inspect the unfiltered mean of condition-aligned RMSD over all 50 candidates, including valid coordinate measurements from candidates whose molecular build failed. Missing or nonfinite geometry cannot be deleted to improve ranking. Do not substitute a trimmed mean, recovery threshold, best sample, or protein-frame pose error for the declared objective.

The alignment uses the same terminal-assigned condition points for fitting and RMSD. It allows translation and proper rotation, with no reflection, scaling, or new atom matching. This is condition conformation error, not whole-molecule RMSD or chemical identity recovery. The per-step guidance objective remains unchanged.

## Comparisons and Coverage

Use `compare_paired_runs` for compatible measured configurations sharing seed/slot identities. Its candidate-minus-reference mean is negative for improvement. Descriptive seed-cluster intervals on an adaptively explored panel are not independent validation.

Read `window_evidence` and `search_path_context` to distinguish same-background doses, broader coverage, and prefix families with different amounts of suffix adaptation. A single unsuccessful prefix/suffix pair does not rule out that prefix in all backgrounds. Large updates and assignment switching indicate sensitivity, not final benefit. Scale diversity alone does not prove molecular diversity.

## Executability

Use `audit_action_batch` on the whole proposed portfolio to verify integer grids, real parents, evidence bindings, early signed directions, novelty, and cumulative cost. Individual executability does not imply that the entire portfolio fits the budget. Audit receipts do not prove that inference has completed.

Use only tools actually exposed by the runtime. Do not request unavailable trajectory inspections, new seeds, new scoring models, or extra sampling. Report unavailable evidence explicitly. Keep endpoints, chemical coverage, gradient scope, and the declared first-window shape fixed; all six window scales remain independent search parameters.
