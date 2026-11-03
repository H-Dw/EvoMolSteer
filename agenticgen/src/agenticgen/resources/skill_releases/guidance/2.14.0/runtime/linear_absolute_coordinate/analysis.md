---
name: jump-guidance-history
description: Review measured guidance paths, early-window coverage, and prefix alternatives before proposing complete six-window schedules.
metadata:
  version: "2.14.0"
---

# History and Evidence

## Objective

Minimize the arithmetic mean of condition-aligned RMSD across all 50 candidates. The evaluator fits only the fixed terminal condition correspondences with a proper rigid transform. Non-condition atoms do not enter the fit or residual. Preserve the calibration root and replace the global best only after a strictly lower measured mean. Protein-frame pose error and molecular quality remain separate diagnostics.

## Read the Search State

Review the calibration root, actual parent path, recent outcomes, search focus, prefix families, and window coverage. Distinguish measured improvement from a plausible opportunity. A prefix is the first four scales; its observed quality depends on the suffixes actually tested. Compare candidate results with both the real parent and the global best without conflating them.

Check B1 and B2 dose coverage, independent B3 and B4 changes, and the number of suffixes tested for each prefix. Repeated arrays sharing one early change do not establish independent early evidence. Unchanged best-path scales do not establish convergence. Different backgrounds do not establish a single-window effect.

## Forward Exploration and Backtracking

Begin with B1/B2. Use measured feedback to refine the current focus or extend toward subsequent windows. Earlier scales are provisional choices, not frozen commitments. Revisit them when a changed suffix, weak coverage, or an alternative prefix creates a concrete opportunity.

After two consecutive completed rounds without global improvement, review failed continuations and backtrack to a useful early window or historical prefix. All compatible measured anchors remain available; restoring the global best does not erase side branches. Give a promising prefix bounded suffix adaptation when its first result merely reused another prefix's suffix. Stop extending a branch when no specific new opportunity remains.

## Evidence Boundaries

Every node is a measured complete schedule. A new branch runs the full inference trajectory; it does not resume a molecular state at the focus window. Early update magnitude and assignment switching may motivate a probe, but do not prove improved final conformation. Report brief source-backed conclusions, not private reasoning traces. Missing measurements, failed attempts, insufficient budget, and convergence are different states. Two rounds without improvement also include rounds whose attempts all failed; retain their failure evidence when reconsidering the path.
