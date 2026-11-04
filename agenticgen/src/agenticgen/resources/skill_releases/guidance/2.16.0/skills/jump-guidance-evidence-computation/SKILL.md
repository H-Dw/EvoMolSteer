---
name: jump-guidance-evidence-computation
description: Look up compatible complete schedules, compare terminal condition-aligned means, and audit stage-based proposals without sampling.
metadata:
  version: "2.16.0"
---

# Evidence Computation

## Read Complete Measurements
Use summarize_population and compare_paired_runs for compatible measured populations. Rank only the unfiltered arithmetic mean of terminal condition-aligned RMSD over all 50 candidates. Include valid terminal geometry even when molecular construction failed. Missing or nonfinite values cannot be deleted or replaced by pose error to create a rankable population. Alignment uses condition points only, fixed terminal correspondences, proper rotation and translation, without reflection, scaling or rematching.

## Reuse and Compare
Use lookup_schedule to retrieve a measured complete array's evidence ID, score and zero new cost. Only verified compatible completed measurements are reusable. Cite existing controls; spend proposal slots on missing complete schedules. Distinguish a prefix's first completion under an inherited suffix from later suffix adaptation. Attribute a window-specific effect only on a matched background; otherwise describe the complete configuration.

## Audit the Decision
Use audit_action_batch with search_focus and, for the first decision, stage_partition. The shared validator checks temporal partitioning, actual-parent Early directions, initial independent probes, focus changes, required Early revisits, branch credit, novelty, evidence and cumulative budget. Audit receipts do not represent completed inference. Read any duplicate response and use its source ID as historical evidence while repairing proposals.

Use only exposed tools. Preserve the configured sampling dynamics, chemical coverage, endpoint behavior, matching, first-window shape and terminal evaluation. Stage labels do not change these operations. Do not request extra seeds, early stopping, an evaluator model or unbudgeted sampling.
