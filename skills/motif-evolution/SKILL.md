---
name: evomolsteer-motif-evolution
description: Coordinate-space motif discovery, lineage controls and bounded evolution of differentiable reward programs.
---

Use coordinate-analyst and coordinate-designer contracts first. The molecular
sampling tree is a genealogy of dependent clones, not a Brownian phylogenetic
tree. Never apply independent contrasts to zero-length cloned branches or count
descendants as independent statistical replicates. Batch-level inference remains
the primary uncertainty unit. Equal-root and within-root statistics diagnose
genealogical frequency confounding; they do not establish causality.

Analyst workflow:
1. Verify trajectory hashes, selection flags, receptor frame, scale and exact
   complete learning window. Distinguish score t, native proposal t+dt and actual
   boundary state. Include selected/unselected seeds and attempted failures.
2. Run mine_spatial_motifs.py on discovery only. Read unweighted, selected,
   end-window retained, native-background and root-balanced contrasts separately.
   Match foreground/background at the same event; earlier SMC history remains.
   For actual-boundary survival targets, also run mine_boundary_motifs.py and
   use state_time to exclude the end score's outside-state proposal. Project
   descendant counts onto PRE-selection candidates; verify conserved mass.
   No final t=1 labels enter reward learning. Distinguish integrated tail tests
   from whole-curve tests and report both FDR families, including nulls.
3. Compare partial affinity and lag gain with nuisance controls, q and time
   coverage. Undefined within-root contrasts at singleton roots are missing,
   never zero or a negative result. Do not pick spatial directions from a signal
   that disappears after global pose/size/type-count adjustment.
4. Shell/pair motifs are smooth geometrical descriptors; they lack the chemical
   prerequisites needed to call PLIP hydrogen bonds or other interactions. Early
   categorical forecasts annotate noisy slots rather than settled chemistry.
5. Use complete-window fitted functions and derivatives descriptively. Bounded
   [0,1] motifs use a sigmoid output transformation; no unbounded polynomial is
   accepted as a physically legal target. Reward targets use measured node
   distributions. Report tiny kernel support and unit-gradient amplification.
6. Store a compact evidence/formula/decision ledger: feature IDs, statistics,
   source hashes, assumptions, counterevidence, hypotheses and falsifiers. Do not
   request or publish an internal reasoning transcript.

Designer workflow:
1. Compare joint centroid/shell/pair mixtures, pair-only, shell-only and bounded
   selected/background density contrast. Treat non-significant motifs as explicit
   exploratory representation ablations. Do not claim discovered affinity causes.
2. Freeze model-independent coordinate observables, conditional forecast masks,
   units, empirical scales/covariances, formula and dynamic support window.
   A direct proposal reward differentiates x; an endpoint loss would require
   FLOWR's real endpoint Jacobian. The derivative path must be explicit.
3. Test live derivatives and full zero/native trajectory equivalence. Native
   atom/bond channels remain free. No graph equality, novel-composition veto or
   ligand pair-distance acceptance threshold. Pair-distance distribution rewards
   are permitted and do not lock the graph.
4. Separate formula weights from external dose. Predictive linear-flow RMS dose
   avoids measuring the stochastic startup contraction as desired control.
   Smooth dose schedules stay positive over the learned window; after its end,
   continue native inference to t=1 without reward or particle resampling.
5. Evolve reward PROGRAMS using a Pareto archive of head, boundary shape, validity,
   MMFF median/p90, surrounding relaxation RMS and coverage. This is inspired by
   multiobjective evolutionary search, not an implementation of full NSGA-II and
   not genetic selection of generated molecular candidates.
6. Regressions trigger restoration/alternative exploration, not premature
   scientific stopping. Respect the explicit round budget and technical failure
   gates. Push each frozen plan before remote pull/inference. Verify report
   retention before cleanup; protect original Steer, checkpoints and runtime.
7. Freeze the discovery winner before validation/heldout. Never tune with their
   results. Report all/valid/unique-head means, failure-inclusive denominators,
   energy coverage and tails; historical Steer is not a paired randomized arm.

Current reference sources and transfer limits are documented in
docs/motif_evolution_workflow_20261006.md. No extra fitted neural model is required;
the LLM proposes code/formulas and the deterministic scripts calculate evidence.
