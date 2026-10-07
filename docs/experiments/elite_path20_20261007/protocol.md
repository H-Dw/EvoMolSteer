# Sequential elite-path exploration

The user authorized a fresh budget of at most 20 rounds. Each round changes one
module or one registered parameter axis, evaluates it, records the result, and
retains the incumbent before starting another intervention. Structural/mining
rounds are explicitly distinguished from actual FLOWR inference rounds.

The primary objective is final predicted target affinity and its valid high-score
tail. Energy/strain is secondary. The initial executable incumbent is frozen R26
from configs/experiments/skill_ablation_v1/incumbent.json. The checkpoint, pocket,
100 integration steps and master seed 42 remain fixed. Guidance has the same
dynamic learning support as its data; it then switches off for native continuation
to t=1. No inference resampling, affinity-head gradient or added production
affinity-head forward is allowed. Chemical graphs may change freely.

Module order: (1) sparse typed path graph, (2) decoded terminal elite credit,
(3) evidence and identity-bearing coordinate library, (4) individual registered
reward interventions, including coverage and continuity, (5) future potential.
Later modules are not optimized while their prerequisite module is unresolved.
Analyst and Designer simulations are sequential and preserve their literal
Skills, prompt, response, compilation and program hashes.

Discovery labels use the existing teacher batches 0–13. Historical results are
retrospective. Inference screens use a fixed paired initial-state panel; parameter
selection on that panel is adaptive. A winner is frozen before independent
confirmation, whose results cannot be used to call its own selection unbiased.
Reports include all attempts, invalid outputs, high-score thresholds, score means,
tail counts/unique graphs and physical diagnostics. Thresholds are frozen from
discovery data, never molecule-specific Skill text.

All source changes are made locally, tested, committed and pushed through the
configured local network interface before the remote checkout pulls that commit.
The remote host only runs inference, packages outputs and retires disposable
files; scientific analysis runs locally. Original Steer data, checkpoints and
source remain protected. Before each inference, previous outputs may be deleted
only after local scientific reports and their checksums are retained.

Each round stores its evidence and scientific derivation, rather than a private
reasoning transcript. Declines trigger diagnosis and a new single-axis trial from
a retained parent, not an overwrite of the best result. No iteration is counted
as an inference round unless it actually launched and completed that inference.
