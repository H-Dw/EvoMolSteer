# CK2 single-target selection imitation: frozen experiment design

Status before generation: a geometric selection-imitation hypothesis, not a validated affinity mechanism.

## Discovery and provenance

The 40 single/unguided raw trajectories on SCNet match the local copies byte for byte (SHA256), 20 independent batches per arm, 50 candidates per batch. Analysis uses only actual 51 resampling events at score times 0, .01, …, .5, with separate predicted-endpoint and proposal representations. Discovery batches are 0–13; validation 14–16 and held-out 17–19 are not used to select reward parameters. Final outcomes never enter reward discovery.

Primary model: `flowr_root_v2.ckpt`, SHA256 `f28e863b2b208718f3d3f85f09837c2f907a71123436db6f98a25d2f1858b6a0`, identical to the historical experiment. The newer v2.2 is not used for this matched comparison.

The single-target evidence exporter now selects the configured `evidence_arm`. The original joint-arm default could not be used unchanged for this experiment. No feature family or inference definition was removed.

## Observations and hypotheses

1. At score time [0,.1), CK2 VAL116 normalized soft-min distance shifts by −0.00845 Å under expected selection, with 14/14 discovery batches negative, BH q≈.00502. This suggests a weak preference for forecast ligand placement near this receptor region. It does not identify a hydrogen bond or demonstrate that making any ligand closer raises affinity.
2. At [.1,.2), ligand radius of gyration shifts by −0.00604 Å, with the same batch direction consistency and q≈.00502. This motivates a separate compactness ablation, not an obligatory second target.
3. Selected and uniform distributions overlap strongly. Same-parent endpoint effects are zero and corresponding proposal effects are not significant. Reward discovery therefore addresses the model's endpoint forecast; it does not claim that the inherited proposals already encode a causal advantage.
4. Neighboring GLU114/HIS115/VAL116/ASN117 signals are correlated candidate explanations. One representative region avoids repeatedly rewarding the same placement change. ASN118 has opposing all-atom and heteroatom directions; uniformly attracting every atom toward the whole patch is not supported. Stage-3/4 null results do not prove no late role, but provide no basis for adding a late reward.

## RewardDesignIR

- **Outcome:** imitate the two declared endpoint selection preferences; test CK2 affinity rescore separately.
- **Coordinate state:** native current coordinates x; prediction hθ(x,t; detached previous self-conditioning); world endpoint y = scale·hθ + target pocket COM. Receptor points and all active ligand slots use this same Å frame.
- **Core target:** decrease VAL116 distance until its discovery p-weighted median 4.374834436 Å during [0,.1). Scale 0.127568599 Å. Optional compactness target 3.587680966 Å, scale 0.074939683 Å, during [.1,.2).
- **Distance definition:** z = −τ log(mean over eligible atom–region pairs of exp(−distance/τ)), τ=.25 Å. It is not a minimum contact distance. Rg is the unweighted all-active-atom radius of gyration.
- **Shape:** u=max((z−target)/scale,0); φ(u)=u²/2 for u≤1, else u−1/2. R=−Σ gate(t)·φ(u). This C1, one-sided Huber penalty stops attracting below the empirical median and bounds feature-space slope. The median is a conservative stopping convention, not a learned optimum.
- **Time support:** exact stage support with .02-time-unit smoothstep taper; the stage starting at zero has full onset at zero. There is no reward after .2, although the historical selection scope extends to .5.
- **Architecture comparison:** a symmetric empirical-IQR well can push a sample with an already smaller feature upward and can suppress diversity. An unbounded linear attraction keeps pulling past the observed population. The chosen one-sided saturating penalty avoids both extrapolations. A two-stage composite is retained only as an ablation against the regional-only program.
- **Function lineage:** MolThinker knowledge table G01 (flat-bottom/one-sided geometric penalties); bounded-slope Huber continuation and discovery-selected median are explicit adaptations. S02 population softmax is a sampling operator, not a coordinate-gradient function. No extra fitted network or LLM numerical regression is used.
- **Live control:** g=∇x R(scale·hθ(x,t)+COM). First perform the unchanged native stochastic/categorical integrator; add strength·dt·g, capped by one scalar per molecule so the largest atom displacement is at most .025 Å. Strength .05 is an uncalibrated, predeclared pilot parameter. This is approximate gradient control, not an exact tilted-distribution sampler.
- **Inherited semantics:** same live-current-coordinate pullback, detached step history, native final head and reward-ascent sign as MolExecutor. Unlike its current direct scalar engine, this implementation explicitly adds evidence-scoped gates and a recorded displacement cap. All molecules in a batch are guided independently, without dividing gradients by batch size.
- **Constraint/diagnostic distinction:** finite-value checks and per-step displacement cap are enforced. Steric clashes, validity, connectivity, diversity and final affinities are measured outcomes, not hidden acceptance filters. A step cap alone does not guarantee steric safety.
- **Uncertainty:** geometry is associated with selection weights; target imitation can leave affinity unchanged or lower it. No claim of generalization, chemical causality, or optimal generation path is made.

## Predeclared validation

1. Local finite-difference, rigid-motion, batch-size scaling, exact-support, saturation, nonfinite-rejection and cap tests.
2. GPU finite differences through the actual FLOWR endpoint to current native coordinates; no model parameter gradients.
3. A full 100-step weight-zero run must reproduce the passive native output bit for bit, with matched initial state and RNG.
4. Pilot: seed 20261004, B=8, regional and composite controls plus weight-zero reference. Retain all failures and diagnostics. Do not select reward parameters from final affinity results.
5. Comparison: new seed 20261005; 4 independent batches of 50 samples, with unguided, single-target SMC, regional gradient, and regional+compact gradient. The same prior and initial sampler RNG are used within each batch. SMC consumes extra resampling RNG; later noise draws are consequently not claimed identical across SMC and gradient arms. Run all 100 native integration steps; SMC selects at 0–.5, while the declared reward acts only during its supported early stages.
6. Primary outcome is final CK2 endpoint rescore; also report valid/connected fraction, unique SMILES, hard clashes, and the actual regional observable. Report batch-paired differences and uncertainty; 4 batches form an exploratory comparison, not a confirmatory efficacy study. Historical 1,000-sample means are background, not paired controls.

## Reference implementation provenance

Remote MolSteer files inspected on 2026-10-04:

- `src/molsteer/molexecutor/flowr.py`: SHA256 `9de70745b16eac4cc21c5de4087913da7017f36f6f9550c3a4c0ace2b95b4df8`.
- `src/molsteer/molexecutor/engine.py`: SHA256 `91655af1cc1d4121fa732673150c77a0ee76aa23d9add55820597cd5fd163da4`.
- `src/molsteer/molthinker/knowledge.py`: SHA256 `32996130511bb625e1027d216fa18fd9e7c82af75c5763d356916f1f48ddc63e`.

Relevant local skill: `C:/Users/Wei/.codex/skills/molthinker-reward-creativity/SKILL.md`. The program is a constrained data document; LLM-produced Python is not executed.
