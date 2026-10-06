# Coordinate motif discovery and reward-program evolution

The objective is higher predicted affinity with terminal geometry and isolated-
ligand energy diagnostics at least comparable to historical Steer. This is an
experimental target, not an achieved result. Native graph changes remain free.

## What transfers from published methods

[Felsenstein (1985)](https://ichthyology.usm.edu/courses/multivariate/Felsenstein_1985.pdf)
explains why related observations cannot be treated as independent and derives
contrasts under an evolutionary model. SMC clone genealogy lacks those Brownian
branch assumptions. We instead retain batch inference, collapse parent copies
for lag gain, and add equal-root/within-root diagnostics. These are adaptations,
not classical phylogenetic independent contrasts.

[STREME (Bailey, 2021)](https://pubmed.ncbi.nlm.nih.gov/33760053/) motivates contrast
between foreground and background motifs. Here a motif is a smooth distribution
of three-dimensional distances, not a sequence motif. The same event's unweighted
candidates are the foreground selection background; native candidates are a
separate reference. Neither removes selection-oracle tautology or earlier SMC
history. Matching time, pose/size/type nuisance and independent batches is essential.

[PLIP (Salentin et al., 2015)](https://academic.oup.com/nar/article/43/W1/W443/2467865)
combines interaction chemistry with geometry. Our distance kernels only measure
geometry. Calling them hydrogen bonds or regional binding energies would require
additional donor/acceptor/aromatic/charge and angular evidence on settled graphs.

[NSGA-II (Deb et al., 2002)](https://web.njit.edu/~horacio/Math451H/download/2002-6-2-DEB-NSGA-II.pdf)
motivates retaining nondominated alternatives. We apply Pareto comparisons to
reward programs, with bounded predeclared architecture/dose variations. This
small campaign is not full NSGA-II: no crossover, crowding-distance tournament
or molecular genetic population is introduced into FLOWR inference.

## Deterministic analysis and compact storage

`scripts/mine_spatial_motifs.py` accepts dataset, campaign, analysis catalogue,
regions, batches and output. It streams every actual selection node across the
single observed window. No 0.1 subwindows and no particle/edge feature cache.
Nine observables per region/channel: centroid x/y/z, smooth radial shell densities
at 2/4/6 A (width 1 A), and pair densities at 1.5/3/5 A (width .75 A). Channels all,
NOS and NOS–C include surrounding carbon in the relative-coordinate gradient.
These widths are declared hypotheses, not learned interaction cutoffs. Existing
geometry/shape/transport/composition modules remain complementary evidence.

The foreground/background, retained descendant and native contrasts are kept
separate. Equal-root weighting removes clone frequency from one diagnostic;
within-root residual correlations remove between-root offsets. Starting head,
global position/size and forecast NOS count control lag associations. The
independent uncertainty unit is batch. Coverage, p/q and missing values remain.
Global Legendre functions choose degree by leave-one-batch-out one-SE error;
bounded motifs use sigmoid output functions. Their temporal derivative is not a
spatial reward gradient. Reward references use exact measured nodes instead.

Outputs are float64 compressed batch/node summaries, evidence CSVs, function JSON,
lineage diagnostics and source hashes. No per-node structures are copied into
LLM prompts. The Analyst gets this compact evidence and the Designer gets its
review, feature definitions, formula contract and counterevidence. Original
trajectories remain the reproducible source.

`scripts/mine_boundary_motifs.py` separately accepts dataset/campaign/mining/output
and channels. It projects descendant counts backwards from the last ACTUAL
proposal state at the learned boundary, not the outside-state proposal of the
last scored event. With native dt=.01 and window [0,.5], score nodes 0 through
.49 correspond to proposal states .01 through .5; score .5 proposes .51 and is
excluded from this ancestry target. This alignment is derived from stored
state_time, never a hardcoded cutoff. No terminal t=1 outcome is read. The new
survival references keep the final .5 score frame as unweighted context only;
the inference controller never injects it beyond the learned state boundary.

Boundary mining preserves retained-vs-background and equal-surviving-root shifts,
weighted quartile-tail enrichment, batch intervals, whole-window change rates,
LOBO one-SE effect curves and derivative intervals. A separate whole-curve
batch sign-flip test catches cancellation of opposing time effects. Tail tests
use declared strict quartiles and are not interaction thresholds. Descendant
counts are retrospective weights, not additional replicates. Survival-conditional
feature density approximates retained ancestral geometry; it is not an exact
Doob transform or a demonstrated causal affinity gradient. The compact working
output is 2.7 MB with no per-candidate/edge expansion. Source and code hashes
permit rebuilding it from protected trajectories.

`scripts/motif_agents.py` exposes export/import/api/compile actions for Analyst
and Designer. The bounded packet includes per-metric counts/null evidence, top
rows, definitions and relevant functions; it does not include structures. Designer
export requires a grounded Analyst response. Compilation checks exact evidence
and reference hashes, whitelist/observable/window/units and declared dose. It
compiles a configuration into registered torch formulas, never runs arbitrary
LLM Python. API transport uses the existing EVOMOLSTEER_BASE_URL/MODEL/API_KEY
variables and strict JSON responses. Subagent simulation follows the same
export/import/compile path without making a billed HTTP call.
Optional `--boundary` attaches compact actual-boundary evidence, its separate
whole-curve/integrated test counts and fitted effect/rate intervals. Designer
declares `target_definition` as instantaneous or boundary_survival; compilation
rejects a reference with a different lineage target. A boundary design requires
measured boundary evidence. The additional request is about 359 KB, with source
hashes and no per-candidate structures.

## Reward and dose

For measured feature vector z=F(x; detached forecast labels/anchor)/sigma, use
selected equal-batch mode means mu_m(t), covariance C_m(t), and
q_m=(z-mu_m)^T C_m^-1 (z-mu_m)/d. The imitation reward is
tau*log(mean_m exp(-delta^2*(sqrt(1+q_m/delta^2)-1)/tau)).
An alternative uses b*tanh((log p_selected(z)-log p_background(z))/b), with
normalized Gaussian determinants and empirical same-event background moments.
The contrasts are statistical approximations, not learned causal affinity forces.
Contrast dose retains both tanh saturation and a weak-contrast amplitude
sqrt(2*k/(1+2*k)), where k is mean paired-Gaussian KL divided by dimension.
This is not the KL of the full mixture or a significance/causality estimate.

Coordinates of all participating pair slots receive gradients, allowing a local
polar group and surrounding carbons to move together. Centroid, shell and pair
ablations test which part affects final head and physical tails. The frozen
conditional forecast is refreshed every native step. No chemical graph is fixed.

Dose is calibrated to dt*(endpoint-current)/(1-t) for the verified linear FLOWR
schedule, rather than the SDE startup contraction. Atom/path caps and a severe
new receptor-clash check bound displacement. They are not graph vetoes. Optional
positive continuous schedule (.1+.9*u)^p varies intensity across the learned
window; the reward remains eligible at every native step in that window. After
the learned end, all remaining native inference completes without gradient or SMC.

## Fifteen-round execution contract

New campaign: `configs/experiments/ck2_motif_seed42_v1/campaign.json`; previous30
rounds are immutable. R1 checks native/zero/new reward equivalence; R1–8 compare
declared representations/contrast/dose. After R1/R2 showed physical tradeoffs,
R9/10 were revised BEFORE their dispatch to compare actual-boundary surviving
ancestry targets (joint all-atom versus polar-carbon pair geometry). R11 restores
a balanced Pareto parent at half dose; R12 restores an affinity Pareto parent
with a positive continuous ramp. Original R1–8 definitions stay frozen.
Before R11/12 dispatch, normalization of immediate-selection targets was also
prospectively restricted to actual in-window proposal states. These two rounds
test restored parents with strict scales plus the originally declared dose
variation; the effects cannot be attributed to either change alone. R13's final
winner is chosen only among strict-scope R9–12. This prevents returning a
historical outside-state scale as the final learned-window program. R1–8 remain
honest historical screens, with their 51-score-grid normalization limitation.
R13 freezes a discovery winner and validates batches14/15. R14–15 replay that
same winner on heldout17–19 without further tuning. Master seed42; batch seed
42+100003*index; native100 integration steps. Whole-window support is read from
the learning reference, never hardcoded in the reward.

`scripts/resume_motif_campaign.py` runs locally. It freezes plans, records evidence,
commits/pushes through the local VPN, requires remote pull to match the commit,
dispatches inference, downloads checksum-verified archives, evaluates locally,
retains reports/failures and deletes disposable generated outputs. Remote raw
outputs are removed before the next inference and after the final retained
report. Original Steer/checkpoints/runtime are protected. Credentials stay in
process memory; reports contain no secret. Technical failures stop safely and
can resume without duplicating immutable dispatch identities.

Report all/valid/unique predicted head, validity/PB yield, diversity, MMFF coverage
and median/p90, surrounding relaxation displacement, actual boundary shape and
per-batch changes. MMFF is isolated-ligand relaxation, not receptor binding free
energy; surrounding displacement is not measured pocket compatibility. Source
selection and terminal head share an oracle. Independent affinity calibration
or experiments would still be required for a biological efficacy claim.

`scripts/summarize_motif_campaign.py --evidence <retained-report-folder> --output
<report-folder>` accepts retained reports only, and `--require-complete` checks
all fifteen outcomes. Heldout aggregation combines all attempted candidates on
batches17/18/19, keeps energy coverage and unique-head separate, and reports
matched per-batch directions. Graph equality is never required. Three independent
batches support a descriptive check, not candidate-level significance. The R9
intervention directly consumes the grounded Analyst/Designer compiled program,
checks its predeclared numeric contract and stores its SHA with the frozen plan.
