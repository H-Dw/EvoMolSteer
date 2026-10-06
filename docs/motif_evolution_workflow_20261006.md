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
declared representations/contrast/dose. R9–12 restore Pareto parents and test
smaller dose or continuous schedules using completed development outcomes.
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
