# Seed 42 terminal comparison — round 28

All arms complete 100 integration steps. New guided arms use no particle resampling; control ends at the learned window boundary.

| Metric | Original Steer | Native | coordinate_r28_affinity30 |
|---|---:|---:|---:|
| Connected valid yield | 0.98 | 0.93 | 0.96 |
| PB dock_fast yield | 0.98 | 0.93 | 0.96 |
| Unique canonical graphs | 24 | 60 | 54 |
| Unique scaffolds | 15 | 52 | 42 |
| Unique graph diversity | 0.643144 | 0.770015 | 0.731137 |
| Valid-pose target head | 7.50114 | 7.47235 | 7.62475 |
| Unique-first-pose target head | 7.26822 | 7.30167 | 7.48299 |
| Valid-pose off-target head | 6.45027 | 6.72047 | 6.91151 |
| MMFF relief / heavy atom | 0.398679 | 0.539055 | 0.514699 |
| Surround relaxation RMS (A) | 0.202921 | 0.367456 | 0.298364 |
| Surround severe clash fraction | 0 | 0 | 0 |
| Final regional q/r2 | 3.49496 | 11.7755 | 5.86253 |

MMFF is same-graph local relaxation relief, not binding free energy. FLOWR affinity-head predictions share the training/selection oracle. PB dock_fast is a structural subset.

The historical Steer arm is an original result, not a new randomized treatment. Fixed-seed development batches cannot establish generalization or causality.

Per-batch effects, missing-energy counts and candidate failures remain in the linked source reports; no composite score substitutes for these checks.
