# Seed 42 terminal comparison — round 15

All arms complete 100 integration steps. New guided arms use no particle resampling; control ends at the learned window boundary.

| Metric | Original Steer | Native | coordinate_r15_motif15 |
|---|---:|---:|---:|
| Connected valid yield | 0.98 | 0.92 | 0.95 |
| PB dock_fast yield | 0.98 | 0.91 | 0.94 |
| Unique canonical graphs | 24 | 69 | 70 |
| Unique scaffolds | 15 | 58 | 60 |
| Unique graph diversity | 0.643144 | 0.787385 | 0.784454 |
| Valid-pose target head | 7.50114 | 7.44288 | 7.4272 |
| Unique-first-pose target head | 7.26822 | 7.27579 | 7.26393 |
| Valid-pose off-target head | 6.45027 | 6.75024 | 6.69818 |
| MMFF relief / heavy atom | 0.398679 | 0.525177 | 0.527247 |
| Surround relaxation RMS (A) | 0.202921 | 0.359514 | 0.339058 |
| Surround severe clash fraction | 0 | 0 | 0 |
| Final regional q/r2 | 3.49496 | 50.5495 | 39.3405 |

MMFF is same-graph local relaxation relief, not binding free energy. FLOWR affinity-head predictions share the training/selection oracle. PB dock_fast is a structural subset.

The historical Steer arm is an original result, not a new randomized treatment. Fixed-seed development batches cannot establish generalization or causality.

Per-batch effects, missing-energy counts and candidate failures remain in the linked source reports; no composite score substitutes for these checks.
