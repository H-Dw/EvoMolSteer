# Seed 42 terminal comparison — round 27

All arms complete 100 integration steps. New guided arms use no particle resampling; control ends at the learned window boundary.

| Metric | Original Steer | Native | coordinate_r27_affinity30 |
|---|---:|---:|---:|
| Connected valid yield | 0.98 | 0.95 | 0.99 |
| PB dock_fast yield | 0.98 | 0.94 | 0.98 |
| Unique canonical graphs | 24 | 72 | 50 |
| Unique scaffolds | 15 | 66 | 45 |
| Unique graph diversity | 0.643144 | 0.783347 | 0.74663 |
| Valid-pose target head | 7.50114 | 7.46381 | 7.59531 |
| Unique-first-pose target head | 7.26822 | 7.35143 | 7.4279 |
| Valid-pose off-target head | 6.45027 | 6.67634 | 6.86766 |
| MMFF relief / heavy atom | 0.398679 | 0.563659 | 0.519157 |
| Surround relaxation RMS (A) | 0.202921 | 0.327379 | 0.298353 |
| Surround severe clash fraction | 0 | 0 | 0 |
| Final regional q/r2 | 3.49496 | 7.50084 | 6.51528 |

MMFF is same-graph local relaxation relief, not binding free energy. FLOWR affinity-head predictions share the training/selection oracle. PB dock_fast is a structural subset.

The historical Steer arm is an original result, not a new randomized treatment. Fixed-seed development batches cannot establish generalization or causality.

Per-batch effects, missing-energy counts and candidate failures remain in the linked source reports; no composite score substitutes for these checks.
