# Seed 42 terminal comparison — round 23

All arms complete 100 integration steps. New guided arms use no particle resampling; control ends at the learned window boundary.

| Metric | Original Steer | Native | coordinate_r23_backtrack |
|---|---:|---:|---:|
| Connected valid yield | 0.98 | 0.98 | 0.99 |
| PB dock_fast yield | 0.98 | 0.98 | 0.99 |
| Unique canonical graphs | 24 | 69 | 73 |
| Unique scaffolds | 15 | 63 | 64 |
| Unique graph diversity | 0.643144 | 0.786638 | 0.777491 |
| Valid-pose target head | 7.50114 | 7.43704 | 7.4351 |
| Unique-first-pose target head | 7.26822 | 7.27187 | 7.29473 |
| Valid-pose off-target head | 6.45027 | 6.72411 | 6.73566 |
| MMFF relief / heavy atom | 0.398679 | 0.548773 | 0.543631 |
| Surround relaxation RMS (A) | 0.202921 | 0.327214 | 0.337812 |
| Surround severe clash fraction | 0 | 0 | 0 |
| Final regional q/r2 | 3.49496 | 35.852 | 30.8803 |

MMFF is same-graph local relaxation relief, not binding free energy. FLOWR affinity-head predictions share the training/selection oracle. PB dock_fast is a structural subset.

The historical Steer arm is an original result, not a new randomized treatment. Fixed-seed development batches cannot establish generalization or causality.

Per-batch effects, missing-energy counts and candidate failures remain in the linked source reports; no composite score substitutes for these checks.
