# Seed 42 terminal comparison — round 13

All arms complete 100 integration steps. New guided arms use no particle resampling; control ends at the learned window boundary.

| Metric | Original Steer | Native | coordinate_r13_motif15 |
|---|---:|---:|---:|
| Connected valid yield | 0.98 | 0.95 | 0.98 |
| PB dock_fast yield | 0.98 | 0.95 | 0.98 |
| Unique canonical graphs | 24 | 67 | 67 |
| Unique scaffolds | 15 | 59 | 59 |
| Unique graph diversity | 0.643144 | 0.775723 | 0.771188 |
| Valid-pose target head | 7.50114 | 7.50592 | 7.51604 |
| Unique-first-pose target head | 7.26822 | 7.37632 | 7.39884 |
| Valid-pose off-target head | 6.45027 | 6.81822 | 6.83713 |
| MMFF relief / heavy atom | 0.398679 | 0.53623 | 0.540903 |
| Surround relaxation RMS (A) | 0.202921 | 0.275824 | 0.277293 |
| Surround severe clash fraction | 0 | 0 | 0 |
| Final regional q/r2 | 3.49496 | 16.264 | 13.3527 |

MMFF is same-graph local relaxation relief, not binding free energy. FLOWR affinity-head predictions share the training/selection oracle. PB dock_fast is a structural subset.

The historical Steer arm is an original result, not a new randomized treatment. Fixed-seed development batches cannot establish generalization or causality.

Per-batch effects, missing-energy counts and candidate failures remain in the linked source reports; no composite score substitutes for these checks.
