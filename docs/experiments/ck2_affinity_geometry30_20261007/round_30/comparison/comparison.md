# Seed 42 terminal comparison — round 30

All arms complete 100 integration steps. New guided arms use no particle resampling; control ends at the learned window boundary.

| Metric | Original Steer | Native | coordinate_r30_affinity30 |
|---|---:|---:|---:|
| Connected valid yield | 0.98 | 0.9 | 0.97 |
| PB dock_fast yield | 0.98 | 0.9 | 0.97 |
| Unique canonical graphs | 24 | 65 | 49 |
| Unique scaffolds | 15 | 57 | 32 |
| Unique graph diversity | 0.643144 | 0.787996 | 0.70324 |
| Valid-pose target head | 7.50114 | 7.35668 | 7.65315 |
| Unique-first-pose target head | 7.26822 | 7.19029 | 7.47293 |
| Valid-pose off-target head | 6.45027 | 6.65112 | 6.97586 |
| MMFF relief / heavy atom | 0.398679 | 0.550581 | 0.497153 |
| Surround relaxation RMS (A) | 0.202921 | 0.346285 | 0.243108 |
| Surround severe clash fraction | 0 | 0 | 0 |
| Final regional q/r2 | 3.49496 | 18.3503 | 10.3692 |

MMFF is same-graph local relaxation relief, not binding free energy. FLOWR affinity-head predictions share the training/selection oracle. PB dock_fast is a structural subset.

The historical Steer arm is an original result, not a new randomized treatment. Fixed-seed development batches cannot establish generalization or causality.

Per-batch effects, missing-energy counts and candidate failures remain in the linked source reports; no composite score substitutes for these checks.
