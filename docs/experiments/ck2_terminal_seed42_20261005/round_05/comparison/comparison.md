# Seed 42 terminal comparison — round 5

All arms complete 100 integration steps. New guided arms use no particle resampling; control ends at the learned window boundary.

| Metric | original_Steer | native | local_interval_r05 |
|---|---:|---:|---:|
| Connected valid yield | 0.96 | 0.95 | 0.95 |
| PB dock_fast yield | 0.96 | 0.95 | 0.94 |
| Unique canonical graphs | 56 | 67 | 70 |
| Unique scaffolds | 41 | 59 | 61 |
| Unique graph diversity | 0.73447 | 0.775723 | 0.774601 |
| Valid-pose target head | 7.57905 | 7.50592 | 7.49607 |
| Unique-first-pose target head | 7.57458 | 7.37632 | 7.38478 |
| Valid-pose off-target head | 6.5996 | 6.81822 | 6.82977 |
| MMFF relief / heavy atom | 0.578302 | 0.53623 | 0.560457 |
| Surround relaxation RMS (A) | 0.189661 | 0.275824 | 0.311036 |
| Surround severe clash fraction | 0 | 0 | 0 |
| Final regional q/r2 | 5.66633 | 16.264 | 13.0577 |

MMFF is same-graph local relaxation relief, not binding free energy. FLOWR affinity-head predictions share the training/selection oracle. PB dock_fast is a structural subset.

The historical Steer arm is an original result, not a new randomized treatment. Fixed-seed development batches cannot establish generalization or causality.

Per-batch effects, missing-energy counts and candidate failures remain in the linked source reports; no composite score substitutes for these checks.
