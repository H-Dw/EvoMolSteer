# Seed 42 terminal comparison — round 29

All arms complete 100 integration steps. New guided arms use no particle resampling; control ends at the learned window boundary.

| Metric | Original Steer | Native | coordinate_r29_affinity30 |
|---|---:|---:|---:|
| Connected valid yield | 0.98 | 0.97 | 0.95 |
| PB dock_fast yield | 0.98 | 0.97 | 0.95 |
| Unique canonical graphs | 24 | 71 | 53 |
| Unique scaffolds | 15 | 61 | 41 |
| Unique graph diversity | 0.643144 | 0.778695 | 0.737774 |
| Valid-pose target head | 7.50114 | 7.40985 | 7.5456 |
| Unique-first-pose target head | 7.26822 | 7.28792 | 7.38741 |
| Valid-pose off-target head | 6.45027 | 6.75642 | 6.87805 |
| MMFF relief / heavy atom | 0.398679 | 0.548085 | 0.515586 |
| Surround relaxation RMS (A) | 0.202921 | 0.329004 | 0.277421 |
| Surround severe clash fraction | 0 | 0 | 0 |
| Final regional q/r2 | 3.49496 | 13.0423 | 10.7027 |

MMFF is same-graph local relaxation relief, not binding free energy. FLOWR affinity-head predictions share the training/selection oracle. PB dock_fast is a structural subset.

The historical Steer arm is an original result, not a new randomized treatment. Fixed-seed development batches cannot establish generalization or causality.

Per-batch effects, missing-energy counts and candidate failures remain in the linked source reports; no composite score substitutes for these checks.
