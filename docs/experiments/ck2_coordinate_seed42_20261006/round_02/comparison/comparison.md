# Seed 42 terminal comparison — round 2

All arms complete 100 integration steps. New guided arms use no particle resampling; control ends at the learned window boundary.

| Metric | Original Steer | Native | Global spread |
|---|---:|---:|---:|
| Connected valid yield | 0.98 | 0.98 | 0.98 |
| PB dock_fast yield | 0.98 | 0.98 | 0.98 |
| Unique canonical graphs | 24 | 69 | 71 |
| Unique scaffolds | 15 | 63 | 64 |
| Unique graph diversity | 0.643144 | 0.786638 | 0.786155 |
| Valid-pose target head | 7.50114 | 7.43704 | 7.42988 |
| Unique-first-pose target head | 7.26822 | 7.27187 | 7.25592 |
| Valid-pose off-target head | 6.45027 | 6.72411 | 6.73474 |
| MMFF relief / heavy atom | 0.398679 | 0.548773 | 0.545973 |
| Surround relaxation RMS (A) | 0.202921 | 0.327214 | 0.340162 |
| Surround severe clash fraction | 0 | 0 | 0 |
| Final regional q/r2 | 3.49496 | 35.852 | 37.8367 |

MMFF is same-graph local relaxation relief, not binding free energy. FLOWR affinity-head predictions share the training/selection oracle. PB dock_fast is a structural subset.

The historical Steer arm is an original result, not a new randomized treatment. Fixed-seed development batches cannot establish generalization or causality.

Per-batch effects, missing-energy counts and candidate failures remain in the linked source reports; no composite score substitutes for these checks.
