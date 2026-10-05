# Seed 42 terminal comparison — round 2

All arms complete 100 integration steps. New guided arms use no particle resampling; control ends at the learned window boundary.

| Metric | original_Steer | native | global_MMD | local_interval_r02 |
|---|---:|---:|---:|---:|
| Connected valid yield | 0.98 | 0.98 | 0.99 | 0.95 |
| PB dock_fast yield | 0.98 | 0.98 | 0.99 | 0.95 |
| Unique canonical graphs | 24 | 69 | 63 | 66 |
| Unique scaffolds | 15 | 63 | 56 | 59 |
| Unique graph diversity | 0.643144 | 0.786638 | 0.774907 | 0.781848 |
| Valid-pose target head | 7.50114 | 7.43704 | 7.56836 | 7.46592 |
| Unique-first-pose target head | 7.26822 | 7.27187 | 7.42209 | 7.31358 |
| Valid-pose off-target head | 6.45027 | 6.72411 | 6.82606 | 6.732 |
| MMFF relief / heavy atom | 0.398679 | 0.548773 | 0.555021 | 0.530126 |
| Surround relaxation RMS (A) | 0.202921 | 0.327214 | 0.301723 | 0.350825 |
| Surround severe clash fraction | 0 | 0 | 0 | 0 |
| Final regional q/r2 | 3.49496 | 35.852 | 15.2509 | 20.8537 |

MMFF is same-graph local relaxation relief, not binding free energy. FLOWR affinity-head predictions share the training/selection oracle. PB dock_fast is a structural subset.

The historical Steer arm is an original result, not a new randomized treatment. Fixed-seed development batches cannot establish generalization or causality.

Per-batch effects, missing-energy counts and candidate failures remain in the linked source reports; no composite score substitutes for these checks.
