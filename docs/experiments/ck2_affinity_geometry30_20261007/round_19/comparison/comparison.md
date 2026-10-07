# Seed 42 terminal comparison — round 19

All arms complete 100 integration steps. New guided arms use no particle resampling; control ends at the learned window boundary.

| Metric | Original Steer | Native | coordinate_r19_affinity30 |
|---|---:|---:|---:|
| Connected valid yield | 0.98 | 0.98 | 0.96 |
| PB dock_fast yield | 0.98 | 0.98 | 0.96 |
| Unique canonical graphs | 24 | 69 | 47 |
| Unique scaffolds | 15 | 63 | 40 |
| Unique graph diversity | 0.643144 | 0.786638 | 0.74366 |
| Valid-pose target head | 7.50114 | 7.43704 | 7.66436 |
| Unique-first-pose target head | 7.26822 | 7.27187 | 7.55808 |
| Valid-pose off-target head | 6.45027 | 6.72411 | 6.86978 |
| MMFF relief / heavy atom | 0.398679 | 0.548773 | 0.500959 |
| Surround relaxation RMS (A) | 0.202921 | 0.327214 | 0.269754 |
| Surround severe clash fraction | 0 | 0 | 0 |
| Final regional q/r2 | 3.49496 | 35.852 | 2.5456 |

MMFF is same-graph local relaxation relief, not binding free energy. FLOWR affinity-head predictions share the training/selection oracle. PB dock_fast is a structural subset.

The historical Steer arm is an original result, not a new randomized treatment. Fixed-seed development batches cannot establish generalization or causality.

Per-batch effects, missing-energy counts and candidate failures remain in the linked source reports; no composite score substitutes for these checks.
