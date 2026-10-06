# Seed 42 terminal comparison — round 6

All arms complete 100 integration steps. New guided arms use no particle resampling; control ends at the learned window boundary.

| Metric | Original Steer | Native | coordinate_r06_motif15 |
|---|---:|---:|---:|
| Connected valid yield | 0.98 | 0.98 | 1 |
| PB dock_fast yield | 0.98 | 0.98 | 1 |
| Unique canonical graphs | 24 | 69 | 72 |
| Unique scaffolds | 15 | 63 | 65 |
| Unique graph diversity | 0.643144 | 0.786638 | 0.786966 |
| Valid-pose target head | 7.50114 | 7.43704 | 7.44234 |
| Unique-first-pose target head | 7.26822 | 7.27187 | 7.30195 |
| Valid-pose off-target head | 6.45027 | 6.72411 | 6.71893 |
| MMFF relief / heavy atom | 0.398679 | 0.548773 | 0.549079 |
| Surround relaxation RMS (A) | 0.202921 | 0.327214 | 0.331955 |
| Surround severe clash fraction | 0 | 0 | 0 |
| Final regional q/r2 | 3.49496 | 35.852 | 31.5766 |

MMFF is same-graph local relaxation relief, not binding free energy. FLOWR affinity-head predictions share the training/selection oracle. PB dock_fast is a structural subset.

The historical Steer arm is an original result, not a new randomized treatment. Fixed-seed development batches cannot establish generalization or causality.

Per-batch effects, missing-energy counts and candidate failures remain in the linked source reports; no composite score substitutes for these checks.
