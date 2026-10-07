# Seed 42 terminal comparison — round 1

All arms complete 100 integration steps. New guided arms use no particle resampling; control ends at the learned window boundary.

| Metric | Original Steer | Native | coordinate_r01_skill_ablation |
|---|---:|---:|---:|
| Connected valid yield | 0.98 | 0.98 | 0.95 |
| PB dock_fast yield | 0.98 | 0.98 | 0.94 |
| Unique canonical graphs | 24 | 72 | 56 |
| Unique scaffolds | 15 | 56 | 44 |
| Unique graph diversity | 0.643144 | 0.76103 | 0.735255 |
| Valid-pose target head | 7.50114 | 7.4423 | 7.59315 |
| Unique-first-pose target head | 7.26822 | 7.30618 | 7.4776 |
| Valid-pose off-target head | 6.45027 | 6.7858 | 6.93068 |
| MMFF relief / heavy atom | 0.398679 | 0.501441 | 0.51029 |
| Surround relaxation RMS (A) | 0.202921 | 0.306274 | 0.325274 |
| Surround severe clash fraction | 0 | 0 | 0 |
| Final regional q/r2 | 3.49496 | 18.4379 | 7.07337 |

MMFF is same-graph local relaxation relief, not binding free energy. FLOWR affinity-head predictions share the training/selection oracle. PB dock_fast is a structural subset.

The historical Steer arm is an original result, not a new randomized treatment. Fixed-seed development batches cannot establish generalization or causality.

Per-batch effects, missing-energy counts and candidate failures remain in the linked source reports; no composite score substitutes for these checks.
