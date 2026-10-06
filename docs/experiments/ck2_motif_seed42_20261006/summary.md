# Spatial-motif seed42 campaign

15/15 rounds retained. Every reported round completed all native 100 steps; gradient acts only inside the learned dynamic window. No SMC. Chemical graphs remain free.

| Round | Split | Head Δ | Shape improvement | MMFF median improvement | MMFF p90 improvement | Surround RMS improvement |
|---:|---|---:|---:|---:|---:|---:|
|1|discovery|0.021735|3.609%|-4.441%|-28.565%|-1.682%|
|2|discovery|0.080461|7.401%|-1.239%|17.693%|-4.570%|
|3|discovery|0.015093|1.067%|0.504%|18.003%|1.159%|
|4|discovery|-0.006818|-1.827%|-0.132%|6.851%|0.763%|
|5|discovery|0.011446|1.187%|-0.234%|-27.256%|-12.552%|
|6|discovery|0.008069|2.611%|-0.056%|19.967%|-1.449%|
|7|discovery|0.001392|-0.118%|-0.035%|13.022%|2.331%|
|8|discovery|0.002342|-0.546%|-0.132%|-14.581%|0.156%|
|9|discovery|0.033956|3.508%|0.005%|31.307%|2.721%|
|10|discovery|0.024121|0.932%|-0.007%|23.853%|-4.356%|
|11|discovery|0.009523|-0.127%|-0.034%|23.667%|-2.547%|
|12|discovery|0.037198|1.775%|-0.721%|-25.979%|-10.642%|
|13|validation|0.011901|1.979%|-0.871%|13.695%|-0.533%|
|14|heldout|0.043394|-0.213%|0.450%|-12.794%|5.331%|
|15|heldout|-0.010991|2.211%|-0.394%|23.553%|5.690%|

All directions remain separate. Head is a shared model prediction; MMFF is isolated-ligand relaxation relief. Historical Steer is not a matched randomized arm.
Rounds 1–12 are adaptive reused discovery screens. Round13 validates the frozen discovery winner; rounds14–15 reuse that same frozen reward on heldout batches. No validation/heldout tuning.
Each round retains candidate failures, coverage, physical tails, frozen formulas/configuration and inference commit; disposable trajectories/archives are retired after verified report retention.

## Frozen heldout reward

|Arm / benchmark|Attempts|Predicted head|MMFF relief/heavy median|Surround relaxation RMS A|Valid / PB|Unique graphs|
|---|---:|---:|---:|---:|---|---:|
|Matched native|150|7.460158|0.526564|0.349219|140 / 139|99|
|Frozen gradient|150|7.467295|0.528508|0.329824|144 / 143|99|
|Historical discovery Steer|100|7.510351|0.398679|0.202921|98 / 98|24|

Historical Steer is not a paired heldout control. All-head, unique-head and energy availability are separate in heldout_summary.json.

|Heldout batch|Head change|MMFF median change|MMFF p90 change|Surround RMS change A|Boundary shape change A|Valid / PB count change|
|---:|---:|---:|---:|---:|---:|---|
|17|0.043394|-0.002393|0.172730|-0.017566|0.000477|1 / 1|
|18|-0.025143|0.026202|-0.734547|-0.001654|-0.007634|3 / 3|
|19|0.003161|0.003622|-0.083119|-0.038261|-0.002480|0 / 0|

Positive head and negative physical changes are favorable. Three batch contrasts are descriptive; no candidate-level significance or biological affinity proof is claimed.
