# Experiment 3 endpoint scoring and input audit

The native raw999 finalizer intentionally returns detached CPU tensors in world
angstrom coordinates. Experiment 1's `runtime.scoring.final_metrics` moves them
to the pocket device before subtracting the protein center. Experiment 3's
standalone scorer omitted that transfer and subtracted a CUDA center from CPU
coordinates. Attempt 006 completed fifty engineering trajectories and failed
only during endpoint rescoring; it reserved no scientific candidates.

`affinity_inputs.scoring_inputs` now prepares independent copies on the actual
model device, with the model floating dtype and integer ligand atom-count mask.
It applies the same world-to-pocket transformation, categorical argmax/one-hot
decoding and t=.9999 as experiment 1. The affinity head remains read-only. The
scorer checks complete, finite results and verifies that sampling RNG is unchanged.
Source endpoints, coordinates, chemistry and pockets are not modified.

The input audit also checks:

- The initial SDF is the unique frozen teacher node with the expected heavy-atom
  count. Its graph, coordinates and PLIP-controlled identities match the guidance
  feature input; the protein file hash is unchanged.
- Each current teacher's features, geometry condition, chemical encoding and
  protein frame are bound together. Proposed promotion conditions are checked
  before selection and their source row, terminal hash and frozen score are saved.
- Fresh priors use dense native adjacency probabilities and integer masks. Their
  shapes/vocabulary widths match frozen native inputs while allowing a changed
  teacher atom count. Fifty valid slots and five independent seed streams retain
  the fixed pocket and COM used by the initial baseline scorer.
- Full-graph baseline affinity scoring does not expand inpainting. Chemistry
  injection still uses only interaction-controlled atoms and their induced bonds,
  with shared geometric Hungarian correspondence during steps 1–98. Raw999 is
  uncovered. Shared generation modules and the fresh-prior adapter are unchanged.

The authorized recovery uses independent `attempt_007` and revision
`arm3_endpoint_device`. It reuses the fifty completed engineering trajectories
from attempt 006, checking an exact artifact manifest, original configuration,
teacher, checkpoint, fixed-scale provenance, all hundred steps and chemical
receipts. It recomputes geometry and affinity, requires identical terminal
correspondences, and compares all fifty pIC50 values against the unchanged
experiment-1 scoring entry point on the real GPU (absolute tolerance 1e-5).
The reuse receipt distinguishes fifty existing engineering trajectories from
zero new engineering trajectories. Missing/incompatible artifacts stop recovery;
there is no fallback resampling.

Only after this acceptance may five fresh scientific rounds run. Scales remain
`[50,100,40,60,80,200]`, fixed to experiment 1's final complete best measurement.
There is no Agent, calibration or scale adjustment. The cap remains 250 new
scientific candidates. Failed scientific reservations are retained and never
refilled. Other jobs and all prior failures remain unchanged.

Tests cover the scoring frame and input isolation, mixed CPU/GPU boundaries,
model float dtype and integer metadata, native size embeddings, teacher/frame
bindings, replay mismatch refusal, and replay followed by five fresh scientific
rounds with idempotent recovery. Local machines without FLOWR/GPU cannot prove
the native GPU path; remote CPU preflight and cached real-GPU acceptance provide
that evidence. These checks establish implementation consistency, not affinity
improvement or scientific validity of predicted pIC50.
