# Extraction validation

Source revision: 6c59bbf5931d65159ec0f09c0cdad8eb04631eee.
The first extraction commit retains the old scientific configuration. It is not
permission to launch the old protocol or bind new source to existing results.

CPU regression: 33 passed, 4 skipped (native Torch-dependent tests require the
remote GPU environment). Wheel build succeeded. Tests cover registered tools,
private analysts, synthesis, review evidence gates, relative host paths and an
entry import from an unrelated working directory without old-project imports.
AST checks confirm terminal rescoring, reward functions and RNG helpers were
copied without changing their function bodies.

The former scoring module imported a whole campaign controller. Only its original
terminal-scoring function and numeric conversion helper are carried forward.
Likewise launch preflight and worker locking are extracted without legacy runners.
Skills are installed resources; review documents still use the containing Git
repository as their committed-evidence boundary.

Remote native tests and new-protocol acceptance remain separate deployment gates.
