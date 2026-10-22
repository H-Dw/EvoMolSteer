# AgenticGen

Independently installable subproject of EvoMolSteer for evidence-driven adaptive
conditional inference control. The Python package does not import the old project.
FLOWR source, model checkpoints and datasets remain external, hash-checked inputs.

Install for CPU development: `python -m pip install -e '.[test]'` from this folder.
Run `python -m agenticgen --help`, or `agenticgen doctor`, from any directory.
Run the unit tests with `python -m pytest tests/unit`.

The native environment must preserve the audited Torch/DTK stack. Do not use a
generic Torch upgrade to satisfy runtime dependencies. See docs/operations.md.
Skills ship in the package resources; host paths are configurable independently
of experiments. No credentials, molecular datasets or weights belong in Git.

The migration manifest records original source hashes. Native scoring, coordinate
reward and RNG semantics are checked separately from subsequent protocol changes.
