# agenticgen

An installable subproject for adaptive conditional reconstruction with FLOWR.

The current protocol is defined in [the experiment plan](docs/plan.zh-CN.md).
Condition fidelity is the primary objective, mean predicted affinity agreement
with the teacher is the functional objective, and best affinity is a bonus.
All are continuous metrics, without hard success cutoffs.

Each hypothesis uses 25 fixed starts (five per historical seed); four hypotheses
share a physical batch of 100. Native random draws preserve twenty-row logical
call shapes before selecting five rows. Every trial resumes fixed R0 at step 10.

Use a virtual environment based on the already audited GPU interpreter; do not
upgrade Torch/DTK. Install this project with `pip install --no-deps .` after its
declared dependencies have been checked. Set `AGENTICGEN_REPOSITORY` to the fixed
Git checkout for deployment/review checks. Set `AGENTICGEN_HOST_CONFIG` to an
explicit host JSON when overriding packaged external resource paths.

The installed `agenticgen` / `python -m agenticgen` entry point works independently
of the current directory. `agenticgen doctor` reports resolved non-secret paths.
See [operations](docs/operations.md). Credentials and model/data files are external.
The package never imports the old evomolsteer implementation.

Local CPU tests validate contracts and deterministic calculations. Native GPU
acceptance and complete prior-version review are separate mandatory launch gates.
