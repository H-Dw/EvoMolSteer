# agenticgen

An installable subproject for adaptive conditional reconstruction with FLOWR.

New absolute-coordinate guidance defaults to skills **2.9.0** and the
`linear-open-domain-v2` control interface. Merge
[the new-run overlay](configs/uniguide_open_domain.json) into a complete new
absolute-coordinate configuration. It separates declared grid bounds from
measured confirmation evidence, checks action budgets before selection, and
keeps the best complete calibration measurement as the initial incumbent.
Rollback prompts carry the saved best-node decision ancestry and poor side
paths. See [the implementation and validation plan](docs/experiments/20261101_open_domain_v29/plan.zh-CN.md).
Historical explicit releases and frozen experiment roots are not migrated.

The optional [six-window overlay](configs/uniguide_six_windows_zero_start.json)
uses skills **2.10.1**, windows 0–10/11–20/21–30/31–40/41–50/51–99,
an actual zero-scale starting trial, and up to 500 calibration attempts per
teacher. Its explicit endpoint policy applies gradient updates at steps 1–98:
step 0 keeps the native transition, and step 99 returns the uncovered prediction
without computing an unused VJP. Chemical state inpainting shares the same
1–98 phase through `activation: guidance_phase`; a zero scale only disables the
extra coordinate gradient. The four-window default remains available.
See [the endpoint and calibration contract](docs/experiments/20261101_six_window_zero_start/plan.zh-CN.md).

The following describes the original conditional reconstruction campaign.

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
