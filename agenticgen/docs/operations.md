# Deployment and evidence boundaries

Use only root@ssh.zzai.scnet.cn:11762. Preserve the audited GPU environment.
Create a task-local venv using the audited interpreter and system site packages;
install this project's wheel with `--no-deps`. Install missing non-GPU dependencies
at their recorded versions into that task environment, then verify imports and
record their origins. Do not inherit legacy dataset `deps` via PYTHONPATH.

`scripts/run.sh` uses the task environment, activates DTK and inherits API/proxy
settings without writing their values. AGENTICGEN_PYTHON and
AGENTICGEN_HOST_CONFIG may name explicit deployment paths. The package itself is
independent of the working directory. Review operations require the exact Git
checkout identified by AGENTICGEN_REPOSITORY.

Every code change is tested and committed locally, pushed to GitHub, and deployed
at that exact commit. Record the package import path, all source hashes, dependency
versions, model/native source identity and config hashes before running.

Existing results are immutable. Never rebind PREPARED or ACCEPTANCE to changed
scientific settings. New protocols get new result roots and acceptance evidence.
Review approval binds actual session evidence, coverage, committed skills and the
deployed revision. A generated index is not a substantive master-agent review.

Quality must finish before verified TOP1 publication and structural cleanup.
Retain all analysis, prompts, model responses, XML, errors and cleanup manifests.
Recovery reuses completed molecules and requests; it cannot fabricate decisions.
