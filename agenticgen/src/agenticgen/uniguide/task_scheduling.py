"""The UG04 teacher/configuration loops are the only supported GPU schedule."""

POLICY = "ug04-930db032-serial"
BASELINE_COMMIT = "930db03280c66c3f8a39f131ae01df3202544af4"


def configure(config):
    if "gpu_execution" in config:
        raise ValueError("gpu_execution was removed; use UG04 serial task scheduling")
    if config.get("task_scheduling", POLICY) != POLICY:
        raise ValueError("Only UG04 serial task scheduling is supported")
    return dict(config, task_scheduling=POLICY)
