"""Existing-molecule quality metrics."""
from .io import read

def metrics_folder(folder, template):
    complete = folder / "complete.json"
    if not complete.exists():
        return {"quality_evaluated": False, "quality_failed": (folder / "failure.json").exists()}
    result = read(complete)
    p = read(folder / "interactions.json")
    teacher = {tuple(x) for x in template["fingerprints"]}
    current = {tuple(x) for x in p["fingerprints"]}
    return {**result["quality"], "quality_evaluated": True, "quality_failed": False,
            "plip_recovery": len(teacher & current) / len(teacher),
            "plip_new_fingerprints": len(current - teacher),
            "plip_interaction_count": len(p["interactions"])}
