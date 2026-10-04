"""Read-only inventory with file sizes; does not equate archive and array bytes."""
import argparse
from collections import defaultdict
from pathlib import Path
import zipfile

from evomolsteer.io import write_json

p = argparse.ArgumentParser()
p.add_argument("--project", required=True)
p.add_argument("--output", required=True)
a = p.parse_args()
root = Path(a.project).resolve()
totals, raw = defaultdict(lambda: {"bytes": 0, "files": 0}), defaultdict(lambda: {"bytes": 0, "files": 0})
trajectories = []
for path in sorted(root.rglob("*")):
    if not path.is_file() or ".venv" in path.parts:
        continue
    rel = path.relative_to(root)
    if rel.parts[:2] in [("data", "optimized"), ("test", "storage_trials")]:
        continue
    key = "/".join(rel.parts[:2])
    totals[key]["bytes"] += path.stat().st_size
    totals[key]["files"] += 1
    if rel.parts[:2] == ("data", "raw"):
        name = path.name
        category = ("trajectory.npz" if name == "trajectory.npz" else
                    "restart_anchors" if name.startswith("restart_step_") else
                    "endpoint_anchors" if name.startswith("endpoint_step_") else
                    "initial_states" if name == "initial_state.pt.gz" else
                    "final_predictions" if name == "final_prediction.pt.gz" else "other")
        raw[category]["bytes"] += path.stat().st_size
        raw[category]["files"] += 1
        if name == "trajectory.npz":
            with zipfile.ZipFile(path) as z:
                trajectories.append({"path": rel.as_posix(), "npz_bytes": path.stat().st_size,
                                     "expanded_npy_members_bytes": sum(i.file_size for i in z.infolist()),
                                     "arrays": len(z.infolist())})
write_json(a.output, {"project": str(root), "size_unit": "file length in bytes, not NTFS allocated clusters",
                      "excluded": [".venv", "data/optimized", "test/storage_trials"],
                      "directory_totals": dict(totals), "raw_categories": dict(raw), "trajectories": trajectories})
