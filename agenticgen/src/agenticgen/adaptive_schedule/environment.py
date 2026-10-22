"""Allowlisted deployment identity; never serialize process environment variables."""
import importlib.metadata
import platform
import socket
from pathlib import Path

from .io import sha, read, write
from .vendor.common import EXPECTED_CHECKPOINT, EXPECTED_SAMPLER


def record_environment(root, config):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Native GPU runtime is unavailable")
    flowr = Path(config["flowr"])
    hashes = {"checkpoint": sha(flowr / "checkpoints/flowr_root_v2.ckpt"),
              "sampler": sha(flowr / "flowr/models/fm_pocket.py")}
    if hashes != {"checkpoint": EXPECTED_CHECKPOINT, "sampler": EXPECTED_SAMPLER}:
        raise ValueError("Deployment model or native sampler differs from the historical experiment")
    packages = {}
    for name in ["numpy", "scipy", "h5py", "rdkit", "plip", "posebusters", "httpx"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "unavailable"
    props = torch.cuda.get_device_properties(0)
    result = {"ssh": config["ssh"], "hostname": socket.gethostname(),
              "python": platform.python_version(), "platform": platform.platform(),
              "torch": str(torch.__version__), "hip": torch.version.hip, "cuda": torch.version.cuda,
              "device": {"name": props.name, "total_memory": props.total_memory,
                         "capability": list(torch.cuda.get_device_capability(0))},
              "packages": packages, "model_hashes": hashes, "flowr": str(flowr),
              "configured_matmul_precision": "high", "cross_server_bitwise_claim": False}
    path = Path(root) / "deployment_environment.json"
    if path.exists() and read(path) != result:
        raise ValueError("Deployment environment changed within this cohort")
    if not path.exists():
        write(path, result)
    return result
