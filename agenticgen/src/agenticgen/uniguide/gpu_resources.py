"""Idle-point GPU telemetry. Missing native counters are explicit, never guessed."""
import ctypes
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime
from zoneinfo import ZoneInfo
import torch

from .io import frozen, sha, value_sha


class NativeCounters:
    def __init__(self, folder):
        self.errors = {}
        self.workspace = None
        self.queue = None
        self.binding = {}
        try:
            self._workspace_adapter(Path(folder))
        except (OSError, RuntimeError, subprocess.SubprocessError) as error:
            self.errors["workspace"] = str(error)
        try:
            self._queue_counter()
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
            self.errors["queue"] = str(error)

    def _workspace_adapter(self, folder):
        if sys.platform != "linux" or not str(torch.__version__).startswith("2.9."):
            raise RuntimeError("Workspace adapter supports Linux PyTorch 2.9 only")
        source = Path(__file__).with_name("workspace_counter.cpp")
        torch_root = Path(torch.__file__).parent
        library = torch_root / "lib" / ("libtorch_hip.so" if torch.version.hip else "libtorch_cuda.so")
        binding = dict(torch_version=str(torch.__version__), source_sha256=sha(source),
            torch_library=str(library), torch_library_sha256=sha(library),
            cxx11_abi=bool(torch._C._GLIBCXX_USE_CXX11_ABI))
        target = folder / ("workspace_" + value_sha(binding)[:16] + ".so")
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            command = ["g++", "-std=c++17", "-shared", "-fPIC", "-O2",
                "-D_GLIBCXX_USE_CXX11_ABI=" + str(int(binding["cxx11_abi"])),
                "-I" + str(torch_root / "include"), str(source), str(library),
                "-Wl,-rpath," + str(torch_root / "lib"), "-o", str(target)]
            compiled = subprocess.run(command, capture_output=True, text=True, timeout=60)
            if compiled.returncode:
                raise RuntimeError("Workspace counter compilation failed: " + compiled.stderr[-3000:])
        adapter = ctypes.CDLL(str(target))
        for name in ("ug_blas_workspace_count", "ug_blaslt_workspace_count"):
            getattr(adapter, name).restype = ctypes.c_size_t
            getattr(adapter, name).argtypes = []
        self.workspace = adapter
        self.binding["workspace"] = dict(binding, adapter_sha256=sha(target))

    def _queue_counter(self):
        if not torch.version.hip or sys.platform != "linux":
            raise RuntimeError("Native queue counter requires a symbol-bearing ROCr runtime")
        maps = Path("/proc/self/maps").read_text().splitlines()
        libraries = {line.split()[-1] for line in maps if "/libhsa-runtime64.so" in line}
        if len(libraries) != 1:
            raise RuntimeError("Expected exactly one loaded ROCr runtime")
        library = next(iter(libraries))
        symbols = subprocess.run(["nm", "-S", "-C", "--defined-only", library],
            capture_output=True, text=True, check=True, timeout=30).stdout.splitlines()
        symbol = "rocr::GPU::AqlQueue::queue_count_"
        hits = [line.split(maxsplit=3) for line in symbols if line.endswith(" " + symbol)]
        if len(hits) != 1 or int(hits[0][1], 16) != ctypes.sizeof(ctypes.c_uint32):
            raise RuntimeError("ROCr queue counter symbol or atomic width unavailable")
        class DlInfo(ctypes.Structure):
            _fields_ = [("filename", ctypes.c_char_p), ("base", ctypes.c_void_p),
                        ("symbol", ctypes.c_char_p), ("symbol_address", ctypes.c_void_p)]
        runtime = ctypes.CDLL(library)
        linker = ctypes.CDLL("libdl.so.2")
        linker.dladdr.argtypes = [ctypes.c_void_p, ctypes.POINTER(DlInfo)]
        linker.dladdr.restype = ctypes.c_int
        info = DlInfo()
        if not linker.dladdr(ctypes.cast(runtime.hsa_init, ctypes.c_void_p), ctypes.byref(info)):
            raise RuntimeError("Cannot resolve the loaded ROCr image base")
        if Path(os.fsdecode(info.filename)).resolve() != Path(library).resolve():
            raise RuntimeError("ROCr symbol resolved to another runtime image")
        base = info.base
        address = base + int(hits[0][0], 16)
        # ELF .bss may extend into an anonymous mapping after the file-backed
        # segment. Validate every readable mapping, using dladdr for the image
        # base, rather than rejecting that legitimate zero-initialized storage.
        mappings = [line.split() for line in maps]
        if not any("r" in m[1] and int(m[0].split("-")[0], 16) <= address
                   and address + 4 <= int(m[0].split("-")[1], 16) for m in mappings):
            raise RuntimeError("Native queue counter is not in a readable runtime mapping")
        self.queue = ctypes.c_uint32.from_address(address)
        self.binding["queue"] = dict(library=library, library_sha256=sha(library),
            symbol=symbol, symbol_size_bytes=4,
            counter_kind="ROCr GPU AQL queue-event reference count; not submitted kernel count")

    def counts(self):
        return dict(blas_workspace_count=(self.workspace.ug_blas_workspace_count() if self.workspace else None),
            blaslt_workspace_count=(self.workspace.ug_blaslt_workspace_count() if self.workspace else None),
            native_gpu_queue_count=(self.queue.value if self.queue is not None else None),
            unavailable=self.errors)


def snapshot(executor, counters):
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return dict(timestamp=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        pid=os.getpid(), allocated_bytes=torch.cuda.memory_allocated(),
        reserved_bytes=torch.cuda.memory_reserved(), free_device_bytes=free, total_device_bytes=total,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        gpu_workers=executor.workers, persistent_worker_pools=len(executor.slots),
        persistent_streams=(1 if executor.workers == 1 else len(executor.slots)),
        owned_side_streams=len(executor.slots), **counters.counts())


def plateau(rows, *, warmup=2, tolerance_bytes=32 << 20):
    """A bounded stability claim: every post-warmup batch must remain on the plateau."""
    stable = rows[warmup:]
    if len(stable) < 4:
        raise ValueError("Resource acceptance requires at least four post-warmup batches")
    if any(row.get("unavailable") for row in stable):
        raise ValueError("Native workspace or queue telemetry unavailable")
    counts = ("blas_workspace_count", "blaslt_workspace_count", "native_gpu_queue_count",
              "persistent_worker_pools", "persistent_streams")
    changes = {key: [row[key] for row in stable] for key in counts
               if any(row[key] != stable[0][key] for row in stable)}
    growth = {key: max(row[key] for row in stable) - min(row[key] for row in stable)
              for key in ("allocated_bytes", "reserved_bytes")}
    if changes or max(growth.values()) > tolerance_bytes:
        raise ValueError("GPU resources did not plateau: " + str(dict(count_changes=changes, byte_ranges=growth)))
    return dict(plateau_observed=True, warmup_batches=warmup, checked_batches=len(stable),
        tolerance_bytes=tolerance_bytes, byte_ranges=growth,
        native_counts={key: stable[0][key] for key in counts},
        finite_observation_not_long_term_guarantee=True)


def engineering_inventory(folder):
    """Count actual engineering passes; none are scientific measurement attempts."""
    import h5py
    evaluations = full = calls = 0
    for path in Path(folder).rglob("trace.h5"):
        with h5py.File(path) as trace:
            count = trace["endpoint_A"].shape[1]
        calls += 1
        evaluations += count
        if (path.parent / "checkpoint_100.pt.gz").exists():
            full += count
    return dict(engineering_trajectory_calls=calls, engineering_trajectory_evaluations=evaluations,
        engineering_full_100_interval_passes=full, engineering_repeated_passes_are_not_independent=True)
