"""Secret-free connectivity preflight and exclusive launch checks."""
import contextlib, os, time
import httpx

def check_api(*, transport=None):
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY missing; no worker started")
    started = time.time()
    try:
        with httpx.Client(timeout=20, trust_env=True, transport=transport) as client:
            response = client.get("https://api.deepseek.com/models",
                                  headers={"Authorization": "Bearer " + key})
    except httpx.HTTPError as error:
        # Exception text can include authenticated proxy URLs. Record only type.
        raise RuntimeError(f"DeepSeek connectivity check failed ({type(error).__name__}); no worker started") from None
    if response.status_code != 200:
        raise RuntimeError(f"DeepSeek connectivity check HTTP {response.status_code}; no worker started")
    return {"authenticated_models_http_status": 200, "elapsed_seconds": time.time() - started,
            "api_key_present": True,
            "proxy_variables_present": sorted(k for k in os.environ
                if k.lower() in {"http_proxy", "https_proxy", "all_proxy", "no_proxy"}
                and os.environ[k])}

@contextlib.contextmanager
def idle_locks(root):
    import fcntl
    with contextlib.ExitStack() as stack:
        for name in ("supervisor.lock", "worker.lock"):
            handle = stack.enter_context((root / name).open("a"))
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError(f"Experiment is active ({name}); no duplicate launch") from None
        yield
