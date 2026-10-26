"""Audited serialization-only recovery against an unchanged frozen UG checkout.

Successful provider outputs are stored with sorted object keys. On replay their
embedding in a synthesis prompt may therefore differ from the original text.
For an existing request only, prove that JSON object order is the sole change,
then supply the ORIGINAL messages to the unchanged provider/cache validator.
This neither edits a request/response nor retries calls or alters their budget.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess


SOURCE_PATH = "agenticgen/scripts/resume_uniguide_cached_requests.py"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def strict_object(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON property")
            result[key] = value
        return result

    def reject(value):
        raise ValueError("Non-finite JSON constant")

    result = json.loads(text, object_pairs_hook=pairs, parse_constant=reject)
    if not isinstance(result, dict):
        raise ValueError("Only JSON object message content can be reordered")
    return json.dumps(result, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":"))


def original_messages(current, saved):
    if len(current) != len(saved):
        raise ValueError("Message count changed")
    for new, old in zip(current, saved):
        if new == old:
            continue
        if (set(new) != {"role", "content"} or set(old) != set(new)
                or new["role"] != "user" or old["role"] != "user"):
            raise ValueError("Non-user message or message metadata changed")
        if strict_object(new["content"]) != strict_object(old["content"]):
            raise ValueError("Scientific request content changed")
    return saved


def wrap_call(provider_call, record):
    def call(messages, schema, output, **kwargs):
        path = Path(output).with_suffix(".request.json")
        if path.exists():
            stored = read(path)
            # Provider still checks the full body/options/schema and all receipt
            # checksums. No input-store or alternate protocol is handled here.
            if stored.get("provider") != "DeepSeek" or kwargs.get("input_store") is not None:
                raise ValueError("Unsupported request store")
            saved = stored["body"]["messages"]
            original_messages(messages, saved)
            if messages != saved:
                record({"event": "reuse_original_serialization", "request": str(path),
                        "request_sha256": sha(path),
                        "reconstructed_messages_sha256": hashlib.sha256(
                            json.dumps(messages, ensure_ascii=False).encode()).hexdigest()})
            messages = saved
        return provider_call(messages, schema, output, **kwargs)
    return call


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--frozen-repo", required=True)
    parser.add_argument("--adapter-commit", required=True)
    parser.add_argument("--preservation-audit", required=True)
    args = parser.parse_args()
    repo = Path(args.frozen_repo).resolve()
    config = read(args.config)
    root = Path(config["root"])
    manifest_path = root / "versions" / args.version / "manifest.json"
    manifest = read(manifest_path)
    git = lambda *rest: subprocess.check_output(["git", "-C", str(repo), *rest])
    if git("rev-parse", "HEAD").decode().strip() != manifest["code_commit"]:
        raise ValueError("Frozen checkout commit changed")
    if git("status", "--porcelain", "--untracked-files=no").strip():
        raise ValueError("Frozen checkout has tracked modifications")
    if git("show", args.adapter_commit + ":" + SOURCE_PATH) != Path(__file__).read_bytes():
        raise ValueError("Recovery entry point differs from its committed blob")
    import fcntl
    import sys
    lock = (root / ".executor.lock").open("a+")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    sys.path.insert(0, config["plip_dependencies"])
    from agenticgen.uniguide import council, pipeline
    if Path(pipeline.__file__).resolve().parents[4] != repo:
        raise ValueError("Recovery imported a different scientific checkout")
    if pipeline.source_binding() != manifest["source_binding"]:
        raise ValueError("Frozen source binding changed")
    preservation = read(args.preservation_audit)

    def verify_preserved():
        for relative, expected in preservation["source_hashes"].items():
            path = (root / relative).resolve()
            if not path.is_relative_to(root.resolve()) or sha(path) != expected:
                raise ValueError("Preserved source changed: " + relative)

    verify_preserved()
    now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))
    audit = root / "engineering_audits" / (args.version + "_request_recovery_" + now.strftime("%Y%m%d_%H%M%S") + ".jsonl")
    with audit.open("x", encoding="utf-8") as stream:
        def record(event):
            event = dict(event, time_shanghai=datetime.datetime.now(
                datetime.timezone(datetime.timedelta(hours=8))).isoformat())
            stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        record({"event": "start", "version": args.version, "pid": os.getpid(),
                "frozen_code_commit": manifest["code_commit"], "adapter_commit": args.adapter_commit,
                "adapter_sha256": sha(__file__), "manifest_sha256": sha(manifest_path),
                "preservation_audit_sha256": sha(args.preservation_audit),
                "preserved_files": len(preservation["source_hashes"]),
                "automatic_length_retries": 0, "llm": config["llm"]})
        council.call_json = wrap_call(council.call_json, record)
        try:
            result = pipeline.run_version(config, args.version)
            record({"event": "version_finalized", "review_required": True})
            print(json.dumps(result, ensure_ascii=False, indent=2))
        except BaseException as error:
            record({"event": "stopped", "error_type": type(error).__name__})
            raise
        finally:
            verify_preserved()
            record({"event": "preserved_files_verified", "count": len(preservation["source_hashes"])})


if __name__ == "__main__":
    main()
