"""Audited serialization-only recovery against an unchanged frozen UG checkout.

Successful provider outputs are stored with sorted object keys. On replay their
embedding in a synthesis prompt may therefore differ from the original text.
For an existing request only, prove that JSON object order is the sole change,
then supply the ORIGINAL messages to the unchanged provider/cache validator.
By default this does not alter budgets. An explicit, hash-bound authorization
may raise one failed request's output cap by 50%, using a separate provider store.
"""
import argparse
import datetime
import hashlib
import json
import os
import inspect
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


def wrap_token_override(provider_call, record, root, authorization_path, version):
    """One authorized call; preserve old evidence and validate effective caches.

    Canonical response aliases carry the effective request/receipt paths, so
    readers can find the actual 24576-token call without changing old requests.
    A durable claim prevents a failed enlarged request being silently retried.
    """
    root = Path(root).resolve()
    authorization = read(authorization_path)
    if (authorization["version"] != version
            or authorization["old_max_output_tokens"] != 16384
            or authorization["new_max_output_tokens"] != 24576
            or not authorization.get("user_instruction")):
        raise ValueError("Invalid single-request token authorization")
    target = (root / authorization["output_relative_path"]).resolve()
    if not target.is_relative_to(root / "versions" / version):
        raise ValueError("Token override target outside frozen version")
    original_request = target.with_suffix(".request.json")
    effective = target.parent / (target.stem + ".token24576") / "response.json"
    authorization_sha = sha(authorization_path)
    signature = inspect.signature(provider_call)

    def call(messages, schema, output, **kwargs):
        if Path(output).resolve() != target:
            return provider_call(messages, schema, output, **kwargs)
        if sha(authorization_path) != authorization_sha or sha(original_request) != authorization["original_request_sha256"]:
            raise ValueError("Authorized request or authorization changed")
        stored = read(original_request)
        original_messages(messages, stored["body"]["messages"])
        messages = stored["body"]["messages"]
        bound = signature.bind(messages, schema, output, **kwargs)
        bound.apply_defaults()
        options = bound.arguments
        if (options["input_store"] is not None or options["response_parser"] is not None
                or options["format_builder"] is not None or options["format_policy"] != "json-interface-v5"
                or options["max_output_tokens"] != 16384):
            raise ValueError("Unsupported original provider options")
        body = dict(model=options["model"], messages=messages,
                    thinking={"type": "disabled"}, temperature=0,
                    max_tokens=16384, response_format={"type": "json_object"})
        if options["thinking"]:
            body.update(thinking={"type": "enabled"}, reasoning_effort=options["reasoning_effort"])
            body.pop("temperature")
        if stored != dict(provider="DeepSeek", body=body, response_schema=schema):
            raise ValueError("Original provider request/schema changed")
        receipt_path = target.with_suffix(".receipt.json")
        claim = effective.parent / "authorization_claim.json"
        effective_receipt = effective.with_suffix(".receipt.json")
        cached = effective.exists() and effective_receipt.exists()
        if target.exists() or receipt_path.exists():
            if not cached:
                raise ValueError("Canonical response exists without effective success")
            if target.exists() and target.read_bytes() != effective.read_bytes():
                raise ValueError("Canonical response alias changed")
        if not cached:
            effective.parent.mkdir(exist_ok=True)
            # Exclusive creation is deliberate: no implicit retry after failure.
            with claim.open("x", encoding="utf-8") as stream:
                json.dump(dict(authorization_sha256=authorization_sha, pid=os.getpid()), stream)
            record({"event": "authorized_token_override", "original_request": str(original_request),
                    "effective_output": str(effective), "authorization_sha256": authorization_sha,
                    "old_max_output_tokens": 16384, "new_max_output_tokens": 24576})
        if read(claim)["authorization_sha256"] != authorization_sha:
            raise ValueError("Token authorization claim changed")
        result = provider_call(messages, schema, effective, **dict(kwargs, max_output_tokens=24576))
        receipt = read(effective_receipt)
        receipt["receipt_kind"] = "audited_alias_of_effective_provider_receipt"
        receipt["protocol_deviation"] = dict(
            authorization_path=str(Path(authorization_path).resolve()), authorization_sha256=authorization_sha,
            original_request_path=str(original_request), original_request_sha256=sha(original_request),
            effective_request_path=str(effective.with_suffix(".request.json")),
            effective_receipt_path=str(effective_receipt), old_max_output_tokens=16384, new_max_output_tokens=24576)
        for key in ("wire_request_path", "raw_response_path"):
            if key in receipt:
                receipt[key] = str((effective.parent / receipt[key]).relative_to(target.parent))
        receipt["interface_repairs"] = [dict(item, path=str((effective.parent / item["path"]).relative_to(target.parent)))
                                         for item in receipt.get("interface_repairs", [])]
        if receipt_path.exists() and read(receipt_path) != receipt:
            raise ValueError("Canonical receipt alias changed")
        # Never overwrite existing bytes. A crash between these two writes can
        # be repaired from the validated effective cache without another HTTP call.
        for path, content in ((target, effective.read_bytes()),
                              (receipt_path, json.dumps(receipt, ensure_ascii=False, indent=2).encode("utf-8"))):
            if not path.exists():
                with path.open("xb") as stream:
                    stream.write(content)
        record({"event": "token_override_success", "cached": cached,
                "effective_receipt_sha256": sha(effective_receipt), "canonical_output_sha256": sha(target)})
        return result
    return call


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--frozen-repo", required=True)
    parser.add_argument("--adapter-commit", required=True)
    parser.add_argument("--preservation-audit", required=True)
    parser.add_argument("--token-override", help="Hash-bound user authorization for one 50% cap increase")
    parser.add_argument("--prompt-revision", help="Audited user authorization to revise unfinished prompts")
    parser.add_argument("--prompt-skills", help="Committed revised skills, separate from the frozen version")
    parser.add_argument("--anchor-revision", help="Audited identity-contract revision with legacy cache bindings")
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
        provider_call = council.call_json
        prompt_commit = args.adapter_commit
        if args.anchor_revision:
            if not args.prompt_revision or not args.token_override:
                raise ValueError("Anchor recovery requires both existing deviation authorizations")
            anchor_authorization = read(args.anchor_revision)
            if (anchor_authorization["manifest_sha256"] != sha(manifest_path)
                    or anchor_authorization["frozen_code_commit"] != manifest["code_commit"]
                    or anchor_authorization["adapter_commit"] != args.adapter_commit
                    or Path(anchor_authorization["legacy_authorization_path"]).resolve() != Path(args.prompt_revision).resolve()
                    or anchor_authorization["legacy_authorization_sha256"] != sha(args.prompt_revision)
                    or anchor_authorization["token_authorization_sha256"] != sha(args.token_override)):
                raise ValueError("Anchor authorization belongs to a different recovery binding")
            prompt_commit = anchor_authorization["legacy_adapter_commit"]
        if args.token_override:
            authorization = read(args.token_override)
            if (authorization["manifest_sha256"] != sha(manifest_path)
                    or authorization["frozen_code_commit"] != manifest["code_commit"]):
                raise ValueError("Token authorization belongs to a different frozen version")
            provider_call = wrap_token_override(provider_call, record, root, args.token_override, args.version)
        if args.prompt_revision:
            import importlib.util
            policy_path = Path(__file__).with_name("uniguide_prompt_policy.py")
            if git("show", prompt_commit + ":agenticgen/scripts/uniguide_prompt_policy.py") != policy_path.read_bytes():
                raise ValueError("Prompt policy differs from committed blob")
            if not args.prompt_skills:
                raise ValueError("Separate committed prompt skills are required")
            authorization = read(args.prompt_revision)
            if (authorization["manifest_sha256"] != sha(manifest_path)
                    or authorization["frozen_code_commit"] != manifest["code_commit"]
                    or authorization["adapter_commit"] != prompt_commit):
                raise ValueError("Prompt authorization belongs to a different binding")
            for name in ["analysis.md", "proposal.md", "synthesis.md"]:
                if git("show", prompt_commit + ":agenticgen/src/agenticgen/uniguide/skills/" + name) != (Path(args.prompt_skills) / name).read_bytes():
                    raise ValueError("Prompt skills differ from committed blobs")
            spec = importlib.util.spec_from_file_location("ug_prompt_policy", policy_path)
            policy = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(policy)
            provider_call = policy.wrap_prompt_revision(provider_call, council.call_json, record, root,
                args.prompt_revision, args.version, args.prompt_skills)
        if args.anchor_revision:
            anchor_path = Path(__file__).with_name("uniguide_anchor_policy.py")
            if git("show", args.adapter_commit + ":agenticgen/scripts/uniguide_anchor_policy.py") != anchor_path.read_bytes():
                raise ValueError("Anchor policy differs from committed blob")
            spec = importlib.util.spec_from_file_location("ug_anchor_policy", anchor_path)
            anchor_policy = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(anchor_policy)
            provider_call = anchor_policy.wrap_anchor_revision(provider_call, council.call_json, record, root,
                args.anchor_revision, args.version, args.prompt_skills)
        council.call_json = wrap_call(provider_call, record)
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
