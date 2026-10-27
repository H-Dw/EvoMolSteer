"""Explicit executable identities over the unchanged lossless v1 representation.

Reference labels locate evidence; they never become action or selection IDs.
The frozen v1 module remains available to validate its successful cached calls.
"""
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path

_legacy_path = Path(__file__).with_name("uniguide_prompt_policy.py")
_spec = importlib.util.spec_from_file_location("ug_anchor_legacy", _legacy_path)
legacy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(legacy)
canonical, file_sha, read_json = legacy.canonical, legacy.file_sha, legacy.read_json
parse_output, format_contract = legacy.parse_output, legacy.format_contract
POLICY = "uniguide-explicit-anchor-v2"


def transform(messages, schema, skills_dir):
    rendered, revised, metrics = legacy.transform(messages, schema, skills_dir)
    context = json.loads(messages[1]["content"])
    history = context["history"]
    ids = [row["id"] for row in history]
    if (not ids or any(not isinstance(key, str) or not key for key in ids)
            or len(set(ids)) != len(ids) or any(row.get("complete") is not True for row in history)):
        raise ValueError("Executable anchors require unique complete measured history IDs")
    contract = dict(allowed_anchor_ids=ids, measured_anchors=[
        dict(anchor_id=row["id"], schedule=row["schedule"], source="history[%d].id" % i)
        for i, row in enumerate(history)])
    if "proposals" in revised["properties"]:
        revised["properties"]["proposals"]["items"]["properties"]["anchor_id"]["enum"] = ids
        default = context["default_anchor"]["id"]
        if default not in ids:
            raise ValueError("Default anchor is outside measured history")
        contract["default_anchor_id"] = default
    elif "selected" in revised["properties"]:
        candidates = context["candidates"]
        expected = {branch: [key for key in candidates if key.startswith(branch + ":")]
                    for branch in ("increase", "decrease", "redistribute")}
        if {k: v["enum"] for k, v in revised["properties"]["selected"]["properties"].items()} != expected:
            raise ValueError("Synthesis candidate identities disagree with frozen schema")
        contract["candidate_ids_by_branch"] = expected
    # Keep this small derived index visible even when the evidence row is a ref.
    # The original context, all numerical evidence and the inverse stay intact.
    packed = json.loads(rendered[1]["content"])
    packed["execution_contract"] = contract
    rendered[1]["content"] = json.dumps(packed, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    rendered[0]["content"] += (
        "\nExecutable identity contract: copy anchor_id exactly from execution_contract.allowed_anchor_ids "
        "(the actual history record id). A $ref label such as r80 only locates a record in shared; "
        "it is not that record's id. Calibration IDs outside history are not executable anchors. "
        "Do not invent, shorten or substitute IDs. Use the matching measured_anchors schedule for "
        "the original action checks. For synthesis, selected values are candidates dictionary keys "
        "listed in candidate_ids_by_branch, never $ref labels or private proposal IDs. "
        "Keep the existing scientific fields and return one complete schema-valid JSON object.")
    metrics.update(policy=POLICY, allowed_anchor_ids=ids,
        execution_contract_sha256=hashlib.sha256(canonical(contract).encode("utf-8")).hexdigest(),
        effective_characters=sum(len(m["content"]) for m in rendered),
        context_effective_characters=len(rendered[1]["content"]))
    return rendered, revised, metrics


def wrap_anchor_revision(previous_provider, raw_provider, record, root, authorization_path, version, skills_dir):
    """Apply the authorized identity contract only to unfinished UG calls.

    Frozen successful calls still traverse their original provider validator.
    Revised successes are explicit aliases of separately stored effective calls.
    A persistent per-call claim stops implicit retries after a new failure.
    """
    root = Path(root).resolve()
    authorization_path = Path(authorization_path).resolve()
    authorization = read_json(authorization_path)
    auth_sha = file_sha(authorization_path)
    skills_dir = Path(skills_dir).resolve()
    if (authorization["version"] != version or authorization["policy"] != POLICY
            or authorization["scope"] != "unfinished_calls_of_frozen_version"
            or not authorization.get("user_instruction")):
        raise ValueError("Invalid prompt revision authorization")
    signature = inspect.signature(raw_provider)
    def verify_assets():
        if file_sha(_legacy_path) != authorization["legacy_policy_sha256"]:
            raise ValueError("Legacy prompt policy changed")
        if file_sha(authorization["legacy_authorization_path"]) != authorization["legacy_authorization_sha256"]:
            raise ValueError("Legacy prompt authorization changed")
        if file_sha(authorization_path) != auth_sha or file_sha(__file__) != authorization["policy_sha256"]:
            raise ValueError("Prompt authorization or policy changed")
        if {p.name: file_sha(p) for p in skills_dir.glob("*.md")} != authorization["skill_sha256"]:
            raise ValueError("Authorized prompt skills changed")
    verify_assets()

    def call(messages, schema, output, **kwargs):
        verify_assets()
        target = Path(output).resolve()
        if not target.is_relative_to(root / "versions" / version):
            raise ValueError("Prompt revision output outside authorized version")
        relative = target.relative_to(root).as_posix()
        baseline = authorization["successful_outputs"].get(relative)
        receipt_path = target.with_suffix(".receipt.json")
        if baseline is not None:
            if file_sha(target) != baseline["output_sha256"] or file_sha(receipt_path) != baseline["receipt_sha256"]:
                raise ValueError("Previously successful response changed")
            return previous_provider(messages, schema, output, **kwargs)
        parts = target.relative_to(root / "versions" / version).parts
        if (len(parts) != 3 or parts[0] not in authorization["teachers"]
                or parts[1] not in ["advice_R%02d" % i for i in range(1, 6)] + ["analysis_R05"]
                or target.stem not in ["increase", "decrease", "redistribute", "synthesis"]):
            raise ValueError("Unexpected unfinished call path")
        options = signature.bind(messages, schema, output, **kwargs)
        options.apply_defaults()
        opts = options.arguments
        if (opts["input_store"] is not None or opts["response_parser"] is not None
                or opts["format_builder"] is not None or opts["format_policy"] != "json-interface-v5"
                or opts["max_output_tokens"] != 16384 or opts["model"] != "deepseek-flash"
                or opts["thinking"] is not True or opts["reasoning_effort"] != "high"
                or opts["attempts"] != 3 or opts["max_format_repairs"] != 1):
            raise ValueError("Frozen provider settings changed")
        original = dict(provider="DeepSeek", response_schema=schema, body=dict(
            model=opts["model"], messages=messages, thinking={"type": "enabled"},
            reasoning_effort="high", max_tokens=16384, response_format={"type": "json_object"}))
        original_path = target.with_suffix(".request.json")
        if original_path.exists():
            if canonical(read_json(original_path)) != canonical(original):
                raise ValueError("Original request/schema changed")
        else:
            with original_path.open("x", encoding="utf-8") as stream:
                json.dump(original, stream, ensure_ascii=False, indent=2)
        if relative == authorization["first_output"] and file_sha(original_path) != authorization["first_request_sha256"]:
            raise ValueError("First revised request changed")
        revised_messages, revised_schema, metrics = transform(messages, schema, skills_dir)
        effective = target.parent / (target.stem + ".anchorids_v2") / "response.json"
        effective_receipt = effective.with_suffix(".receipt.json")
        cached = effective.exists() and effective_receipt.exists()
        if target.exists() or receipt_path.exists():
            if not cached or (target.exists() and target.read_bytes() != effective.read_bytes()):
                raise ValueError("Canonical revised response alias changed")
        claim = effective.parent / "authorization_claim.json"
        if not cached:
            effective.parent.mkdir(exist_ok=True)
            with claim.open("x", encoding="utf-8") as stream:
                json.dump(dict(authorization_sha256=auth_sha, original_request_sha256=file_sha(original_path), pid=os.getpid()), stream)
            record(dict(event="authorized_anchor_revision", original_request=str(original_path),
                        effective_output=str(effective), authorization_sha256=auth_sha, **metrics))
        if read_json(claim)["authorization_sha256"] != auth_sha or read_json(claim)["original_request_sha256"] != file_sha(original_path):
            raise ValueError("Prompt revision claim changed")
        result = raw_provider(revised_messages, revised_schema, effective, **dict(kwargs,
            response_parser=parse_output, format_builder=format_contract, format_policy=POLICY))
        receipt = read_json(effective_receipt)
        receipt["receipt_kind"] = "audited_alias_of_effective_provider_receipt"
        receipt["protocol_deviation"] = dict(kind="explicit_anchor_revision",
            authorization_path=str(authorization_path), authorization_sha256=auth_sha,
            original_request_path=str(original_path), original_request_sha256=file_sha(original_path),
            effective_request_path=str(effective.with_suffix(".request.json")),
            effective_receipt_path=str(effective_receipt), output_token_cap_unchanged=16384, **metrics)
        for key in ("wire_request_path", "raw_response_path"):
            if key in receipt:
                receipt[key] = str((effective.parent / receipt[key]).relative_to(target.parent))
        receipt["interface_repairs"] = [dict(item, path=str((effective.parent / item["path"]).relative_to(target.parent)))
                                         for item in receipt.get("interface_repairs", [])]
        if receipt_path.exists() and read_json(receipt_path) != receipt:
            raise ValueError("Canonical revised receipt alias changed")
        for path, content in ((target, effective.read_bytes()),
                (receipt_path, json.dumps(receipt, ensure_ascii=False, indent=2).encode("utf-8"))):
            if not path.exists():
                with path.open("xb") as stream:
                    stream.write(content)
        record(dict(event="anchor_revision_success", cached=cached, effective_receipt_sha256=file_sha(effective_receipt),
                    canonical_output_sha256=file_sha(target), **metrics))
        return result
    return call
