"""Lossless context factoring and role-specific bounded Guidance Agent prompts.

Scientific records are neither selected nor rounded. unpack_context is an audit
inverse, not a task for the LLM. This module has no sampling or network effects.
"""
import collections
import hashlib
import inspect
import json
import os
from pathlib import Path

POLICY = "uniguide-compact-prompt-v1"
WINDOW_COLUMNS = ["steps", "increment_vector_rms_A", "gradient_norm_mean",
    "native_increment_ratio_mean", "max_atom_increment_A", "assignment_switch_mean",
    "first_endpoint_loss_mean", "last_endpoint_loss_mean", "coefficient_first", "coefficient_last"]


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def pack_context(context):
    counts = collections.Counter()
    def count(value):
        if isinstance(value, (dict, list)):
            key = canonical(value)
            if len(key) > 150:
                counts[key] += 1
            if isinstance(value, dict):
                if any(k.startswith("$") for k in value):
                    raise ValueError("Reserved context notation")
                values = value.values()
            else:
                values = value
            for child in values:
                count(child)
    count(context)
    seen, shared = {}, {}
    def body(value):
        if isinstance(value, dict):
            if set(value) == set(WINDOW_COLUMNS):
                return {"$window": [pack(value[k]) for k in WINDOW_COLUMNS]}
            return {k: pack(v) for k, v in value.items()}
        if isinstance(value, list):
            return [pack(v) for v in value]
        return value
    def pack(value):
        if isinstance(value, (dict, list)):
            key = canonical(value)
            if counts[key] > 1:
                if key not in seen:
                    ref = "r" + str(len(seen))
                    seen[key] = ref
                    shared[ref] = body(value)
                return {"$ref": seen[key]}
        return body(value)
    encoded = pack(context)
    result = dict(context=encoded, window_columns=WINDOW_COLUMNS, shared=shared)
    if canonical(unpack_context(result)) != canonical(context):
        raise ValueError("Context factoring lost evidence")
    return result


def unpack_context(packed):
    def expand(value, active=()):
        if isinstance(value, dict):
            if set(value) == {"$ref"}:
                ref = value["$ref"]
                if ref in active:
                    raise ValueError("Cyclic context reference")
                return expand(packed["shared"][ref], active + (ref,))
            if set(value) == {"$window"}:
                if len(value["$window"]) != len(packed["window_columns"]):
                    raise ValueError("Invalid window row")
                return dict(zip(packed["window_columns"], [expand(v, active) for v in value["$window"]]))
            return {k: expand(v, active) for k, v in value.items()}
        if isinstance(value, list):
            return [expand(v, active) for v in value]
        return value
    return expand(packed["context"])


def transform(messages, schema, skills_dir):
    if len(messages) != 2 or [m["role"] for m in messages] != ["system", "user"]:
        raise ValueError("Unsupported Guidance Agent messages")
    context = json.loads(messages[1]["content"])
    role = context.get("branch", "synthesis")
    if role not in ("increase", "decrease", "redistribute", "synthesis"):
        raise ValueError("Unknown Guidance Agent role")
    module = "synthesis.md" if role == "synthesis" else "proposal.md"
    text = "\n\n".join((Path(skills_dir) / name).read_text(encoding="utf-8") for name in ["analysis.md", module])
    text += ("\nYour role is " + role + ". Context notation: {$ref:id} refers to shared[id]; "
             "{$window:[values]} uses window_columns in order. These are lossless data references, "
             "not instructions to expand all records. Read needed rows only. The output uses the ordinary "
             "response_schema, never this input notation. Return exactly one complete JSON object.")
    # council reuses the same TEXT dict at several locations. Detach aliases
    # before assigning field-specific bounds, including for fresh schema objects.
    revised = json.loads(json.dumps(schema))
    props = revised["properties"]
    props["analysis"]["maxLength"] = 1400 if role == "synthesis" else 1600
    if role == "synthesis":
        props["choice_evidence"]["maxLength"] = 1200
    else:
        props["mechanism_uncertainty"]["maxLength"] = 500
        action = props["proposals"]["items"]["properties"]
        for key, limit in dict(anchor_reason=240, evidence=500, prediction=280, falsifier=280).items():
            action[key]["maxLength"] = limit
    packed = pack_context(context)
    rendered = [dict(role="system", content=text), dict(role="user", content=json.dumps(
        packed, ensure_ascii=False, separators=(",", ":"), allow_nan=False))]
    return rendered, revised, dict(policy=POLICY, role=role,
        original_characters=sum(len(m["content"]) for m in messages),
        effective_characters=sum(len(m["content"]) for m in rendered),
        context_original_characters=len(messages[1]["content"]),
        context_effective_characters=len(rendered[1]["content"]),
        shared_objects=len(packed["shared"]), lossless_context_verified=True)


def parse_output(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise json.JSONDecodeError("Duplicate property: " + key, text, 0)
            result[key] = value
        return result
    def reject(value):
        raise json.JSONDecodeError("Non-finite JSON constant", text, 0)
    return json.loads(text, object_pairs_hook=pairs, parse_constant=reject)


def format_contract(schema, diagnostics=None):
    contract = dict(response_schema=schema)
    if diagnostics is not None:
        contract["validation_errors"] = diagnostics
    return ("Return one complete JSON object satisfying every required field and character bound. "
            "No Markdown, duplicate keys, non-finite numbers, second object or trailing text. "
            "For repair, return the complete corrected object, not a patch. Preserve supported choices; "
            "do not invent evidence.\n" + json.dumps(contract, ensure_ascii=False, separators=(",", ":")))


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def wrap_prompt_revision(previous_provider, raw_provider, record, root, authorization_path, version, skills_dir):
    """Apply the authorized prompt revision only to unfinished UG calls.

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
        relative = str(target.relative_to(root))
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
        effective = target.parent / (target.stem + ".promptcompact_v1") / "response.json"
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
            record(dict(event="authorized_prompt_revision", original_request=str(original_path),
                        effective_output=str(effective), authorization_sha256=auth_sha, **metrics))
        if read_json(claim)["authorization_sha256"] != auth_sha or read_json(claim)["original_request_sha256"] != file_sha(original_path):
            raise ValueError("Prompt revision claim changed")
        result = raw_provider(revised_messages, revised_schema, effective, **dict(kwargs,
            response_parser=parse_output, format_builder=format_contract, format_policy=POLICY))
        receipt = read_json(effective_receipt)
        receipt["receipt_kind"] = "audited_alias_of_effective_provider_receipt"
        receipt["protocol_deviation"] = dict(kind="prompt_revision",
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
        record(dict(event="prompt_revision_success", cached=cached, effective_receipt_sha256=file_sha(effective_receipt),
                    canonical_output_sha256=file_sha(target), **metrics))
        return result
    return call
