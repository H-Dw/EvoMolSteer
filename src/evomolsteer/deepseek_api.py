"""Real DeepSeek JSON calls with bounded retries and secret-free receipts."""
import json
import hashlib
import os
import time
from pathlib import Path

import httpx
import jsonschema

from evomolsteer.io import digest, read_json, write_json


FORMAT_POLICY = "json-interface-v4"


class LLMInterfaceError(ValueError):
    """An invalid role response, never an implicit incumbent decision."""

    def __init__(self, diagnostics, output):
        self.diagnostics = diagnostics
        self.output = str(output)
        super().__init__("LLM interface validation failed: " + json.dumps(diagnostics, ensure_ascii=False))


def validation_diagnostics(error):
    """Report actionable format errors without dumping response instances."""
    if isinstance(error, json.JSONDecodeError):
        from evomolsteer.json_syntax import container_diagnostic
        result = {"error_type": "JSONDecodeError", "line": error.lineno,
                  "column": error.colno, "reason": error.msg, "character_offset": error.pos}
        structure = container_diagnostic(error.doc)
        if structure:
            result["structure"] = structure
        return result
    pointer = lambda path: "/" + "/".join(str(v).replace("~", "~0").replace("/", "~1") for v in path) if path else ""
    info = {"error_type": "ValidationError", "instance_pointer": pointer(error.absolute_path),
            "schema_pointer": pointer(error.absolute_schema_path), "validator": error.validator}
    if error.validator in {"additionalProperties", "required"}:
        info["allowed_properties"] = sorted(error.schema.get("properties", {}))
        if error.validator == "additionalProperties":
            # patternProperties remain valid even when additionalProperties=False.
            import re
            patterns = error.schema.get("patternProperties", {})
            info["unexpected_properties"] = sorted(k for k in error.instance
                if k not in info["allowed_properties"] and not any(re.search(p, k) for p in patterns))
        else:
            info["missing_required_properties"] = [k for k in error.validator_value if k not in error.instance]
    elif error.validator == "type":
        info.update(expected_type=error.validator_value, actual_type=type(error.instance).__name__)
    elif error.validator in {"const", "enum"}:
        # Large source-ID enums already occur in the response schema. Repeating
        # thousands of valid IDs obscures the offending value and repair task.
        info["invalid_value"] = error.instance
        values = error.validator_value
        if error.validator == "enum" and len(values) > 64:
            info.update(allowed_value_count=len(values),
                allowed_values_location="supplied source-ID mapping and response_schema source-ID enums")
        else:
            info["allowed_value_or_values"] = values
    elif error.validator in {"oneOf", "anyOf"}:
        info["branch_count"] = len(error.validator_value)
        info["reason"] = "Use one permitted action branch; see child diagnostics and supplied response_schema."
    else:
        info["constraint"] = error.validator_value
        if error.validator=="verbatim_quote":
            info["invalid_value"] = error.instance
    return info


def response_diagnostics(error, result, schema):
    """One bounded repair addresses all invalid references, not only the last.

    Values are diagnostics only; this never rewrites a response, substitutes
    evidence IDs or supplies a new scientific finding.
    """
    info = validation_diagnostics(error)
    if not isinstance(error, jsonschema.ValidationError):
        return info
    validator = jsonschema.validators.validator_for(schema)(schema)
    errors = list(validator.iter_errors(result)) + list(error.context)
    rest = [e for e in errors if (list(e.absolute_path), e.validator) !=
            (list(error.absolute_path), error.validator)]
    if rest:
        info["additional_errors"] = [validation_diagnostics(e) for e in rest[:31]]
        info["unreported_error_count"] = max(0, len(rest) - 31)
    return info


def format_message(schema, diagnostics=None):
    """Shared transport guidance; no scientific or module-specific advice."""
    contract = {"format_policy": FORMAT_POLICY, "response_schema": schema,
        "allowed_root_properties": sorted(schema.get("properties", {})),
        "required_root_properties": schema.get("required", []),
        "additional_root_properties": schema.get("additionalProperties", True)}
    if diagnostics is not None:
        contract["validation_errors"] = diagnostics
    return ("Output-format contract only. Return JSON satisfying the supplied schema. "
            "Optional fields remain optional; this adds no research requirements. "
            "Preserve supported substantive findings and choices in schema-compatible fields. "
            "Inspect each invalid_value at its instance_pointer. For invalid source citations, consult the supplied ID-to-record mapping; "
            "correct a reference only when that record supports the original claim, otherwise state that claim's lack of support in permitted fields. "
            "Never guess a replacement ID or invent a source. "
            "For verbatim_quote errors, copy an exact supporting excerpt from the supplied role response; "
            "if no supporting excerpt exists, correct the assessment using the supplied rubric. "
            "Do not add research advice. Do not emit another role's decision fields outside the permitted properties.\n"
            + json.dumps(contract, ensure_ascii=False, separators=(",", ":")))


def unused_path(output, template):
    """Allocate an audit filename without overwriting earlier failed runs."""
    index = 1
    while True:
        path = output.with_suffix(template.format(index=index))
        if not path.exists():
            return path
        index += 1


def call_json(messages, schema, output, *, model="deepseek-flash", timeout=240,
              transport=None, attempts=3, sleep=time.sleep, input_store=None, max_output_tokens=8192,
              response_validator=None, response_parser=None, format_policy=FORMAT_POLICY,
              format_builder=None, max_format_repairs=1):
    if model != "deepseek-flash":
        raise ValueError("This study requires exactly deepseek-flash; no fallback")
    if attempts < 1:
        raise ValueError("A positive network attempt budget is required")
    if isinstance(max_format_repairs, bool) or not isinstance(max_format_repairs, int) or not 0 <= max_format_repairs <= 2:
        raise ValueError("At most two neutral format repairs are allowed")
    if isinstance(max_output_tokens,bool) or not isinstance(max_output_tokens,int) or not 512<=max_output_tokens<=65536:
        raise ValueError("Bounded positive output token budget required")
    output = Path(output)
    request_path = output.with_suffix(".request.json")
    receipt_path = output.with_suffix(".receipt.json")
    body = {"model": model, "messages": messages,
            "thinking": {"type": "disabled"}, "temperature": 0,
            "max_tokens": max_output_tokens, "response_format": {"type": "json_object"}}
    def store_request(body, path, *, policy=None):
        stored_body = dict(body)
        if input_store is not None:
            store = Path(input_store)
            store.mkdir(parents=True, exist_ok=True)
            references = []
            for message in body["messages"]:
                raw = message["content"].encode("utf-8")
                sha = hashlib.sha256(raw).hexdigest()
                content_path = store / (sha + ".txt")
                if content_path.exists() and digest(content_path) != sha:
                    raise ValueError("Corrupt shared input cache")
                if not content_path.exists(): content_path.write_bytes(raw)
                references.append({"role": message["role"], "content_sha256": sha,
                    "content_relative_path": os.path.relpath(content_path, path.parent).replace("\\", "/")})
            stored_body["messages"] = references
        stored = {"provider": "DeepSeek", "body": stored_body, "response_schema": schema}
        if policy is not None:
            stored["format_policy"] = policy
        if path.exists():
            if read_json(path) != stored:
                raise ValueError("Existing call belongs to different instructions; original request preserved")
        else:
            write_json(path, stored)
    if response_parser is not None and format_policy == FORMAT_POLICY:
        raise ValueError("A custom parser requires a versioned format policy")
    store_request(body, request_path, policy=None if format_policy == FORMAT_POLICY else format_policy)
    if receipt_path.exists() and output.exists():
        receipt = read_json(receipt_path)
        if receipt["request_sha256"] != digest(request_path):
            raise ValueError("Existing call belongs to different instructions")
        if receipt.get("response_sha256") != digest(output):
            raise ValueError("Cached response checksum mismatch")
        result = (response_parser(output.read_text(encoding="utf-8"))
                  if response_parser is not None else read_json(output))
        if format_policy != FORMAT_POLICY:
            raw = output.parent / receipt.get("raw_response_path", "")
            if (receipt.get("format_policy") != format_policy or not raw.is_file() or
                raw.resolve().parent != output.parent.resolve() or
                digest(raw) != receipt.get("raw_response_sha256")):
                raise ValueError("Cached raw API response/protocol checksum mismatch")
            raw_result = (response_parser(raw.read_text(encoding="utf-8")) if response_parser is not None
                          else json.loads(raw.read_text(encoding="utf-8")))
            if raw_result != result:
                raise ValueError("Cached extraction differs from raw API response")
        jsonschema.validate(result, schema)
        # Cached schema-valid responses can still violate source/quote
        # contracts. Never silently accept, replace, or retry such a cache.
        if response_validator is not None:
            response_validator(result)
        return result
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise ValueError("DEEPSEEK_API_KEY is absent from this process")
    started = time.time()
    repairs = []
    # Keep the scientific request/hash stable. Record the complete wire request
    # separately so cached v1 successes remain readable and prompt changes are
    # visible in receipts, rather than being hidden in the old request file.
    builder = format_builder or format_message
    wire_messages = messages + [{"role": "user", "content": builder(schema)}]
    body = dict(body, messages=wire_messages)
    wire_path = unused_path(output, ".wire_{index}.request.json")
    store_request(body, wire_path, policy=format_policy)
    http_attempts = 0
    endpoint = "https://api.deepseek.com/chat/completions"
    with httpx.Client(timeout=timeout, transport=transport, trust_env=True) as client:
        # Network retries do not consume the one neutral format repair.
        for interface_attempt in range(max_format_repairs + 1):
            for attempt in range(attempts):
                try:
                    http_attempts += 1
                    response = client.post(endpoint, json=body,
                        headers={"Authorization": "Bearer " + key})
                except (httpx.TimeoutException, httpx.NetworkError):
                    if attempt + 1 == attempts:
                        raise RuntimeError("DeepSeek network/timeout retries exhausted") from None
                    sleep(min(30, 5 * 2 ** attempt))
                    continue
                if response.status_code in {429, 500, 502, 503, 504} and attempt + 1 < attempts:
                    sleep(min(30, 5 * 2 ** attempt))
                    continue
                if response.status_code != 200:
                    # Do not log response bodies or request headers: either may
                    # contain credentials echoed by a gateway.
                    raise RuntimeError(f"DeepSeek HTTP {response.status_code}")
                break
            wire = response.json()
            choice = wire["choices"][0]
            if choice.get("finish_reason") != "stop":
                incomplete=unused_path(output,".incomplete_{index}.txt")
                incomplete.write_text(choice["message"].get("content", ""),encoding="utf-8")
                write_json(incomplete.with_suffix(".receipt.json"),{
                    "requested_model":model,"returned_model":wire.get("model"),"valid":False,
                    "finish_reason":choice.get("finish_reason"),"usage":wire.get("usage"),
                    "provider_request_id":wire.get("id"),"content_sha256":digest(incomplete),
                    "request_sha256":digest(request_path),"max_output_tokens":max_output_tokens})
                raise ValueError("Incomplete JSON answer; no incumbent masquerading as success")
            if wire.get("model") != model:
                write_json(unused_path(output, ".model_mismatch_{index}.json"), {
                    "requested_model": model, "returned_model": wire.get("model"),
                    "provider_request_id": wire.get("id"), "valid": False})
                raise ValueError("Provider returned a different model; fallback is prohibited")
            content = choice["message"]["content"]
            if format_policy != FORMAT_POLICY and content is None:
                # Some JSON-mode empty answers have null content. Treat them as
                # empty text and use the same bounded neutral format repair.
                content = ""
            if format_policy != FORMAT_POLICY and not isinstance(content, str):
                raise ValueError("Provider content is not text; no valid JSON response")
            result = None
            try:
                result = (response_parser(content) if response_parser is not None else json.loads(content))
                jsonschema.validate(result, schema)
                if response_validator is not None:
                    response_validator(result)
            except (json.JSONDecodeError, jsonschema.ValidationError) as error:
                diagnostics = response_diagnostics(error, result, schema)
                invalid_path = unused_path(output, ".attempt_{index}.invalid.txt")
                invalid_path.write_text(content, encoding="utf-8")
                write_json(invalid_path.with_suffix(".receipt.json"), {
                    "provider": "DeepSeek", "returned_model": wire.get("model"),
                    "usage": wire.get("usage"), "provider_request_id": wire.get("id"),
                    "finish_reason": choice.get("finish_reason"), "max_output_tokens": max_output_tokens,
                    "error_type": type(error).__name__, "diagnostics": diagnostics,
                    "format_policy": format_policy, "http_attempt": http_attempts,
                    "valid": False, "content_sha256": digest(invalid_path)})
                if interface_attempt == max_format_repairs:
                    raise LLMInterfaceError(diagnostics, output) from None
                body = dict(body, messages=body['messages'] + [
                    {"role": "assistant", "content": content},
                    {"role": "user", "content": builder(schema, diagnostics)}])
                repair_path = unused_path(output, ".repair_{index}.request.json")
                store_request(body, repair_path, policy=format_policy)
                repairs.append({"request_sha256": digest(repair_path), "path": repair_path.name})
                continue
            raw_path = None
            if format_policy != FORMAT_POLICY:
                raw_path = unused_path(output, ".raw_{index}.txt")
                raw_path.write_text(content, encoding="utf-8")
            write_json(output, result)
            write_json(receipt_path, {
                "provider": "DeepSeek", "endpoint": endpoint,
                "requested_model": model, "returned_model": wire.get("model"),
                "request_sha256": digest(request_path), "wire_request_sha256": digest(wire_path),
                "wire_request_path": wire_path.name, "format_policy": format_policy,
                **({"raw_response_path": raw_path.name, "raw_response_sha256": digest(raw_path)} if raw_path else {}),
                "response_sha256": digest(output), "usage": wire.get("usage"),
                "provider_request_id": wire.get("id"), "attempts": http_attempts,
                "interface_repairs": repairs,
                "started_unix": started, "completed_unix": time.time(),
                "thinking": "disabled", "backend": "real_api",
                "finish_reason": choice.get("finish_reason"), "max_output_tokens": max_output_tokens,
            })
            return result
    raise RuntimeError("DeepSeek retry budget exhausted")
