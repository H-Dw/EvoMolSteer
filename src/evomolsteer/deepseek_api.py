"""Real DeepSeek JSON calls with bounded retries and secret-free receipts."""
import json
import hashlib
import os
import time
from pathlib import Path

import httpx
import jsonschema

from evomolsteer.io import digest, read_json, write_json


FORMAT_POLICY = "json-interface-v2"


class LLMInterfaceError(ValueError):
    """An invalid role response, never an implicit incumbent decision."""

    def __init__(self, diagnostics, output):
        self.diagnostics = diagnostics
        self.output = str(output)
        super().__init__("LLM interface validation failed: " + json.dumps(diagnostics, ensure_ascii=False))


def validation_diagnostics(error):
    """Report actionable format errors without dumping response instances."""
    if isinstance(error, json.JSONDecodeError):
        return {"error_type": "JSONDecodeError", "line": error.lineno,
                "column": error.colno, "reason": error.msg}
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
        info["allowed_value_or_values"] = error.validator_value
    else:
        info["constraint"] = error.validator_value
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
            "Preserve the same substantive findings and choices in schema-compatible fields. "
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
              transport=None, attempts=3, sleep=time.sleep, input_store=None):
    if model != "deepseek-flash":
        raise ValueError("This study requires exactly deepseek-flash; no fallback")
    if attempts < 1:
        raise ValueError("A positive network attempt budget is required")
    output = Path(output)
    request_path = output.with_suffix(".request.json")
    receipt_path = output.with_suffix(".receipt.json")
    body = {"model": model, "messages": messages,
            "thinking": {"type": "disabled"}, "temperature": 0,
            "max_tokens": 8192, "response_format": {"type": "json_object"}}
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
    store_request(body, request_path)
    if receipt_path.exists() and output.exists():
        receipt = read_json(receipt_path)
        if receipt["request_sha256"] != digest(request_path):
            raise ValueError("Existing call belongs to different instructions")
        if receipt.get("response_sha256") != digest(output):
            raise ValueError("Cached response checksum mismatch")
        result = read_json(output)
        jsonschema.validate(result, schema)
        return result
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise ValueError("DEEPSEEK_API_KEY is absent from this process")
    started = time.time()
    repairs = []
    # Keep the scientific request/hash stable. Record the complete wire request
    # separately so cached v1 successes remain readable and prompt changes are
    # visible in receipts, rather than being hidden in the old request file.
    wire_messages = messages + [{"role": "user", "content": format_message(schema)}]
    body = dict(body, messages=wire_messages)
    wire_path = unused_path(output, ".wire_{index}.request.json")
    store_request(body, wire_path, policy=FORMAT_POLICY)
    http_attempts = 0
    endpoint = "https://api.deepseek.com/chat/completions"
    with httpx.Client(timeout=timeout, transport=transport, trust_env=True) as client:
        # Network retries do not consume the one neutral format repair.
        for interface_attempt in range(2):
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
                raise ValueError("Incomplete JSON answer; no incumbent masquerading as success")
            content = choice["message"]["content"]
            try:
                result = json.loads(content)
                jsonschema.validate(result, schema)
            except (json.JSONDecodeError, jsonschema.ValidationError) as error:
                diagnostics = validation_diagnostics(error)
                invalid_path = unused_path(output, ".attempt_{index}.invalid.txt")
                invalid_path.write_text(content, encoding="utf-8")
                write_json(invalid_path.with_suffix(".receipt.json"), {
                    "provider": "DeepSeek", "returned_model": wire.get("model"),
                    "usage": wire.get("usage"), "provider_request_id": wire.get("id"),
                    "error_type": type(error).__name__, "diagnostics": diagnostics,
                    "format_policy": FORMAT_POLICY, "http_attempt": http_attempts,
                    "valid": False, "content_sha256": digest(invalid_path)})
                if interface_attempt:
                    raise LLMInterfaceError(diagnostics, output) from None
                body = dict(body, messages=wire_messages + [
                    {"role": "assistant", "content": content},
                    {"role": "user", "content": format_message(schema, diagnostics)}])
                repair_path = unused_path(output, ".repair_{index}.request.json")
                store_request(body, repair_path, policy=FORMAT_POLICY)
                repairs.append({"request_sha256": digest(repair_path), "path": repair_path.name})
                continue
            write_json(output, result)
            write_json(receipt_path, {
                "provider": "DeepSeek", "endpoint": endpoint,
                "requested_model": model, "returned_model": wire.get("model"),
                "request_sha256": digest(request_path), "wire_request_sha256": digest(wire_path),
                "wire_request_path": wire_path.name, "format_policy": FORMAT_POLICY,
                "response_sha256": digest(output), "usage": wire.get("usage"),
                "provider_request_id": wire.get("id"), "attempts": http_attempts,
                "interface_repairs": repairs,
                "started_unix": started, "completed_unix": time.time(),
                "thinking": "disabled", "backend": "real_api",
            })
            return result
    raise RuntimeError("DeepSeek retry budget exhausted")
