"""Real DeepSeek JSON calls with bounded retries and secret-free receipts."""
import json
import hashlib
import os
import time
from pathlib import Path

import httpx
import jsonschema

from evomolsteer.io import digest, read_json, write_json


def call_json(messages, schema, output, *, model="deepseek-flash", timeout=240,
              transport=None, attempts=3, sleep=time.sleep, input_store=None):
    if model != "deepseek-flash":
        raise ValueError("This study requires exactly deepseek-flash; no fallback")
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise ValueError("DEEPSEEK_API_KEY is absent from this process")
    output = Path(output)
    request_path = output.with_suffix(".request.json")
    receipt_path = output.with_suffix(".receipt.json")
    body = {"model": model, "messages": messages,
            "thinking": {"type": "disabled"}, "temperature": 0,
            "max_tokens": 8192, "response_format": {"type": "json_object"}}
    def store_request(body, path):
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
        write_json(path, {"provider": "DeepSeek", "body": stored_body, "response_schema": schema})
    store_request(body, request_path)
    if receipt_path.exists() and output.exists():
        receipt = read_json(receipt_path)
        if receipt["request_sha256"] != digest(request_path):
            raise ValueError("Existing call belongs to different instructions")
        result = read_json(output)
        jsonschema.validate(result, schema)
        return result
    started = time.time()
    repairs = []
    endpoint = "https://api.deepseek.com/chat/completions"
    with httpx.Client(timeout=timeout, transport=transport, trust_env=True) as client:
        for attempt in range(attempts):
            try:
                response = client.post(endpoint, json=body,
                    headers={"Authorization": "Bearer " + key})
                if response.status_code in {429, 500, 502, 503, 504}:
                    if attempt + 1 < attempts:
                        sleep(min(30, 5 * 2 ** attempt))
                        continue
                if response.status_code != 200:
                    # Do not log response bodies or request headers: either may
                    # contain credentials echoed by a gateway.
                    raise RuntimeError(f"DeepSeek HTTP {response.status_code}")
                wire = response.json()
                choice = wire["choices"][0]
                if choice.get("finish_reason") != "stop":
                    raise ValueError("Incomplete JSON answer; no incumbent masquerading as success")
                content = choice["message"]["content"]
                try:
                    result = json.loads(content)
                    jsonschema.validate(result, schema)
                except (json.JSONDecodeError, jsonschema.ValidationError) as error:
                    invalid_path = output.with_suffix(f".attempt_{attempt + 1}.invalid.txt")
                    invalid_path.write_text(content, encoding="utf-8")
                    write_json(invalid_path.with_suffix(".receipt.json"), {
                        "provider": "DeepSeek", "returned_model": wire.get("model"),
                        "usage": wire.get("usage"), "provider_request_id": wire.get("id"),
                        "error_type": type(error).__name__, "valid": False,
                        "content_sha256": digest(invalid_path)})
                    if repairs or attempt + 1 == attempts: raise
                    # One neutral interface repair, applied identically to all
                    # treatments. No new research hint, reward choice or model.
                    body = dict(body, messages=messages + [
                        {"role": "assistant", "content": content},
                        {"role": "user", "content": "Your answer failed JSON parsing/schema validation. Return valid JSON matching the supplied schema, preserving the same substantive findings and choices. Do not add research advice. Error type: " + type(error).__name__}])
                    repair_path = output.with_suffix(".repair.request.json")
                    store_request(body, repair_path)
                    repairs.append({"request_sha256": digest(repair_path), "path": repair_path.name})
                    continue
                write_json(output, result)
                write_json(receipt_path, {
                    "provider": "DeepSeek", "endpoint": endpoint,
                    "requested_model": model, "returned_model": wire.get("model"),
                    "request_sha256": digest(request_path),
                    "response_sha256": digest(output), "usage": wire.get("usage"),
                    "provider_request_id": wire.get("id"), "attempts": attempt + 1,
                    "interface_repairs": repairs,
                    "started_unix": started, "completed_unix": time.time(),
                    "thinking": "disabled", "backend": "real_api",
                })
                return result
            except (httpx.TimeoutException, httpx.NetworkError):
                if attempt + 1 == attempts:
                    raise RuntimeError("DeepSeek network/timeout retries exhausted") from None
                sleep(min(30, 5 * 2 ** attempt))
    raise RuntimeError("DeepSeek retry budget exhausted")
