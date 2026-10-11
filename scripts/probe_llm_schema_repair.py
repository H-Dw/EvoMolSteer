"""Replay a recorded schema-invalid JSON answer as a real-API format probe.

This never modifies the original call, invokes Designer or starts generation.
Credentials are provided through DEEPSEEK_API_KEY in the calling process.
"""
import argparse
from pathlib import Path

import jsonschema

from evomolsteer.deepseek_api import call_json, format_message, validation_diagnostics
from evomolsteer.io import digest, read_json, write_json


def hydrate(request_path):
    request_path = Path(request_path)
    request = read_json(request_path)
    body = request.get("body", request)
    messages = []
    for message in body["messages"]:
        if "content" in message:
            messages.append(message)
            continue
        path = (request_path.parent / message["content_relative_path"]).resolve()
        if digest(path) != message["content_sha256"]:
            raise ValueError("Recorded prompt input checksum mismatch")
        messages.append({"role": message["role"], "content": path.read_text(encoding="utf-8")})
    return messages, request["response_schema"]


def contains_payload(tree, payload):
    if tree == payload:
        return True
    children = tree.values() if isinstance(tree, dict) else tree if isinstance(tree, list) else []
    return any(contains_payload(value, payload) for value in children)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--invalid", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source, invalid, output = Path(args.request), Path(args.invalid), Path(args.output)
    messages, schema = hydrate(source)
    original = read_json(invalid)
    try:
        jsonschema.validate(original, schema)
    except jsonschema.ValidationError as error:
        diagnostics = validation_diagnostics(error)
    else:
        raise ValueError("This probe requires an actually invalid JSON answer")
    messages += [{"role": "assistant", "content": invalid.read_text(encoding="utf-8")},
                 {"role": "user", "content": format_message(schema, diagnostics)}]
    repaired = call_json(messages, schema, output / "Analyst.format_probe.json", input_store=output / "inputs")
    fields = {key: contains_payload(repaired, value) for key, value in original.items()}
    original_hypotheses = original.get("hypotheses", [])
    summary = {"schema_valid": True, "source_request_sha256": digest(source),
        "source_invalid_sha256": digest(invalid), "original_error": diagnostics,
        "original_field_payloads_preserved_exactly_somewhere": fields,
        "original_findings_identical": repaired.get("findings") == original.get("findings"),
        "original_hypotheses_exact_item_matches": sum(contains_payload(repaired, item) for item in original_hypotheses),
        "original_hypotheses_items": len(original_hypotheses),
        "semantic_equivalence_not_automatically_proved": True,
        "new_scientific_role_chain": False, "original_call_rewritten": False,
        "generation_started": False, "designer_called": False,
        "probe_response_sha256": digest(output / "Analyst.format_probe.json"),
        "receipt": read_json(output / "Analyst.format_probe.receipt.json")}
    write_json(output / "probe_summary.json", summary)
    print({"schema_valid": True, "findings_identical": summary["original_findings_identical"],
           "all_original_payloads_preserved": all(fields.values()), "output": str(output)})


if __name__ == "__main__":
    main()
