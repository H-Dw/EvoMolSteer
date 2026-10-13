"""One schema for prompting, parsing and validation; no scientific directions."""
import json

import jsonschema

POLICY = "evidence-tools-json-v1"


def strict_json(text):
    """Accept exactly one JSON object. Never salvage prose or partial responses."""
    def fail(message):
        raise json.JSONDecodeError(message, text, 0)

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                fail("Duplicate JSON property: " + key)
            result[key] = value
        return result

    def constant(value):
        fail("Non-finite JSON number: " + value)

    def number(value):
        import math
        parsed = float(value)
        if not math.isfinite(parsed):
            fail("Non-finite JSON number")
        return parsed

    result = json.loads(text, object_pairs_hook=pairs, parse_constant=constant, parse_float=number)
    if not isinstance(result, dict):
        fail("Root must be one JSON object")
    return result


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def obj(properties, required=None):
    return {"type": "object", "additionalProperties": False,
            "properties": properties, "required": list(properties) if required is None else required}


def refs(known):
    return {"type": "array", "uniqueItems": True,
            "items": {"type": "string", "enum": sorted(known)} if known else {"not": {}}}


def final_schema(role, known, choices=None, hypotheses=()):
    string = {"type": "string"}
    if role == "Analyst":
        finding = obj({"claim": string, "evidence_refs": refs(known),
                       "priority": {"type": "integer", "minimum": 1}}, ["claim", "evidence_refs"])
        hypothesis = obj({"statement": string, "evidence_refs": refs(known),
                          **{k: string for k in ["observable", "direction", "scope", "falsifier"]}},
                         ["statement", "evidence_refs"])
        return obj({"role": {"const": role}, "findings": {"type": "array", "items": finding},
                    "hypotheses": {"type": "array", "items": hypothesis},
                    "limitations": {"type": "array", "items": string}})
    if role != "Designer":
        raise ValueError("Only Analyst and Designer are registered")
    edit = {"type": "string", "enum": sorted(choices or {})} if choices else {"not": {}}
    schema = obj({"role": {"const": role}, "decision": {"enum": ["retain", "defer", "edit"]},
                  "edit_choice": {"anyOf": [{"type": "null"}, edit]},
                  "evidence_refs": refs(known), "hypothesis_refs": refs(hypotheses),
                  "rationale": string, "failure_modes": {"type": "array", "items": string}})
    schema["allOf"] = [
        {"if": {"properties": {"decision": {"const": "edit"}}},
         "then": {"properties": {"edit_choice": edit}}},
        {"if": {"properties": {"decision": {"enum": ["retain", "defer"]}}},
         "then": {"properties": {"edit_choice": {"type": "null"}}}}]
    return schema


def action_schema(role, known, tools, choices=None, hypotheses=()):
    return {"oneOf": [obj({"action": {"const": "tool"}, "tool": {"enum": sorted(tools)},
                           "arguments": {"type": "object"}}),
                      obj({"action": {"const": "finish"},
                           "result": final_schema(role, known, choices, hypotheses)})]}


def format_contract(schema, diagnostics=None, *, policy=POLICY):
    contract = {"interface_version": policy, "json_schema": schema}
    syntax_instruction = ''
    if policy != POLICY:
        contract['syntax_example_only'] = {'failure_modes': ['A first failure condition', 'A second failure condition']}
        syntax_instruction = 'Close each array with ] and each object with }; failure_modes remains an array, even when empty. '
    if diagnostics is not None:
        contract["format_errors"] = diagnostics
    return ("Return exactly one JSON object matching this JSON schema, with no Markdown or prose. "
            "Use action=tool to request one registered read-only operation, or action=finish with result. "
            "Citations are evidence_refs containing ONLY handles delivered in observations. "
            "Do not spell source IDs, add another role's fields, or reproduce numeric arrays. "
            "Optional scientific details remain optional. Unsupported findings may be omitted and limitations stated. "
            + syntax_instruction +
            "If repairing, change only formatting/references supported by delivered evidence; never guess sources.\n"
            + canonical(contract))


def validate(value, schema):
    jsonschema.Draft202012Validator(schema).validate(value)
