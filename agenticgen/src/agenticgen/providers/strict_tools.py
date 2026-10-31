"""Provider-supported strict tool schemas; full application validation stays local."""

from copy import deepcopy

FINAL_TOOL = "submit_final_response"
ENDPOINT = "https://api.deepseek.com/beta/chat/completions"
CLIENT_ONLY = {"minLength", "maxLength", "minItems", "maxItems", "uniqueItems"}
SCALAR_KEYS = {
    "type", "description", "enum", "pattern", "format", "minimum", "maximum",
    "exclusiveMinimum", "exclusiveMaximum", "multipleOf", "const", "default",
}


def schema_for_wire(schema, path="$", client_checks=None):
    """Translate only the documented subset; never relax the local schema.

    All supplied object fields must already be mandatory. Unsupported length /
    uniqueness constraints remain in the original schema and are recorded as
    client checks. Unknown constraints fail before any provider request.
    """
    client_checks = [] if client_checks is None else client_checks
    unknown = set(schema) - SCALAR_KEYS - CLIENT_ONLY - {
        "properties", "required", "additionalProperties", "items", "anyOf",
    }
    if unknown:
        raise ValueError(f"Unsupported strict schema keywords at {path}: {sorted(unknown)}")
    result = {k: deepcopy(v) for k, v in schema.items() if k in SCALAR_KEYS}
    if "type" not in result and "enum" in result:
        if result["enum"] and all(type(v) is str for v in result["enum"]):
            result["type"] = "string"
        else:
            raise ValueError("Strict enum requires an explicit unambiguous type")
    for key in sorted(CLIENT_ONLY & set(schema)):
        client_checks.append(dict(path=path, keyword=key, value=deepcopy(schema[key])))
    if result.get("type") == "object":
        properties = schema.get("properties", {})
        if set(schema.get("required", [])) != set(properties) or schema.get("additionalProperties") is not False:
            raise ValueError("Strict objects require all declared fields and no additional properties")
        result.update(
            properties={k: schema_for_wire(v, path + "." + k, client_checks)
                        for k, v in properties.items()},
            required=list(properties), additionalProperties=False,
        )
    elif result.get("type") == "array":
        if "items" not in schema:
            raise ValueError("Strict arrays require an item schema")
        result["items"] = schema_for_wire(schema["items"], path + "[]", client_checks)
    if "anyOf" in schema:
        result["anyOf"] = [schema_for_wire(v, path + f".anyOf[{i}]", client_checks)
                           for i, v in enumerate(schema["anyOf"])]
    if "type" not in result and "anyOf" not in result:
        raise ValueError("Strict schema requires a type or anyOf")
    return result


def definitions(registry_definitions, final_schema):
    """Add a terminal submission capability; it never enters the evidence registry."""
    names = [d["function"]["name"] for d in registry_definitions]
    if len(set(names)) != len(names) or FINAL_TOOL in names:
        raise ValueError("Duplicate/reserved function name in strict tool registry")
    tools, checks = [], []
    for definition in registry_definitions:
        function = definition["function"]
        tool = deepcopy(definition)
        tool["function"].update(strict=True, parameters=schema_for_wire(
            function["parameters"], function["name"], checks))
        tools.append(tool)
    tools.append(dict(type="function", function=dict(
        name=FINAL_TOOL, strict=True,
        description=("Submit the complete final answer exactly once, alone in its tool-call batch. "
                     "Include every required field. This records a decision only; it never samples or measures."),
        parameters=schema_for_wire(final_schema, FINAL_TOOL, checks),
    )))
    return tools, checks
