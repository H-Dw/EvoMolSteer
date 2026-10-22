"""Host-enforced scientific action boundary; amplitudes are integer tenths."""
import math

import jsonschema
from .grid import INTERVALS


def validate_ticks(ticks):
    if not isinstance(ticks, list) or len(ticks) != len(INTERVALS):
        raise ValueError("Exactly four interval amplitudes are required")
    if any(type(x) is not int or not 0 <= x <= 10 for x in ticks):
        raise ValueError("Amplitude ticks must be integers in [0,10]; tick 5 means lambda=0.5")
    return ticks


def expanded(ticks):
    validate_ticks(ticks)
    values = [0.] * 100
    for value, (start, end) in zip(ticks, INTERVALS):
        values[start:end + 1] = [value / 10] * (end - start + 1)
    return values


def earliest(old, new):
    validate_ticks(old)
    validate_ticks(new)
    changed = [i for i, (a, b) in enumerate(zip(old, new)) if a != b]
    if not changed:
        raise ValueError("No-op or duplicate schedule is not an experiment")
    return INTERVALS[min(changed)][0]


def error(message):
    raise jsonschema.ValidationError(message, validator="scientific_contract", validator_value=message)


def step_allowances(history, parent_id):
    """A double tick needs two consecutive improving ancestral edits of that interval/direction."""
    by_id = {x["id"]: x for x in history}
    allowances = {}
    p = by_id[parent_id]
    for interval in range(len(INTERVALS)):
        signs = []
        node = p
        for _ in range(2):
            previous = by_id.get(node.get("parent_id"))
            if previous is None or not node.get("improved_parent", False):
                break
            d = node["ticks"][interval] - previous["ticks"][interval]
            if not d:
                break
            signs.append(1 if d > 0 else -1)
            node = previous
        if len(signs) == 2 and signs[0] == signs[1]:
            allowances[interval] = signs[0]
    return allowances


def validate_action(action, history, evidence_ids, parent_ids=None):
    by_id = {x["id"]: x for x in history}
    parent = by_id.get(action["parent_id"])
    if parent is None:
        error("Parent is not in this candidate's immutable history")
    if parent_ids is not None and action["parent_id"] not in parent_ids:
        error("Only the retained incumbent is restartable; use past trials as documentary evidence")
    try:
        new = validate_ticks(action["ticks"])
        earliest(parent["ticks"], new)
    except ValueError as exc:
        error(str(exc))
    if any(x["ticks"] == new for x in history):
        error("This exact schedule has already been evaluated; use its recorded outcome")
    changed = [(i, b - a) for i, (a, b) in enumerate(zip(parent["ticks"], new)) if a != b]
    if len(changed) > 2:
        error("At most two intervals may change in one proposal")
    allowances = step_allowances(history, parent["id"])
    for i, delta in changed:
        if abs(delta) > 2 or (abs(delta) == 2 and allowances.get(i) != (1 if delta > 0 else -1)):
            error("Use a one-tick change; two ticks require two consecutive improving ancestral edits in that direction")
    if not set(action["evidence_ids"]).issubset(set(evidence_ids)):
        error("Evidence reference was not supplied to this candidate")
    if not action["evidence_ids"]:
        error("A proposal must cite observed evidence")
    return earliest(parent["ticks"], new)


def schema(history, evidence_ids, parent_ids=None):
    string = {"type": "string", "minLength": 1}
    props = {
        "parent_id": {"enum": parent_ids if parent_ids is not None else [x["id"] for x in history]},
        "ticks": {"type": "array", "minItems": len(INTERVALS), "maxItems": len(INTERVALS),
                  "items": {"type": "integer", "minimum": 0, "maximum": 10}},
        "evidence_ids": {"type": "array", "minItems": 1, "uniqueItems": True, "items": {"enum": evidence_ids}},
        "diagnosis": string, "hypothesis": string, "expected_change": string,
        "expected_mean_effect": string, "expected_best_effect": string,
        "falsifier": string, "memory_update": string,
    }
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}
