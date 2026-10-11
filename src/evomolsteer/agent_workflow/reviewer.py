"""Reviewer selects source cards; the host extracts exact quotes and pointers."""
import hashlib

import jsonschema

from evomolsteer.deepseek_api import call_json
from evomolsteer.io import write_json
from .contracts import POLICY, canonical, obj, strict_json


def source_cards(calls, chunk_size=160, overlap=32):
    if not 1 <= chunk_size <= 200 or not 0 <= overlap < chunk_size:
        raise ValueError("Invalid source-card size")
    cards = {}

    def walk(value, chain, pointer):
        if isinstance(value, str) and value:
            for start in range(0, len(value), chunk_size - overlap):
                text = value[start:start + chunk_size]
                card = {"chain": chain, "pointer": pointer, "start": start,
                        "end": start + len(text), "text": text}
                ref = "S" + hashlib.sha256(canonical(card).encode()).hexdigest()[:16]
                cards[ref] = card
        elif isinstance(value, dict):
            for key, item in value.items():
                # Formatting IDs/role names cannot prove a scientific function.
                if key in {"role", "id", "evidence_ids", "hypothesis_ids"}:
                    continue
                walk(item, chain, pointer + "/" + key.replace("~", "~0").replace("/", "~1"))
        elif isinstance(value, list):
            for i, item in enumerate(value):
                walk(item, chain, pointer + "/" + str(i))

    for chain, call in enumerate(calls):
        walk(call, chain, "")
    return cards


def extract_review(result, calls, rubrics, cards):
    expected = {(i, m["module"]) for i in range(len(calls)) for m in rubrics}
    actual = [(a["chain"], a["module"]) for a in result["assessments"]]
    if set(actual) != expected or len(actual) != len(expected):
        raise jsonschema.ValidationError("Every chain/rubric must appear exactly once")
    assessments = []
    for a in result["assessments"]:
        ref = a["support_ref"]
        if ref is not None and (ref not in cards or cards[ref]["chain"] != a["chain"]):
            raise jsonschema.ValidationError("Source card not from assessed chain")
        if a["score"] in {1, 2} and ref is None:
            raise jsonschema.ValidationError("Positive score needs an actual source card")
        card = cards[ref] if ref is not None else None
        if card:
            source = calls[a["chain"]]
            for key in card["pointer"].split("/")[1:]:
                key = key.replace("~1", "/").replace("~0", "~")
                source = source[int(key)] if isinstance(source, list) else source[key]
            if not isinstance(source, str) or source[card["start"]:card["end"]] != card["text"]:
                raise jsonschema.ValidationError("Source-card content changed")
        assessments.append({k: a[k] for k in ["chain", "module", "score", "explanation"]} |
                           {"quote": card["text"] if card else "", "source_ref": ref,
                            "source_pointer": card["pointer"] if card else None})
    return {"assessments": assessments}


def run_review(calls, rubrics, output, *, caller=call_json, input_store=None, evidence_context=None):
    cards = source_cards(calls)
    shape = obj({"chain": {"type": "integer", "minimum": 0, "maximum": len(calls) - 1},
                 "module": {"enum": [m["module"] for m in rubrics]}, "score": {"enum": [0, 1, 2, "NA"]},
                 "support_ref": {"anyOf": [{"type": "null"}, {"enum": sorted(cards)}]},
                 "explanation": {"type": "string", "maxLength": 250}})
    schema = obj({"assessments": {"type": "array", "items": shape}})
    messages = [{"role": "system", "content": "You are a blinded functional Reviewer. Return JSON. "
                 "Assess every supplied chain and rubric once. 0=absent/wrong, 1=generic mention, "
                 "2=evidence-specific function affecting a priority/hypothesis/design, NA=required input absent. "
                 "Select a supplied support_ref from the SAME chain for positive scores; null for absence. "
                 "Never compose or paraphrase a quote. The host extracts exact source text. "
                 "Do not infer generation quality, instruction condition or module necessity. "
                 "Evidence-correctness checks require supplied source context; use NA when that input is absent."},
                {"role": "user", "content": canonical({"cards": cards, "rubrics": rubrics, "evidence_context": evidence_context})}]
    validator = lambda value: extract_review(value, calls, rubrics, cards)
    def contract(shape, errors=None):
        return "Return JSON matching this schema. " + canonical({"schema": shape, "format_errors": errors})
    # Final selected policies can contain several focal modules across six
    # chains. Allocate response space for complete coverage, never truncate it.
    raw = caller(messages, schema, output, input_store=input_store,
                 max_output_tokens=min(16384, max(4096, len(calls) * len(rubrics) * 256)),
                 response_parser=strict_json, response_validator=validator, format_policy=POLICY,
                 format_builder=contract)
    result = validator(raw)
    write_json(output.with_name(output.stem + ".extracted.json"), result)
    return result
