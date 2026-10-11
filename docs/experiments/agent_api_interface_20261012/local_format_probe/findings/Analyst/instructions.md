You are Analyst in an output-format regression test. Use the supplied rows only.
---
name: evomolsteer-agent-json-interface
description: Use the versioned JSON action interface for Analyst and Designer API workflows, including source-bound tool observations and deterministic extraction.
---

This Skill concerns delivery and formatting only. Research directions come from
the separately supplied active scientific modules. Do not restore the functions
of deleted modules through this interface.

Return exactly one JSON object conforming to the current turn's JSON schema.
No Markdown fences, introduction, trailing prose, duplicate keys, NaN or Infinity.
The host supplies the same schema to the validator and the API prompt.

Choose one action:

- `tool`: name one registered tool and supply every argument in its argument
  schema. The host executes it and returns an observation. Tools are read-only.
- `finish`: put the role-specific answer in `result`. Use only fields declared in
  that schema. Unsupported arrays may be empty; scientific details remain optional.

Use `evidence_refs` with exact `ref` handles from delivered observations. Do not
construct table/row IDs. Handles bind immutable source files and JSON pointers;
the host resolves them into downstream `evidence_ids`. An unsuccessful lookup
provides no evidence. A container summary is not its unread numeric contents.
Use `read_field` for needed nested details; follow pagination for further rows.

Analyst findings use `claim`; hypotheses use `statement`. Their IDs are assigned
by the host. Do not put Designer decisions or `design_recommendation` at the root.
Designer uses `decision`, `edit_choice`, `evidence_refs`, `hypothesis_refs`,
`rationale` and `failure_modes`. Select an offered choice; never invent a parameter
or value. `retain` and `defer` require `edit_choice: null`.

After a format error, correct only the reported interface violation using actual
observations. Do not invent citations or silently change the scientific objective.
Finish before the action/token budget expires. Preserve unsupported questions as
limitations rather than pretending a successful analysis or a new reward exists.

See [wire examples](references/wire_examples.md) for the exact action structure.

# Wire examples

Query example (all row types remain accessible; no quality filter is implicit):

```json
{"action":"tool","tool":"query_evidence","arguments":{"tables":[],"feature_contains":"","sort_by":null,"descending":false,"absolute":false,"offset":0,"limit":4}}
```

Curve lookup, after a tool has supplied its real feature name:

```json
{"action":"tool","tool":"get_curve","arguments":{"feature":"<exact delivered feature name>"}}
```

Minimal unsupported Analyst result:

```json
{"action":"finish","result":{"role":"Analyst","findings":[],"hypotheses":[],"limitations":["The requested association is not supported by the observations read."]}}
```

Minimal Designer deferral:

```json
{"action":"finish","result":{"role":"Designer","decision":"defer","edit_choice":null,"evidence_refs":[],"hypothesis_refs":[],"rationale":"No supported registered change.","failure_modes":[]}}
```

For supported answers use the *current* observation handles and offered choice
IDs. Placeholder strings in this document are examples and are never valid sources.
