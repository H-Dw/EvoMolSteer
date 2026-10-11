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
