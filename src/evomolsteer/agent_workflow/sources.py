"""Bounded read-only retrieval from immutable, compact mathematical evidence."""
import copy
import hashlib
from pathlib import Path

import jsonschema

from evomolsteer.io import digest, read_json
from .contracts import canonical, obj


class EvidenceTools:
    """No shell, arbitrary paths, external network, generated code or refitting.

    Every row, including null/negative evidence, is accessible by stable paging.
    No elite filter or scientific ranking is imposed by the interface.
    """

    def __init__(self, evidence_dir):
        self.root = Path(evidence_dir).resolve()
        self.packet = read_json(self.root / "packet.json")
        self.files = {n: digest(self.root / n) for n in
                      ["packet.json", "index.json", "continuous_functions.json", "observable_definitions.json"]}
        self.index = read_json(self.root / "index.json")
        self.curves = read_json(self.root / "continuous_functions.json")
        self.definitions = read_json(self.root / "observable_definitions.json")
        clocks = self.packet["clocks"]
        if self.curves["window"] != clocks["learning_window"] or self.curves["times"] != clocks["observed_score_times"]:
            raise ValueError("Evidence clocks disagree")
        self.records, self.tables, self.pointers = {}, {}, {}
        for name, table in self.packet["tables"].items():
            rows = []
            for i, values in enumerate(table["rows"]):
                if len(values) != len(table["columns"]):
                    raise ValueError("Malformed columnar source")
                record = dict(zip(table["columns"], values))
                identifier = record["evidence_id"]
                if identifier in self.records:
                    raise ValueError("Duplicate evidence source ID")
                self.records[identifier] = record
                self.pointers[identifier] = f"/tables/{name}/rows/{i}"
                rows.append(record)
            self.tables[name] = rows
        if set(self.records) != set(self.index["ids"]):
            raise ValueError("Source-ID index is not the complete evidence dataset")
        self.delivered = {}
        names = sorted(self.tables)
        integer = lambda low, high: {"type": "integer", "minimum": low, "maximum": high}
        strings = {"type": "array", "items": {"type": "string"}, "uniqueItems": True}
        self.schemas = {
            "query_evidence": obj({"tables": {"type": "array", "items": {"enum": names}, "uniqueItems": True},
                "feature_contains": {"type": "string"}, "sort_by": {"type": ["string", "null"]},
                "descending": {"type": "boolean"}, "absolute": {"type": "boolean"},
                "offset": integer(0, 100000), "limit": integer(1, 24)}),
            "get_evidence": obj({"source_ids": dict(strings, minItems=1, maxItems=24)}),
            "get_curve": obj({"feature": {"type": "string"}}),
            "read_metadata": obj({"section": {"enum": ["clocks", "objective", "limitations", "teacher_modes", "prerequisites"]}}),
            "read_definition": obj({"key": {"enum": sorted(self.definitions)}}),
            "read_field": obj({"ref": {"type": "string"}, "path": {"type": "array", "items": {"type": "string"}},
                "offset": integer(0, 100000), "limit": integer(1, 24)})}

    @property
    def known(self):
        return set(self.delivered)

    def provenance(self):
        return {"sources": dict(self.files), "references": copy.deepcopy(self.delivered)}

    def verify_sources(self):
        if any(digest(self.root / name) != sha for name, sha in self.files.items()):
            raise ValueError("Evidence files changed during the workflow")

    def catalogue(self):
        return {"source_sha256": self.files, "tables": {
            name: {"rows": len(rows), "columns": self.packet["tables"][name]["columns"]}
            for name, rows in self.tables.items()},
            "tools": {name: {"arguments": schema} for name, schema in self.schemas.items()},
            "definitions": sorted(self.definitions), "curve_features": len(self.curves["functions"]),
            "paging": "All rows, including null/weak/negative evidence, are available. No automatic ranking/filtering.",
            "containers": "Large nested fields have explicit summaries. read_field retrieves a source-bound page."}

    @staticmethod
    def _preview(value):
        if isinstance(value, (dict, list)) and len(canonical(value).encode("utf-8")) > 3000:
            return {"container_summary_only": True, "type": type(value).__name__, "items": len(value),
                    "keys_preview": list(value)[:16] if isinstance(value, dict) else None,
                    "read_with": "read_field"}
        return copy.deepcopy(value)

    def _deliver(self, source_id, file, pointer, value, kind="row", table=None):
        # Handle binds the source bytes, source ID and precise pointer. It never
        # means a row with the same number in a different table.
        identity = {"source_id": source_id, "file": file, "pointer": pointer,
                    "source_sha256": self.files[file], "kind": kind, "table": table}
        ref = "E" + hashlib.sha256(canonical(identity).encode()).hexdigest()[:16]
        if ref in self.delivered and self.delivered[ref] != identity:
            raise ValueError("Reference collision")
        self.delivered[ref] = identity
        data = ({k: self._preview(v) for k, v in value.items()} if isinstance(value, dict) else self._preview(value))
        return {"ref": ref, "source": identity, "data": data}

    def _row(self, record):
        identifier = record["evidence_id"]
        pointer = self.pointers[identifier]
        return self._deliver(identifier, "packet.json", pointer, record, table=pointer.split("/")[2])

    def validate_call(self, name, arguments):
        if name not in self.schemas:
            raise jsonschema.ValidationError("Unregistered tool", validator="enum", validator_value=sorted(self.schemas), instance=name)
        jsonschema.Draft202012Validator(self.schemas[name]).validate(arguments)

    def execute(self, name, arguments):
        """Invalid scientific lookups are explicit observations, never fabricated rows."""
        self.validate_call(name, arguments)
        snapshot = copy.deepcopy(self.delivered)
        try:
            observation = getattr(self, "_" + name)(**arguments)
        except (KeyError, IndexError, TypeError, ValueError) as error:
            self.delivered = snapshot
            return {"ok": False, "error": {"type": type(error).__name__, "message": str(error)},
                    "instruction": "Query the catalogue or an existing delivered reference; never invent a record."}
        if len(canonical(observation).encode("utf-8")) > 48000:
            self.delivered = snapshot
            return {"ok": False, "error": {"type": "ObservationBudget", "message": "Use a smaller page or read_field"}}
        return {"ok": True, "observation": observation}

    def _query_evidence(self, tables, feature_contains, sort_by, descending, absolute, offset, limit):
        rows = [r for t in (tables or sorted(self.tables)) for r in self.tables[t]
                if feature_contains in str(r.get("feature", ""))]
        if sort_by is not None:
            # Missing/null values are retained at the end, never turned into 0.
            present = [r for r in rows if r.get(sort_by) is not None]
            if any(not isinstance(r[sort_by], (int, float)) or isinstance(r[sort_by], bool) for r in present):
                raise ValueError("sort_by must name a numeric field")
            present.sort(key=lambda r: abs(r[sort_by]) if absolute else r[sort_by], reverse=descending)
            rows = present + [r for r in rows if r.get(sort_by) is None]
        return {"matched": len(rows), "offset": offset, "next_offset": offset + limit if offset + limit < len(rows) else None,
                "records": [self._row(r) for r in rows[offset:offset + limit]]}

    def _get_evidence(self, source_ids):
        for identifier in source_ids:
            if identifier not in self.records:
                raise KeyError("Unknown source ID: " + identifier)
        return {"records": [self._row(self.records[i]) for i in source_ids]}

    def _get_curve(self, feature):
        if feature not in self.curves["functions"]:
            raise KeyError("Unknown curve feature: " + feature)
        pointer = "/functions/" + feature.replace("~", "~0").replace("/", "~1")
        # Preserve fit diagnostics, empirical values, coefficients and clocks.
        return self._deliver("curve/" + feature, "continuous_functions.json", pointer,
                             self.curves["functions"][feature], kind="curve")

    def _read_metadata(self, section):
        return self._deliver("metadata/" + section, "packet.json", "/" + section,
                             self.packet[section], kind="metadata")

    def _read_definition(self, key):
        return self._deliver("definition/" + key, "observable_definitions.json", "/" + key,
                             self.definitions[key], kind="definition")

    def _source_value(self, identity):
        if identity["kind"] == "row":
            return self.records[identity["source_id"]]
        if identity["kind"] == "curve":
            return self.curves["functions"][identity["source_id"][6:]]
        if identity["kind"] == "definition":
            return self.definitions[identity["source_id"][11:]]
        if identity["kind"] == "metadata":
            return self.packet[identity["source_id"][9:]]
        # Field references resolve back to their original immutable base.
        base = identity["base"]
        value = self._source_value(base)
        for key in identity["field_path"]:
            if isinstance(value, list) and (not key.isdigit() or int(key) >= len(value)):
                raise KeyError("Invalid nonnegative source array index")
            value = value[int(key)] if isinstance(value, list) else value[key]
        return value

    def _read_field(self, ref, path, offset, limit):
        if ref not in self.delivered:
            raise KeyError("Reference has not been delivered: " + ref)
        base = self.delivered[ref]
        value = self._source_value(base)
        for key in path:
            if isinstance(value, list) and (not key.isdigit() or int(key) >= len(value)):
                raise KeyError("Invalid nonnegative source array index")
            value = value[int(key)] if isinstance(value, list) else value[key]
        if isinstance(value, dict):
            keys = list(value)[offset:offset + limit]
            page = {k: self._preview(value[k]) for k in keys}
            total = len(value)
        elif isinstance(value, list):
            page = [self._preview(v) for v in value[offset:offset + limit]]
            total = len(value)
        else:
            page, total = value, 1
        identity = {"source_id": base["source_id"], "file": base["file"], "source_sha256": base["source_sha256"],
                    "kind": "field", "base": base, "field_path": path,
                    "pointer": base["pointer"] + ("/" + "/".join(k.replace("~", "~0").replace("/", "~1") for k in path) if path else ""),
                    "offset": offset, "limit": limit}
        field_ref = "E" + hashlib.sha256(canonical(identity).encode()).hexdigest()[:16]
        self.delivered[field_ref] = identity
        return {"ref": field_ref, "source": identity, "data": page, "total": total, "offset": offset,
                "next_offset": offset + limit if offset + limit < total else None}

    def import_references(self, provenance):
        """Designer receives exactly the validated Analyst references, never fresh IDs."""
        if provenance["sources"] != self.files:
            raise ValueError("Analyst and Designer evidence sources differ")
        for ref, identity in provenance["references"].items():
            expected = "E" + hashlib.sha256(canonical(identity).encode()).hexdigest()[:16]
            if ref != expected:
                raise ValueError("Corrupt reference identity")
            if identity.get("source_sha256") != self.files.get(identity.get("file")):
                raise ValueError("Reference source digest is not bound to the evidence dataset")
            self._source_value(identity)
            self.delivered[ref] = copy.deepcopy(identity)

    def resolve(self, refs):
        """Legacy citations contain real delivered source IDs; no guessed replacements."""
        result = []
        for ref in refs:
            identity = self.delivered[ref]
            identifier = identity["source_id"]
            if identifier.startswith("metadata/"):
                result.append(identifier[9:])
            elif identifier.startswith("definition/"):
                result.append("observable_definitions")
            elif identifier.startswith("curve/"):
                feature = identifier[6:]
                matches = [r["evidence_id"] for r in self.tables["continuous_trends"] if r["feature"] == feature]
                result.extend(matches + ["curve_support"])
            else:
                result.append(identifier)
        return list(dict.fromkeys(result))
