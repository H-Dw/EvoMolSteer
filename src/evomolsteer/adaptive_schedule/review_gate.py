"""Fail closed unless committed review evidence covers every completed V1 decision."""
import hashlib
import json
import re
import subprocess
from pathlib import Path

from .cohort import counts, progress
from .io import read, sha, repository, commit


def contained(base, relative):
    base = Path(base).resolve()
    p = (base / relative).resolve()
    if Path(relative).is_absolute() or not p.is_relative_to(base):
        raise ValueError("Review reference escapes its artifact root")
    return p


def committed(repo, revision, relative, expected_hash):
    path = contained(repo, relative)
    data = subprocess.check_output(["git", "-C", str(repo), "show", f"{revision}:{relative}"])
    if hashlib.sha256(data).hexdigest() != expected_hash or sha(path) != expected_hash:
        raise ValueError("Review or skill differs from its committed content")
    return data


def decision_key(row):
    return row["teacher"], row["batch"], row["slot"], row["id"]


def validate_review(root, config, skill):
    root, repo = Path(root), repository()
    approval = read(root / "versions/v2/review_approval.json")
    if approval.get("schema") != 1:
        raise ValueError("Unsupported review approval schema")
    templates = read(root / "teachers.json")
    if not progress(root, "v1", config, templates)["complete"]:
        raise ValueError("V1 histories or retention are incomplete")
    total = counts(config, templates)["decisions_per_version"]
    manifest_path = root / "review/v1/manifest.json"
    for relative, key in [("review/v1/manifest.json", "manifest_sha256"),
                          ("FINALIZED_v1.json", "finalized_sha256"),
                          ("versions/v1/binding.json", "v1_binding_sha256")]:
        if sha(root / relative) != approval.get(key):
            raise ValueError("V1 review provenance mismatch: " + relative)
    manifest = read(manifest_path)
    finalized = read(root / "FINALIZED_v1.json")
    if (manifest["version"] != "v1" or manifest["decisions"] != total or
            finalized["review_decisions"] != total or
            read(root / "versions/v1/COMPLETE.json")["decisions"] != total):
        raise ValueError("V1 review decision count does not match the cohort")
    revision = approval.get("review_commit", "")
    if not re.fullmatch(r"[0-9a-f]{40}", revision) or revision != commit():
        raise ValueError("Run V2 from the exact committed review and skill revision")
    skill_path = contained(repo, approval["skill_path"])
    if Path(skill).resolve() != skill_path:
        raise ValueError("Requested skill is not the approved committed skill")
    committed(repo, revision, approval["skill_path"], approval["skill_sha256"])
    if approval["skill_sha256"] == read(root / "versions/v1/binding.json")["skill_sha256"]:
        raise ValueError("V2 requires an actual skill revision")
    document = committed(repo, revision, approval["review_document"], approval["review_document_sha256"])
    if not document.strip():
        raise ValueError("Review report is empty")
    coverage = approval["coverage"]
    if len({c["path"] for c in coverage}) != len(coverage):
        raise ValueError("Duplicate review coverage entries")
    covered = {c["path"]: c for c in coverage}
    if set(covered) != {s["path"] for s in manifest["shards"]}:
        raise ValueError("Every V1 review shard must be covered exactly once")
    observed = set()
    for shard in manifest["shards"]:
        item = covered[shard["path"]]
        if sha(contained(root, shard["path"])) != shard["sha256"] or item["sha256"] != shard["sha256"]:
            raise ValueError("Review shard checksum mismatch")
        rows = read(contained(root, shard["path"]))
        review = json.loads(committed(repo, revision, item["review_file"], item["review_sha256"]))
        if review["shard_sha256"] != shard["sha256"] or len(rows) != shard["decisions"]:
            raise ValueError("Review shard identity or count mismatch")
        entries = review["decisions"]
        keys = [decision_key(x) for x in rows]
        reviewed_keys = [decision_key(x) for x in entries]
        if len(set(keys)) != len(keys) or len(set(reviewed_keys)) != len(entries) or set(keys) != set(reviewed_keys):
            raise ValueError("Review must assess every individual decision exactly once")
        if observed.intersection(keys):
            raise ValueError("Decision duplicated across shards")
        observed.update(keys)
        by_key = {decision_key(x): x for x in rows}
        for entry in entries:
            if not isinstance(entry.get("assessment"), str) or not entry["assessment"].strip():
                raise ValueError("Missing per-decision review assessment")
            row = by_key[decision_key(entry)]
            evidence_path = contained(root, row["prompt_evidence"])
            if sha(evidence_path) != row["prompt_evidence_sha256"]:
                raise ValueError("Actual prompt evidence changed")
            references = entry.get("evidence_ids", [])
            if not references or not set(references).issubset(read(evidence_path)["evidence"]):
                raise ValueError("Review must cite real prompt evidence")
    expected = {(t["teacher_id"], b, s, f"r{r:02d}") for t in templates
                for b in range(config["batches"]) for s in range(config["batch_size"])
                for r in range(1, config["rounds"] + 1)}
    if observed != expected or len(observed) != total:
        raise ValueError("Review coverage does not match every teacher, batch, slot and round")
    return {"review_commit": revision, "decisions": total, "shards": len(coverage),
            "skill_sha256": approval["skill_sha256"]}
