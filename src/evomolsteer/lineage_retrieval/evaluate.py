"""Deterministic evaluation for the CK2 lineage retrieval study.

This module deliberately treats the oracle as an evaluation-only input.  A
candidate with no observed descendants is censored; it is never assigned a
negative affinity label.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _targets(pool: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return the canonical target list, synthesizing it from groups if needed."""
    if "targets" in pool:
        return list(pool["targets"])
    rows = []
    for group in pool.get("groups", []):
        for node in group.get("targets", []):
            rows.append({"node_id": node, "batch": group.get("batch"), "ancestor_slot": group.get("representative_slot"), "key": group.get("key")})
    return rows


def coverage_at_k(groups: Iterable[Mapping[str, Any]], target_ids: Iterable[Any], k: int, denominator: int | None = None) -> float:
    """Fraction of targets covered by the first k groups (small fixtures supported)."""
    ids = list(dict.fromkeys(target_ids))
    denom = len(ids) if denominator is None else denominator
    if denom <= 0:
        return 0.0
    hit = set()
    for group in list(groups)[:k]:
        hit.update(group.get("targets", []))
    return len(hit.intersection(ids)) / denom


def _at_k(values: list[dict[str, Any]], target_ids: list[Any], ancestors: set[Any], k: int) -> dict[str, float | int]:
    chosen = values[:k]
    covered = set().union(*(set(c["targets"]) for c in chosen)) if chosen else set()
    covered.intersection_update(target_ids)
    covered_ancestors = {a for c in chosen for a in c.get("ancestor_keys", []) if a in ancestors}
    positive = sum(bool(c["targets"]) for c in chosen)
    return {
        "coverage": len(covered) / len(target_ids) if target_ids else 0.0,
        "covered_targets": len(covered),
        "unique_ancestor_recall": len(covered_ancestors) / len(ancestors) if ancestors else 0.0,
        "unique_ancestors_recalled": len(covered_ancestors),
        "precision": positive / k if k else 0.0,
        "hit": int(bool(covered)),
    }


def evaluate_selection(pool_oracle: Mapping[str, Any], answer: Mapping[str, Any], mapping: Mapping[str, Any]) -> dict[str, Any]:
    """Score one selector answer without relying on generation order or IDs."""
    step = int(pool_oracle.get("step", answer.get("step", -1)))
    if int(answer.get("step", -2)) != step:
        raise ValueError(f"answer step {answer.get('step')} does not match pool step {step}")
    groups = list(pool_oracle.get("groups", []))
    by_key = {str(g["key"]): g for g in groups}
    anon_to_key = mapping.get("anonymous_to_key", mapping)
    if not isinstance(anon_to_key, Mapping):
        raise ValueError("mapping must contain anonymous_to_key mapping")
    ranking = answer.get("ranking", [])
    if len(ranking) != 5:
        raise ValueError("ranking must contain exactly five candidates")
    ranks = [row.get("rank") for row in ranking]
    if ranks != [1, 2, 3, 4, 5]:
        raise ValueError("ranking ranks must be exactly 1 through 5 in order")
    anon_ids = [str(row.get("candidate_id")) for row in ranking]
    if len(set(anon_ids)) != 5:
        raise ValueError("ranking candidate IDs must be distinct")
    resolved = []
    for row, anon in zip(ranking, anon_ids):
        if anon not in anon_to_key:
            raise ValueError(f"unknown anonymous candidate ID: {anon}")
        key = str(anon_to_key[anon])
        if key not in by_key:
            raise ValueError(f"candidate maps to unknown group key: {key}")
        resolved.append((row, by_key[key]))
    keys = [str(g["key"]) for _, g in resolved]
    if len(set(keys)) != 5:
        raise ValueError("ranking must resolve to five distinct groups")

    target_rows = _targets(pool_oracle)
    target_ids = list(dict.fromkeys(t["node_id"] for t in target_rows))
    if len(target_ids) != len(target_rows):
        raise ValueError("oracle target list contains duplicate node_id entries")
    membership: dict[Any, str] = {}
    for group in groups:
        for node in group.get("targets", []):
            if node in membership:
                raise ValueError(f"target {node!r} belongs to multiple groups; oracle groups must partition targets")
            membership[node] = str(group["key"])
    for target in target_rows:
        node = target["node_id"]
        if node not in membership:
            raise ValueError(f"target {node!r} has no group membership")
        if str(target.get("key")) != membership[node]:
            raise ValueError(f"target {node!r} key disagrees with group membership")
    ancestors = {str(t.get("key")) for t in target_rows if t.get("key") is not None}
    original_branches = {(t.get("batch"), t.get("ancestor_slot")) for t in target_rows if t.get("ancestor_slot") is not None}
    ordered = []
    target_rank: dict[Any, int] = {}
    alias_matches: dict[Any, bool] = {}
    strict_matches: dict[Any, bool] = {}
    alias_by_ancestor: dict[Any, bool] = {}
    strict_by_ancestor: dict[Any, bool] = {}
    strict_cover: dict[Any, bool] = {}
    for rank, (_, group) in enumerate(resolved, 1):
        gtargets = set(group.get("targets", []))
        aliases = set(group.get("aliases", []))
        representative = group.get("representative_slot")
        local_targets = []
        for target in target_rows:
            node = target["node_id"]
            if node in gtargets:
                target_rank.setdefault(node, rank)
                local_targets.append(node)
                slot = target.get("ancestor_slot")
                same_batch = target.get("batch") == group.get("batch")
                alias_matches[node] = same_batch and slot in aliases
                strict_matches[node] = same_batch and slot == representative
                ancestor_key = str(target.get("key"))
                alias_by_ancestor[ancestor_key] = alias_by_ancestor.get(ancestor_key, False) or alias_matches[node]
                strict_by_ancestor[ancestor_key] = strict_by_ancestor.get(ancestor_key, False) or strict_matches[node]
                strict_cover[node] = same_batch and slot == representative
        ordered.append({"rank": rank, "key": str(group["key"]), "target_count": len(gtargets.intersection(target_ids)),
                        "observed_descendants": group.get("observed_descendants"), "target_node_ids": local_targets,
                        "representative_slot": representative, "aliases": sorted(aliases, key=str)})

    selected = [{"targets": list(g.get("targets", [])), "ancestor_keys": list({str(t.get("key")) for t in target_rows if t["node_id"] in set(g.get("targets", [])) and t.get("key") is not None})} for _, g in resolved]
    target_by_id = {t["node_id"]: t for t in target_rows}
    metrics: dict[str, Any] = {"step": step, "target_count": len(target_ids), "unique_ancestor_count": len(ancestors),
        "unique_original_branch_count": len(original_branches),
        "ranking": ordered, "target_ranks": [{"node_id": n, "batch": target_by_id[n].get("batch"), "rank": target_rank.get(n), "ancestor_slot": target_by_id[n].get("ancestor_slot"), "ancestor_key": target_by_id[n].get("key"),
                           "alias_matched": alias_matches.get(n, False), "strict_representative_match": strict_matches.get(n, False)} for n in target_ids],
        "alias_matched_targets": sum(alias_matches.values()), "strict_representative_matched_targets": sum(strict_matches.values()),
        "applied_clauses": list(answer.get("applied_clauses", []))}
    for k in (1, 3, 5):
        metrics[f"at_{k}"] = _at_k(selected, target_ids, ancestors, k)
        selected_ancestors = {a for g in selected[:k] for a in g["ancestor_keys"]}
        metrics[f"at_{k}"]["alias_match_rate"] = sum(alias_by_ancestor.get(a, False) for a in selected_ancestors) / len(ancestors) if ancestors else 0.0
        metrics[f"at_{k}"]["strict_representative_match_rate"] = sum(strict_by_ancestor.get(a, False) for a in selected_ancestors) / len(ancestors) if ancestors else 0.0
        strict_target_hits = set().union(*(set(g.get("targets", [])) for g in selected[:k])) if k else set()
        metrics[f"at_{k}"]["strict_representative_coverage"] = sum(strict_cover.get(n, False) for n in strict_target_hits) / len(target_ids) if target_ids else 0.0
        strict_branches = {(target_by_id[n].get("batch"), target_by_id[n].get("ancestor_slot")) for n in strict_target_hits if strict_cover.get(n, False)}
        metrics[f"at_{k}"]["strict_original_branch_recall"] = len(strict_branches) / len(original_branches) if original_branches else 0.0

    # Ideal weighted coverage ceiling: greedy max-marginal target coverage, with
    # stable group-key tie breaking. This is an attainable oracle ceiling at k.
    all_groups = sorted(groups, key=lambda g: str(g["key"]))
    uncovered = set(target_ids)
    ceiling = {}
    for k in (1, 3, 5):
        if k == 1:
            uncovered = set(target_ids)
            chosen_ceiling: list[Mapping[str, Any]] = []
        while len(chosen_ceiling) < k:
            remaining = [g for g in all_groups if g not in chosen_ceiling]
            if not remaining:
                break
            best = max(remaining, key=lambda g: (len(set(g.get("targets", [])) & uncovered), -all_groups.index(g)))
            chosen_ceiling.append(best)
            uncovered -= set(best.get("targets", []))
        ceiling[str(k)] = len(set(target_ids) - uncovered) / len(target_ids) if target_ids else 0.0
    metrics["ideal_weighted_coverage_ceiling"] = ceiling
    return metrics


def _csv_write(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fieldnames or (list(rows[0]) if rows else [])
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def evaluate_round(study: str | Path, round_dir: str | Path) -> dict[str, Any]:
    """Evaluate all available selectors in a round and write sealed artifacts."""
    study, round_dir = Path(study), Path(round_dir)
    selector_root = round_dir / "selectors"
    metrics_rows, target_rows, branches, hashes = [], [], [], {}
    for answer_path in sorted(selector_root.glob("t*/answer.json")):
        match = re.search(r"t(\d+)$", answer_path.parent.name)
        step = int(match.group(1)) if match else None
        oracle_path = study / "oracle" / "pools" / f"t{step:03d}.json"
        mapping_path = answer_path.parent / "mapping.json"
        if not oracle_path.exists() or not mapping_path.exists():
            continue
        result = evaluate_selection(_read_json(oracle_path), _read_json(answer_path), _read_json(mapping_path))
        metrics_rows.append(result)
        hashes[str(answer_path.relative_to(round_dir))] = _sha256(answer_path)
        hashes[str(mapping_path.relative_to(round_dir))] = _sha256(mapping_path)
        hashes[str(oracle_path.relative_to(study))] = _sha256(oracle_path)
        target_rows.extend({"step": step, **row} for row in result["target_ranks"])
        branches.extend({"step": step, "rank": row["rank"], **row} for row in result["ranking"])
    complete = len(metrics_rows) == 10 and sorted(r["step"] for r in metrics_rows) == list(range(5, 51, 5))
    macro = {}
    if complete:
        for k in (1, 3, 5):
            macro[f"coverage_at_{k}"] = sum(r[f"at_{k}"]["coverage"] for r in metrics_rows) / len(metrics_rows)
            macro[f"precision_at_{k}"] = sum(r[f"at_{k}"]["precision"] for r in metrics_rows) / len(metrics_rows)
            macro[f"hit_at_{k}"] = sum(r[f"at_{k}"]["hit"] for r in metrics_rows) / len(metrics_rows)
    bundle = {"schema_version": 1, "round": round_dir.name, "evaluated_steps": [r["step"] for r in metrics_rows],
              "expected_steps": list(range(5, 51, 5)), "complete": len(metrics_rows) == 10,
              "macro": macro, "metrics": metrics_rows, "input_sha256": hashes}
    _write_json(round_dir / "metrics.json", bundle)
    _csv_write(round_dir / "target_ranks.csv", target_rows, ["step", "node_id", "batch", "rank", "ancestor_slot", "ancestor_key", "alias_matched", "strict_representative_match"])
    _csv_write(round_dir / "selected_branches.csv", branches)
    return bundle


def evaluate_baselines(study: str | Path) -> dict[str, Any]:
    """Compute deterministic random-selection expectations and intermediate-affinity oracle baselines."""
    study = Path(study)
    pool_paths = sorted((study / "oracle" / "pools").glob("t*.json"))
    rows = []
    random_rows = []
    for path in pool_paths:
        pool = _read_json(path)
        step = int(pool.get("step", re.search(r"t(\d+)", path.stem).group(1)))
        groups = list(pool.get("groups", [])); targets = _targets(pool)
        target_ids = list(dict.fromkeys(t["node_id"] for t in targets))
        ancestors = {str(t.get("key")) for t in targets if t.get("key") is not None}
        positive = sum(bool(set(g.get("targets", [])) & set(target_ids)) for g in groups)
        group_by_key = {str(g["key"]): g for g in groups}
        strict_target_count = sum(
            t.get("batch") == group_by_key[str(t.get("key"))].get("batch") and
            t.get("ancestor_slot") == group_by_key[str(t.get("key"))].get("representative_slot")
            for t in targets if str(t.get("key")) in group_by_key
        )
        # Exact analytic expectation for uniform sampling without replacement.
        # Oracle groups partition target membership, so each target and each
        # equivalence ancestor has inclusion probability min(k,N)/N.
        n = len(groups)
        for k in (1, 3, 5):
            kk = min(k, n)
            inclusion = kk / n if n else 0.0
            no_hit = math.comb(n - positive, kk) / math.comb(n, kk) if n and n - positive >= kk else (0.0 if positive else 1.0)
            random_rows.append({"step": step, "k": k, "coverage": inclusion,
                                "unique_ancestor_recall": inclusion, "hit": 1.0 - no_hit,
                                "precision": positive / n if n else 0.0,
                                "strict_representative_coverage": inclusion * strict_target_count / len(target_ids) if target_ids else 0.0,
                                "positive_groups": positive, "group_count": n, "method": "exact_analytic"})

        # Baseline selects top-five by current online pIC50. The final-time
        # affinity is intentionally excluded from this intermediate baseline.
        evidence_path = study / "evidence" / "pools" / f"t{step:03d}.json"
        evidence = _read_json(evidence_path) if evidence_path.exists() else {"cards": []}
        score = {c["key"]: c.get("now", {}).get("online_pic50") for c in evidence.get("cards", [])}
        ordered = sorted(groups, key=lambda g: (score.get(str(g["key"])) is not None, score.get(str(g["key"])) if score.get(str(g["key"])) is not None else float("-inf"), str(g["key"])), reverse=True)[:5]
        selection = [{"targets": list(g.get("targets", [])), "ancestor_keys": list({str(t.get("key")) for t in targets if t["node_id"] in set(g.get("targets", [])) and t.get("key") is not None})} for g in ordered]
        row = {"step": step, "method": "top5_intermediate_online_pic50", "groups_ranked": sum(score.get(str(g["key"])) is not None for g in ordered)}
        for k in (1, 3, 5):
            row[f"coverage_at_{k}"] = _at_k(selection, target_ids, ancestors, k)["coverage"]
            row[f"precision_at_{k}"] = _at_k(selection, target_ids, ancestors, k)["precision"]
            row[f"hit_at_{k}"] = _at_k(selection, target_ids, ancestors, k)["hit"]
        rows.append(row)
    bundle = {"schema_version": 1, "random_expectation": random_rows, "intermediate_affinity": rows,
              "note": "Intermediate baseline uses only online_pic50 available at that step; terminal/final affinity is excluded."}
    _write_json(study / "baselines.json", bundle)
    _csv_write(study / "baselines.csv", random_rows)
    _csv_write(study / "intermediate_affinity_baseline.csv", rows)
    return bundle
