"""Human-readable reports for lineage retrieval studies."""
from __future__ import annotations

import csv
import hashlib
import html
import json
from pathlib import Path
from typing import Any

from .evaluate import _csv_write, _read_json


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _functional_job_status(job: Path) -> tuple[bool, str]:
    """Verify a functional audit is bound to this request and this answer receipt."""
    answer_path, audit_path = job / "answer.json", job / "functional_audit.json"
    binding_path, receipt_path = job / "controller_binding.json", job / "receipt.json"
    if not answer_path.exists() or not audit_path.exists():
        return False, "missing answer or functional audit"
    if not binding_path.exists():
        return False, "missing controller binding"
    if not receipt_path.exists():
        return False, "missing model-call receipt"
    try:
        audit = _read_json(audit_path)
        binding = _read_json(binding_path)
        receipt = _read_json(receipt_path)
    except (OSError, json.JSONDecodeError):
        return False, "unreadable audit, binding, or receipt"
    request_hash = binding.get("request_sha256")
    if not request_hash or audit.get("request_sha256") != request_hash:
        return False, "functional audit request hash does not match controller binding"
    try:
        answer_hash = _canonical_json_sha256(_read_json(answer_path))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return False, "current answer is unreadable or noncanonical"
    if receipt.get("status") != "completed" or receipt.get("answer_sha256") != answer_hash:
        return False, "receipt is incomplete or does not identify the current answer"
    if not audit.get("clause_id") or audit.get("prompt_contains_active_clause") is not True:
        return False, "functional audit does not verify the active clause"
    if audit.get("application_claim_not_a_causal_improvement") is not True:
        return False, "functional audit is missing its causal limitation"
    return True, ""


def _selection_metrics_complete(metrics: dict[str, Any]) -> bool:
    expected = set(range(5, 51, 5))
    rows = metrics.get("metrics", [])
    try:
        return bool(metrics.get("complete") is True and len(rows) == 10
                    and {int(row.get("step", -1)) for row in rows} == expected
                    and {int(step) for step in metrics.get("evaluated_steps", [])} == expected)
    except (TypeError, ValueError):
        return False


def _selection_run_complete(folder: Path, metrics: dict[str, Any]) -> tuple[bool, list[str]]:
    issues = []
    if not _selection_metrics_complete(metrics):
        issues.append("metrics do not contain ten unique expected cutoffs")
    for step in range(5, 51, 5):
        ok, reason = _functional_job_status(folder / "selectors" / f"t{step:03d}")
        if not ok:
            issues.append(f"t{step:03d}: {reason}")
    return not issues, issues


def _load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _metric_macro(metrics: dict[str, Any]) -> dict[str, float]:
    macro = metrics.get("macro", {})
    rows = metrics.get("metrics", [])
    return {f"coverage_at_{k}": float(macro.get(f"coverage_at_{k}", sum(r.get(f"at_{k}", {}).get("coverage", 0.0) for r in rows) / len(rows) if rows else 0.0)) for k in (1, 3, 5)}


def _escape_md(value: Any) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", " ")


def _rank_table_markdown(node_rows: list[dict[str, Any]], rank_by_node_step: dict[tuple[str, int], Any], steps: list[int]) -> str:
    if not node_rows or not steps:
        return "（尚无完整轮次排名。）"
    header = "| node_id | batch | " + " | ".join(f"t{s:03d}" for s in steps) + " |\n|---|---:|" + "---:|" * len(steps)
    lines = [header]
    for row in node_rows:
        node = str(row.get("node_id", ""))
        vals = [rank_by_node_step.get((node, s)) for s in steps]
        lines.append("| " + node + " | " + str(row.get("batch", "")) + " | " + " | ".join("—" if x is None else str(x) for x in vals) + " |")
    return "\n".join(lines)


def _scientist_source(study: Path, round_dir: Path) -> tuple[Path, str | None]:
    """Resolve round-local Scientist outputs, including frozen R26+ outputs."""
    meta_path = round_dir / "_version.json"
    meta = _read_json(meta_path) if meta_path.exists() else {}
    frozen = meta.get("scientist_outputs_frozen_from")
    frozen_receipt = round_dir / "scientist_frozen.json"
    if frozen_receipt.exists():
        frozen = _read_json(frozen_receipt).get("source_round") or frozen
    if frozen:
        return study / "rounds" / str(frozen), str(frozen)
    return round_dir, None


def _round_audit_status(study: Path, round_dir: Path) -> dict[str, Any]:
    """Check that functional audits and the main evidence audit exist and pass."""
    meta_path = round_dir / "_version.json"
    meta = _read_json(meta_path) if meta_path.exists() else {}
    source, frozen_round = _scientist_source(study, round_dir)
    functional = []
    for step in range(5, 51, 5):
        job = round_dir / "selectors" / f"t{step:03d}"
        path = job / "functional_audit.json"
        data = _read_json(path) if path.exists() else None
        ok, reason = _functional_job_status(job)
        functional.append({"role": "selector", "step": step, "path": path, "ok": ok, "reason": reason, "data": data})
    scientist_path = source / "scientist" / "functional_audit.json"
    scientist_data = _read_json(scientist_path) if scientist_path.exists() else None
    scientist_ok, scientist_reason = _functional_job_status(source / "scientist")
    functional.append({"role": "scientist", "step": None, "path": scientist_path, "ok": scientist_ok, "reason": scientist_reason, "data": scientist_data})
    semantic_path = round_dir / "semantic_audit.json"
    semantic = _read_json(semantic_path) if semantic_path.exists() else None
    assessments = semantic.get("selector_assessments", []) if semantic else []
    semantic_ok = bool(semantic and semantic.get("passes") is True
                       and len(assessments) == 10
                       and {int(a.get("step", -1)) for a in assessments} == set(range(5, 51, 5))
                       and all(a.get("evidence_truth") is True and a.get("clause_applied") is True for a in assessments)
                       and semantic.get("scientist_evidence_truth") is True
                       and semantic.get("scientist_clause_applied") is True)
    issues = [f"missing/invalid {x['role']} functional audit" + (f" at t{x['step']:03d}" if x["step"] is not None else "") for x in functional if not x["ok"]]
    if not semantic_ok:
        issues.append("missing or failed main-model xhigh evidence audit")
    semantic_failure = semantic.get("failure_analysis", "") if semantic else ""
    return {"complete": all(x["ok"] for x in functional) and semantic_ok,
            "functional_complete": all(x["ok"] for x in functional),
            "functional": functional, "scientist_source_round": frozen_round,
            "semantic_path": semantic_path, "semantic": semantic, "semantic_complete": semantic_ok,
            "issues": issues, "semantic_failure": semantic_failure}


def _write_plot_curves(study: Path, curves: list[dict[str, Any]], heat: list[list[float | None]], node_ids: list[str], steps: list[int]) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return []
    emitted = []
    if curves:
        fig, ax = plt.subplots(figsize=(9, 5.2))
        for k, color in ((1, "#4477AA"), (3, "#228833"), (5, "#CC6677")):
            usable = [r for r in curves if r.get(f"coverage_at_{k}") is not None]
            if usable:
                ax.plot([int(str(r["round"])[1:]) for r in usable], [r[f"coverage_at_{k}"] for r in usable], marker="o", color=color, label=f"Selector Coverage@{k}")
                ceilings = [r.get(f"ceiling_at_{k}") for r in usable]
                if any(v is not None for v in ceilings):
                    ax.plot([int(str(r["round"])[1:]) for r in usable], [v if v is not None else float("nan") for v in ceilings], linestyle="--", color=color, alpha=.7, label=f"Oracle ceiling@{k}")
        ax.set(xlabel="Optimizer round (R00–R25 Scientist; R26–R50 Selector)", ylabel="Macro top-30 coverage", ylim=(0, 1), title="Mean selector retrieval by complete round")
        ax.grid(alpha=.25)
        handles, labels = ax.get_legend_handles_labels()
        if handles: ax.legend(ncol=2, fontsize=8)
        fig.tight_layout()
        for ext in ("png", "pdf", "svg"):
            path = study / f"coverage_curves.{ext}"; fig.savefig(path, dpi=180); emitted.append(path.name)
        plt.close(fig)
    if heat and node_ids and steps:
        arr = np.array([[float("nan") if v is None else v for v in row] for row in heat], dtype=float)
        fig, ax = plt.subplots(figsize=(10, max(6, len(node_ids) * .19)))
        masked = np.ma.masked_invalid(arr)
        im = ax.imshow(masked, aspect="auto", interpolation="nearest", cmap="viridis_r", vmin=1, vmax=5)
        ax.set_xticks(range(len(steps)), [str(s) for s in steps]); ax.set_yticks(range(len(node_ids)), node_ids, fontsize=5)
        ax.set(xlabel="Generation step", ylabel="Top-30 target node", title="Selector rank of each target (blank = not selected)")
        fig.colorbar(im, ax=ax, label="Rank (1 is best)"); fig.tight_layout()
        for ext in ("png", "pdf", "svg"):
            path = study / f"target_rank_heatmap.{ext}"; fig.savefig(path, dpi=180); emitted.append(path.name)
        plt.close(fig)
    return emitted


def report(study: str | Path) -> dict[str, Any]:
    """Create Chinese Markdown/HTML report and iteration/rank artifacts.

    Partial studies are reported as partial and never silently promoted to
    complete. This function only summarizes on-disk evidence.
    """
    study = Path(study)
    study.mkdir(parents=True, exist_ok=True)
    expected_steps = list(range(5, 51, 5))
    expected_rounds = [f"R{i:02d}" for i in range(51)]
    status_path = study / "run_status.json"
    status = _read_json(status_path) if status_path.exists() else {"status": "unknown"}
    baseline_path = study / "baselines.json"
    baselines = _read_json(baseline_path) if baseline_path.exists() else {}
    rounds_root = study / "rounds"
    round_dirs = sorted((p for p in rounds_root.glob("R*") if p.is_dir()), key=lambda p: p.name) if rounds_root.exists() else []
    round_metrics = []
    version_manifests = []
    all_metrics = []
    audits = []
    rejected_trials = []
    audit_status_by_round: dict[str, dict[str, Any]] = {}
    confirmation_dirs = sorted((study / "confirmations").glob("*") if (study / "confirmations").exists() else [])
    for round_dir in round_dirs:
        rejected_root = round_dir / "rejected"
        if rejected_root.exists():
            for trial_dir in sorted((p for p in rejected_root.glob("T*") if p.is_dir()), key=lambda p: p.name):
                manifest_path = trial_dir / "manifest.json"
                manifest = _read_json(manifest_path) if manifest_path.exists() else {}
                rejected_trials.append({"round": round_dir.name, "trial": trial_dir.name,
                    "reason": manifest.get("reason", "archive manifest missing"),
                    "manifest_sha256": _file_sha256(manifest_path) if manifest_path.exists() else None,
                    "archived_file_count": len(manifest.get("files", [])),
                    "contains_rejected_metrics": any(r.get("source") == "metrics.json" for r in manifest.get("files", [])),
                    "manifest": manifest})
        try:
            audit_status_by_round[round_dir.name] = _round_audit_status(study, round_dir)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            audit_status_by_round[round_dir.name] = {"complete": False, "functional_complete": False,
                "functional": [], "scientist_source_round": None, "semantic": None, "semantic_complete": False}
        metrics_path = round_dir / "metrics.json"
        metrics = _read_json(metrics_path) if metrics_path.exists() else None
        if metrics:
            all_metrics.append((round_dir.name, metrics))
            version_path = round_dir / "_version.json"
            version_manifests.append({"round": round_dir.name, "manifest": _read_json(version_path) if version_path.exists() else metrics.get("version", metrics.get("manifest")), "complete": metrics.get("complete", False)})
            for item in metrics.get("metrics", []):
                round_metrics.append((round_dir.name, item))
        else:
            version_path = round_dir / "_version.json"
            version_manifests.append({"round": round_dir.name, "manifest": _read_json(version_path) if version_path.exists() else None, "complete": False})
        for p in round_dir.rglob("*audit*.json"):
            try:
                audits.append({"round": round_dir.name, "file": str(p.relative_to(study)), "data": _read_json(p)})
            except (OSError, json.JSONDecodeError):
                continue
    present = {s for _, m in round_metrics for s in m.get("evaluated_steps", [])}
    complete_rounds = [name for name, m in all_metrics if m.get("complete") and len(m.get("metrics", [])) == 10
                       and set(m.get("evaluated_steps", [])) == set(expected_steps)
                       and audit_status_by_round.get(name, {}).get("complete")]
    study_complete = status.get("status") == "complete" and set(complete_rounds) >= set(expected_rounds)
    # A best version is chosen only among complete rounds, using the mean over
    # all ten observation times. Input ordering preserves earliest-round ties.
    best = None
    for name, metrics in all_metrics:
        if name not in complete_rounds:
            continue
        macro = metrics.get("macro") or {}
        if not all(f"coverage_at_{k}" in macro for k in (1, 3, 5)):
            rows = metrics.get("metrics", [])
            macro = {f"coverage_at_{k}": sum(r.get(f"at_{k}", {}).get("coverage", 0) for r in rows) / len(rows) for k in (1, 3, 5)}
        score = (macro["coverage_at_5"], macro["coverage_at_3"], macro["coverage_at_1"])
        if best is None or score > best["score"]:
            best = {"round": name, "score": score, "metrics": metrics}
    if best:
        best = {"round": best["round"], "macro_coverage_at_5": best["score"][0], "macro_coverage_at_3": best["score"][1], "macro_coverage_at_1": best["score"][2], "tie_break": "rounds ranked by mean Coverage@5, @3, @1 across steps 5..50; exact tie keeps earliest round"}
    status_best = status.get("best_round")
    if status_best in complete_rounds:
        best_round_name = status_best
    else:
        best_round_name = best.get("round") if best else None

    targets_path = study / "oracle" / "top30.csv"
    target_rows = _load_csv(targets_path)
    node_col = next((c for c in ("node_id", "target_node_id", "id") if target_rows and c in target_rows[0]), "node_id")
    node_ids = [row.get(node_col, "") for row in target_rows[:30]]
    rank_lookup = {(name, int(m["step"]), str(t["node_id"])): t.get("rank") for name, m in round_metrics for t in m.get("target_ranks", []) if t.get("node_id") is not None}
    selected_round = best_round_name if best_round_name else (round_metrics[0][0] if round_metrics else None)
    heat_steps = sorted({int(m["step"]) for n, m in round_metrics if n == selected_round})
    heat = [[rank_lookup.get((selected_round, step, node)) for step in heat_steps] for node in node_ids]

    ceilings = {}
    for _, m in round_metrics:
        for k, v in m.get("ideal_weighted_coverage_ceiling", {}).items():
            ceilings[(int(m["step"]), int(k))] = v
    curve_rows = []
    metrics_by_name = {name: m for name, m in all_metrics}
    versions_by_name = {row["round"]: row.get("manifest") or {} for row in version_manifests}
    for name in expected_rounds:
        m = metrics_by_name.get(name)
        manifest = versions_by_name.get(name, {})
        round_number = int(name[1:])
        role = "scientist" if round_number <= 25 else "selector"
        row: dict[str, Any] = {"round": name, "active_role": role,
            "active_version": manifest.get(f"{role}_version"),
            "scientist_version": manifest.get("scientist_version"),
            "selector_version": manifest.get("selector_version"),
            "complete": bool(m and name in complete_rounds)}
        macro = (m or {}).get("macro", {})
        if m and not macro and name in complete_rounds:
            mr = m.get("metrics", [])
            macro = {f"coverage_at_{k}": sum(r.get(f"at_{k}", {}).get("coverage", 0) for r in mr) / len(mr) for k in (1, 3, 5)}
        for k in (1, 3, 5):
            row[f"coverage_at_{k}"] = macro.get(f"coverage_at_{k}") if name in complete_rounds else None
            ceiling_values = [float(r.get("ideal_weighted_coverage_ceiling", {}).get(str(k), 0)) for r in (m or {}).get("metrics", [])]
            row[f"ceiling_at_{k}"] = sum(ceiling_values) / len(ceiling_values) if ceiling_values and name in complete_rounds else None
        curve_rows.append(row)
    _csv_write(study / "iteration_curves.csv", curve_rows)
    _csv_write(study / "rejected_trials.csv", [{k: v for k, v in row.items() if k != "manifest"} for row in rejected_trials],
               ["round", "trial", "reason", "manifest_sha256", "archived_file_count", "contains_rejected_metrics"])
    raw_rows = []
    for name, m in round_metrics:
        for m in m.get("metrics", []):
            r = {"round": name, "step": m.get("step")}
            for k in (1, 3, 5):
                v = m.get(f"at_{k}", {})
                r.update({f"coverage_at_{k}": v.get("coverage"), f"precision_at_{k}": v.get("precision"), f"hit_at_{k}": v.get("hit"), f"unique_ancestor_recall_at_{k}": v.get("unique_ancestor_recall"), f"strict_rep_match_at_{k}": v.get("strict_representative_match_rate"), f"strict_rep_coverage_at_{k}": v.get("strict_representative_coverage"), f"alias_match_at_{k}": v.get("alias_match_rate")})
            raw_rows.append(r)
    _csv_write(study / "iteration_scores.csv", raw_rows)
    plot_files = _write_plot_curves(study, curve_rows, heat, node_ids, heat_steps)

    # Private Scientist prose is joined to true teacher identities only in this
    # evaluation report; decision workers never receive teacher_map.
    teacher_map_path = study / "oracle" / "teacher_map.json"
    teacher_map = _read_json(teacher_map_path) if teacher_map_path.exists() else {}
    descriptions = []
    hypotheses = []
    scientist_source_round = None
    if best_round_name:
        scientist_source_dir, scientist_source_round = _scientist_source(study, study / "rounds" / best_round_name)
        scientist_answer_path = scientist_source_dir / "scientist" / "answer.json"
        hypotheses_path = study / "rounds" / best_round_name / "hypotheses.json"
        if not hypotheses_path.exists():
            hypotheses_path = scientist_source_dir / "hypotheses.json"
        if scientist_answer_path.exists():
            answer = _read_json(scientist_answer_path)
            for item in answer.get("interval_descriptions", []):
                mapped = teacher_map.get(item.get("donor_id"), {})
                descriptions.append({"donor_id": item.get("donor_id"), "node_id": mapped.get("node_id"),
                                     "batch": mapped.get("batch"), "slot": mapped.get("slot"),
                                     "interval": item.get("interval"), "text": item.get("text"),
                                     "evidence": json.dumps(item.get("evidence", []), ensure_ascii=False)})
        if hypotheses_path.exists():
            hypotheses = _read_json(hypotheses_path)
    descriptions.sort(key=lambda r: (str(r.get("node_id", "")), int(r.get("interval") or 0)))
    _csv_write(study / "private_interval_descriptions.csv", descriptions,
               ["node_id", "batch", "slot", "donor_id", "interval", "text", "evidence"])
    hypotheses_md = "\n".join(f"| {h.get('interval')} | {h.get('start_step')}–{h.get('end_step')} | {h.get('hypothesis', '')} | {h.get('minority_modes', '')} |" for h in hypotheses)
    if hypotheses_md:
        hypotheses_md = "| 区间 | steps | 假设 | 少数模式 |\n|---:|---:|---|---|\n" + hypotheses_md
    else:
        hypotheses_md = "（最佳轮次没有可读假设文件。）"

    best_target_rows = []
    rank_by_node_step: dict[tuple[str, int], Any] = {}
    for name, item in round_metrics:
        if name == best_round_name:
            for target in item.get("target_ranks", []):
                rank_by_node_step[(str(target.get("node_id")), int(item["step"]))] = target.get("rank")
    for row in sorted(target_rows, key=lambda x: int(x.get("rank") or 999)):
        node = str(row.get("node_id", ""))
        summary = {"node_id": node, "batch": row.get("batch"), "slot": row.get("slot"),
                   "terminal_pic50_on_rescore": row.get("pic50_on_rescore"),
                   "strain_or_relief": row.get("mmff_relief_per_heavy", row.get("strain_MMFF94s_kcal_mol"))}
        for step in expected_steps:
            summary[f"rank_t{step:03d}"] = rank_by_node_step.get((node, step))
        best_target_rows.append(summary)
    _csv_write(study / "best_round_target_summary.csv", best_target_rows)
    rank_table_md = _rank_table_markdown(target_rows[:30], rank_by_node_step, expected_steps)

    # Track requested instruction versions separately from observed call
    # receipts. Version text is binding metadata, not proof of behavior.
    role_version_rows = []
    version_root = study / "versions"
    for role in ("scientist", "selector"):
        for version_dir in sorted((version_root / role).glob("*") if (version_root / role).exists() else []):
            meta_path = version_dir / "version.json"
            if not meta_path.exists():
                continue
            meta = _read_json(meta_path)
            diagnosis_path = version_dir / "diagnosis.json"
            diag = _read_json(diagnosis_path) if diagnosis_path.exists() else {}
            role_version_rows.append({"role": role, "version": meta.get("version", version_dir.name), "parent": meta.get("parent"),
                "active_clause": meta.get("active_clause"), "model": meta.get("model"), "effort": meta.get("effort"),
                "controller_model": meta.get("controller_model"), "controller_effort": meta.get("controller_effort"),
                "skill_sha256": meta.get("skill_sha256"), "workflow_sha256": meta.get("workflow_sha256"),
                "diagnosis": diag.get("diagnosis"), "mechanism": diag.get("mechanism"), "observable_effect": diag.get("observable_effect")})
    _csv_write(study / "role_versions.csv", role_version_rows)
    receipt_rows = []
    for root in (study / "rounds", study / "controls", study / "confirmations"):
        if not root.exists():
            continue
        for receipt_path in root.rglob("receipt.json"):
            try:
                receipt = _read_json(receipt_path)
            except (OSError, json.JSONDecodeError):
                continue
            receipt_rows.append({"job": str(receipt_path.parent.relative_to(study)), "role": receipt.get("role"),
                                 "requested_model": receipt.get("requested_model"), "model": receipt.get("model"),
                                 "effort": receipt.get("effort"), "status": receipt.get("status"),
                                 "prompt_sha256": receipt.get("prompt_sha256"), "packet_sha256": receipt.get("packet_sha256")})
    _csv_write(study / "role_call_receipts.csv", receipt_rows)

    # Main controls and confirmation repetitions remain distinct. Repetition 1
    # is the already-recorded main round or structure-only control.
    control_rows = []
    controls_root = study / "controls" / "structure_only"
    if controls_root.exists():
        for path in sorted(controls_root.glob("R*")):
            mp = path / "metrics.json"
            if mp.exists():
                cm = _read_json(mp); macro = _metric_macro(cm)
                valid, issues = _selection_run_complete(path, cm)
                control_rows.append({"version_round": path.name, "condition": "structure_only", "repeat": 1, "source": str(path.relative_to(study)), "complete": valid, "validation_issues": "; ".join(issues), **macro})
    confirmation_rows = []
    for label in dict.fromkeys(("R00", status_best or (best_round_name or ""))):
        if not label:
            continue
        source = study / "rounds" / label / "metrics.json"
        if source.exists():
            cm = _read_json(source); macro = _metric_macro(cm)
            valid, issues = _selection_run_complete(source.parent, cm)
            # The main repeat also needs its independent xhigh evidence audit.
            valid = valid and audit_status_by_round.get(label, {}).get("semantic_complete", False)
            confirmation_rows.append({"version_round": label, "condition": "affinity_visible", "repeat": 1, "source": str(source.parent.relative_to(study)), "complete": valid, "validation_issues": "; ".join(issues), **macro})
        control = study / "controls" / "structure_only" / label / "metrics.json"
        if control.exists():
            cm = _read_json(control); macro = _metric_macro(cm)
            valid, issues = _selection_run_complete(control.parent, cm)
            confirmation_rows.append({"version_round": label, "condition": "structure_only", "repeat": 1, "source": str(control.parent.relative_to(study)), "complete": valid, "validation_issues": "; ".join(issues), **macro})
        for condition in ("affinity_visible", "structure_only"):
            for repeat in (2, 3):
                cp = study / "confirmations" / label / condition / f"repeat{repeat}" / "metrics.json"
                if cp.exists():
                    cm = _read_json(cp); macro = _metric_macro(cm)
                    valid, issues = _selection_run_complete(cp.parent, cm)
                    confirmation_rows.append({"version_round": label, "condition": condition, "repeat": repeat,
                        "source": str(cp.parent.relative_to(study)), "complete": valid, "validation_issues": "; ".join(issues), **macro})
    _csv_write(study / "structure_only_controls.csv", control_rows)
    _csv_write(study / "confirmation_repeats.csv", confirmation_rows)
    confirmation_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in confirmation_rows:
        confirmation_groups.setdefault((row["version_round"], row["condition"]), []).append(row)
    confirmation_summary = []
    for (label, condition), rows in sorted(confirmation_groups.items()):
        valid = [r for r in rows if r.get("complete")]
        summary = {"version_round": label, "condition": condition, "n_repeats": len(valid),
                   "expected_repeats": 3, "complete": len(valid) == 3}
        for k in (1, 3, 5):
            summary[f"mean_coverage_at_{k}"] = sum(r[f"coverage_at_{k}"] for r in valid) / len(valid) if valid else None
        confirmation_summary.append(summary)
    _csv_write(study / "confirmation_summary.csv", confirmation_summary)
    random_baseline = baselines.get("random_expectation", [])
    affinity_baseline = baselines.get("intermediate_affinity", [])
    baseline_md = "Random reference and intermediate online-pIC50 selection are stored in `baselines.csv` and `intermediate_affinity_baseline.csv`. The latter uses only contemporaneous online affinity, never terminal affinity."
    if not baselines:
        baseline_md = "Baseline results are not available yet; no reference scores are imputed."
    control_md = "\n".join(f"| {r['version_round']} | {r['coverage_at_1']:.3f} | {r['coverage_at_3']:.3f} | {r['coverage_at_5']:.3f} | {r['complete']} |" for r in control_rows)
    control_md = ("| 版本 | C@1 | C@3 | C@5 | complete |\n|---|---:|---:|---:|---|\n" + control_md) if control_md else "（尚无结构专用控制结果。）"
    confirmation_md = "\n".join(f"| {r['version_round']} | {r['condition']} | {r['repeat']} | {r['coverage_at_1']:.3f} | {r['coverage_at_3']:.3f} | {r['coverage_at_5']:.3f} | {r['complete']} |" for r in confirmation_rows)
    confirmation_md = ("| 版本 | 条件 | repeat | C@1 | C@3 | C@5 | complete |\n|---|---|---:|---:|---:|---:|---|\n" + confirmation_md) if confirmation_md else "（尚无确认重复结果。）"
    rejected_md = "\n".join(f"| {r['round']} | {r['trial']} | {_escape_md(r['reason'])} | {r['archived_file_count']} | {r['contains_rejected_metrics']} | `{r['manifest_sha256'] or 'missing'}` |" for r in rejected_trials)
    rejected_md = ("| round | trial | rejection reason | archived files | rejected metrics archived | archive manifest SHA-256 |\n|---|---|---|---:|---:|---|\n" + rejected_md) if rejected_md else "（尚无归档拒绝轮次。）"

    # Describe iteration instructions and hypotheses only when provenance exists.
    teacher_text = []
    for pattern in ("**/*teacher*.json", "**/*provenance*.json"):
        for p in study.glob(pattern):
            try:
                data = _read_json(p)
                teacher_text.append({"file": str(p.relative_to(study)), "data": data})
            except (OSError, json.JSONDecodeError):
                continue
    config_candidates = [study / "experiment.json", study / "config.json", study / "experiment_config.json"]
    config = next((_read_json(p) for p in config_candidates if p.exists()), {})
    model = config.get("model", config.get("primary_model", status.get("subagent_model", "配置未记录")))
    reasoning = config.get("reasoning", config.get("reasoning_effort", status.get("subagent_effort", "配置未记录")))
    study_hypotheses = hypotheses
    config_hypotheses = config.get("qualitative_hypotheses", config.get("hypotheses", []))
    interval_lines = []
    if teacher_text:
        for item in teacher_text:
            interval_lines.append(f"- `{item['file']}`：`{json.dumps(item['data'], ensure_ascii=False)[:1000]}`")
    else:
        interval_lines.append("- 教师来源或区间描述尚未写入 study；本报告不推测 300 次迭代的具体描述。")
    configured_hypotheses_md = "\n".join(f"- {h}" for h in config_hypotheses) if config_hypotheses else "- 配置中未记录定性假设。"
    actual_hypotheses_md = "\n".join(f"| {h.get('interval')} | {h.get('start_step')}–{h.get('end_step')} | {_escape_md(h.get('hypothesis'))} | {_escape_md(h.get('minority_modes'))} |" for h in study_hypotheses)
    actual_hypotheses_md = ("| 区间 | steps | 假设 | 少数模式 |\n|---:|---:|---|---|\n" + actual_hypotheses_md) if actual_hypotheses_md else "（最佳完整轮次没有可读假设文件。）"
    private_description_md = "\n".join(
        "| {node} | {batch} | {slot} | {interval} | {text} |".format(
            node=_escape_md(r.get("node_id")), batch=r.get("batch", ""), slot=r.get("slot", ""),
            interval=r.get("interval", ""), text=_escape_md(r.get("text")))
        for r in descriptions)
    private_description_md = ("| node_id | batch | final slot | interval | Scientist private description |\n|---|---:|---:|---:|---|\n" + private_description_md) if private_description_md else "（最佳完整轮次的 Scientist 私有描述尚不可用。）"
    completeness = "完整" if study_complete else "未完成或状态未知"
    audit_rows = []
    for name in expected_rounds:
        state = audit_status_by_round.get(name, {})
        audit_rows.append({"round": name, "functional_audit": bool(state.get("functional_complete")),
                           "semantic_audit": bool(state.get("semantic_complete")),
                           "scientist_source_round": state.get("scientist_source_round") or name,
                           "complete_for_selection": name in complete_rounds})
    for row in audit_rows:
        info = audit_status_by_round.get(row["round"], {})
        functional_issues = [f"{item['role']} {item['reason']}" + (f" (t{item['step']:03d})" if item["step"] is not None else "")
                             for item in info.get("functional", []) if not item.get("ok")]
        row["audit_issue"] = "; ".join(info.get("issues", []) + functional_issues)
        row["semantic_failure"] = info.get("semantic_failure", "")
    audit_table = "\n".join(f"| {r['round']} | {r['scientist_source_round']} | {r['functional_audit']} | {r['semantic_audit']} | {r['complete_for_selection']} | {_escape_md(r['audit_issue'])} |" for r in audit_rows if r['round'] in {n for n, _ in all_metrics})
    audit_table = ("| round | Scientist output source | functional audits | xhigh evidence audit | eligible for best | audit gaps |\n|---|---|---:|---:|---:|---|\n" + audit_table) if audit_table else "（尚无主实验审计记录。）"
    failed_audit_lines = [f"- {r['round']}: {_escape_md(r['semantic_failure'])}" for r in audit_rows if r["semantic_failure"] and not r["semantic_audit"]]
    failed_audit_md = "\n".join(failed_audit_lines) if failed_audit_lines else "（尚无已记录的失败审计分析。）"
    best_semantic = audit_status_by_round.get(best_round_name, {}).get("semantic") if best_round_name else None
    best_semantic_text = ("Main-model xhigh semantic audit for the selected round: passes=" + str(best_semantic.get("passes")) +
        "; Scientist assessment: " + _escape_md(best_semantic.get("scientist_assessment")) +
        "; instruction effect: " + _escape_md(best_semantic.get("instruction_effect_assessment")) +
        "; failure analysis: " + _escape_md(best_semantic.get("failure_analysis")) if best_semantic else "尚无合格最佳轮次的 xhigh 语义审计。")
    best_semantic_rows = best_semantic.get("selector_assessments", []) if best_semantic else []
    best_semantic_table = "\n".join(f"| {a.get('step')} | {a.get('evidence_truth')} | {a.get('clause_applied')} | {_escape_md(a.get('evidence'))} | {_escape_md(a.get('correction_needed'))} |" for a in best_semantic_rows)
    best_semantic_table = ("| step | evidence truth | clause applied | audit evidence | correction needed |\n|---:|---:|---:|---|---|\n" + best_semantic_table) if best_semantic_table else "（无逐时点语义审计。）"
    best_line = ("最佳完整版本：轮次 " + str(best["round"]) + "，宏平均 Coverage@5=" + f"{best['macro_coverage_at_5']:.3f}" + "，Coverage@3=" + f"{best['macro_coverage_at_3']:.3f}" + "，Coverage@1=" + f"{best['macro_coverage_at_1']:.3f}" + "。按完整轮次 10 个时间点的均值排序；并列按 Coverage@5、@3、@1，完全并列保留较早轮次。" if best else "尚无完整主实验轮次可供版本选择。")
    md = f"""# CK2 TOP30 Scientist–Selector 评估报告

**状态：{completeness}。** 运行状态文件记录为 `{status.get('status', 'unknown')}`；主实验完整轮次为 {len(complete_rounds)}/51，已发现 {len(confirmation_dirs)} 个确认目录（单独报告，不计入 51 个主轮次）。选择器答案覆盖步骤 {', '.join(map(str, sorted(present))) or '无'}。缺失的答案不会按零分填充。

## 实验配置与假设

- 主模型：{model}
- 推理强度：{reasoning}
- 定性假设：
{configured_hypotheses_md}

最佳完整版本的十条假设：
{actual_hypotheses_md}

最佳完整版本的 300 条 Scientist 私有描述（donor ID 已映射回真实目标 ID，仅供事后分析；Scientist 输出来源轮次：{scientist_source_round or '未解析'}）：
{private_description_md}

## 迭代指导

每 300 次迭代的教师指导描述仅从教师溯源文件摘录：
{chr(10).join(interval_lines)}

## 结果

{best_line}

每个优化轮次的十个时间点宏平均见 `iteration_curves.csv`；R00–R25 优化 Scientist，R26–R50 冻结 Scientist 并优化 Selector。每轮每时间点原始分数见 `iteration_scores.csv`。目标节点排名热图基于主实验最佳完整版本；空格表示该目标未进入所选五个分支。`best_round_target_summary.csv` 列出 TOP30 全部真实 node_id、batch、final slot、最终 readout 和十个时点排名。下表为该版本精确的 30×10 ancestry rank 矩阵。

已拒绝并归档的试验不会进入曲线、版本选择或指标平均；归档试验及其原因/清单哈希见 `rejected_trials.csv`：

{rejected_md}

{rank_table_md}

结构专用控制结果（完整行另见 `structure_only_controls.csv`）：

{control_md}

独立重复（原始记录见 `confirmation_repeats.csv`，按条件汇总见 `confirmation_summary.csv`）：repeat 1 是原始主实验/控制结果，repeat 2–3 是新上下文与新匿名 ID 的确认调用；两个输入条件分开列出。均值仅在完整重复上计算，并同时列出有效重复数。

{confirmation_md}

随机基线与中间亲和力基线：{baseline_md}

## 指标定义

- Coverage@k：前 k 个分支覆盖的 TOP30 节点数除以 30（若 oracle 清单不足 30，则按实际唯一目标数）。
- Unique-ancestor recall：前 k 个分支触及的不同祖先等价组 key 数除以 TOP30 涉及的不同等价组数；跨 batch 的相同槽位号不会合并。另报告以 (batch, ancestor_slot) 区分的原始分支召回。
- Precision@k / Hit@k：前 k 个分支中含目标的分支比例 / 是否至少覆盖一个目标。
- Alias match 与 strict representative match 分别检查目标祖先槽位是否属于分支 alias 集、是否严格等于默认代表槽位。
- observed_descendants 是观察到的后代数。没有观察到后代代表右删失或无观测，不代表低亲和力。

## 审计与局限

报告保留选择器回答、匿名映射、oracle 池的 SHA-256 输入清单（见各轮 `metrics.json`），并核对功能审计的 request hash 与 controller binding、当前 answer hash 与调用 receipt。主轮次只有在所有十个 Selector 的绑定有效 `functional_audit.json`、有效 Scientist functional audit（R26+ 指向冻结 Scientist 的来源轮次）和通过的 main-model xhigh `semantic_audit.json`（Scientist evidence-truth/application 两项及十个 Selector 时点均显式通过）均存在时才可进入版本比较。审计覆盖情况：

{audit_table}

已记录但未通过的 xhigh 审计失败分析：
{failed_audit_md}

所选完整轮次的实际 xhigh 语义审计：{best_semantic_text}

{best_semantic_table}

运行状态请求配置：main `{status.get('main_model', '未记录')} / {status.get('main_effort', '未记录')}`；worker `{status.get('subagent_model', '未记录')} / {status.get('subagent_effort', '未记录')}`。观测到 {len(receipt_rows)} 份角色调用 receipt；逐条记录见 `role_call_receipts.csv`，角色版本与指令哈希见 `role_versions.csv`。当前 Scientist bridge 按顺序只解锁一个五步区间，下一段要等本段提交后才可读取；调用日志可审计这一访问顺序。功能审计和语义审计是实验流程中的审核证据，不等于证明指令改变了内部推理或带来因果改进。评估是对既有 TOP30 成员关系的回顾性检索；它不证明未来泛化能力。

## 版本清单

```json
{json.dumps(version_manifests, ensure_ascii=False, indent=2)}
```

## 角色调用审计发现

发现 {len(audits)} 个文件名含 audit 的 JSON 文件。R26+ 的 Scientist 描述与功能审计均来自 `_version.json` / `scientist_frozen.json` 指定的冻结来源轮次；版本清单和本节逐轮明确标出来源。详细原始内容留在各轮目录中；匿名映射还原后的节点明细仅保留在本地研究报告与 CSV 中，不会放入 Selector 输入包。
"""
    (study / "report.md").write_text(md, encoding="utf-8")
    # Escape all Markdown as text in a minimal HTML wrapper: preserves the
    # content safely and avoids adding an undeclared markdown dependency.
    html_doc = "<!doctype html><html lang=\"zh-CN\"><meta charset=\"utf-8\"><title>CK2 TOP30 评估报告</title><body><pre style=\"white-space:pre-wrap;font-family:system-ui;max-width:1000px;margin:2rem auto\">" + html.escape(md) + "</pre></body></html>"
    (study / "report.html").write_text(html_doc, encoding="utf-8")
    result = {"status": "complete" if study_complete else "incomplete", "complete_rounds": complete_rounds, "expected_rounds": expected_rounds,
              "evaluated_steps": sorted(present), "best": best, "plots": plot_files,
              "artifacts": ["report.md", "report.html", "iteration_curves.csv", "iteration_scores.csv", "rejected_trials.csv", "private_interval_descriptions.csv", "best_round_target_summary.csv", "role_versions.csv", "role_call_receipts.csv", "structure_only_controls.csv", "confirmation_repeats.csv", "confirmation_summary.csv", *plot_files],
              "audits_found": len(audits), "round_audit_status": {name: {"complete": info.get("complete"), "functional_complete": info.get("functional_complete"), "semantic_complete": info.get("semantic_complete"), "scientist_source_round": info.get("scientist_source_round")} for name, info in audit_status_by_round.items()}, "rejected_trials": [{k: v for k, v in row.items() if k != "manifest"} for row in rejected_trials], "confirmation_directories": [p.name for p in confirmation_dirs],
              "confirmation_records": confirmation_rows, "confirmation_summary": confirmation_summary,
              "structure_only_control_records": control_rows,
              "scientist_descriptions": len(descriptions), "random_baseline_rows": len(random_baseline),
              "intermediate_affinity_baseline_rows": len(affinity_baseline), "teacher_provenance_files": [x["file"] for x in teacher_text]}
    (study / "report_manifest.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result
