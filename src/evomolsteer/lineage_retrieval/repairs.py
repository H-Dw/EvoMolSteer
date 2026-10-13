"""Safe archival and target selection for semantic CK2 round repairs.

This module never generates scientific content or alters scores. It preserves
the original attempts while moving only the current, accepted round outputs so
the controller can create a corrected versioned retry.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_ROUND_FILE_ALLOWLIST = {
    "metrics.json",
    "hypotheses.json",
    "_version.json",
    "semantic_audit.json",
    "quality_retry.json",
    "target_ranks.csv",
    "selected_branches.csv",
}
_CURRENT_JOB_FILE_ALLOWLIST = {
    "answer.json",
    "functional_audit.json",
    "receipt.json",
    "events.jsonl",
    "controller_binding.json",
}
_VALID_SELECTOR_STEPS = set(range(5, 51, 5))
_SCIENTIST_SCOPE = re.compile(
    r"\b(scientist|teacher|donor|hypothes(?:is|es)|interval|description|paragraph|scientific prose)\b",
    re.IGNORECASE,
)
_NEGATIVE_CLAIM = re.compile(
    r"\b(unsupported|unsubstantiated|inaccurate|incorrect|false|overstated?|"
    r"misstates?|misrepresents?|not supported|not grounded|does not match|"
    r"fails? to match|contradicts?|fabricated)\b",
    re.IGNORECASE,
)
_NEGATION_PREFIX = re.compile(r"\b(no|none|not|without|never|neither)\b[^.!?]{0,24}$", re.IGNORECASE)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _contained(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _has_positive_negative_claim(text: str) -> bool:
    """Find a negative claim while ignoring plain statements that none exist."""
    for match in _NEGATIVE_CLAIM.finditer(text):
        prefix = text[max(0, match.start() - 32):match.start()]
        if not _NEGATION_PREFIX.search(prefix):
            return True
    return False


def semantic_repair_targets(audit: dict[str, Any]) -> dict[str, Any]:
    """Return the Scientist flag and exact failed Selector steps from an audit.

    Current audits use ``scientist_assessment`` text. Newer schemas may also
    provide ``scientist_evidence_truth`` and ``scientist_clause_applied`` booleans.
    Selector targeting is limited to the ten declared study cutoffs.
    """
    if not isinstance(audit, dict):
        raise TypeError("audit must be a JSON object")

    scientist = False
    for key in ("scientist_evidence_truth", "scientist_clause_applied"):
        if key in audit and audit[key] is False:
            scientist = True

    assessment = audit.get("scientist_assessment", "")
    if isinstance(assessment, str) and _has_positive_negative_claim(assessment):
        scientist = True

    claims = audit.get("unsupported_claims", [])
    if isinstance(claims, list):
        for claim in claims:
            if not isinstance(claim, str):
                continue
            # The array name already says these are unsupported; scope them to
            # Scientist content only when the claim identifies that content.
            if _SCIENTIST_SCOPE.search(claim):
                scientist = True

    failed_steps: set[int] = set()
    assessments = audit.get("selector_assessments", [])
    if isinstance(assessments, list):
        for row in assessments:
            if not isinstance(row, dict):
                continue
            step = row.get("step")
            if isinstance(step, bool) or not isinstance(step, int) or step not in _VALID_SELECTOR_STEPS:
                continue
            if row.get("evidence_truth") is False or row.get("clause_applied") is False:
                failed_steps.add(step)

    return {"scientist": scientist, "selector_steps": sorted(failed_steps)}


def _archive_sources(round_path: Path) -> list[Path]:
    sources: list[Path] = []
    for name in sorted(_ROUND_FILE_ALLOWLIST):
        path = round_path / name
        if path.exists():
            if path.is_file():
                sources.append(path)
            else:
                raise ValueError(f"allowlisted round artifact is not a file: {name}")

    for subtree in ("scientist", "selectors"):
        root = round_path / subtree
        if not root.exists():
            continue
        if not root.is_dir():
            raise ValueError(f"allowlisted round subtree is not a directory: {subtree}")
        for path in sorted(root.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"symlinks are not allowed in archived subtrees: {path}")
            if not path.is_file() or path.name not in _CURRENT_JOB_FILE_ALLOWLIST:
                continue
            rel_parts = path.relative_to(round_path).parts
            # Preserve mappings, packets, prompts, and every attempt-level
            # original (including failures). Only root Scientist outputs and
            # direct per-cutoff Selector outputs are current round results.
            current_job_file = (
                subtree == "scientist" and len(rel_parts) == 2
            ) or (
                subtree == "selectors" and len(rel_parts) == 3
                and re.fullmatch(r"t\d{3}", rel_parts[1]) is not None
            )
            if current_job_file:
                sources.append(path)
    return sources


def archive_round_trial(study: str | Path, round_dir: str | Path, reason: str) -> Path:
    """Move allowlisted current outputs into ``round/rejected/Txx``.

    Attempt directories (including failed model outputs), candidate mappings,
    prompts, and packets remain in place. Every moved file is recorded with its
    relative source/archive paths, byte count, and SHA-256 digest.
    """
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("reason is required")
    study_arg = Path(study).expanduser().absolute()
    round_arg = Path(round_dir).expanduser().absolute()
    study_root = study_arg.resolve(strict=True)
    round_path = round_arg.resolve(strict=True)
    rounds_root = (study_root / "rounds").resolve(strict=True)
    if not _contained(rounds_root, study_root):
        raise ValueError("study rounds directory resolves outside study")
    if not _contained(round_path, rounds_root) or round_path.parent != rounds_root:
        raise ValueError("round_dir must be a direct child of study/rounds")
    if not re.fullmatch(r"R\d{2,}", round_path.name):
        raise ValueError("round directory must be named R followed by a round number")
    if round_arg.resolve(strict=True) != round_path:
        raise ValueError("round path did not resolve consistently")

    rejected_root = round_path / "rejected"
    if rejected_root.exists() and (rejected_root.is_symlink() or not rejected_root.is_dir()):
        raise ValueError("round rejected path must be a real directory")
    rejected_root.mkdir(exist_ok=True)
    rejected_root = rejected_root.resolve(strict=True)
    if not _contained(rejected_root, round_path):
        raise ValueError("rejected archive resolves outside its round")

    archive_dir = None
    for index in range(1, 10000):
        candidate = rejected_root / f"T{index:02d}"
        if not candidate.exists():
            archive_dir = candidate
            break
    if archive_dir is None:
        raise RuntimeError("no unused rejected trial identifier is available")
    archive_dir.mkdir()
    if not _contained(archive_dir.resolve(strict=True), round_path):
        raise ValueError("archive directory resolves outside its round")

    records = []
    moved: list[tuple[Path, Path]] = []
    try:
        for source in _archive_sources(round_path):
            resolved_source = source.resolve(strict=True)
            if not _contained(resolved_source, round_path) or source.is_symlink():
                raise ValueError(f"archive source resolves outside round: {source}")
            rel = source.relative_to(round_path)
            destination = archive_dir / rel
            if not _contained(destination.parent.resolve(strict=False), archive_dir.resolve(strict=True)):
                raise ValueError(f"archive destination resolves outside trial directory: {destination}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise FileExistsError(destination)
            digest = _sha256(source)
            size = source.stat().st_size
            shutil.move(str(source), str(destination))
            moved.append((source, destination))
            records.append({
                "source": rel.as_posix(),
                "archived_as": destination.relative_to(archive_dir).as_posix(),
                "size_bytes": size,
                "sha256": digest,
            })
        manifest = {
            "archive_id": archive_dir.name,
            "round": round_path.name,
            "reason": reason.strip(),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "files": records,
        }
        manifest_path = archive_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except Exception:
        # Restore moved files if a pre-manifest operation fails. Nothing is
        # deleted; partially created empty archive directories may remain.
        for source, destination in reversed(moved):
            source.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() and not source.exists():
                shutil.move(str(destination), str(source))
        raise
    return archive_dir
