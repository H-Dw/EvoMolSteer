"""Publish verified per-chain incumbents before removing obsolete structural payloads.

JSON/XML/logs and immutable R0 are never garbage collected. No molecule is selected
across slots: each of the twenty native slots retains its own historical TOP1.
"""
import os
import re
from pathlib import Path

from .io import read, write, sha, value_sha, better, sync_directory


def enabled(config):
    return config.get("retention", {}).get("mode") == "top1_per_chain"


def batch_folder(root, version, teacher, batch):
    return Path(root) / "versions" / version / teacher / f"batch_{batch:03d}"


def checked_path(root, relative, prefix):
    root = Path(root).resolve()
    p = root / relative
    allowed = root / prefix
    if not p.resolve().is_relative_to(allowed.resolve()) or not allowed.resolve().is_relative_to(root):
        raise ValueError("Retention path escapes its authorized batch")
    for part in [p, *p.parents]:
        if part == root:
            break
        if part.is_symlink():
            raise ValueError("Retention never follows symlinks")
    return p


def winners(chains):
    if len(chains) != 20 or [c["slot"] for c in chains] != list(range(20)):
        raise ValueError("TOP1 retention requires the complete ordered twenty-slot batch")
    if len({len(c["history"]) for c in chains}) != 1:
        raise ValueError("Only atomically committed rounds can be compacted")
    result = []
    for c in chains:
        best = c["history"][0]
        for h in c["history"][1:]:
            if better(h["record"], best["record"]):
                best = h
        if c["best_id"] != best["id"]:
            raise ValueError("Stored incumbent disagrees with eligible pIC50 ranking/tie policy")
        result.append(best)
    return result


def verify_snapshot(root, relative):
    p = checked_path(root, relative, "retained")
    complete = read(p / "COMPLETE.json")
    if not complete.get("payload_sha256"):
        raise ValueError("Snapshot has no verified payload inventory")
    for name, digest in complete["payload_sha256"].items():
        f = checked_path(p, name, ".")
        if not f.is_file() or sha(f) != digest:
            raise ValueError("Retained snapshot checksum mismatch: " + str(f))
    return complete


def restart_refs(root, version, teacher, batch, chains):
    best = winners(chains)
    p = batch_folder(root, version, teacher, batch) / "retention.json"
    if not p.exists():
        return [h["branch"] for h in best]
    index = read(p)
    if index["best_ids"] != [h["id"] for h in best]:
        raise ValueError("Finish the committed round's retention transaction before resuming")
    verify_snapshot(root, index["snapshot"])
    return index["restart_refs"]


def artifact_folder(root, version, teacher, batch, branch, slot):
    """Locate a retained winner for presentation; documentary history keeps original IDs."""
    p = batch_folder(root, version, teacher, batch) / "retention.json"
    if p.exists():
        index = read(p)
        if index["original_branches"][slot] == branch:
            return Path(root) / index["snapshot"]
    return Path(root) / branch


def prune_payload(root, relative, prefix):
    folder = checked_path(root, relative, prefix)
    ledger_path = folder / "PRUNED.json"
    ledger = read(ledger_path) if ledger_path.exists() else {"files": {}, "policy": "top1_per_chain"}
    for p in sorted(folder.rglob("*")):
        # Small analyses and audit hashes survive even when the matching binary is removed.
        if not p.is_file() or not p.name.endswith((".pt.gz", ".h5", ".sdf", ".pdb", ".pdb.gz", ".mol2")):
            continue
        checked_path(root, p.relative_to(Path(root)), prefix)
        key = str(p.relative_to(folder))
        if key not in ledger["files"]:
            ledger["files"][key] = {"bytes": p.stat().st_size, "sha256": sha(p)}
            write(ledger_path, ledger)  # An interruption cannot erase deletion provenance.
        elif sha(p) != ledger["files"][key]["sha256"]:
            raise ValueError("Payload changed after deletion was journaled")
        p.unlink()
    ledger["complete"] = True
    ledger["removed_bytes"] = sum(x["bytes"] for x in ledger["files"].values())
    write(ledger_path, ledger)
    return ledger["removed_bytes"]


def compact_batch(root, config, template, batch, version, chains, builder=None, profiler=None):
    if not enabled(config):
        return None
    root = Path(root).resolve()
    teacher = template["teacher_id"]
    best = winners(chains)
    folder = batch_folder(root, version, teacher, batch)
    round_no = len(chains[0]["history"]) - 1
    # Profile every scientific attempt before its SDF/complex can disappear.
    if profiler is None:
        from .quality import profile_chains
        profiler = profile_chains
    profiler(root, template, chains)
    for c in chains:
        for h in c["history"]:
            q = root / h["branch"] / "profiles" / f"slot_{c['slot']:03d}"
            if not ((q / "complete.json").exists() or (q / "failure.json").exists()):
                raise ValueError("Quality evaluation has not reached a recorded terminal state")
    if round_no == 0:
        return None  # R0 is the shared immutable reference and already restartable.
    index_path = folder / "retention.json"
    previous = read(index_path) if index_path.exists() else None
    identity = [{"id": h["id"], "branch": h["branch"], "slot": i} for i, h in enumerate(best)]
    identity_sha = value_sha(identity)
    relative = f"retained/{version}/{teacher}/batch_{batch:03d}/best_{identity_sha[:16]}"
    if previous and previous["identity_sha256"] == identity_sha:
        index = previous
        verify_snapshot(root, relative)
    else:
        sources = []
        for i, h in enumerate(best):
            old = previous and previous["best_ids"][i] == h["id"]
            sources.append(previous["restart_refs"][i] if old else h["branch"])
        # Sources for molecular/head/trace payloads use the archive root, not a fallback context.
        payload_sources = [previous["snapshot"] if previous and previous["best_ids"][i] == h["id"]
                           else h["branch"] for i, h in enumerate(best)]
        if builder is None:
            from .snapshot import materialize
            builder = materialize
        marker = root / relative / "COMPLETE.json"
        if not marker.exists():
            builder(root, relative, sources, payload_sources, best, template, batch)
        complete = verify_snapshot(root, relative)
        if complete["identity_sha256"] != identity_sha:
            raise ValueError("Snapshot identity changed")
        # Make the new structural data durable before publishing the deletion gate.
        snapshot = root / relative
        for name in complete["payload_sha256"]:
            with (snapshot / name).open("r+b") as f:
                os.fsync(f.fileno())
        for directory in sorted((p for p in snapshot.rglob("*") if p.is_dir()),
                                key=lambda p: len(p.parts), reverse=True):
            sync_directory(directory)
        sync_directory(snapshot)
        sync_directory(snapshot.parent)
        index = {"schema": 1, "identity_sha256": identity_sha, "snapshot": relative,
                 "best_ids": [h["id"] for h in best], "original_branches": [h["branch"] for h in best],
                 "restart_refs": complete["restart_refs"], "selection": config["selection"]}
        write(index_path, index)  # Publication precedes every deletion.
    prefix = f"branches/{teacher}/batch_{batch:03d}"
    removed = 0
    for branch in sorted((root / prefix).glob(version + "_r*")):
        m = re.fullmatch(re.escape(version) + r"_r(\d+)(?:_s\d+)?", branch.name)
        if m and int(m[1]) <= round_no and (branch / "COMPLETE.json").exists():
            removed += prune_payload(root, branch.relative_to(root), prefix)
    archive_prefix = f"retained/{version}/{teacher}/batch_{batch:03d}"
    for old in sorted((root / archive_prefix).glob("best_*")):
        if old != root / index["snapshot"]:
            removed += prune_payload(root, old.relative_to(root), archive_prefix)
    write(folder / f"round_{round_no:02d}" / "RETENTION_COMPLETE.json",
          {"snapshot": index["snapshot"], "history_sha256": value_sha(chains),
           "non_r0_structures_retained": 20, "shared_r0_preserved": True,
           "journaled_removed_bytes": removed})
    return index
