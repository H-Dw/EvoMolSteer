"""Verify and restore the immutable generation archive format, without overwrites."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile

from evomolsteer.io import digest, read_json, safe_extract


MANIFEST = "GENERATION_ARCHIVE_MANIFEST.json"


def verify_generation_archive(archive, metadata, destination):
    """Check transport and every payload before restoring into a new directory.

    ``metadata`` is the JSON sidecar written by ``archive_generation.py``.
    Refuse existing destinations, unmanifested members, links and unsafe paths.
    The sibling .partial directory only becomes the destination after validation.
    """
    archive = Path(archive).resolve()
    destination = Path(destination).resolve()
    partial = destination.with_name(destination.name + ".partial")
    if destination.exists() or partial.exists():
        raise FileExistsError("Restore requires a new destination and .partial directory")
    meta = read_json(metadata) if not isinstance(metadata, dict) else metadata
    if archive.stat().st_size != meta["bytes"] or digest(archive) != meta["archive_sha256"]:
        raise ValueError("Generation archive transport checksum or size mismatch")
    with tarfile.open(archive, "r:gz") as tf:
        members = tf.getmembers()
        names = [m.name for m in members]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate archive member")
        for member in members:
            name = member.name
            path = PurePosixPath(name)
            if (not member.isfile() or path.is_absolute() or ".." in path.parts
                    or "\\" in name or ":" in name or str(path) != name
                    or not (partial / name).resolve().is_relative_to(partial)):
                raise ValueError("Unsafe archive member: " + name)
        manifest = json.load(tf.extractfile(MANIFEST))
        if manifest.get("format") != "evomolsteer.generation_archive.v1" or manifest.get("complete") is not True:
            raise ValueError("Unsupported or incomplete generation archive")
        campaign = manifest.get("campaign")
        if (not isinstance(campaign, str) or campaign in {"", ".", ".."}
                or any(c in campaign for c in "/\\:")):
            raise ValueError("One campaign directory name required")
        checks = manifest["files"]
        if set(names) != set(checks) | {MANIFEST} or MANIFEST in checks:
            raise ValueError("Archive members differ from manifest")
        if len(checks) != meta["files"] or sum(r["bytes"] for r in checks.values()) != meta["source_bytes"]:
            raise ValueError("Archive metadata count or size mismatch")
        for name, record in checks.items():
            if tf.getmember(name).size != record["bytes"]:
                raise ValueError("Payload size mismatch: " + name)
            h = hashlib.sha256()
            with tf.extractfile(name) as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    h.update(chunk)
            if h.hexdigest() != record["sha256"]:
                raise ValueError("Payload checksum mismatch: " + name)
    safe_extract(archive, partial, meta["archive_sha256"])
    for name, record in checks.items():
        path = partial / name
        if path.stat().st_size != record["bytes"] or digest(path) != record["sha256"]:
            raise ValueError("Restored payload checksum mismatch: " + name)
    completion = read_json(partial / "results" / manifest["campaign"] / "COMPLETE.json")
    if completion.get("status") != "complete":
        raise ValueError("Campaign completion record missing or incomplete")
    partial.rename(destination)
    return {"verified": True, "campaign": manifest["campaign"],
            "archive": str(archive), "archive_sha256": meta["archive_sha256"],
            "destination": str(destination), "files_verified": len(checks),
            "source_bytes": meta["source_bytes"], "code_commit": manifest.get("code_commit")}
