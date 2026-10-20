"""Audited input-cap increase; preserve sampler, prompts, caches and failure history."""
import argparse
import copy
import time
import zipfile
from contextlib import ExitStack
from pathlib import Path

from evomolsteer.adaptive_schedule.io import DEFAULT_ROOT, read, write, sha, commit, canonical


def apply_budget(root, new_limit=131072, revision=None):
    root = Path(root)
    revision = revision or commit()
    operation = root / "operations" / f"input_budget_{new_limit}"
    journal = operation / "journal.json"
    config_path = root / "config.json"
    if not journal.exists():
        config = read(config_path)
        previous = config["llm"]["input_hard_tokens"]
        if type(new_limit) is not int or not previous < new_limit <= 131072:
            raise ValueError("Only a bounded input-cap increase up to 128 KiB is supported")
        changed = copy.deepcopy(config)
        changed["llm"]["input_hard_tokens"] = new_limit
        updates = [{"path": "config.json", "before": config, "after": changed}]
        # Use the actual atomic writer, including platform-specific line endings.
        write(operation / "proposed_config.json", changed)
        new_hash = sha(operation / "proposed_config.json")
        for binding in sorted((root / "versions").glob("*/binding.json")):
            old = read(binding)
            if old["config_sha256"] != sha(config_path):
                raise ValueError("Cohort binding does not match the pre-change configuration")
            updated = dict(old, config_sha256=new_hash)
            updates.append({"path": str(binding.relative_to(root)), "before": old, "after": updated})
        operation.mkdir(parents=True, exist_ok=True)
        # A later retry may refresh these operational records. Retain their originals.
        paths = set((root / "versions").rglob("*.budget.json"))
        paths.update((root / "versions").rglob("*error*.json"))
        paths.update(root / name for name in ["status.json", "launch.json", "workers.json", "supervisor_v1_exit.json"])
        archive = operation / "pre_change_records.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for path in sorted(paths):
                if path.is_file():
                    z.write(path, str(path.relative_to(root)))
        write(journal, {"revision": revision, "created": time.time(), "old_limit": previous,
                        "new_limit": new_limit, "units": "UTF-8 byte upper bound, not provider token count",
                        "reason": "Deduplicated full histories exceed 64 KiB at rounds 9-10; no evidence is dropped",
                        "unchanged": ["messages", "reward", "model", "output_budget", "skill", "sampler", "seeds"],
                        "archive_sha256": sha(archive), "updates": updates})
    record = read(journal)
    if record["new_limit"] != new_limit:
        raise ValueError("Operation journal belongs to a different input cap")
    if sha(operation / "pre_change_records.zip") != record["archive_sha256"]:
        raise ValueError("Pre-change audit archive checksum mismatch")
    # Validate every file before applying or resuming a partially committed update.
    for update in record["updates"]:
        value = read(root / update["path"])
        if canonical(value) not in [canonical(update["before"]), canonical(update["after"])]:
            raise ValueError("Unexpected concurrent change: " + update["path"])
    for update in record["updates"]:
        write(root / update["path"], update["after"])
    write(operation / "COMPLETE.json", {"revision": record["revision"], "new_limit": new_limit,
                                       "journal_sha256": sha(journal)})
    return {"operation": str(operation), "old_limit": record["old_limit"], "new_limit": new_limit}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    import fcntl
    with ExitStack() as stack:
        for name in ["supervisor.lock", "worker.lock"]:
            handle = stack.enter_context((args.root / name).open("a"))
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(apply_budget(args.root))
