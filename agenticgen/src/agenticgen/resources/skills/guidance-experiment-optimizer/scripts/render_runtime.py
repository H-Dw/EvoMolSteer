#!/usr/bin/env python3
"""Render the standard skill into the existing three-module prompt interface.

No model, experiment, network, or deployment operations. Hashes use UTF-8 text
with LF newlines so a Windows checkout and a Linux checkout produce one bundle.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

PACKAGE = Path(__file__).resolve().parents[1]
SHARED_START = "<!-- runtime:shared:start -->"
SHARED_END = "<!-- runtime:shared:end -->"
INPUTS = (
    "SKILL.md", "references/history-learning.md", "references/control-diagnostics.md",
    "references/proposal.md", "references/synthesis.md",
)


def read_text(path):
    text = Path(path).read_text(encoding="utf-8")
    if "\ufffd" in text:
        raise ValueError(f"Unicode replacement character: {path}")
    return text


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def render(package=PACKAGE):
    """Validate all source inputs before returning any files to write."""
    package = Path(package)
    inputs = {name: read_text(package / name) for name in INPUTS}
    entry = inputs["SKILL.md"]
    match = re.match(r"\A---\n(.*?)\n---\n", entry, re.DOTALL)
    if not match:
        raise ValueError("SKILL.md requires YAML frontmatter")
    frontmatter = match.group(1)
    name_match = re.search(r"^name: ([a-z0-9]+(?:-[a-z0-9]+)*)$", frontmatter, re.MULTILINE)
    version_match = re.search(r'^  version: "(\d+\.\d+\.\d+)"$', frontmatter, re.MULTILINE)
    if not name_match or not version_match:
        raise ValueError("Missing skill name or quoted metadata version")
    name, version = name_match.group(1), version_match.group(1)
    if entry.count(SHARED_START) != 1 or entry.count(SHARED_END) != 1:
        raise ValueError("Exactly one shared runtime section is required")
    start = entry.index(SHARED_START) + len(SHARED_START)
    end = entry.index(SHARED_END)
    if start >= end or not entry[start:end].strip():
        raise ValueError("Shared runtime section is empty or reversed")
    header = f"<!-- Generated from {name} {version}; edit its standard skill sources. -->\n\n"
    outputs = {
        "analysis.md": header + "\n\n".join((
            entry[start:end].strip(), inputs["references/history-learning.md"].strip(),
            inputs["references/control-diagnostics.md"].strip())) + "\n",
        "proposal.md": header + inputs["references/proposal.md"].strip() + "\n",
        "synthesis.md": header + inputs["references/synthesis.md"].strip() + "\n",
    }
    manifest = {
        "skill": name, "version": version, "hash_encoding": "UTF-8 with LF newlines",
        "source_sha256": {path: digest(value) for path, value in inputs.items()},
        "output_sha256": {path: digest(value) for path, value in outputs.items()},
        "role_inputs": {"private": ["analysis.md", "proposal.md"],
                        "synthesis": ["analysis.md", "synthesis.md"]},
        "scope": "instruction-source update; no scientific execution or deployment",
    }
    outputs["BUNDLE.json"] = json.dumps(manifest, ensure_ascii=True, indent=2) + "\n"
    return outputs


def sync(output_dir, *, check=False, package=PACKAGE):
    outputs = render(package)
    output_dir = Path(output_dir)
    changed = [name for name, value in outputs.items()
               if not (output_dir / name).is_file() or read_text(output_dir / name) != value]
    if not check:
        output_dir.mkdir(parents=True, exist_ok=True)
        for name in changed:
            (output_dir / name).write_text(outputs[name], encoding="utf-8", newline="\n")
    return changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Explicit destination; never target a frozen experiment")
    parser.add_argument("--check", action="store_true", help="Check drift without writing")
    args = parser.parse_args()
    try:
        changed = sync(args.output_dir, check=args.check)
    except (OSError, ValueError) as error:
        parser.exit(2, f"Cannot render skill: {error}\n")
    print(json.dumps({"mode": "check" if args.check else "render", "changed": changed},
                     ensure_ascii=True))
    return 1 if args.check and changed else 0


if __name__ == "__main__":
    raise SystemExit(main())
