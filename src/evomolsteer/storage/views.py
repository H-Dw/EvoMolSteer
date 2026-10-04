"""Reference-only views of the verified shared packages; no duplicated features."""
import json
import os
from pathlib import Path

import pyarrow.parquet as pq

from ..io import digest, write_json
from .features import FORMAT, FeaturePackage, table_digest


def build_shard_views(packages, shard_root, destination):
    """Verify each existing Parquet shard before replacing its data by references.

    Originals are untouched. A view is a normal restore_features/verify_features
    manifest whose package paths point at shared HDF5 files.
    """
    packages, shard_root, destination = map(Path, (packages, shard_root, destination))
    if destination.exists():
        raise FileExistsError(destination)
    main = json.loads((packages / "manifest.json").read_text())
    lookup = {}
    for item in main["packages"]:
        with FeaturePackage(packages / item["package"]) as package:
            for rep, expected in zip(package.representations, item["verified_table_sha256"]):
                lookup[(item["campaign"], item["arm"], item["batch"], rep)] = (item, expected)
    summary = []
    for source in sorted(shard_root.glob("batch_*/features.parquet")):
        reader = pq.ParquetFile(source)
        view_dir = destination / source.parent.name
        view_dir.mkdir(parents=True)
        references = {}
        checked = 0
        for i in range(reader.num_row_groups):
            table = reader.read_row_group(i)
            key = tuple(table[c][0].as_py() for c in ["campaign", "arm", "batch", "representation"])
            item, expected = lookup[key]
            if table_digest(table) != expected:
                raise AssertionError("Shard is not an exact semantic duplicate: " + str(source))
            ref = references.setdefault(item["package"], {**item, "source_row_groups": [], "verified_table_sha256": [], "_reps": []})
            ref["source_row_groups"].append(i)
            ref["verified_table_sha256"].append(expected)
            ref["_reps"].append(key[-1])
            checked += table.num_rows
        for name, ref in references.items():
            with FeaturePackage(packages / name) as package:
                if ref.pop("_reps") != package.representations:
                    raise ValueError("Shard view must contain complete ordered representation groups")
            ref["package"] = Path(os.path.relpath((packages/name).resolve(), view_dir.resolve())).as_posix()
        manifest = {"format": FORMAT, "source": str(source.resolve()), "source_sha256": digest(source),
                    "source_bytes": source.stat().st_size, "source_rows": reader.metadata.num_rows,
                    "rows": checked, "complete": checked == reader.metadata.num_rows,
                    "feature_columns": main["feature_columns"], "packages": list(references.values()),
                    "storage": "reference-only view; HDF5 package bytes are shared with the main dataset"}
        write_json(view_dir / "manifest.json", manifest)
        summary.append({"view": source.parent.name, "source_bytes": source.stat().st_size,
                        "manifest_bytes": (view_dir/"manifest.json").stat().st_size,
                        "source_rows": checked, "source_sha256": manifest["source_sha256"]})
        print(f"Verified shard view {source.parent.name}: {checked} rows", flush=True)
    if not summary:
        raise ValueError("No source shards")
    result = {"views": summary, "source_shard_bytes": sum(r["source_bytes"] for r in summary),
              "view_manifest_bytes": sum(r["manifest_bytes"] for r in summary),
              "source_main_bytes": main["source_bytes"], "shared_package_bytes": main["package_bytes"],
              "source_rows_verified": sum(r["source_rows"] for r in summary)}
    write_json(destination / "views_manifest.json", result)
    return result
