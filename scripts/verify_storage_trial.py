"""Verify the entire source, actual exports, and all trial trajectory arrays."""
import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from evomolsteer.io import digest, write_json
from evomolsteer.storage.arrays import byte_equal
from evomolsteer.storage.features import restore_features, table_digest, verify_features
from evomolsteer.storage.trajectory import TrajectoryPackage

p = argparse.ArgumentParser()
p.add_argument("--feature-source", required=True)
p.add_argument("--feature-packages", required=True)
p.add_argument("--trajectory-trials", required=True)
p.add_argument("--output", required=True)
a = p.parse_args()
output = Path(a.output).resolve()
output.mkdir(parents=True, exist_ok=True)
verification = verify_features(a.feature_packages, a.feature_source)
print("Verified all source feature row groups", flush=True)
restored = output / "temporary_restored_features.parquet"
assert restored.resolve().is_relative_to(output)
restore_features(a.feature_packages, restored)
original, returned = pq.ParquetFile(a.feature_source), pq.ParquetFile(restored)
assert original.num_row_groups == returned.num_row_groups
assert original.schema_arrow.equals(returned.schema_arrow, check_metadata=True)
for i in range(original.num_row_groups):
    assert table_digest(original.read_row_group(i)) == table_digest(returned.read_row_group(i))
verification["actual_parquet_roundtrip"] = {"row_groups": returned.num_row_groups,
                                          "rows": returned.metadata.num_rows,
                                          "uncompressed_export_bytes": restored.stat().st_size,
                                          "sha256": digest(restored),
                                          "retained": False, "all_schema_and_content_hashes_match": True}
returned.close()
original.close()
restored.unlink()  # Only the exact, newly created and successfully verified export.
print("Verified actual full Parquet export and removed temporary copy", flush=True)
raw = []
for item in json.loads((Path(a.trajectory_trials)/"manifest.json").read_text()):
    if not item["lineage_aliases"]:
        continue
    source, package_path = Path(item["source"]), Path(item["package"])
    assert digest(source) == item["source_sha256"]
    assert digest(package_path) == item["package_sha256"]
    export = output / (package_path.stem + "_temporary_restored.npz")
    assert export.resolve().is_relative_to(output)
    with TrajectoryPackage(package_path) as package:
        verified = package.verify()
        package.restore(export)
    with np.load(source, allow_pickle=False) as x, np.load(export, allow_pickle=False) as y:
        assert x.files == y.files
        assert all(byte_equal(x[k], y[k]) for k in x.files)
    raw.append({"source": str(source), "arrays": verified, "all_dtype_shape_and_bits_match": True,
                "uncompressed_export_bytes": export.stat().st_size, "export_retained": False})
    export.unlink()
verification["actual_npz_roundtrips"] = raw
write_json(output/"verification.json", verification)
print(json.dumps(verification, indent=2))
