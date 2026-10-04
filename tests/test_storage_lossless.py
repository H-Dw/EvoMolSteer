from pathlib import Path

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from evomolsteer.storage.arrays import assert_no_filters, byte_equal, read_array, write_array
from evomolsteer.storage.features import FeaturePackage, iter_feature_tables, pack_features, restore_features, table_digest, verify_features
from evomolsteer.storage.trajectory import TrajectoryPackage, pack_trajectory
from evomolsteer.storage.views import build_shard_views


def test_numeric_bits_and_sparse_node_access(tmp_path):
    # Positive/negative zero, infinities and distinct NaN payloads may never merge.
    floats = np.array([0, 0x80000000, 0x7fc00001, 0x7fc00002, 0x7f800000, 0xff800000], dtype=np.uint32).view(np.float32)
    sparse = np.zeros((50, 100), dtype=np.float32)
    sparse[0, 9] = floats[1]
    sparse[20, 7:11] = floats[2:]
    examples = [floats, np.broadcast_to(floats, (20, 30, 6)), sparse,
                np.full((3, 7), floats[2]), np.array([0, 2, 255, 256], dtype=np.int64),
                np.array([-1, 0, 1], dtype=np.int64), np.zeros((4, 0), dtype=np.float32),
                np.arange(6, dtype=">f4").reshape(2,3)]
    with h5py.File(tmp_path / "arrays.h5", "w") as f:
        for i, a in enumerate(examples):
            write_array(f, str(i), a)
            assert byte_equal(read_array(f[str(i)]), a)
            if a.size:
                assert byte_equal(read_array(f[str(i)], (0,)), a[0])
        assert assert_no_filters(f) > 0


def make_trajectory():
    rng = np.random.default_rng(49)
    t, b, n = 4, 5, 3
    indices = rng.integers(0, b, (t,b), dtype=np.int64)
    proposal = rng.normal(size=(t,b,n,3)).astype(np.float32)
    current = np.concatenate([rng.normal(size=(1,b,n,3)).astype(np.float32), proposal[np.arange(t-1)[:,None], indices[:-1]]])
    bonds = np.zeros((t,b,n,n), dtype=np.uint8)
    bonds[...,0,1] = bonds[...,1,0] = 1
    return {"selected_indices": indices, "current_coords": current, "proposal_coords": proposal,
            "predicted_bonds": bonds, "score_time": np.zeros((t,3), dtype=np.float32),
            "mask": np.ones((t,b,n), dtype=np.uint8), "pic50_on": rng.normal(size=(t,b)).astype(np.float32)}


def test_lineage_alias_rejected_siblings_and_restore(tmp_path):
    data = make_trajectory()
    np.savez(tmp_path / "source.npz", **data)
    report = pack_trajectory(tmp_path / "source.npz", tmp_path / "nodes.h5")
    assert report["lineage_aliases"] == ["current_coords"]
    with TrajectoryPackage(tmp_path / "nodes.h5") as p:
        for t in range(4):
            for b in range(5):
                node = p.node(t, b)
                for k in data:
                    expected = data[k][t] if k == "score_time" else data[k][t,b]
                    assert byte_equal(node[k], expected)
                np.testing.assert_array_equal(node["children_next_slots"], np.flatnonzero(data["selected_indices"][t] == b))
        p.restore(tmp_path / "restored.npz")
        with pytest.raises(IndexError):
            p.read("current_coords", -1, 0)
    with np.load(tmp_path / "restored.npz") as r:
        assert list(r.files) == list(data)
        assert all(byte_equal(r[k], data[k]) for k in data)


def test_failed_alias_and_asymmetry_fall_back(tmp_path):
    data = make_trajectory()
    data["current_coords"][1,0,0,0] += 1
    data["predicted_bonds"][2,1,0,2] = 4
    np.savez(tmp_path / "source.npz", **data)
    report = pack_trajectory(tmp_path / "source.npz", tmp_path / "nodes.h5")
    assert report["lineage_aliases"] == []
    with TrajectoryPackage(tmp_path / "nodes.h5") as p:
        assert p.verify(data) == len(data)


def test_compressed_trajectory_exact_bits_and_analysis_adapter(tmp_path):
    from evomolsteer.trajectory_source import open_trajectory,trajectory_paths
    from evomolsteer.storage.arrays import assert_lossless_filters
    data=make_trajectory()
    bits=np.array([0,0x80000000,0x7fc00001,0x7fc00002,0x7f800000,0xff800000],dtype=np.uint32)
    data['adversarial_float_payload']=np.tile(bits,2000).view(np.float32).reshape(4,5,-1)
    src=tmp_path/'source.npz';np.savez_compressed(src,**data)
    path=tmp_path/'joint/batch_000/trajectory.h5';path.parent.mkdir(parents=True)
    r=pack_trajectory(src,path,codec='gzip_shuffle')
    assert r['filter_audit']['gzip_shuffle_datasets']>0
    with TrajectoryPackage(path) as p:
        assert p.verify(data)==len(data)
        assert assert_lossless_filters(p.file)['gzip_shuffle_datasets']>0
        assert byte_equal(p.read('adversarial_float_payload',1,2),data['adversarial_float_payload'][1,2])
        p.restore(tmp_path/'restored.npz')
    with open_trajectory(path) as z:
        assert all(byte_equal(z[k],data[k]) for k in data)
    with np.load(tmp_path/'restored.npz') as z:
        assert all(byte_equal(z[k],data[k]) for k in data)
    assert trajectory_paths(tmp_path)==[path]
    (path.parent/'trajectory.npz').write_bytes(src.read_bytes())
    with pytest.raises(ValueError):trajectory_paths(tmp_path)


def test_incomplete_or_corrupt_input_bundle_is_rejected(tmp_path):
    from evomolsteer.trajectory_source import validate_input_bundle
    from evomolsteer.io import write_json,digest
    manifest={'format':'evomolsteer.analysis_inputs.v1','complete':False,'campaign':'c',
              'packages':[],'copied_files':{}}
    record=tmp_path/'input_bundle_manifest.json';write_json(record,manifest)
    with pytest.raises(ValueError):validate_input_bundle(tmp_path,{'campaign':'c'})
    data=tmp_path/'geometry.json';data.write_text('{}')
    manifest.update(complete=True,copied_files={'geometry.json':{'sha256':digest(data)}})
    write_json(record,manifest);validate_input_bundle(tmp_path,{'campaign':'c'})
    data.write_text('{"changed":true}')
    with pytest.raises(ValueError):validate_input_bundle(tmp_path,{'campaign':'c'})


def test_feature_null_schema_order_node_and_export(tmp_path):
    tables = []
    nan = np.array([0x7ff8000000000001], dtype=np.uint64).view(np.float64)[0]
    for rep in ["current_state", "predicted_endpoint", "proposal_state"]:
        tables.append(pa.table({"campaign": ["c"]*4, "arm": ["joint"]*4, "batch": [0]*4,
                                "node_id": ["n0","n1","n2","n3"], "step": [0,0,1,1], "slot": [0,1,0,1],
                                "representation": [rep]*4, "optional_text": [None,"","中文","α"],
                                "value::geometry": pa.array([nan, -0., 0., None], type=pa.float64(), from_pandas=False)}).replace_schema_metadata({b"custom": b"preserve"}))
    source = tmp_path / "features.parquet"
    with pq.ParquetWriter(source, tables[0].schema, compression=None) as writer:
        for t in tables:
            writer.write_table(t)
    r = pack_features(source, tmp_path / "packed")
    path = tmp_path / "packed" / r["packages"][0]["package"]
    with FeaturePackage(path) as p:
        for rep, t in zip(p.representations, tables):
            assert table_digest(p.table(rep)) == table_digest(t)
            assert table_digest(p.node(rep, 1, 0)) == table_digest(t.slice(2,1))
    restored = restore_features(tmp_path / "packed", tmp_path / "restored.parquet")
    pf = pq.ParquetFile(restored)
    for i, t in enumerate(tables):
        assert table_digest(pf.read_row_group(i)) == table_digest(t)
    assert verify_features(tmp_path / "packed", source)["verified_rows"] == 12
    assert len(list(iter_feature_tables(tmp_path / "packed", representations=["proposal_state"]))) == 1
    shards = tmp_path / "shards" / "batch_000"
    shards.mkdir(parents=True)
    (shards / "features.parquet").write_bytes(source.read_bytes())
    view_report = build_shard_views(tmp_path / "packed", shards.parent, tmp_path / "views")
    assert view_report["source_rows_verified"] == 12
    assert verify_features(tmp_path / "views" / "batch_000", shards / "features.parquet")["verified_rows"] == 12
