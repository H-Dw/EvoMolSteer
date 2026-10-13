from scripts.export_ck2_interval_trace import build_trace_tables


def fixture_tables():
    top30, teacher_map, ancestry, pools = [], {}, [], {}
    for i in range(30):
        batch, final_slot = i % 20, i % 50
        node_id, donor_id = f"node-{i:02d}", f"T{i + 1:02d}"
        top30.append({"rank": i + 1, "node_id": node_id, "batch": batch, "slot": final_slot})
        teacher_map[donor_id] = {"node_id": node_id, "batch": batch, "slot": final_slot}
        for step in range(101):
            ancestor = (final_slot + step) % 50 if step < 100 else (final_slot + 100) % 50
            ancestry.append({
                "node_id": node_id, "batch": batch, "slot": final_slot, "step": step,
                "ancestor_slot": ancestor, "root_slot": final_slot,
                "selected_parent_slot": ancestor if step == 0 else (ancestor - 1) % 50,
                "time": step / 100,
                "locator": {
                    "path": f"raw/batch_{batch:03d}/trajectory.npz" if step < 100 else f"raw/batch_{batch:03d}/final_prediction.pt.gz",
                    "sha256": f"sha-{batch:03d}-{step:03d}", "step": step,
                    "slot": ancestor if step < 100 else final_slot,
                    "parent_slot": ancestor if step == 100 else None,
                    "channel": "final_readout" if step == 100 else None,
                },
            })
    for step in range(5, 51, 5):
        pools[step] = {"targets": [
            {"node_id": f"node-{i:02d}", "batch": i % 20, "slot": i % 50,
             "ancestor_slot": (i % 50 + step) % 50, "key": f"g-{step}-{i}"}
            for i in range(30)
        ]}
    return top30, teacher_map, ancestry, pools


def test_export_builds_exact_300_intervals_and_1530_step_rows():
    tables = fixture_tables()
    interval_rows, full_rows = build_trace_tables(*tables)
    assert len(interval_rows) == 300
    assert len(full_rows) == 1530
    row = next(r for r in interval_rows if r["donor_id"] == "T01" and r["interval"] == 2)
    assert (row["start_step"], row["mid_step"], row["end_step"]) == (5, 7, 10)
    assert row["start_ancestor_slot"] == 5
    assert row["mid_ancestor_slot"] == 7
    assert row["end_ancestor_slot"] == 10
    assert row["end_oracle_group_key"] == "g-10-0"
    assert row["end_source_path"] == "raw/batch_000/trajectory.npz"
    exact = next(r for r in full_rows if r["donor_id"] == "T01" and r["step"] == 7)
    assert exact["state_locator_key"] == "b000:t007:s007"
    assert exact["source_sha256"] == "sha-000-007"


def test_export_rejects_mismatched_oracle_ancestry():
    tables = list(fixture_tables())
    tables[3][5]["targets"][0]["ancestor_slot"] += 1
    import pytest
    with pytest.raises(ValueError, match="oracle target ancestry does not match"):
        build_trace_tables(*tables)
