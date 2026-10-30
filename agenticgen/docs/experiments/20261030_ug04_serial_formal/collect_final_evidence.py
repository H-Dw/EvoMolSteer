"""Read-only CPU audit and evidence export; never sample or call a model API."""
import argparse
import collections
import datetime
import gzip
import hashlib
import json
from pathlib import Path
import re
import tarfile

import h5py
import numpy as np
import torch

RMSD = "condition_coordinate_root_mean_square_deviation_angstrom"
RANKING = "ranking_" + RMSD
SEEDS = [42, 100045, 200048, 300051, 400054]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def sdf_rows(path):
    result = []
    for block in path.read_text().split("$$$$"):
        if not block.strip():
            continue
        props = dict(re.findall(r">\s*<([^>]+)>[^\n]*\n([^\n]*)", block))
        result.append(props)
    return result


def audit_group(root, folder, mappings):
    aggregate = read(folder / "aggregate.json")
    rows = read(folder / "records.json")
    receipt = read(folder / "COMPLETE.json")
    assert len(rows) == 50
    hashes = {name: sha(folder / name) == expected for name, expected in receipt["files"].items()}
    assert all(hashes.values()), str(folder)
    x = np.asarray([r[RMSD] for r in rows], dtype=np.float64)
    assert np.isfinite(x).all()
    q1, q3 = np.quantile(x, [.25, .75])
    threshold = q3 + max(3 * (q3 - q1), 1)
    keep = x <= threshold
    stats = aggregate["coordinate_loss_statistics"]
    assert stats["excluded_high_indices"] == np.flatnonzero(~keep).tolist()
    assert abs(aggregate[RMSD] - x.mean()) < 1e-12
    assert abs(aggregate[RANKING] - x[keep].mean()) < 1e-12
    assert abs(stats["raw_mean_coordinate_loss_angstrom_squared"] - np.mean(x*x/2)) < 1e-12
    condition = read(root / "conditions" / (aggregate["teacher_id"] + ".json"))
    frame = condition["protein_alignment"]
    with h5py.File(folder / "trace.h5", "r") as trace:
        assignment = trace["assignment"][10:100]
        frozen = bool(np.all(assignment == assignment[:1]))
        assert frozen
        slots = assignment[-1].astype(np.int64)
        endpoint = trace["endpoint_A"][99].astype(np.float64)
        reference = trace["reference_condition_coordinates_angstrom"][99]
        aligned = endpoint @ np.asarray(frame["rotation"]) + np.asarray(frame["translation_angstrom"])
        error = aligned[np.arange(50)[:, None], slots] - reference
        recomputed = np.sqrt(np.sum(error*error, axis=(1, 2))/9)
        rmsd_difference = float(np.max(np.abs(recomputed - x)))
        assert rmsd_difference < 1e-5
        for row, selected in zip(rows, slots):
            assert row["condition_generated_slots"] == selected.tolist()
            key = (aggregate["teacher_id"], row["logical_seed"], row["logical_slot"])
            previous = mappings.setdefault(key, selected.tolist())
            assert previous == selected.tolist()
        delta = trace["guidance_delta_A"][:].astype(np.float64)
        increment = np.linalg.norm(delta, axis=-1).max(axis=-1)
        above_three = np.argwhere(increment > 3)
        maximum_index = np.unravel_index(increment.argmax(), increment.shape)
        chemistry = read(folder / "CHEMICAL_CONDITION.json")["features"]
        injected = {}
        for key, expected in chemistry.items():
            values = trace["inpainted_chemical_" + key][10:100]
            expected = np.asarray(expected).argmax(-1)
            if key == "bonds":
                selected = values[:, np.arange(50)[None, :, None, None],
                                  slots[None, :, :, None], slots[None, :, None, :]]
                selected = selected[:, 0]
            else:
                selected = np.take_along_axis(values, slots[None], axis=2)
            injected[key] = bool(np.all(selected == expected))
        assert all(injected.values())
    terminal_path = folder / "terminal.pt.gz"
    assert sha(terminal_path) == read(terminal_path.with_suffix(".sha.json"))["sha256"]
    with gzip.open(terminal_path, "rb") as stream:
        terminal = torch.load(stream, map_location="cpu", weights_only=False)
    chemistry_matches = {}
    for key, expected in chemistry.items():
        expected = np.asarray(expected).argmax(-1)
        values = terminal[key].argmax(-1).numpy()
        if key == "bonds":
            selected = values[np.arange(50)[:, None, None], slots[:, :, None], slots[:, None, :]]
            matched = np.all(selected == expected, axis=(1, 2))
        else:
            selected = np.take_along_axis(values, slots, axis=1)
            matched = np.all(selected == expected, axis=1)
        chemistry_matches[key] = matched
    all_match = np.logical_and.reduce(list(chemistry_matches.values()))
    output = read(folder / "OUTPUT_POLICY.json")
    assert output["integration_intervals"] == 100 and output["grid_end"] == .999
    assert not output["coordinate_projection"] and not output["extra_terminal_forward"]
    return dict(
        id=str(folder.relative_to(root)), teacher_id=aggregate["teacher_id"],
        complete_files_verified=len(hashes), assignment_frozen_10_to_99=frozen,
        mappings_equal_across_configurations=True, rmsd_recalculation_maximum_difference=rmsd_difference,
        ranking_statistics_recomputed=True, injected_chemistry_exact_all_steps=injected,
        raw_terminal_chemistry_all_channels_match_count=int(all_match.sum()),
        raw_terminal_chemistry_channel_match_counts={k:int(v.sum()) for k,v in chemistry_matches.items()},
        raw_terminal_chemistry_denominator=50,
        maximum_single_atom_guidance_increment_angstrom=float(increment.max()),
        maximum_increment_step=int(maximum_index[0]), maximum_increment_tensor_row=int(maximum_index[1]),
        steps_with_increments_above_three_angstrom=sorted(set(above_three[:, 0].tolist())),
        candidate_step_events_above_three_angstrom=len(above_three),
        unique_candidates_above_three_angstrom=len(set(above_three[:, 1].tolist())),
        no_three_angstrom_cap_applied=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    root, destination = args.root.resolve(), args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    assert (root / "CAMPAIGN_FINISHED.json").exists()
    assert not (root / "CAMPAIGN_FAILED.json").exists()
    awaiting = read(root / "versions/GD01/AWAITING_GPT_REVIEW.json")
    groups = sorted(root.glob("versions/GD01/*/iteration_*/*/COMPLETE.json"))
    assert len(groups) == 30 and len(awaiting["strategy_contexts"]) == 4
    mappings = {}
    audits = [audit_group(root, p.parent, mappings) for p in groups]
    provider = []
    for p in sorted(root.glob("versions/GD01/**/*.receipt.json")):
        receipt = read(p)
        output = p.with_name(p.name.removesuffix(".receipt.json") + ".json")
        session = p.with_name(p.name.removesuffix(".receipt.json") + ".session")
        assert sha(session / "BINDING.json") == receipt["binding_sha256"]
        assert sha(output) == receipt["response_sha256"]
        assert all(sha(session / name) == h for name,h in receipt["transcript_files"].items())
        for request in session.glob("step_*/REQUEST.json"):
            wire = read(request)
            assert wire["model"] == "deepseek-flash" and wire["max_tokens"] == 16384
            assert wire["thinking"] == {"type":"enabled"} and wire["reasoning_effort"] == "high"
        provider.append(dict(path=str(p.relative_to(root)),steps=receipt["steps"],
                             interface_repairs=receipt["interface_repairs"],all_transcript_hashes_valid=True))
    budgets = []
    for folder in sorted(root.glob("versions/GD01/**/model_budget")):
        calls = [read(p) for p in folder.glob("call_*.json")]
        charged = sum(r.get("completion_tokens",r["reserved_output_tokens"]) for r in calls)
        assert len(calls) <= 30 and charged <= 491520
        budgets.append(dict(path=str(folder.relative_to(root)),calls=len(calls),charged_output_tokens=charged))
    retentions = []
    for folder in sorted(root.glob("versions/GD01/*/retention")):
        rows = read(folder / "per_slot_top1.json")
        receipt = read(folder / "RETENTION_COMPLETE.json")
        assert len(rows) == 50 and len({(r['logical_seed'],r['logical_slot']) for r in rows}) == 50
        valid = True
        built = {}
        for row in rows:
            source = root / row["source_group"]
            original = next(r for r in read(source / "records.json") if r["node_id"] == row["node_id"])
            assert all(row[k] == v for k,v in original.items())
            built[(row["source_group"],row["node_id"])] = original["build_success"]
        molecules = sdf_rows(folder / "per_slot_top1_built.sdf")
        actual = {(m["source_group"],m["node_id"]) for m in molecules}
        assert actual == {k for k,v in built.items() if v}
        rep = root / receipt["representative"]
        assert sha(rep / "terminal.pt.gz") == receipt["representative_terminal_sha256"]
        assert sha(rep / "trace.h5") == receipt["representative_trace_sha256"]
        retentions.append(dict(teacher=folder.parent.name,entries=50,built_sdf=len(molecules),
                               source_identity_valid=valid,representative=receipt["representative"],
                               representative_terminal_and_trace_valid=True))
    report = dict(timestamp=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
                  audit_scope="read_only_cpu_postprocessing_no_sampling_no_api_calls",groups=audits,
                  mappings=len(mappings),provider=provider,budgets=budgets,retentions=retentions)
    write(destination / "FINAL_INTEGRITY_AUDIT.json", report)
    selected = set()
    for base in ["calibration", "versions", "slot_registry", "measurement_registry", "provenance", "conditions"]:
        for p in (root / base).rglob("*.json"):
            if p.name != "steps.json" and "profiles" not in p.parts:
                selected.add(p)
    for p in root.glob("*.json"):
        selected.add(p)
    for p in root.glob("sample_shards/*/*.json"):
        if p.name not in {"steps.json"}:
            selected.add(p)
    for p in root.glob("versions/GD01/*/retention/*.sdf"):
        selected.add(p)
    for p in root.glob("quality_queue/*.json"):
        selected.add(p)
    selected.update((root / "quality_queue/requests").glob("*.json"))
    selected.add(root / "acceptance/COMPLETE.json")
    selected.add(root / "monitoring/heartbeat_20261030_2353.json")
    source_index = {str(p.relative_to(root)):{"sha256":sha(p),"bytes":p.stat().st_size} for p in sorted(selected)}
    write(destination / "SOURCE_INDEX.json", source_index)
    archive = destination / "FINAL_EVIDENCE.tar.gz"
    with tarfile.open(archive,"w:gz",compresslevel=3) as tar:
        for p in sorted(selected):
            tar.add(p,arcname="results/"+str(p.relative_to(root)),recursive=False)
        tar.add(destination / "FINAL_INTEGRITY_AUDIT.json",arcname="audit/FINAL_INTEGRITY_AUDIT.json")
        tar.add(destination / "SOURCE_INDEX.json",arcname="audit/SOURCE_INDEX.json")
    print(json.dumps(dict(archive=str(archive),sha256=sha(archive),bytes=archive.stat().st_size,
                          source_files=len(source_index),formal_groups=len(audits),role_receipts=len(provider),
                          actual_http_calls=sum(b["calls"] for b in budgets),retentions=retentions)))


if __name__ == "__main__":
    main()
