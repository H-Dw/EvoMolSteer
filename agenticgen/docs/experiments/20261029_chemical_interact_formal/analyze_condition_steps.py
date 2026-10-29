"""Read saved endpoints and traces only; never load a model or sample candidates."""

import argparse
import collections
import datetime
import gzip
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import h5py
import numpy as np
import torch


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rms(values):
    return np.sqrt(np.mean(np.square(values), axis=-1))


def coordinate_metrics(coordinates, reference):
    reference_centered = reference - reference.mean(axis=0)
    result = []
    for xyz in coordinates:
        centered = xyz - xyz.mean(axis=0)
        left, _, right = np.linalg.svd(centered.T @ reference_centered)
        correction = np.diag([1.0, 1.0, np.sign(np.linalg.det(left @ right))])
        residual = centered @ left @ correction @ right - reference_centered
        result.append(
            {
                "same_frame_coordinate_rmsd_A": float(
                    np.sqrt(np.mean(np.sum((xyz - reference) ** 2, axis=-1)))
                ),
                "proper_rotation_aligned_coordinate_rmsd_A": float(
                    np.sqrt(np.mean(np.sum(residual**2, axis=-1)))
                ),
                "centroid_displacement_A": float(
                    np.linalg.norm(xyz.mean(axis=0) - reference.mean(axis=0))
                ),
            }
        )
    return result


def pair_metrics(coordinates, condition):
    pairs = condition["pairs"]
    slots = np.asarray([pair["generated_slots"] for pair in pairs])
    reference = np.asarray([pair["reference_distance_angstrom"] for pair in pairs])
    errors = (
        np.linalg.norm(
            coordinates[..., slots[:, 0], :] - coordinates[..., slots[:, 1], :], axis=-1
        )
        - reference
    )
    controlled = [
        i for i, p in enumerate(pairs) if p["category"] == "controlled_internal"
    ]
    outer = [
        i for i, p in enumerate(pairs) if p["category"] == "controlled_neighborhood"
    ]
    interactions = condition["interaction_pairs"]
    interaction_errors = np.linalg.norm(
        coordinates[..., [p["generated_slot"] for p in interactions], :]
        - np.asarray([p["target_coordinates_angstrom"] for p in interactions]),
        axis=-1,
    ) - np.asarray([p["reference_distance_angstrom"] for p in interactions])
    return {
        "controlled_pair_distance_rmsd_A": rms(errors[..., controlled]),
        "outer_pair_distance_rmsd_A": rms(errors[..., outer]),
        "all_intramolecular_pair_distance_rmsd_A": rms(errors),
        "interaction_pair_distance_rmsd_A": rms(interaction_errors),
    }


def summarize(values):
    return {
        name: float(function(values))
        for name, function in (
            ("mean", np.mean),
            ("minimum", np.min),
            ("maximum", np.max),
            ("median", np.median),
        )
    }


def analyze(root, destination):
    frozen = next((root / "calibration/frozen").glob("*/coarse_results.json"))
    coarse = read(frozen)
    condition = read(root / "conditions/6pyy_H7S_A_402.json")
    atoms = {atom["id"]: atom for atom in condition["atoms"]}
    indices = {atom["id"]: i for i, atom in enumerate(condition["atoms"])}
    controlled_ids = condition["controlled_atom_ids"]
    controlled_slots = [condition["atom_to_generated_slot"][i] for i in controlled_ids]
    slot_to_atom = {
        slot: atom for atom, slot in condition["atom_to_generated_slot"].items()
    }
    reference = np.asarray(condition["reference_coordinates_angstrom"])[
        [indices[i] for i in controlled_ids]
    ]
    sources = {
        "conditions/6pyy_H7S_A_402.json": sha(root / "conditions/6pyy_H7S_A_402.json")
    }
    configurations = []
    first_guided_step_reference = None
    for measurement in coarse:
        source = measurement["sampling_provenance"][0]["source_group"]
        folder = root / source
        for name in ("terminal.pt.gz", "trace.h5", "steps.json", "records.json"):
            sources[f"{source}/{name}"] = sha(folder / name)
        with gzip.open(folder / "terminal.pt.gz", "rb") as stream:
            terminal = torch.load(stream, map_location="cpu", weights_only=False)
        coords = terminal["coords"].double().numpy()
        records = read(folder / "records.json")
        increment_limit = read(folder / "OUTPUT_POLICY.json")[
            "maximum_single_atom_guidance_increment_angstrom"
        ]
        geometry = coordinate_metrics(coords[:, controlled_slots], reference)
        distances = pair_metrics(coords, condition)
        assert np.allclose(
            distances["controlled_pair_distance_rmsd_A"],
            [
                r["controlled_atom_pair_distance_root_mean_square_deviation_angstrom"]
                for r in records
            ],
            atol=2e-6,
        )
        steps = read(folder / "steps.json")
        per_step = []
        events = []
        with h5py.File(folder / "trace.h5", "r") as trace:
            first_step_values = {
                key: trace[key][10]
                for key in ("current_A", "endpoint_A", "input_gradient")
            }
            if first_guided_step_reference is None:
                first_guided_step_reference = first_step_values
            first_step_comparison = {
                key: bool(np.array_equal(value, first_guided_step_reference[key]))
                for key, value in first_step_values.items()
            }
            endpoints = trace["endpoint_A"][:].astype(np.float64)
            step_distances = pair_metrics(endpoints, condition)
            raw = trace["unconstrained_guidance_delta_A"][:]
            applied = trace["guidance_delta_A"][:].astype(np.float64)
            raw_norm = np.linalg.norm(raw, axis=-1)
            applied_norm = np.linalg.norm(applied, axis=-1)
            terminal_trace_difference = float(np.max(np.abs(coords - endpoints[-1])))
            for row in steps:
                step = row["step"]
                factors = np.asarray(row["guidance_increment_scale_factor"])
                flags = np.asarray(row["guidance_increment_limited"], dtype=bool)
                assert np.allclose(
                    raw_norm[step].max(axis=-1),
                    row["unconstrained_max_atom_increment_A"],
                )
                assert np.array_equal(
                    flags, raw_norm[step].max(axis=-1) > increment_limit
                )
                assert applied_norm[step].max() <= increment_limit + 1e-6
                item = {
                    "step": step,
                    "network_time": row["time"],
                    "coefficient": row["coefficient"][0],
                    "limited_candidates": int(flags.sum()),
                    "maximum_raw_increment_A": float(raw_norm[step].max()),
                    "maximum_applied_increment_A": float(applied_norm[step].max()),
                    "minimum_scale_factor": float(factors.min()),
                    "mean_input_gradient_norm": float(np.mean(row["gradient_norm"])),
                    "maximum_input_gradient_norm": float(np.max(row["gradient_norm"])),
                    "mean_guidance_to_native_increment_ratio": float(
                        np.mean(row["delta_native_ratio"])
                    ),
                    "mean_normalized_total_loss": float(np.mean(row["loss"])),
                    "mean_detached_length_denominator_A": float(
                        trace["distance_normalization_denominator_angstrom"][
                            step
                        ].mean()
                    ),
                    **{
                        key: float(value[step].mean())
                        for key, value in step_distances.items()
                    },
                }
                per_step.append(item)
                for candidate in np.flatnonzero(flags):
                    atom_slot = int(raw_norm[step, candidate].argmax())
                    atom_id = slot_to_atom[atom_slot]
                    events.append(
                        {
                            "step": step,
                            "candidate": int(candidate),
                            "node_id": records[candidate]["node_id"],
                            "logical_seed": records[candidate]["logical_seed"],
                            "logical_slot": records[candidate]["logical_slot"],
                            "peak_atom_id": atom_id,
                            "peak_atom_element": atoms[atom_id]["element"],
                            "peak_atom_is_controlled": atom_id in controlled_ids,
                            "raw_increment_A": float(
                                raw_norm[step, candidate, atom_slot]
                            ),
                            "raw_maximum_controlled_atom_increment_A": float(
                                raw_norm[step, candidate, controlled_slots].max()
                            ),
                            "scale_factor": float(factors[candidate]),
                        }
                    )
            raw_max_step, raw_max_candidate, raw_max_atom = np.unravel_index(
                raw_norm.argmax(), raw_norm.shape
            )
            weights = {
                key: np.unique(trace[key][:]).tolist()
                for key in (
                    "condition_intra_weight",
                    "condition_outer_weight",
                    "interact_weigh",
                )
            }
        configurations.append(
            {
                "actual_scale": measurement["effective_guidance_scale_schedule"][0],
                "source_group": source,
                "metrics": {key: summarize(value) for key, value in distances.items()},
                "condition_coordinate_metrics": {
                    key: summarize([g[key] for g in geometry]) for key in geometry[0]
                },
                "condition_coordinate_metrics_by_candidate": [
                    dict(node_id=r["node_id"], **g) for r, g in zip(records, geometry)
                ],
                "terminal_vs_step99_maximum_difference_A": terminal_trace_difference,
                "loss_weight_values_across_all_steps": weights,
                "step10_exact_to_weakest_configuration": first_step_comparison,
                "limited_candidate_steps": len(events),
                "limited_steps": [
                    r["step"] for r in per_step if r["limited_candidates"]
                ],
                "raw_maximum": {
                    "step": int(raw_max_step),
                    "candidate": int(raw_max_candidate),
                    "atom_slot": int(raw_max_atom),
                    "atom_id": slot_to_atom[int(raw_max_atom)],
                    "increment_A": float(raw_norm.max()),
                },
                "limit_peak_atom_categories": dict(
                    collections.Counter(
                        "controlled" if event["peak_atom_is_controlled"] else "outer"
                        for event in events
                    )
                ),
                "per_step": per_step,
                "limiting_events": events,
            }
        )
    archived = read(root / "monitoring/terminal_boundary_audit_20261029.json")
    mismatches = [
        name
        for name, digest in sources.items()
        if digest != archived["artifact_sha256"][name]
    ]
    assert not mismatches, mismatches
    result = {
        "timestamp": datetime.datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "analysis_only": True,
        "new_candidates": 0,
        "new_model_forwards": 0,
        "new_llm_calls": 0,
        "source_hash_mismatches": mismatches,
        "condition_pair_categories": dict(
            collections.Counter(p["category"] for p in condition["pairs"])
        ),
        "interaction_pairs": len(condition["interaction_pairs"]),
        "condition_control_equals_interact": set(controlled_ids)
        == {i["atom_id"] for i in condition["interact_atoms"]},
        "aggregation": "mean of per-candidate RMSD, not pooled root mean square",
        "configurations": configurations,
        "source_sha256": sources,
    }
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    best = configurations[-1]
    print(
        json.dumps(
            {
                "path": str(destination),
                "condition_pair_categories": result["condition_pair_categories"],
                "condition_control_equals_interact": result[
                    "condition_control_equals_interact"
                ],
                "coordinate_metrics_by_scale": [
                    {
                        k: c[k]
                        for k in (
                            "actual_scale",
                            "condition_coordinate_metrics",
                            "limited_steps",
                            "limited_candidate_steps",
                            "raw_maximum",
                        )
                    }
                    for c in configurations
                ],
                "best": {
                    k: best[k] for k in best if k not in ("per_step", "limiting_events")
                },
                "best_limiting_steps": [
                    r for r in best["per_step"] if r["limited_candidates"]
                ],
                "worst_event": max(
                    best["limiting_events"], key=lambda e: e["raw_increment_A"]
                ),
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("destination", type=Path)
    arguments = parser.parse_args()
    analyze(arguments.root, arguments.destination)
