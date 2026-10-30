"""Separate output-policy identity from telemetry; reuse complete promotion shards."""

import ast
import copy
from pathlib import Path

import pytest
import torch

from agenticgen.uniguide import geometry_shards as shards
from agenticgen.uniguide import promotion_recovery as recovery
from agenticgen.uniguide.geometry_protocol import EFFECTIVE
from agenticgen.uniguide.io import Budget, frozen, read, sha, value_sha, write
from agenticgen.uniguide.protocol import SEEDS


def policy(calls=2):
    return dict(
        output_policy="raw999",
        prefix_inpainting=dict(enabled=True, start_step=0, stop_step_exclusive=10),
        prefix_native_times={
            str(i): {
                channel: dict(minimum=1.0, maximum=1.0, calls=calls)
                for channel in ("coordinates", "chemistry")
            }
            for i in range(10)
        },
    )


def test_merge_preserves_nonadditive_observations_and_leaves_sources_unchanged():
    policies = [policy(2), policy(2), policy(6), policy(6), policy(6)]
    original = copy.deepcopy(policies)
    merged = shards.merge_output_policies(policies, list("abcde"))
    assert policies == original and "prefix_native_times" not in merged
    assert merged["prefix_native_times_by_source_shard"] == dict(
        zip("abcde", policies_times(policies))
    )
    assert "not additive" in merged["prefix_telemetry_scope"]
    assert shards.merge_output_policies(policies[:2], ["a", "b"]) == policy(2)
    legacy = dict(output_policy="raw999", prefix_inpainting=dict(enabled=False))
    assert shards.merge_output_policies([legacy, legacy], ["a", "b"]) == legacy


def test_five_window_screening_promotes_and_keeps_four_window_results_identical():
    from test_adaptive_calibration import row
    from agenticgen.uniguide import calibration_search, linear_calibration
    from agenticgen.uniguide.linear_scale import DEFAULTS

    original = [row(10, 3), row(20, 2), row(30, 1)]
    expanded = copy.deepcopy(original)
    for item in expanded:
        item[EFFECTIVE].append(item[EFFECTIVE][0])
        item["window_diagnostics"].append(copy.deepcopy(item["window_diagnostics"][0]))
    left = calibration_search.assess(original)
    right = calibration_search.assess(expanded)
    assert [calibration_search.strength(r) for r in left[0]] == [10, 20, 30]
    assert left[1:] == right[1:]
    settings = copy.deepcopy(DEFAULTS)
    assert linear_calibration.next_screen(
        original, settings
    ) == linear_calibration.next_screen(expanded, settings)
    with pytest.raises(ValueError, match="window layouts"):
        calibration_search.assess([original[0], expanded[1]])
    broken = copy.deepcopy(expanded[0])
    broken["window_diagnostics"].pop()
    with pytest.raises(ValueError, match="match the guidance"):
        calibration_search.assess([broken])


def policies_times(policies):
    return [p["prefix_native_times"] for p in policies]


def test_full_five_window_linear_calibration_reaches_frozen_domain(
    tmp_path, monkeypatch
):
    from test_linear_guidance import config, install_fake
    from agenticgen.uniguide import linear_calibration, linear_scale
    from agenticgen.uniguide.guidance_windows import FIVE

    executor, calls, _ = install_fake(monkeypatch, tmp_path)
    settings = dict(config(tmp_path), windows=[list(w) for w in FIVE])
    subject = dict(
        teacher=dict(target="teacher"),
        condition=dict(condition_sha256="teachercondition"),
    )

    def five_window_executor(*args, **kwargs):
        result = executor(*args, **kwargs)
        result["window_diagnostics"].append(
            copy.deepcopy(result["window_diagnostics"][0])
        )
        folder = tmp_path / result["id"]
        write(folder / "aggregate.json", result)
        receipt = read(folder / "COMPLETE.json")
        receipt["files"]["aggregate.json"] = sha(folder / "aggregate.json")
        write(folder / "COMPLETE.json", receipt)
        return result

    domain = linear_calibration.run(
        settings, [subject], lambda *a: None, five_window_executor
    )["teacher"]
    assert domain[linear_scale.REFERENCE] == 60
    assert domain["initial_guidance_scale_tick_schedule"] == [0] * 5
    assert domain["guidance_windows"] == [list(w) for w in FIVE]
    assert all(len(call[1]) == 5 for call in calls)
    assert linear_calibration.used(tmp_path, "teacher") <= 250


@pytest.mark.parametrize("mutation", ["static", "time", "missing", "calls"])
def test_merge_still_rejects_policy_and_native_time_errors(mutation):
    changed = policy(6)
    if mutation == "static":
        changed["output_policy"] = "native"
    elif mutation == "time":
        changed["prefix_native_times"]["9"]["coordinates"]["maximum"] = 0.9
    elif mutation == "missing":
        changed["prefix_native_times"].pop("9")
    else:
        changed["prefix_native_times"]["0"]["chemistry"]["calls"] = 0
    with pytest.raises(ValueError):
        shards.merge_output_policies([policy(), changed], ["a", "b"])


def old_package(tmp_path):
    package = tmp_path / "old_package"
    package.mkdir()
    tree = ast.parse(Path(shards.__file__).read_text(encoding="utf-8"))
    tree.body = [
        node
        for node in tree.body
        if getattr(node, "name", None) != "merge_output_policies"
    ]
    next(
        node for node in tree.body if getattr(node, "name", None) == "combine_products"
    ).body = [ast.Pass()]
    (package / "geometry_shards.py").write_text(ast.unparse(tree), encoding="utf-8")
    return package


def bindings(package):
    before = dict(
        teacher_id="teacher",
        condition_sha256="condition",
        scientific_sources={
            "geometry_shards": sha(package / "geometry_shards.py"),
            "sampler": "unchanged",
        },
    )
    after = copy.deepcopy(before)
    after["scientific_sources"]["geometry_shards"] = sha(shards.__file__)
    return before, after


def test_only_postprocessor_change_can_rebind_completed_measurements(tmp_path):
    package = old_package(tmp_path)
    before, after = bindings(package)
    proof = recovery.verify_compatibility(
        before, after, package, Path(shards.__file__).parent
    )
    assert proof["allowed_changed_function"] == "combine_products"
    assert "SlotStreams" in proof["unchanged_sampling_definitions"]
    for field in ("condition_sha256", "teacher_id"):
        with pytest.raises(ValueError, match="differ"):
            recovery.verify_compatibility(
                before,
                dict(after, **{field: "other"}),
                package,
                Path(shards.__file__).parent,
            )
    changed = copy.deepcopy(after)
    changed["scientific_sources"]["sampler"] = "changed"
    with pytest.raises(ValueError, match="exactly one"):
        recovery.verify_compatibility(
            before, changed, package, Path(shards.__file__).parent
        )
    text = (package / "geometry_shards.py").read_text(encoding="utf-8")
    (package / "geometry_shards.py").write_text(
        text.replace("class SlotStreams", "class OtherStreams"), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="additions or removals"):
        recovery.verify_module_equivalence(
            package / "geometry_shards.py", shards.__file__
        )


def source_shard(root, binding, slots, strength=100):
    actual = {EFFECTIVE: [strength] * 4}
    identity = dict(compatibility=binding, **actual, population=10, logical_slots=slots)
    relative = "sample_shards/" + value_sha(
        dict(compatibility=binding, **actual, slots=slots)
    )
    folder = root / relative
    frozen(folder / "manifest.json", dict(id=relative, identity=identity))
    write(
        folder / "records.json",
        [
            dict(logical_seed=seed, logical_slot=slot)
            for seed in SEEDS
            for slot in slots
        ],
    )
    write(
        folder / "aggregate.json",
        dict(id=relative, attempts=10, teacher_id="teacher", **actual),
    )
    (folder / "terminal.pt.gz").write_bytes(b"unchanged raw numerical artifact")
    write(folder / "terminal.pt.sha.json", dict(sha256=sha(folder / "terminal.pt.gz")))
    write(folder / "OUTPUT_POLICY.json", policy(2 if slots == [0, 1] else 6))
    frozen(
        folder / "COMPLETE.json", dict(files={p.name: sha(p) for p in folder.iterdir()})
    )
    Budget(root).reserve(relative, "calibration", 10, value_sha(identity))
    ledger = read(root / "attempts.json")
    ledger[relative]["teacher_id"] = "teacher"
    write(root / "attempts.json", ledger)
    return relative


def test_import_is_an_attributed_budget_alias_and_cache_prevents_resampling(
    tmp_path, monkeypatch
):
    from agenticgen.uniguide import absolute_runtime, geometry_batch

    package = old_package(tmp_path)
    before, after = bindings(package)
    source, root = tmp_path / "source", tmp_path / "repair"
    ids = [source_shard(source, before, slots) for slots in ([0, 1], [2, 3])]
    preserved = {
        str(p.relative_to(source)): sha(p) for p in source.rglob("*") if p.is_file()
    }
    manifest = tmp_path / "preserved.json"
    write(manifest, preserved)
    write(root / "acceptance/COMPLETE.json", dict(real_acceptance_fixture=True))
    config = dict(
        root=str(root),
        rounds=8,
        guidance_control=dict(maximum_selected=3),
        postprocessing_recovery=dict(
            source_root=str(source),
            source_commit="original-commit",
            source_package=str(package),
            preservation_manifest=str(manifest),
            preservation_sha256=sha(manifest),
        ),
    )
    subjects = [
        dict(
            teacher=dict(target="teacher"), condition=dict(condition_sha256="condition")
        )
    ]
    monkeypatch.setattr(absolute_runtime, "require_acceptance", lambda c: None)
    monkeypatch.setattr(absolute_runtime, "calibrated", lambda c: subjects)
    monkeypatch.setattr(recovery, "compatibility", lambda *args: after)
    result = recovery.run(config)
    assert (
        result["imported_attempts"] == 20
        and result["actual_new_sampling_attempts"] == 0
    )
    assert result["old_and_new_ledgers_are_alias_views"]
    assert sum(row["count"] for row in read(root / "attempts.json").values()) == 20
    assert read(root / "ATTEMPT_LIMITS.json")["stage_caps"] == dict(
        calibration=350, formal=1200
    )
    for alias, original_id in zip(result["aliases"], ids):
        original, imported = source / original_id, root / alias["alias_id"]
        assert (original / "terminal.pt.gz").read_bytes() == (
            imported / "terminal.pt.gz"
        ).read_bytes()
        assert read(imported / "terminal.pt.sha.json")["sha256"] == sha(
            imported / "terminal.pt.gz"
        )
        assert (
            read(imported / "COMPLETE.json")["actual_sampling_commit"]
            == "original-commit"
        )
        assert read(imported / "RECOVERY_PROVENANCE.json")[
            "ledger_alias_not_independent_measurement"
        ]
    assert all(sha(source / name) == digest for name, digest in preserved.items())
    job = dict(stage="calibration", schedule=[100] * 4, count=20)
    assert (
        geometry_batch.compile_units(
            config,
            subjects[0]["teacher"],
            subjects[0]["condition"],
            [job],
            _compatibility=after,
        )
        == []
    )
    with pytest.raises(ValueError, match="one-shot"):
        recovery.run(config)


@pytest.mark.parametrize(
    "failure", ["incomplete", "budget", "identity", "wrong_teacher"]
)
def test_import_cannot_refund_incomplete_or_reinterpret_other_attempts(
    tmp_path, failure
):
    package = old_package(tmp_path)
    before, after = bindings(package)
    source = tmp_path / "source"
    relative = source_shard(source, before, [0, 1])
    if failure == "incomplete":
        (source / relative / "FAILED.json").write_text("{}")
    else:
        ledger = read(source / "attempts.json")
        field, value = dict(
            budget=("count", 20),
            identity=("binding", "wrong"),
            wrong_teacher=("teacher_id", "other"),
        )[failure]
        ledger[relative][field] = value
        write(source / "attempts.json", ledger)
    with pytest.raises(ValueError):
        recovery.source_items(source, "teacher", after, package)


def test_promotion_merges_twenty_and_thirty_row_contexts_without_rng_or_sampling(
    tmp_path, monkeypatch
):
    import h5py
    import numpy as np
    from agenticgen.uniguide.native_tools import save_pt, load_pt

    parts = []
    for index in range(5):
        folder = tmp_path / str(index)
        folder.mkdir()
        slots = [2 * index, 2 * index + 1]
        values = [seed_index * 10 + slot for seed_index in range(5) for slot in slots]
        write(
            folder / "records.json",
            [
                dict(
                    logical_seed=seed,
                    logical_slot=slot,
                    tensor_row=s * 2 + offset,
                    node_id=str(value),
                )
                for s, seed in enumerate(SEEDS)
                for offset, (slot, value) in enumerate(
                    zip(slots, values[s * 2 : s * 2 + 2])
                )
            ],
        )
        save_pt(
            folder / "terminal.pt.gz", dict(coords=torch.tensor(values)[:, None, None])
        )
        for name in ("molecules_all_built.sdf", "molecules_raw_decodable.sdf"):
            (folder / name).write_bytes(b"")
        with h5py.File(folder / "trace.h5", "w") as trace:
            trace["fixture"] = np.tile(np.array(values)[None, :], (100, 1))
        write(
            folder / "steps.json",
            [
                dict(
                    step=i,
                    **{
                        k: values
                        for k in (
                            "loss",
                            "assignment_switch_fraction",
                            "gradient_norm",
                            "coefficient",
                            "max_atom_increment_A",
                            "delta_native_ratio",
                        )
                    },
                    sampling_rng_after_sha256=str(index),
                )
                for i in range(100)
            ],
        )
        write(folder / "rng_ledger.json", dict(original=index))
        write(folder / "OUTPUT_POLICY.json", policy(2 if index < 2 else 6))
        parts.append(dict(id=str(index)))
    before = {str(p): sha(p) for p in tmp_path.rglob("*") if p.is_file()}
    output = tmp_path / "fifty"
    output.mkdir()
    rng = torch.get_rng_state().clone()
    records = shards.combine_products(tmp_path, parts, output)
    assert torch.equal(rng, torch.get_rng_state())
    assert [r["logical_slot"] for r in records] == list(range(10)) * 5
    assert load_pt(output / "terminal.pt.gz")["coords"].flatten().tolist() == list(
        range(50)
    )
    with h5py.File(output / "trace.h5") as trace:
        assert np.array_equal(
            trace["fixture"][:], np.tile(np.arange(50)[None, :], (100, 1))
        )
    assert read(output / "steps.json")[99]["loss"] == list(range(50))
    assert set(
        read(output / "OUTPUT_POLICY.json")["prefix_native_times_by_source_shard"]
    ) == set("01234")
    assert all(sha(name) == digest for name, digest in before.items())
