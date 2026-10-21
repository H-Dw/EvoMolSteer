import copy
from pathlib import Path

import jsonschema
import pytest

from evomolsteer.adaptive_schedule.io import defaults, read, write, sha, value_sha
from evomolsteer.adaptive_schedule.contracts import schema, validate_action
from evomolsteer.adaptive_schedule.retention import (
    compact_batch, restart_refs, winners, checked_path, verify_snapshot, artifact_folder)
from evomolsteer.adaptive_schedule.agent import packet, compact


TID = "5nn4_SC2_A_1017__top1"


def record(score, slot):
    return {"build_success": True, "connected": True, "failure_reason": "", "pic50_on_rescore": score,
            "smiles": "C", "slot": slot, "batch": 0, "node_id": f"candidate_{slot}"}


def source(root, name, parent=None, start=0):
    ref = f"branches/{TID}/batch_000/{name}"
    p = root / ref
    write(p / "manifest.json", {"parents": [parent] * 20, "start_step": start, "schedules": [[5]*8]*20})
    write(p / "COMPLETE.json", {"attempts": 20})
    write(p / "nodes.json", {str(t): {"step": t, "local_loss": [t/100]*20,
                                       "local_endpoint_A": [[[1, 2, 3]]]*20}
                             for t in range(start, 101, 10)})
    write(p / "steps.json", [{"step": t, "clip_factor": [1.]*20, "loss": [1.]*20}
                             for t in range(start, 100)])
    for name in ["terminal.pt.gz", "trace.h5", "molecules_all_built.sdf"]:
        (p / name).write_bytes(b"structural payload")
    for slot in range(20):
        q = p / "profiles" / f"slot_{slot:03d}"
        write(q / "complete.json", {"quality": {"posebusters_pass": True}})
        write(q / "interactions.json", {"fingerprints": [], "interactions": []})
        (q / "complex.pdb").write_text("protein and ligand")
        (q / "report.xml").write_text("<analysis/>")
    return ref


def make_chains(root):
    r0 = source(root, "r0")
    r1 = source(root, "v1_r01", r0, 10)
    chains = []
    for slot in range(20):
        h0 = {"id": "r00", "branch": r0, "ticks": [5]*8, "record": record(5, slot), "parent_id": None}
        h1 = {"id": "r01", "branch": r1, "ticks": [6]+[5]*7,
              "record": record(6 if slot % 2 == 0 else 4, slot), "parent_id": "r00"}
        chains.append({"slot": slot, "best_id": "r01" if slot % 2 == 0 else "r00", "history": [h0, h1]})
    return chains


def fake_builder(root, relative, sources, payloads, best, template, batch):
    # Independent fixture builder models self-contained restart assets without Torch.
    folder = root / relative
    write(folder / "manifest.json", {"parents": [None]*20, "start_step": 0, "schedules": [h["ticks"] for h in best]})
    write(folder / "nodes.json", {str(t): {"step": t, "local_loss": [t/100]*20,
                                           "local_endpoint_A": [[[1, 2, 3]]]*20} for t in range(0, 101, 10)})
    write(folder / "steps.json", [{"step": t, "clip_factor": [1.]*20, "loss": [1.]*20} for t in range(100)])
    for step in range(0, 101, 10):
        (folder / f"checkpoint_{step:03d}.pt.gz").write_bytes(b"restartable all slots")
    (folder / "molecules_all_built.sdf").write_bytes(b"20 retained winners")
    identity = [{"id": h["id"], "branch": h["branch"], "slot": i} for i, h in enumerate(best)]
    write(folder / "COMPLETE.json", {"identity_sha256": value_sha(identity), "restart_refs": [relative]*20,
          "payload_sha256": {str(p.relative_to(folder)): sha(p) for p in folder.rglob("*") if p.is_file()}})


def run_compact(root, chains, builder=fake_builder, profiler=lambda *args: None):
    return compact_batch(root, defaults(), {"teacher_id": TID}, 0, "v1", chains, builder, profiler)


def test_top1_per_chain_survives_while_failed_structures_are_removed(tmp_path):
    chains = make_chains(tmp_path)
    before = copy.deepcopy(chains)
    index = run_compact(tmp_path, chains)
    assert chains == before and len(index["best_ids"]) == 20
    assert index["best_ids"] == ["r01" if i % 2 == 0 else "r00" for i in range(20)]
    old = tmp_path / chains[0]["history"][-1]["branch"]
    assert not (old / "trace.h5").exists() and not (old / "molecules_all_built.sdf").exists()
    assert (old / "profiles/slot_000/report.xml").exists()
    assert (old / "profiles/slot_000/interactions.json").exists()
    assert not (old / "profiles/slot_000/complex.pdb").exists()
    assert (tmp_path / chains[0]["history"][0]["branch"] / "trace.h5").exists()
    assert len(restart_refs(tmp_path, "v1", TID, 0, chains)) == 20
    assert artifact_folder(tmp_path, "v1", TID, 0, chains[0]["history"][-1]["branch"], 0) == tmp_path / index["snapshot"]
    assert run_compact(tmp_path, chains) == index  # repeated cleanup is idempotent


def test_full_documentary_packet_is_unchanged_by_pruning(tmp_path):
    chains = make_chains(tmp_path)
    template = {"teacher_id": TID, "source_record": {"pic50_on_rescore": 8}, "core_atoms": [0],
                "context_atoms": [], "coords_A": [[0, 0, 0]], "atom_labels": [1], "charge_labels": [0]}
    def evidence():
        return [compact(packet(tmp_path, template, c["slot"], c["history"], c["best_id"], True)[0]) for c in chains]
    before = evidence()
    run_compact(tmp_path, chains)
    assert before == evidence()


def test_quality_and_snapshot_verification_precede_every_deletion(tmp_path):
    chains = make_chains(tmp_path)
    old = tmp_path / chains[0]["history"][-1]["branch"]
    (old / "profiles/slot_003/complete.json").unlink()
    with pytest.raises(ValueError, match="Quality"):
        run_compact(tmp_path, chains)
    assert (old / "trace.h5").exists()
    write(old / "profiles/slot_003/failure.json", {"error": "bad pose"})
    def damaged(*args):
        fake_builder(*args)
        (args[0] / args[1] / "checkpoint_040.pt.gz").write_bytes(b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        run_compact(tmp_path, chains, damaged)
    assert (old / "trace.h5").exists()
    assert not list((tmp_path / "versions").rglob("retention.json"))


def test_failed_builder_can_resume_without_losing_original_sources(tmp_path):
    chains = make_chains(tmp_path)
    def interrupted(*args):
        fake_builder(*args)
        (args[0] / args[1] / "COMPLETE.json").unlink()
        raise RuntimeError("interrupted before publication")
    with pytest.raises(RuntimeError):
        run_compact(tmp_path, chains, interrupted)
    assert (tmp_path / chains[0]["history"][-1]["branch"] / "trace.h5").exists()
    index = run_compact(tmp_path, chains)
    verify_snapshot(tmp_path, index["snapshot"])


def test_next_improvement_replaces_old_snapshot_but_keeps_its_documents(tmp_path):
    chains = make_chains(tmp_path)
    first = run_compact(tmp_path, chains)
    r2 = source(tmp_path, "v1_r02", first["snapshot"], 20)
    for c in chains:
        c["history"].append({"id": "r02", "branch": r2, "parent_id": c["best_id"],
                              "ticks": [6, 6]+[5]*6, "record": record(7, c["slot"])})
        c["best_id"] = "r02"
    second = run_compact(tmp_path, chains)
    assert first["snapshot"] != second["snapshot"]
    assert not (tmp_path / first["snapshot"] / "checkpoint_000.pt.gz").exists()
    assert (tmp_path / first["snapshot"] / "nodes.json").exists()
    verify_snapshot(tmp_path, second["snapshot"])


def test_invalid_scores_ties_and_cross_chain_selection_do_not_change_winner(tmp_path):
    chains = make_chains(tmp_path)
    chains[0]["history"][-1]["record"]["pic50_on_rescore"] = 5
    chains[0]["best_id"] = "r00"
    chains[2]["history"][-1]["record"]["connected"] = False
    chains[2]["best_id"] = "r00"
    assert [winners(chains)[i]["id"] for i in [0, 2]] == ["r00", "r00"]
    chains[2]["best_id"] = "r01"
    with pytest.raises(ValueError, match="ranking"):
        winners(chains)


def test_nonbest_parent_rejected_even_though_documentary_history_remains():
    history = [{"id": "r00", "ticks": [5]*8}, {"id": "r01", "ticks": [6]+[5]*7}]
    action = {"parent_id": "r00", "ticks": [4]+[5]*7, "evidence_ids": ["r00:t050"]}
    with pytest.raises(jsonschema.ValidationError, match="incumbent"):
        validate_action(action, history, ["r00:t050"], ["r01"])
    assert schema(history, ["r00:t050"], ["r01"])["properties"]["parent_id"]["enum"] == ["r01"]


def test_delete_scope_cannot_escape_batch(tmp_path):
    with pytest.raises(ValueError, match="escapes"):
        checked_path(tmp_path, "../outside", "branches")
    with pytest.raises(ValueError, match="escapes"):
        checked_path(tmp_path, "versions/v1", "branches")


@pytest.mark.parametrize("different_rng,empty_r0", [(False, False), (True, False), (False, True)])
def test_real_tensor_snapshot_preserves_every_checkpoint_and_noise_after_gc(tmp_path, different_rng, empty_r0):
    torch = pytest.importorskip("torch")
    h5py = pytest.importorskip("h5py")
    import numpy as np
    import random
    Chem = pytest.importorskip("rdkit.Chem")
    from evomolsteer.adaptive_schedule.flowr import save_pt, load_pt, compose, tensor_digest
    from evomolsteer.adaptive_schedule.runner import resolve_checkpoint
    chains = make_chains(tmp_path)
    r0, r1 = [h["branch"] for h in chains[0]["history"]]
    rng = {"cpu": torch.get_rng_state(), "python": random.getstate(), "numpy": np.random.get_state()}

    def native_files(ref, start, value):
        p = tmp_path / ref
        coords = torch.full((20, 2, 3), float(value))
        for step in range(start, 101, 10):
            state = {"step": step, "context_sha256": "same-context", "rng_sha256": tensor_digest(rng),
                     "integrator": {"guard_warned": False}, "times": [torch.full((20,), step/100)]*3,
                     "rng": rng, "current": {"coords": coords + step}, "cond": {"coords": coords},
                     "previous_assignment": torch.zeros((20, 1), dtype=torch.long)}
            if different_rng and ref == r1:
                state["rng"] = dict(rng, marker="different recorded stream")
                state["rng_sha256"] = tensor_digest(state["rng"])
            save_pt(p / f"checkpoint_{step:03d}.pt.gz", state)
            if step < 100:
                save_pt(p / f"head_{step:03d}.pt.gz", {"prediction": {"coords": coords + step}, "rng": rng})
        save_pt(p / "terminal.pt.gz", {"coords": coords + 100})
        save_pt(p / "rng_terminal.pt.gz", rng)
        with h5py.File(p / "trace.h5", "w") as f:
            f.create_dataset("native_epsilon", data=np.full((100, 20, 2, 3), value, dtype=np.float32))
        writer = Chem.SDWriter(str(p / "molecules_all_built.sdf"))
        for slot in range(20):
            mol = Chem.MolFromSmiles("CO")
            conf = Chem.Conformer(2)
            conf.SetAtomPosition(0, (0, 0, 0)); conf.SetAtomPosition(1, (1.3, 0, 0))
            mol.AddConformer(conf)
            mol.SetProp("slot", str(slot)); mol.SetProp("node_id", f"candidate_{slot}")
            writer.write(mol)
        writer.close()

    native_files(r0, 0, 1)
    native_files(r1, 10, 2)
    if empty_r0:
        # No decoded R0 molecule: preserve its restart state for unsuccessful slots.
        (tmp_path / r0 / "molecules_all_built.sdf").write_bytes(b"")
        for c in chains:
            c["history"][0]["record"].update(build_success=False, pic50_on_rescore=None)
            if c["slot"] % 2:
                c["history"][1]["record"].update(build_success=False, pic50_on_rescore=None)
    expected = {step: [resolve_checkpoint(tmp_path, h["branch"], step) for h in winners(chains)]
                for step in range(0, 101, 10)}
    index = run_compact(tmp_path, chains, builder=None)
    verify_snapshot(tmp_path, index["snapshot"])
    assert not (tmp_path / r1 / "terminal.pt.gz").exists()
    for step, states in expected.items():
        for slot, ref in enumerate(index["restart_refs"]):
            restored = resolve_checkpoint(tmp_path, ref, step)
            if different_rng:
                assert tensor_digest(restored) == tensor_digest(states[slot])
            else:
                assert tensor_digest(restored) == tensor_digest(compose(states))
    packed = tmp_path / index["snapshot"]
    if empty_r0:
        assert len([m for m in Chem.SDMolSupplier(str(packed / "molecules_all_built.sdf")) if m]) == 10
    with h5py.File(packed / "trace.h5", "r") as f:
        assert np.all(f["native_epsilon"][:10] == 1)
        assert np.all(f["native_epsilon"][10:, ::2] == 2)
        assert np.all(f["native_epsilon"][10:, 1::2] == 1)
    heads = load_pt(packed / "head_050.pt.gz")["_retained_slots"]
    assert torch.all(heads[0]["prediction"]["coords"] == 52)
    assert torch.all(heads[1]["prediction"]["coords"] == 51)
    assert tensor_digest(load_pt(packed / "rng_terminal.pt.gz")["_retained_slots"][0]) == tensor_digest(rng)
    # A second round must reuse already compacted winners and materialize their
    # prefixes without reopening the deleted r01 payloads.
    r2 = source(tmp_path, "v1_r02", index["restart_refs"][0], 20)
    native_files(r2, 20, 3)
    for c in chains:
        c["history"].append({"id": "r02", "branch": r2, "parent_id": c["best_id"],
                              "ticks": [6, 6]+[5]*6,
                              "record": record(7 if c["slot"] == 0 else 3, c["slot"])})
        if empty_r0 and c["slot"] % 2:
            c["history"][-1]["record"].update(build_success=False, pic50_on_rescore=None)
        if c["slot"] == 0:
            c["best_id"] = "r02"
    second = run_compact(tmp_path, chains, builder=None)
    verify_snapshot(tmp_path, second["snapshot"])
    assert not (packed / "trace.h5").exists()
    with h5py.File(tmp_path / second["snapshot"] / "trace.h5", "r") as f:
        assert np.all(f["native_epsilon"][10:20, 0] == 2)
        assert np.all(f["native_epsilon"][20:, 0] == 3)
        assert np.all(f["native_epsilon"][10:, 2] == 2)
