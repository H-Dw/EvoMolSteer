"""Internal-distance control using the unchanged native integrator and full input VJP."""
from .geometry_protocol import sampling_reference

import functools
import math
from pathlib import Path
import h5py
import numpy as np
import torch
from rdkit import Chem
from . import sampler
from .geometry_condition import (
    loss,
    metrics,
    validate,
    DISTANCE,
    CONTROL_DISTANCE,
    NEIGHBOR_DISTANCE,
    MAX_ERROR,
)
from .geometry_protocol import (
    effective,
    expand_effective,
    validate_domain,
    GROWTH_LIMIT,
    NORMALIZED,
    EFFECTIVE,
    MULTIPLIER,
    POLICY,
)
from .io import read, write, frozen, value_sha, sha, Budget
from .native_tools import load_pt
from .protocol import SEEDS
from .geometry_dynamics import (
    Raw999Sampling,
    WINDOWS,
    LIMIT,
    policy,
    INCREMENT_OPERATORS,
    increment_summary,
)
from .geometry_reward import WEIGHTS, weights, INTERACTION_DISTANCE
from .geometry_chemistry import ChemicalStateInpainting


def metric_names(condition):
    from .absolute_coordinate import is_absolute, METRICS
    return METRICS if is_absolute(condition) else (DISTANCE,CONTROL_DISTANCE,NEIGHBOR_DISTANCE,MAX_ERROR,INTERACTION_DISTANCE)


def is_maximum_metric(name):
    from .absolute_coordinate import MAX_ERROR as coordinate_maximum
    return name in (MAX_ERROR, coordinate_maximum)


def coordinate_population_statistics(folder, rows, condition, config, aggregate):
    from .absolute_coordinate import is_absolute
    if is_absolute(condition):
        from .coordinate_statistics import aggregate_fields, SUMMARY
        aggregate.update(aggregate_fields(rows, config))
        write(folder / "COORDINATE_STATISTICS.json", aggregate[SUMMARY])
        from .absolute_coordinate import dynamic_assignment
        if dynamic_assignment(condition):
            fields=("initial_assignment_condition_coordinate_root_mean_square_deviation_angstrom",
                    "last_guidance_assignment_condition_coordinate_root_mean_square_deviation_angstrom")
            aggregate["assignment_diagnostics_not_used_for_ranking"] = dict(
                aggregation="raw arithmetic mean over the complete population",population=len(rows),
                **{name:sum(row[name] for row in rows)/len(rows) for name in fields})


class GeometryControl(ChemicalStateInpainting, Raw999Sampling, sampler.Control):
    absolute_state = None

    def __init__(self, *args, condition, scale_binding, **kwargs):
        self.geometry_condition = condition
        self.scale_binding = scale_binding
        from .absolute_coordinate import is_absolute, AssignmentState
        self.absolute_state = AssignmentState(self) if is_absolute(condition) else None
        self.geometry_condition = (
            self.absolute_state.objective.condition
            if self.absolute_state is not None else validate(condition)
        )
        super().__init__(*args, **kwargs)
        if "logical_slots" in self.initial:
            from .geometry_shards import SlotStreams

            self.random_streams = SlotStreams(
                (
                    self.resume["stream_states"]
                    if self.resume
                    else self.initial["stream_states"]
                ),
                self.initial["logical_slots"],
            )

    def expand_schedule(self, schedule):
        if schedule != self.scale_binding[EFFECTIVE]:
            raise ValueError("Effective schedule binding differs")
        return expand_effective(schedule)

    def sampling_function(self, config):
        self.absolute_config = config
        function = super().sampling_function(config)
        self.configure_chemistry()
        return function

    def condition_loss(self, endpoint, valid):
        return self.condition_loss_for_rows(endpoint, valid, list(range(len(endpoint))))

    def row_loss_weights(self, ids):
        return self.scale_binding.get(WEIGHTS)

    def condition_loss_for_rows(self, endpoint, valid, ids):
        if self.absolute_state is not None:
            return self.absolute_state.evaluate(endpoint, valid, ids)
        if not self.geometry_condition.get("reward_policy"):
            return loss(
                endpoint,
                self.geometry_condition,
                valid,
                check=lambda name, x: self.check_rows(name, x, ids),
            )
        from .geometry_reward import evaluate, KEYS, component_weights

        row_weights = self.row_loss_weights(ids)
        result, diagnostics = evaluate(
            endpoint,
            self.geometry_condition,
            valid,
            row_weights,
            check=lambda name, x: self.check_rows(name, x, ids),
        )
        value, reference, selected, assignment = result
        resolved = component_weights(
            self.geometry_condition, row_weights, len(endpoint)
        )
        diagnostics.update(
            {key: endpoint.new_tensor([w[key] for w in resolved]) for key in KEYS}
        )
        return (
            value,
            dict(reference_pair_distances_angstrom=reference, **diagnostics),
            selected,
            assignment,
        )

    def condition_trace(self, target):
        if self.absolute_state is not None:
            return {"reference_condition_coordinates_angstrom":target}
        return (
            target
            if isinstance(target, dict)
            else {"reference_pair_distances_angstrom": target}
        )

    def constrain_guidance(self, delta, valid):
        self.unconstrained_delta = delta
        result, self.increment_audit = INCREMENT_OPERATORS[self.sampling_policy["limiter"]](
            delta, valid, self.model.coord_scale, self.sampling_policy.get(LIMIT)
        )
        return result

    def guidance_limit_diagnostics(self):
        return self.increment_audit

    def guidance_limit_trace(self):
        result = {
            "unconstrained_guidance_delta_A": self.unconstrained_delta.double()
            * self.model.coord_scale,
        }
        if "guidance_increment_scale_factor" in self.increment_audit:
            result["guidance_increment_scale_factor"] = self.increment_audit["guidance_increment_scale_factor"]
        return result

    def checkpoint_metadata(self):
        return dict(
            **self.assignment_checkpoint(),
            geometry_binding=dict(
                condition_sha256=self.geometry_condition["condition_sha256"],
                fixed_correspondence_sha256=value_sha(
                    self.geometry_condition.get("atom_to_generated_slot", self.geometry_condition.get("assignment_policy"))
                ),
                scale_binding=self.scale_binding,
                logical_slots=getattr(self, "initial", {}).get("logical_slots"),
                sampling_policy=self.sampling_policy,
                chemical_binding=getattr(self, "chemical_binding", None),
            ),
            **self.raw_output_checkpoint(),
        )

    def initialize(self, *args):
        self.restore_assignments()
        self.restore_raw_output()
        if (
            self.resume
            and self.resume.get("geometry_binding")
            != self.checkpoint_metadata()["geometry_binding"]
        ):
            raise ValueError(
                "Resume requires identical distance condition, mapping and scale binding"
            )
        return super().initialize(*args)

    def close(self):
        super().close()
        self.write_output_policy()

    def after(self, curr, step):
        result = super().after(curr, step)
        self.rows[-1].update(self.scale_binding)
        self.rows[-1]["condition_sha256"] = self.geometry_condition["condition_sha256"]
        fixed = self.absolute_state is None or not self.absolute_state.dynamic
        if fixed and (self.absolute_state is None or step > 10) and self.rows[-1]["assignment_switch_fraction"] != [0.0] * len(curr["coords"]):
            raise ValueError("Fixed correspondence changed")
        if self.absolute_state is not None:
            self.absolute_state.save_cache()
        return self.apply_chemical_inpainting(result, step)

    def assignment_checkpoint(self):
        return {"absolute_assignment_state": self.absolute_state.rows,
                "absolute_assignment_sha256":value_sha(self.absolute_state.rows)} if self.absolute_state is not None else {}

    def restore_assignments(self):
        if self.absolute_state is not None:
            self.absolute_state.restore()


def scale_binding(
    schedule, multiplier=None, enabled=True, condition=None, loss_weights=None
):
    from . import linear_scale
    if isinstance(multiplier,dict) and linear_scale.enabled(multiplier):
        if loss_weights is not None:
            raise ValueError("Linear absolute guidance has one coordinate objective")
        return linear_scale.binding(schedule,multiplier,enabled)
    extra = (
        {WEIGHTS: weights(condition, loss_weights)}
        if condition and condition.get("reward_policy")
        else {}
    )
    if loss_weights is not None and not extra:
        raise ValueError("Loss controls require the component reward protocol")
    if multiplier is None:
        expand_effective(schedule)
        if not enabled or any(x <= 0 for x in schedule):
            raise ValueError("Calibration uses positive actual scales")
        return {
            NORMALIZED: None,
            EFFECTIVE: schedule,
            MULTIPLIER: None,
            "guidance_enabled": True,
            "scale_mode": "pre_experiment_actual_scale",
            **extra,
        }
    return {
        NORMALIZED: schedule,
        EFFECTIVE: effective(schedule, multiplier, enabled),
        MULTIPLIER: multiplier,
        "guidance_enabled": enabled,
        "scale_mode": "normalized_log2",
        **extra,
    }


def candidate_identity(scale):
    return {key: scale[key] for key in (EFFECTIVE, WEIGHTS) if key in scale}


def compatibility(config, teacher, condition, domain=None):
    import inspect
    from .geometry_protocol import effective, expand_effective, ticks

    package = Path(__file__).parent
    # Only sampling dependencies: LLM skills, prompts, actions and round counts do not refit lambda.
    names = [
        "sampler",
        "random_streams",
        "chunking",
        "batching",
        "condition",
        "geometry_condition",
        "absolute_coordinate",
        "geometry_sampling",
        "geometry_reward",
        "geometry_safety",
        "geometry_dynamics",
        "geometry_chemistry",
        "chem_inpainting",
        "geometry_shards",
        "geometry_batch",
        "task_scheduling",
        "native_tools",
        "prepare",
        "protocol",
        "joint_endpoint",
    ]
    sources = {name: sha(package / (name + ".py")) for name in names}
    sources["runtime/rng"] = sha(package.parent / "runtime/rng.py")
    sources.update(
        {
            fn.__name__: value_sha(inspect.getsource(fn))
            for fn in (effective, expand_effective, ticks)
        }
    )
    roots = Path(config.get("initials_root", config["root"]))
    result = dict(
        policy=config.get("protocol",POLICY),
        teacher_id=teacher["target"],
        condition_sha256=condition["condition_sha256"],
        mapping_sha256=value_sha(condition.get("atom_to_generated_slot",condition.get("assignment_policy"))),
        checkpoint_sha256=config["checkpoint_sha256"],
        native_hashes=config["native_hashes"],
        teacher_native_config_sha256=(
            sha(teacher["native_config"]) if "native_config" in teacher else None
        ),
        numeric_environment=dict(
            torch=str(torch.__version__),
            cuda=torch.version.cuda,
            hip=torch.version.hip,
            numpy=np.__version__,
            rdkit=Chem.rdBase.rdkitVersion,
        ),
        scientific_sources=sources,
        output_policy=config.get("output_policy"),
        sampling_dynamics=policy(config),
        steps=config.get("steps", 100),
        noise_level=config.get("noise_level", 0.2),
        seeds=config.get("seeds", SEEDS),
        windows=config.get("windows", [list(w) for w in WINDOWS]),
        population=50,
        task_scheduling=config.get("task_scheduling", "ug04-930db032-serial"),
        initials={
            str(i): sha(roots / "initials" / teacher["target"] / f"batch_{i:03d}.pt.gz")
            for i in range(5)
        },
    )
    from .absolute_coordinate import is_absolute
    if is_absolute(condition):
        from .coordinate_statistics import policy as statistics_policy
        result["coordinate_statistics_policy"] = statistics_policy(config)
        result["scientific_sources"]["coordinate_statistics"] = sha(package / "coordinate_statistics.py")
    from . import linear_scale
    if linear_scale.enabled(config):
        result["guidance_scale_protocol"] = dict(policy=linear_scale.POLICY,
            **{linear_scale.UNIT:config["guidance_control"][linear_scale.UNIT]})
        result["scientific_sources"].update({"linear_scale/"+fn.__name__:value_sha(inspect.getsource(fn))
            for fn in (linear_scale.effective,linear_scale.encode,linear_scale.binding,
                       linear_scale.check_strength,linear_scale.grid)})
    return result


def validate_receipt(folder):
    folder = Path(folder)
    receipt = read(folder / "COMPLETE.json")
    for name, digest in receipt["files"].items():
        if sha(folder / name) != digest:
            raise ValueError("Completed geometry source changed: " + name)
    return read(folder / "aggregate.json")


def measurement_binding(config, teacher, condition, prepared=None, *, domain=None):
    """Reuse a binding only inside its current task; entry/resume recompute file hashes."""
    if domain is not None:
        validate_domain(domain, teacher["target"], condition["condition_sha256"])
    if prepared is None:
        prepared = compatibility(config, teacher, condition)
    elif (prepared["teacher_id"] != teacher["target"]
              or prepared["condition_sha256"] != condition["condition_sha256"]):
        raise ValueError("Prepared binding belongs to another teacher or condition")
    if domain is not None and domain["sampling_protocol_sha256"] != value_sha(prepared):
        raise ValueError("Another sampling protocol's multiplier; new calibration required")
    return prepared


def reserve_group(
    root,
    run_id,
    stage,
    count,
    identity,
    registry_name="measurement_registry",
    charge=True,
):
    """Caller holds the experiment executor lock; claims are never automatically removed."""
    root = Path(root)
    registry = root / registry_name
    registry.mkdir(exist_ok=True)
    fingerprint = value_sha(identity)
    entry = registry / (fingerprint + ".json")
    if entry.exists():
        saved = read(entry)
        if saved["identity"] != identity:
            raise ValueError("Fingerprint identity mismatch")
        source = root / saved["source_group"]
        if not (source / "COMPLETE.json").exists():
            raise RuntimeError(
                "Existing failed/interrupted geometry claim; explicit audit required"
            )
        return fingerprint, validate_receipt(source)
    if charge:
        Budget(root).reserve(run_id, stage, count, fingerprint)
    with entry.open("x", encoding="utf-8") as stream:
        import json

        json.dump(dict(source_group=run_id, identity=identity), stream, sort_keys=True)
    return fingerprint, None


def terminal_records(model, output, condition, folder, count, slots=None):
    output_policy = (
        read(folder / "OUTPUT_POLICY.json")
        if (folder / "OUTPUT_POLICY.json").exists()
        else {}
    )
    from .absolute_coordinate import is_absolute, dynamic_assignment, match, metrics as coordinate_metrics
    if is_absolute(condition):
        with h5py.File(folder / "trace.h5", "r") as trace:
            last_assignment = torch.as_tensor(trace["assignment"][99], device=output["coords"].device)
            first_assignment = torch.as_tensor(trace["assignment"][10], device=output["coords"].device)
        rematched = dynamic_assignment(condition)
        assignment = match(output["coords"],condition,output["mask"].bool()) if rematched else last_assignment
        values = coordinate_metrics(output["coords"],condition,output["mask"].bool(),assignment)
        correspondence = dict(assignment=assignment.cpu().tolist(),condition_sha256=condition["condition_sha256"],
            assignment_policy=condition["assignment_policy"],no_terminal_rematching=not rematched,
            terminal_rematched=rematched,coordinate_source="returned_raw999_uncovered_prediction")
        if rematched:
            from .absolute_coordinate import RMSD
            diagnostic_coordinates=output["coords"].detach().cpu().double()
            diagnostic_mask=output["mask"].detach().cpu().bool()
            values["initial_assignment_condition_coordinate_root_mean_square_deviation_angstrom"] = coordinate_metrics(
                diagnostic_coordinates,condition,diagnostic_mask,first_assignment.cpu())[RMSD]
            values["last_guidance_assignment_condition_coordinate_root_mean_square_deviation_angstrom"] = coordinate_metrics(
                diagnostic_coordinates,condition,diagnostic_mask,last_assignment.cpu())[RMSD]
            correspondence.update(initial_assignment=first_assignment.cpu().tolist(),
                last_guidance_assignment=last_assignment.cpu().tolist(),
                terminal_to_last_guidance_switch_fraction=(assignment != last_assignment).float().mean(1).tolist(),
                interpretation="Terminal RMSD measures geometric coverage; generated atom identity may change.")
        else:
            correspondence["frozen_from_step"] = 10
        write(folder/"COORDINATE_ASSIGNMENT.json",correspondence)
        geometry_diagnostics = [dict(protein_alignment_sha256=value_sha(condition["protein_alignment"]),
            fitted_protein_rmsd_angstrom=condition["protein_alignment"]["fitted_protein_rmsd_angstrom"],
            stereochemistry_assessment="not_guaranteed_by_coordinate_loss") for _ in range(count)]
    else:
        values = metrics(output["coords"], condition, output["mask"].bool())
        geometry_diagnostics = pose_diagnostics(output["coords"], condition)
    built = model._generate_mols(output)
    raw = model._generate_mols(output, sanitise=False)
    if len(built) != count or len(raw) != count:
        raise ValueError("Native builder population changed")
    rows = []
    with (
        Chem.SDWriter(str(folder / "molecules_all_built.sdf")) as writer,
        Chem.SDWriter(str(folder / "molecules_raw_decodable.sdf")) as rawwriter,
    ):
        for index, (mol, unvalidated) in enumerate(zip(built, raw)):
            seed_index, slot = divmod(index, count // 5)
            if slots is not None:
                slot = slots[slot]
            row = dict(
                node_id=f"geometry_b{seed_index:03d}_final_{slot:03d}",
                logical_seed=SEEDS[seed_index],
                logical_slot=slot,
                tensor_row=index,
                build_success=mol is not None,
                connected=False,
                smiles="",
                failure_reason=None,
                chemistry_identity_guaranteed=False,
                chemical_state_inpainting_enabled=output_policy.get(
                    "chemical_state_projection", False
                ),
                terminal_chemistry_projected=False,
                coordinate_source="raw999 last-loop uncovered prediction (step 99, network time about 0.98901); no terminal forward or projection",
            )
            row.update(
                {
                    name: float(v[index]) if v is not None else None
                    for name, v in values.items()
                }
            )
            if is_absolute(condition):
                row["condition_generated_slots"] = assignment[index].cpu().tolist()
                row["assignment_policy"] = condition["assignment_policy"]
                row["terminal_rematched"] = rematched
            row["geometry_diagnostics_not_used_for_ranking"] = geometry_diagnostics[
                index
            ]
            if mol is not None:
                try:
                    Chem.SanitizeMol(mol)
                    row["smiles"] = Chem.MolToSmiles(mol)
                    row["connected"] = len(Chem.GetMolFrags(mol)) == 1
                except Exception as error:
                    row.update(
                        build_success=False,
                        failure_reason=type(error).__name__ + ": " + str(error),
                    )
            else:
                row["failure_reason"] = "native_builder_returned_none"
                if unvalidated is not None:
                    try:
                        Chem.SanitizeMol(Chem.Mol(unvalidated))
                    except Exception as error:
                        row["failure_reason"] = type(error).__name__ + ": " + str(error)
            for molecule, out in [
                (unvalidated, rawwriter),
                (mol if row["build_success"] else None, writer),
            ]:
                if molecule is not None:
                    for key in (
                        "node_id",
                        "logical_seed",
                        "logical_slot",
                        "tensor_row",
                    ):
                        molecule.SetProp(key, str(row[key]))
                    out.write(molecule)
            rows.append(row)
    return rows


def pose_diagnostics(coordinates, condition):
    """Report rigid-frame and reflection ambiguity separately from the distance objective."""
    ids = [a["id"] for a in condition["atoms"]]
    mapping = condition["atom_to_generated_slot"]
    reference = np.asarray(condition["reference_coordinates_angstrom"])
    generated = (
        coordinates.detach().cpu().double().numpy()[:, [mapping[i] for i in ids]]
    )
    centered_reference = reference - reference.mean(0)
    rows = []
    for xyz in generated:
        centered = xyz - xyz.mean(0)
        left, _, right = np.linalg.svd(centered.T @ centered_reference)
        reflected = np.linalg.det(left @ right) < 0
        correction = np.diag([1.0, 1.0, -1.0 if reflected else 1.0])
        residual = centered @ left @ correction @ right - centered_reference
        rows.append(
            dict(
                mapped_centroid_translation_angstrom=float(
                    np.linalg.norm(xyz.mean(0) - reference.mean(0))
                ),
                mapped_proper_rotation_aligned_root_mean_square_deviation_angstrom=float(
                    np.sqrt(np.mean(np.sum(residual**2, -1)))
                ),
                unconstrained_alignment_prefers_reflection=bool(reflected),
                stereochemistry_assessment="not_evaluated_by_distance_constraints",
                chemical_identity_assessment="not_enforced_or_validated_by_distance_constraints",
            )
        )
    return rows


def execute_shard(
    model,
    config,
    teacher,
    condition,
    schedule,
    folder,
    stage,
    slots,
    domain=None,
    enabled=True,
    _materialized=None,
    _reservation=None,
    loss_weights=None,
    _compatibility=None,
):
    from .geometry_shards import initial_shard

    count = 10
    if domain is None and stage != "calibration":
        raise ValueError("Actual strengths are allowed only in pre-experiments")
    binding = measurement_binding(config, teacher, condition, _compatibility, domain=domain)
    if domain is not None:
        from .geometry_protocol import validate_sampling_schedule
        validate_sampling_schedule(schedule, domain)
    root = Path(config["root"])
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    run_id = folder.relative_to(root).as_posix()
    scale = scale_binding(
        schedule,
        sampling_reference(domain),
        enabled,
        condition,
        loss_weights,
    )
    identity = dict(
        compatibility=binding,
        **candidate_identity(scale),
        population=count,
        logical_slots=slots,
    )
    frozen(folder / "manifest.json", dict(id=run_id, identity=identity))
    if _reservation is None:
        fingerprint, cached = reserve_group(
            root, run_id, stage, count, identity, registry_name="slot_registry"
        )
    else:
        fingerprint = value_sha(identity)
        cached = None
        if _reservation != fingerprint or read(
            root / "slot_registry" / (fingerprint + ".json")
        ) != dict(source_group=run_id, identity=identity):
            raise ValueError(
                "Materialized batch differs from its reserved candidate identity"
            )
    if cached is not None:
        if cached["id"] != run_id:
            frozen(
                folder / "REUSED.json",
                dict(
                    source_group=cached["id"],
                    fingerprint=fingerprint,
                    independent_measurement=False,
                    new_attempts=0,
                ),
            )
        return dict(cached, **scale)
    write(folder / "STARTED.json", dict(fingerprint=fingerprint))
    try:
        factory = functools.partial(
            GeometryControl, condition=condition, scale_binding=scale
        )
        if _materialized is None:
            output, _, _, _ = sampler.sample(
                model,
                config,
                teacher,
                scale[EFFECTIVE],
                folder,
                count,
                control_factory=factory,
                check_affinity=False,
                initial_override=initial_shard(config, teacher["target"], slots),
            )
        else:
            output = _materialized
        rows = terminal_records(model, output, condition, folder, count, slots)
        write(folder / "records.json", rows)
        aggregate = dict(
            id=run_id,
            teacher_id=teacher["target"],
            condition_sha256=condition["condition_sha256"],
            compatibility_sha256=value_sha(binding),
            fingerprint=fingerprint,
            attempts=count,
            complete=True,
            built=sum(r["build_success"] for r in rows),
            connected=sum(r["connected"] for r in rows),
            unique_smiles=len({r["smiles"] for r in rows if r["smiles"]}),
            **scale,
        )
        for name in metric_names(condition):
            measured = [r[name] for r in rows if r.get(name) is not None]
            if measured and (
                len(measured) != count or not all(math.isfinite(v) for v in measured)
            ):
                raise FloatingPointError("Incomplete or nonfinite distance metric")
            aggregate[name] = (
                float(max(measured) if is_maximum_metric(name) else np.mean(measured))
                if measured
                else None
            )
        aggregate["window_diagnostics"] = sampler.diagnostics(folder, WINDOWS)
        aggregate.update(increment_summary(folder))
        aggregate["maximum_single_atom_guidance_increment_angstrom"] = max(
            w["max_atom_increment_A"] for w in aggregate["window_diagnostics"]
        )
        aggregate["finite_values_do_not_prove_stability"] = True
        aggregate["stability_status"] = "pre_experiment_displacement_evidence_only"
        if domain is not None:
            reference = max(
                r["absolute_displacement"]["max_atom_increment_A"]
                for r in domain["constant_comparison"]
                if r["eligible"]
            )
            growth = aggregate.get(
                "maximum_unconstrained_single_atom_guidance_increment_angstrom",
                aggregate["maximum_single_atom_guidance_increment_angstrom"],
            ) / max(reference, 1e-12)
            aggregate.update(
                maximum_displacement_to_calibrated_envelope_ratio=growth,
                stability_status=(
                    "large_finite_displacement_growth"
                    if growth > GROWTH_LIMIT
                    else "within_measured_envelope_not_a_stability_guarantee"
                ),
            )
        coordinate_population_statistics(folder, rows, condition, config, aggregate)
        write(folder / "aggregate.json", aggregate)
        files = [
            "manifest.json",
            "records.json",
            "aggregate.json",
            "terminal.pt.gz",
            "trace.h5",
            "steps.json",
            "rng_ledger.json",
            "molecules_all_built.sdf",
            "molecules_raw_decodable.sdf",
        ]
        files.extend(name for name in (
            "OUTPUT_POLICY.json", "CHEMICAL_CONDITION.json",
            "BATCH_PROVENANCE.json", "COORDINATE_ASSIGNMENT.json", "COORDINATE_STATISTICS.json",
        ) if (folder / name).exists())
        frozen(
            folder / "COMPLETE.json",
            dict(files={name: sha(folder / name) for name in files}),
        )
        return aggregate
    except Exception as error:
        write(
            folder / "FAILED.json",
            dict(
                type=type(error).__name__,
                error=str(error),
                attempts=count,
                failed_attempts_remain_charged=True,
                no_weaker_retry=True,
                fingerprint=fingerprint,
            ),
        )
        raise


def execute(
    model,
    config,
    teacher,
    condition,
    schedule,
    folder,
    stage,
    count=50,
    domain=None,
    enabled=True,
    loss_weights=None,
    _compatibility=None,
):
    from .geometry_shards import combine_products

    if count not in (10, 20, 50):
        raise ValueError("Supported promotion populations are ten, twenty and fifty")
    if domain is None and stage != "calibration":
        raise ValueError("Actual strengths are allowed only in pre-experiments")
    binding = measurement_binding(config, teacher, condition, _compatibility, domain=domain)
    if domain is not None:
        from .geometry_protocol import validate_sampling_schedule
        validate_sampling_schedule(schedule, domain)
    root = Path(config["root"])
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    scale = scale_binding(
        schedule,
        sampling_reference(domain),
        enabled,
        condition,
        loss_weights,
    )
    identity = dict(
        compatibility=binding,
        **candidate_identity(scale),
        population=count,
        logical_slots_per_seed=count // 5,
    )
    run_id = folder.relative_to(root).as_posix()
    frozen(folder / "manifest.json", dict(id=run_id, identity=identity))
    fingerprint, cached = reserve_group(
        root, run_id, stage, count, identity, charge=False
    )
    if cached is not None:
        if cached["id"] != run_id:
            frozen(
                folder / "REUSED.json",
                dict(
                    source_group=cached["id"],
                    fingerprint=fingerprint,
                    independent_measurement=False,
                    new_attempts=0,
                ),
            )
        return dict(cached, **scale)
    try:
        shards = []
        for start in range(0, count // 5, 2):
            key = value_sha(
                dict(
                    compatibility=binding,
                    **candidate_identity(scale),
                    slots=[start, start + 1],
                )
            )
            shards.append(
                execute_shard(
                    model,
                    config,
                    teacher,
                    condition,
                    schedule,
                    root / "sample_shards" / key,
                    stage,
                    [start, start + 1],
                    domain,
                    enabled,
                    loss_weights=loss_weights,
                    _compatibility=binding,
                )
            )
        rows = combine_products(root, shards, folder)
        aggregate = dict(
            shards[0], id=run_id, fingerprint=fingerprint, attempts=count, **scale
        )
        aggregate.update(
            built=sum(r["build_success"] for r in rows),
            connected=sum(r["connected"] for r in rows),
            unique_smiles=len({r["smiles"] for r in rows if r["smiles"]}),
            sampling_provenance=[
                dict(
                    source_group=s["id"],
                    receipt_sha256=sha(root / s["id"] / "COMPLETE.json"),
                )
                for s in shards
            ],
            population_view=True,
            independent_validation=False,
        )
        for name in metric_names(condition):
            values = [r[name] for r in rows if r.get(name) is not None]
            aggregate[name] = (
                float(max(values) if is_maximum_metric(name) else np.mean(values))
                if values
                else None
            )
        aggregate["window_diagnostics"] = sampler.diagnostics(folder, WINDOWS)
        aggregate.update(increment_summary(folder))
        aggregate["maximum_single_atom_guidance_increment_angstrom"] = max(
            w["max_atom_increment_A"] for w in aggregate["window_diagnostics"]
        )
        if domain is not None:
            reference = max(
                r["absolute_displacement"]["max_atom_increment_A"]
                for r in domain["constant_comparison"]
                if r["eligible"]
            )
            growth = aggregate.get(
                "maximum_unconstrained_single_atom_guidance_increment_angstrom",
                aggregate["maximum_single_atom_guidance_increment_angstrom"],
            ) / max(reference, 1e-12)
            aggregate.update(
                maximum_displacement_to_calibrated_envelope_ratio=growth,
                stability_status=(
                    "large_finite_displacement_growth"
                    if growth > GROWTH_LIMIT
                    else "within_measured_envelope_not_a_stability_guarantee"
                ),
            )
        coordinate_population_statistics(folder, rows, condition, config, aggregate)
        write(folder / "aggregate.json", aggregate)
        files = [
            "manifest.json",
            "records.json",
            "aggregate.json",
            "terminal.pt.gz",
            "trace.h5",
            "steps.json",
            "rng_ledger.json",
            "molecules_all_built.sdf",
            "molecules_raw_decodable.sdf",
        ]
        if (folder / "OUTPUT_POLICY.json").exists():
            files.append("OUTPUT_POLICY.json")
        if (folder / "CHEMICAL_CONDITION.json").exists():
            files.append("CHEMICAL_CONDITION.json")
        if (folder / "COORDINATE_STATISTICS.json").exists():
            files.append("COORDINATE_STATISTICS.json")
        frozen(
            folder / "COMPLETE.json",
            dict(files={name: sha(folder / name) for name in files}),
        )
        return aggregate
    except Exception as error:
        write(
            folder / "FAILED.json",
            dict(
                type=type(error).__name__,
                error=str(error),
                requested_population=count,
                failed_attempts_remain_charged=True,
                no_weaker_retry=True,
                fingerprint=fingerprint,
            ),
        )
        raise
