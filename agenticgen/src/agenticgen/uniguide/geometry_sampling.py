"""Internal-distance control using the unchanged native integrator and full input VJP."""
import functools
import math
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem
from . import sampler
from .geometry_condition import loss, metrics, validate, DISTANCE, CONTROL_DISTANCE, NEIGHBOR_DISTANCE, MAX_ERROR
from .geometry_protocol import effective, expand_effective, validate_domain, GROWTH_LIMIT, NORMALIZED, EFFECTIVE, MULTIPLIER, POLICY
from .io import read, write, frozen, value_sha, sha, Budget
from .native_tools import load_pt
from .protocol import SEEDS


class GeometryControl(sampler.Control):
    def __init__(self,*args,condition,scale_binding,**kwargs):
        self.geometry_condition=validate(condition);self.scale_binding=scale_binding
        super().__init__(*args,**kwargs)

    def expand_schedule(self,schedule):
        if schedule!=self.scale_binding[EFFECTIVE]:raise ValueError("Effective schedule binding differs")
        return expand_effective(schedule)

    def condition_loss(self,endpoint,valid):
        return loss(endpoint,self.geometry_condition,valid)

    def condition_trace(self,target):
        return {"reference_pair_distances_angstrom":target}

    def checkpoint_metadata(self):
        return dict(geometry_binding=dict(condition_sha256=self.geometry_condition["condition_sha256"],
            fixed_correspondence_sha256=value_sha(self.geometry_condition["atom_to_generated_slot"]),
            scale_binding=self.scale_binding))

    def initialize(self,*args):
        if self.resume and self.resume.get("geometry_binding")!=self.checkpoint_metadata()["geometry_binding"]:
            raise ValueError("Resume requires identical distance condition, mapping and scale binding")
        return super().initialize(*args)

    def after(self,curr,step):
        result=super().after(curr,step)
        self.rows[-1].update(self.scale_binding)
        self.rows[-1]["condition_sha256"]=self.geometry_condition["condition_sha256"]
        if self.rows[-1]["assignment_switch_fraction"]!=[0.]*len(curr["coords"]):
            raise ValueError("Fixed correspondence changed")
        return result


def scale_binding(schedule,multiplier=None,enabled=True):
    if multiplier is None:
        expand_effective(schedule)
        if not enabled or any(x<=0 for x in schedule):raise ValueError("Calibration uses positive actual scales")
        return {NORMALIZED:None,EFFECTIVE:schedule,MULTIPLIER:None,"guidance_enabled":True,
                "scale_mode":"pre_experiment_actual_scale"}
    return {NORMALIZED:schedule,EFFECTIVE:effective(schedule,multiplier,enabled),MULTIPLIER:multiplier,
            "guidance_enabled":enabled,"scale_mode":"normalized_log2"}


def compatibility(config,teacher,condition,domain=None):
    from .geometry_protocol import source_binding
    roots=Path(config.get("initials_root",config["root"]))
    return dict(policy=POLICY,teacher_id=teacher["target"],condition_sha256=condition["condition_sha256"],
        mapping_sha256=value_sha(condition["atom_to_generated_slot"]),
        checkpoint_sha256=config["checkpoint_sha256"],native_hashes=config["native_hashes"],
        scientific_sources=source_binding(),output_policy="native_final_corrector",
        steps=100,noise_level=.2,seeds=SEEDS,domain_sha256=domain["domain_sha256"] if domain else None,
        initials={str(i):sha(roots/"initials"/teacher["target"]/f"batch_{i:03d}.pt.gz") for i in range(5)})


def validate_receipt(folder):
    folder=Path(folder);receipt=read(folder/"COMPLETE.json")
    for name,digest in receipt["files"].items():
        if sha(folder/name)!=digest:raise ValueError("Completed geometry source changed: "+name)
    return read(folder/"aggregate.json")


def reserve_group(root,run_id,stage,count,identity):
    """Caller holds the experiment executor lock; claims are never automatically removed."""
    root=Path(root);registry=root/"measurement_registry";registry.mkdir(exist_ok=True)
    fingerprint=value_sha(identity);entry=registry/(fingerprint+".json")
    if entry.exists():
        saved=read(entry)
        if saved["identity"]!=identity:raise ValueError("Fingerprint identity mismatch")
        source=root/saved["source_group"]
        if not (source/"COMPLETE.json").exists():
            raise RuntimeError("Existing failed/interrupted geometry claim; explicit audit required")
        return fingerprint,validate_receipt(source)
    Budget(root).reserve(run_id,stage,count,fingerprint)
    with entry.open("x",encoding="utf-8") as stream:
        import json
        json.dump(dict(source_group=run_id,identity=identity),stream,sort_keys=True)
    return fingerprint,None


def terminal_records(model,output,condition,folder,count):
    values=metrics(output["coords"],condition,output["mask"].bool())
    geometry_diagnostics=pose_diagnostics(output["coords"],condition)
    built=model._generate_mols(output);raw=model._generate_mols(output,sanitise=False)
    if len(built)!=count or len(raw)!=count:raise ValueError("Native builder population changed")
    rows=[]
    with Chem.SDWriter(str(folder/"molecules_all_built.sdf")) as writer, Chem.SDWriter(str(folder/"molecules_raw_decodable.sdf")) as rawwriter:
        for index,(mol,unvalidated) in enumerate(zip(built,raw)):
            seed_index,slot=divmod(index,count//5)
            row=dict(node_id=f"geometry_b{seed_index:03d}_final_{slot:03d}",logical_seed=SEEDS[seed_index],
                logical_slot=slot,tensor_row=index,build_success=mol is not None,connected=False,smiles="",
                failure_reason=None,chemistry_identity_guaranteed=False,
                coordinate_source="native final corrector; no inpainting or projection")
            row.update({name:float(v[index]) if v is not None else None for name,v in values.items()})
            row["geometry_diagnostics_not_used_for_ranking"]=geometry_diagnostics[index]
            if mol is not None:
                try:
                    Chem.SanitizeMol(mol);row["smiles"]=Chem.MolToSmiles(mol);row["connected"]=len(Chem.GetMolFrags(mol))==1
                except Exception as error:
                    row.update(build_success=False,failure_reason=type(error).__name__+": "+str(error))
            else:
                row["failure_reason"]="native_builder_returned_none"
                if unvalidated is not None:
                    try:Chem.SanitizeMol(Chem.Mol(unvalidated))
                    except Exception as error:row["failure_reason"]=type(error).__name__+": "+str(error)
            for molecule,out in [(unvalidated,rawwriter),(mol if row["build_success"] else None,writer)]:
                if molecule is not None:
                    for key in ("node_id","logical_seed","logical_slot","tensor_row"):molecule.SetProp(key,str(row[key]))
                    out.write(molecule)
            rows.append(row)
    return rows


def pose_diagnostics(coordinates,condition):
    """Report rigid-frame and reflection ambiguity separately from the distance objective."""
    ids=[a["id"] for a in condition["atoms"]];mapping=condition["atom_to_generated_slot"]
    reference=np.asarray(condition["reference_coordinates_angstrom"])
    generated=coordinates.detach().cpu().double().numpy()[:,[mapping[i] for i in ids]]
    centered_reference=reference-reference.mean(0);rows=[]
    for xyz in generated:
        centered=xyz-xyz.mean(0)
        left,_,right=np.linalg.svd(centered.T@centered_reference)
        reflected=np.linalg.det(left@right)<0
        correction=np.diag([1.,1.,-1. if reflected else 1.])
        residual=centered@left@correction@right-centered_reference
        rows.append(dict(mapped_centroid_translation_angstrom=float(np.linalg.norm(xyz.mean(0)-reference.mean(0))),
            mapped_proper_rotation_aligned_root_mean_square_deviation_angstrom=float(np.sqrt(np.mean(np.sum(residual**2,-1)))),
            unconstrained_alignment_prefers_reflection=bool(reflected),
            stereochemistry_assessment="not_evaluated_by_distance_constraints",
            chemical_identity_assessment="not_enforced_or_validated_by_distance_constraints"))
    return rows


def execute(model,config,teacher,condition,schedule,folder,stage,count=50,domain=None,enabled=True):
    if count not in (10,20,25,50):raise ValueError("Unsupported population")
    if domain is None and stage!="calibration":raise ValueError("Actual strengths are allowed only in pre-experiments")
    if domain is not None:
        validate_domain(domain,teacher["target"],condition["condition_sha256"])
        if any(not domain["normalized_minimum"]<=x<=domain["normalized_maximum"] for x in schedule):
            raise ValueError("Normalized schedule exceeds teacher's empirical interval")
    root=Path(config["root"]);folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    run_id=folder.relative_to(root).as_posix()
    scale=scale_binding(schedule,domain[MULTIPLIER] if domain else None,enabled)
    binding=compatibility(config,teacher,condition)
    identity=dict(compatibility=binding,effective_guidance_scale_schedule=scale[EFFECTIVE],population=count,
        logical_slots_per_seed=count//5)
    frozen(folder/"manifest.json",dict(id=run_id,identity=identity))
    fingerprint,cached=reserve_group(root,run_id,stage,count,identity)
    if cached is not None:
        if cached["id"]!=run_id:
            frozen(folder/"REUSED.json",dict(source_group=cached["id"],fingerprint=fingerprint,
                independent_measurement=False,new_attempts=0))
        return dict(cached,**scale)
    write(folder/"STARTED.json",dict(fingerprint=fingerprint))
    try:
        factory=functools.partial(GeometryControl,condition=condition,scale_binding=scale)
        output,_,_,_=sampler.sample(model,config,teacher,scale[EFFECTIVE],folder,count,
            control_factory=factory,check_affinity=False)
        rows=terminal_records(model,output,condition,folder,count)
        write(folder/"records.json",rows)
        aggregate=dict(id=run_id,teacher_id=teacher["target"],condition_sha256=condition["condition_sha256"],
            compatibility_sha256=value_sha(binding),fingerprint=fingerprint,attempts=count,complete=True,
            built=sum(r["build_success"] for r in rows),connected=sum(r["connected"] for r in rows),
            unique_smiles=len({r["smiles"] for r in rows if r["smiles"]}),**scale)
        for name in (DISTANCE,CONTROL_DISTANCE,NEIGHBOR_DISTANCE,MAX_ERROR):
            measured=[r[name] for r in rows if r[name] is not None]
            if measured and (len(measured)!=count or not all(math.isfinite(v) for v in measured)):
                raise FloatingPointError("Incomplete or nonfinite distance metric")
            aggregate[name]=float(max(measured) if name==MAX_ERROR else np.mean(measured)) if measured else None
        aggregate["window_diagnostics"]=sampler.diagnostics(folder)
        aggregate["maximum_single_atom_guidance_increment_angstrom"]=max(w["max_atom_increment_A"] for w in aggregate["window_diagnostics"])
        aggregate["finite_values_do_not_prove_stability"]=True
        aggregate["stability_status"]="pre_experiment_displacement_evidence_only"
        if domain is not None:
            reference=max(r["absolute_displacement"]["max_atom_increment_A"] for r in domain["constant_comparison"] if r["eligible"])
            growth=aggregate["maximum_single_atom_guidance_increment_angstrom"]/max(reference,1e-12)
            aggregate.update(maximum_displacement_to_calibrated_envelope_ratio=growth,
                stability_status="large_finite_displacement_growth" if growth>GROWTH_LIMIT else "within_measured_envelope_not_a_stability_guarantee")
        write(folder/"aggregate.json",aggregate)
        files=["manifest.json","records.json","aggregate.json","terminal.pt.gz","trace.h5","steps.json",
               "rng_ledger.json","molecules_all_built.sdf","molecules_raw_decodable.sdf"]
        frozen(folder/"COMPLETE.json",dict(files={name:sha(folder/name) for name in files}))
        return aggregate
    except Exception as error:
        write(folder/"FAILED.json",dict(type=type(error).__name__,error=str(error),attempts=count,
            failed_attempts_remain_charged=True,no_weaker_retry=True,fingerprint=fingerprint))
        raise
