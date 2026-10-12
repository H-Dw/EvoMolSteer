"""One operative edit per condition; immutable temperatures and provenance."""
import copy
import gzip
import json
import re
from pathlib import Path

import numpy as np

from evomolsteer.continuous.m2_skill_api import BASE, REF, effective_signature, validate_fixed_temperatures
from evomolsteer.io import digest, read_json, write_json


def compile_program(repo, study, response, incumbent, *, lane="G", folder=None):
    repo,study=Path(repo).resolve(),Path(study).resolve()
    if folder is not None and Path(folder).resolve().is_relative_to(study/"diagnostics"):
        raise ValueError("Diagnostic responses are quarantined from reward compilation")
    activation=read_json(study/"activation.json")
    allowed=activation["protocol"]["lanes"][lane]["allowed"]
    index=read_json(study/"evidence/index.json")
    # Survival, lineage and branch summaries encode the region in their typed
    # feature name even when they have no separate `region` column. Do not
    # reject a genuine region-bound hypothesis because of that storage choice.
    for table in read_json(study/"evidence/packet.json")["tables"].values():
        if "feature" not in table["columns"]:continue
        j=table["columns"].index("feature")
        for row in table["rows"]:
            match=re.match(r"landmark_(\d+)_",str(row[j]))
            if match:index["regions"].setdefault(row[0],int(match.group(1)))
    p=copy.deepcopy(incumbent)
    validate_fixed_temperatures(repo,p)
    # Every incumbent has explicit file provenance; shape phases can build upon
    # a confirmed prior shape without silently returning to the original M2.
    ref_relative=p.pop("_reference_relative_path", REF)
    edit=response["edit"]
    if response["decision"] in {"retain","defer"}:
        if edit is not None:raise ValueError("Retain/defer requires null edit")
    elif response["decision"]=="edit":
        if edit is None:raise ValueError("Missing operative edit")
        parameter,value=edit["parameter"],edit["value"]
        if parameter not in allowed or isinstance(value,bool) or not np.isfinite(value):
            raise ValueError("Illegal parameter, lane or nonfinite value")
        reference=json.loads(gzip.decompress((repo/ref_relative).read_bytes()))
        if parameter in {"region_index","temporal_region_index"}:
            if value!=int(value) or not 0<=int(value)<len(reference["landmarks_A"]):raise ValueError("Unknown region")
            region=int(value)
            if not any(index["regions"].get(e)==region for e in response["evidence_ids"]):
                raise ValueError("Region is not bound to claimed evidence")
            weights=[0.]*len(reference["landmarks_A"]);weights[region]=1.
            if p["reward_view"]=="endpoint_temporal_pointcloud":
                p["temporal_base_view"]="endpoint_regional_pointcloud"
            else:p["reward_view"]="endpoint_regional_pointcloud"
            p.update(coordinate_region_weights=weights,coordinate_region_radius_A=5.,coordinate_background_weight=.25)
            p.pop("coordinate_region_node_salience",None)
            p.pop("scientific_salience_source_sha256",None)
            if parameter=="temporal_region_index":
                salience=read_json(study/"evidence/regional_salience.json")
                if salience["times"]!=reference["times"] or salience["window"]!=reference["window"]:
                    raise ValueError("Temporal attention clock mismatch")
                values=salience["regions"].get(str(region))
                if values is None:raise ValueError("No supplied regional temporal function")
                array=np.ones((len(values),len(weights)))
                array[:,region]=values
                p["coordinate_region_node_salience"]=array.tolist()
                p["scientific_salience_source_sha256"]=digest(study/"evidence/regional_salience.json")
        elif parameter=="temporal_teacher_blend":
            if value not in allowed[parameter]:raise ValueError("Illegal temporal blend")
            if p["reward_view"]!="endpoint_temporal_pointcloud":
                p["temporal_base_view"]=p["reward_view"]
            p["reward_view"]="endpoint_temporal_pointcloud"
            p["temporal_teacher_blend"]=float(value)
            ref_relative=(study/"evidence/temporal_reference.json.gz").relative_to(repo).as_posix()
            p["reference_sha256"]=digest(repo/ref_relative)
        elif isinstance(allowed[parameter],list) and value in allowed[parameter]:
            p[parameter]=int(value) if parameter=="teacher_neighbors" else float(value)
        else:raise ValueError("Illegal bounded edit")
    else:raise ValueError("Unknown design decision")
    validate_fixed_temperatures(repo,p)
    reference=json.loads(gzip.decompress((repo/ref_relative).read_bytes()))
    if p["reference_sha256"]!=digest(repo/ref_relative) or p["window"]!=reference["window"]:
        raise ValueError("Reference/source or learned-support mismatch")
    if p.get("affinity_head_gradient") is not False or p.get("additional_per_step_affinity_calls")!=0 or p["derivative_path"]!="flowr_endpoint_vjp":
        raise ValueError("Coordinate-only real FLOWR derivative contract required")
    sha=effective_signature(p)
    p["program_id"]="scientific_"+sha[:12]
    path=study/"programs"/(sha+".json")
    if path.exists():
        # Metadata and explicit default values do not constitute a new
        # executable reward. Keep the first immutable numerical program.
        if effective_signature(read_json(path))!=sha:raise ValueError("Program signature collision")
    if not path.exists():write_json(path,p)
    receipt={"effective_sha256":sha,"program":path.relative_to(repo).as_posix(),"program_sha256":digest(path),
        "reference":ref_relative,"reference_sha256":digest(repo/ref_relative),"edit":edit,"lane":lane,
        "decision":response["decision"],"unchanged_temperatures":True,
        "time_support":reference["window"],"particle_selection":False,"affinity_head_gradient":False}
    if folder is not None:
        folder=Path(folder)
        receipt.update(response_sha256=digest(folder/"Designer.json"),instruction_sha256=digest(folder/"Designer.instructions.md"))
        write_json(folder/"compiler.json",receipt)
    return path,repo/ref_relative,sha,receipt
