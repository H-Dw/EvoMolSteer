"""Quarantined real-API sign, missingness and physical-unit response probes.

These are interface diagnostics, not molecular experiments or reward evidence.
The actual submodule effects are separately measured on six real role chains.
"""
import copy
import json
from pathlib import Path
from evomolsteer.continuous.m2_skill_api import ANALYST_SCHEMA, DESIGN_SCHEMA, BASE
from evomolsteer.deepseek_api import call_json
from evomolsteer.io import read_json, write_json, digest
from .agents import role_prompt
from .protocol import validate_activation


def contrast_fixture(packet, mode):
    if mode not in {"observed", "sign_changed", "missing"}:
        raise ValueError("Unknown diagnostic contrast")
    p = copy.deepcopy(packet)
    p["diagnostic_only"] = True
    p["diagnostic_transformation"] = mode
    p["tables"] = {"regional_effects": p["tables"]["regional_effects"]}
    p["prerequisites"] = {"selection_contrasts": mode != "missing"}
    p["limitations"] = ["Only regional contrast evidence is supplied in this isolated interface probe.",
        "Transformed copies are never production evidence or inputs to compilation."]
    if mode == "missing":
        p["tables"] = {}
    elif mode == "sign_changed":
        table = p["tables"]["regional_effects"]
        columns = table["columns"]
        for row in table["rows"]:
            for key in ["effect_z"]:
                j=columns.index(key);row[j] = -row[j]
            lo, hi = columns.index("CI_low"), columns.index("CI_high")
            row[lo], row[hi] = -row[hi], -row[lo]
            j=columns.index("positive_batch_fraction");row[j] = 1-row[j]
    return p


def geometry_fixture(packet, unit):
    if unit not in {"A", "nm"}: raise ValueError("Unsupported physical-unit probe")
    scale = 1. if unit == "A" else .1
    source=packet["teacher_modes"]
    return {"diagnostic_only": True, "geometry_unit": unit,
        "teacher_modes": {"count": source["count"], "source": source["source"],
            "assignment_rms_quantiles": [v*scale for v in source["assignment_rms_A_quantiles"]]},
        "clocks": packet["clocks"],
        "limitations": ["An isolated teacher-dispersion probe; no physical change between unit variants.",
            "Registered curvature parameters have units Angstrom; no temperature edit is legal."]}


def run_probes(repo, study, *, caller=call_json):
    repo,study=Path(repo),Path(study);validate_activation(repo,study)
    output=study/"diagnostics";output.mkdir(parents=True,exist_ok=True)
    packet=read_json(study/"evidence/packet.json")
    responses=[]
    for mode in ["observed", "sign_changed", "missing"]:
        payload=contrast_fixture(packet,mode)
        messages=[{"role":"system","content":role_prompt(repo,study,"Analyst",["A1.1"])},
            {"role":"user","content":json.dumps(payload,separators=(",",":"))},
            {"role":"user","content":"Analyze these supplied coordinate observations using the Analyst response interface. This isolated diagnostic is never used for generation."}]
        path=output/("Analyst_"+mode+".json")
        result=caller(messages,ANALYST_SCHEMA,path,input_store=study/"inputs")
        responses.append({"id":mode,"role":"Analyst","input":payload,"response":result,"sha256":digest(path)})
    schema=copy.deepcopy(DESIGN_SCHEMA)
    schema["properties"]["edit"]["properties"]["parameter"]["enum"]=["teacher_neighbors","pointcloud_delta_A"]
    for unit in ["A", "nm"]:
        payload=geometry_fixture(packet,unit)
        messages=[{"role":"system","content":role_prompt(repo,study,"Designer",["D2.1"])},
            {"role":"user","content":json.dumps(payload,separators=(",",":"))},
            {"role":"user","content":json.dumps({"incumbent":read_json(repo/BASE),
                "allowed_edits":{"teacher_neighbors":[2,4,8],"pointcloud_delta_A":[.5,1.,2.]},
                "parameter_unit":"A", "maximum_edits":1,
                "Analyst_response":{"role":"Analyst","findings":[{"evidence_ids":["teacher_modes"],
                    "claim":"The supplied intact-teacher dispersion is a spatial scale, not an intramolecular distance or mixture bandwidth."}]},
                "task":"Answer the Designer interface for this diagnostic. No generation follows."},separators=(",",":"))}]
        path=output/("Designer_unit_"+unit+".json")
        result=caller(messages,schema,path,input_store=study/"inputs")
        responses.append({"id":unit,"role":"Designer","input":payload,"response":result,"sha256":digest(path)})
    schema={"type":"object","additionalProperties":False,"required":["checks"],"properties":{
        "checks":{"type":"array","items":{"type":"object","additionalProperties":False,
            "required":["id","passed","explanation"],"properties":{
                "id":{"enum":["signed_contrast_response","missing_claim_withheld","physical_unit_consistency"]},
                "passed":{"type":"boolean"},"explanation":{"type":"string","maxLength":600}}}}}}
    messages=[{"role":"system","content":"You are an interface Reviewer. Use actual inputs and outputs to check interpretation, not performance. Return JSON only."},
        {"role":"user","content":json.dumps({"cases":responses,"checks":{
            "signed_contrast_response":"Observed and sign-changed contrasts must be described with their actual sign; a ranking may legitimately change. Do not require a reward edit.",
            "missing_claim_withheld":"No supplied contrast must lead to withholding unsupported selection-enrichment claims, not invented numerical effects.",
            "physical_unit_consistency":"Equivalent Angstrom/nm scales must be interpreted consistently in physical units; exploratory decisions may differ, but a tenfold unit error fails."}},separators=(",",":"))}]
    result=caller(messages,schema,output/"functional_preflight.json",input_store=study/"inputs")
    if {r["id"] for r in result["checks"]}!={"signed_contrast_response","missing_claim_withheld","physical_unit_consistency"} or len(result["checks"])!=3:
        raise ValueError("Diagnostic review coverage mismatch")
    write_json(output/"quarantine.json",{"no_compilation":True,"no_inference":True,
        "responses":{r["id"]+r["role"]:r["sha256"] for r in responses},
        "review_sha256":digest(output/"functional_preflight.json"),"role_api_calls":6,
        "passed":all(c["passed"] for c in result["checks"]),"not_module_performance_evidence":True})
    if not all(c["passed"] for c in result["checks"]):
        raise ValueError("Scientific role interface did not respond correctly to functional probes")
    return result
