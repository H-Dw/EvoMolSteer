"""Real stateless role calls and blinded functional assessment."""
import copy
import hashlib
import json
from pathlib import Path

from evomolsteer.continuous.m2_skill_api import ANALYST_SCHEMA, DESIGN_SCHEMA
from evomolsteer.deepseek_api import call_json
from evomolsteer.io import digest, read_json, write_json
from .protocol import PLAN, validate_activation
from .modules import assemble


def full_input_packet(study):
    """Bind complete time functions, not only endpoint/rate summaries.

    All scalar weak/null evidence remains in the primary packet. Shared clock
    storage avoids repeating the same fifty times in every fallback function.
    The original tool output/hash remains immutable and independently readable.
    """
    study=Path(study)
    packet=read_json(study/"evidence/packet.json")
    path=study/"evidence/continuous_functions.json"
    source=read_json(path)
    if source["times"]!=packet["clocks"]["observed_score_times"] or source["window"]!=packet["clocks"]["learning_window"]:
        raise ValueError("Full function input clock mismatch")
    functions={}
    for feature,value in source["functions"].items():
        functions[feature]={"method":value["method"],"observed_support":value["observed_support"]}
        key="coefficients" if value["method"]=="legendre" else "values_z"
        functions[feature][key]=value[key]
    packet["continuous_functions"]={"source_sha256":digest(path),"shared_times":source["times"],
        "Legendre_argument":"u=2*(t-support_start)/(support_end-support_start)-1",
        "linear_argument":"observed values_z at shared_times; interpolate within observed support only",
        "observable":"favorable minus comparison coordinate feature, standardized by original feature scale",
        "not_absolute_target":True,"functions":functions}
    return packet


def role_prompt(repo, study, role, active, alternative=None):
    modules=read_json(Path(study)/"module_registry.json")["modules"]
    contract=(Path(repo)/PLAN/"contracts/shared_contract.md").read_text(encoding="utf-8")
    return f"You are {role}. Return JSON only.\n\n"+contract+"\n\n"+assemble(modules,role,active,alternative)


def design_schema(allowed, region_count):
    """Legal interface bounds, without required scientific recommendations."""
    schema=copy.deepcopy(DESIGN_SCHEMA)
    shape=schema["properties"]["edit"]
    shape["properties"]["parameter"]["enum"]=list(allowed)
    shape["allOf"]=[]
    for name,values in allowed.items():
        values=list(range(region_count)) if isinstance(values,str) else values
        shape["allOf"].append({"if":{"type":"object","properties":{"parameter":{"const":name}}},
            "then":{"properties":{"value":{"enum":values}}}})
    schema["allOf"]=[{"if":{"properties":{"decision":{"enum":["retain","defer"]}}},
        "then":{"properties":{"edit":{"type":"null"}}}},
        {"if":{"properties":{"decision":{"const":"edit"}}},
         "then":{"properties":{"edit":{"type":"object"}}}}]
    return schema


def operative_incumbent(program):
    """Expose the executing pointcloud formula, not inert legacy metadata.

    Compilation continues to use the original frozen program. This read-only
    interface prevents `robust_delta` and motif labels from being mistaken for
    active loss terms in the historical endpoint-pointcloud checkpoint.
    """
    if program is None:return None
    view=copy.deepcopy(program)
    if view.get("reward_view") in {"endpoint_pointcloud","endpoint_regional_pointcloud","endpoint_temporal_pointcloud"}:
        for key in ["robust_delta","geometry_block_weights","motif_components","target_definition","round"]:
            view.pop(key,None)
        view.setdefault("pointcloud_delta_A",1.)
    return view


def call_role(repo, study, case, chain, role, active, *, alternative=None,
              analyst=None, incumbent=None, lane="G", delivery=None, caller=call_json):
    repo,study=Path(repo),Path(study)
    activation=validate_activation(repo,study)
    protocol=activation["protocol"]
    schema=copy.deepcopy(ANALYST_SCHEMA if role=="Analyst" else DESIGN_SCHEMA)
    if role=="Analyst":
        schema["properties"]["findings"]["maxItems"]=12
        schema["properties"]["hypotheses"]["maxItems"]=6
    packet=full_input_packet(study)
    allowed=protocol["lanes"][lane]["allowed"]
    if role=="Designer":
        schema=design_schema(allowed,len(packet["teacher_modes"]["landmarks_A"]))
    prompt=role_prompt(repo,study,role,active,alternative)
    folder=study/"calls"/case/f"chain_{chain:02d}";folder.mkdir(parents=True,exist_ok=True)
    (folder/(role+".instructions.md")).write_text(prompt,encoding="utf-8")
    task={"role":role,"case_id":hashlib.sha256((case+str(chain)).encode()).hexdigest()[:16],
        "lane":lane,"response_schema":schema,"maximum_edits":1,
        "task_wording":("Interpret the supplied evidence and answer the coordinate-design task." if chain<3 else
            "Use these observations to provide your response to the supplied molecular coordinate task."),
        "interface_notes":"Optional findings/hypotheses/details may be empty when unsupported. No submodule function is required by the schema. Prioritize at most twelve concise findings and six hypotheses. Cite supplied curves rather than reproducing numeric arrays. Analyst hypotheses belong in hypotheses; operative decisions belong to Designer.",
        "allowed_edits":allowed,
        "incumbent":operative_incumbent(incumbent),
        "fixed_parameters":protocol["temperature_policy"]["parameters"],
        "formula_registry":{
            "pointcloud":"R=tau*logsumexp(log_pi-rho_delta(q)/tau); q=Hungarian mean squared endpoint distance; rho_delta=delta^2*(sqrt(1+q/delta^2)-1)",
            "prior":"Detached nearest-k teachers, log_pi=-q_anchor/prior_temperature, scores beta=0; k is a count",
            "regional":"Detached background + Gaussian landmark attention, positive surrounding weight",
            "temporal_region":"Same static teacher coordinates; only validated absolute contrast changes regional attention at actual score nodes",
            "temporal_teacher":"(1-blend)*R_static + blend*R_observed_teacher(time); one shared FLOWR endpoint, no velocity target",
            "dose":"Actual endpoint-VJP force normalized and RMS-scaled relative to native predictive update; eta and ramp only in U lane",
            "diagnostic_limit":"Moment effects, pair kernels and turnover rates are diagnostic evidence only. The executing pointcloud scalar has no pair-kernel, moment, joint-motif or geometry-block term. Legacy robust_delta is ignored; the executing curvature is pointcloud_delta_A (default 1 A). Compilation uses the original frozen program."}}
    if analyst is not None:task["Analyst_response"]=analyst
    if delivery is not None:task["delivery_diagnostics"]=delivery
    messages=[{"role":"system","content":prompt},
        {"role":"user","content":json.dumps(packet,separators=(",",":"))},
        {"role":"user","content":json.dumps(task,separators=(",",":"))}]
    result=caller(messages,schema,folder/(role+".json"),input_store=study/"inputs",max_output_tokens=16384)
    known=set(read_json(study/"evidence/index.json")["ids"]) | set(packet) | set(task)
    def citations(x):
        if isinstance(x,dict):
            for k,v in x.items():
                if k=="evidence_ids":yield from v
                else:yield from citations(v)
        elif isinstance(x,list):
            for v in x:yield from citations(v)
    if not set(citations(result))<=known:
        raise ValueError("Unknown scientific evidence citation")
    write_json(folder/(role+".delivery.json"),{"active_modules":active,"alternative":alternative,
        "instruction_sha256":digest(folder/(role+".instructions.md")),
        "evidence_sha256":activation["packet_sha256"],"role":role,
        "continuous_function_source_sha256":digest(study/"evidence/continuous_functions.json"),
        "complete_input_sha256":hashlib.sha256(json.dumps(packet,separators=(",",":")).encode()).hexdigest(),
        "model":"deepseek-flash","backend":"real_api","schema_forces_module_function":False})
    return result


def _strings(x):
    if isinstance(x,str):return [x]
    if isinstance(x,dict):return sum((_strings(v) for v in x.values()),[])
    if isinstance(x,list):return sum((_strings(v) for v in x),[])
    return []


def review(study, case, role, focal_ids, *, caller=call_json):
    """Assess actual functions in full/drop/alternative without condition labels."""
    study=Path(study);registry=read_json(study/"module_registry.json")["modules"]
    modules=[m for m in registry if m["role"]==role and m["id"] in focal_ids]
    calls=[read_json(study/"calls"/case/f"chain_{i:02d}"/(role+".json")) for i in range(6)]
    schema={"type":"object","additionalProperties":False,"required":["assessments"],"properties":{
        "assessments":{"type":"array","items":{"type":"object","additionalProperties":False,
            "required":["chain","module","score","quote","explanation"],"properties":{
                "chain":{"type":"integer","minimum":0,"maximum":5},
                "module":{"enum":[m["id"] for m in modules]},"score":{"enum":[0,1,2,"NA"]},
                "quote":{"type":"string","maxLength":200},"explanation":{"type":"string","maxLength":250}}}}}}
    prompt=("You are a blinded functional Reviewer. Assess supplied role responses, not their instruction condition or molecular performance. "
        "0 absent/wrong despite adequate input; 1 generic mention; 2 evidence-correct function affecting priority/hypothesis/design; NA required input absent. "
        "Quote exact response text, empty for absence. Assess every chain and every supplied rubric once. "
        "A correct statement of unavailable decoded labels is not proof of final-quality credit analysis. "
        "Do not reward mere keywords or infer necessity. Return JSON only.")
    messages=[{"role":"system","content":prompt},
        {"role":"user","content":json.dumps(full_input_packet(study),separators=(",",":"))},
        {"role":"user","content":json.dumps({"calls":[{"chain":i,"response":v} for i,v in enumerate(calls)],
            "rubrics":[{"module":m["id"],"goal":m["label_zh"],"checks":m["functional_rubric"]} for m in modules]},separators=(",",":"))}]
    path=study/"reviews"/(case+"_"+role+".json")
    result=caller(messages,schema,path,input_store=study/"inputs",max_output_tokens=16384)
    expected={(i,m["id"]) for i in range(6) for m in modules}
    actual={(a["chain"],a["module"]) for a in result["assessments"]}
    if actual!=expected or len(result["assessments"])!=len(expected):raise ValueError("Functional review coverage incomplete")
    for a in result["assessments"]:
        if a["quote"] and not any(a["quote"] in s for s in _strings(calls[a["chain"]])):
            raise ValueError("Reviewer quote not in real role output")
    return result
