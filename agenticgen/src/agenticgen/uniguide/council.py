"""Three private Guidance Agents and a selector that cannot invent schedules."""
import json
from pathlib import Path
import jsonschema
from .io import read, sha
from .protocol import BRANCHES, validate_action, choose
from .bootstrap import action_anchor
from .scale_domain import POLICY as SCALE_POLICY
from agenticgen.providers.deepseek_api import call_json


def object_schema(properties):
    return {"type":"object","properties":properties,"required":list(properties),"additionalProperties":False}


TEXT={"type":"string","minLength":1}
ACTION=object_schema({"id":TEXT,"branch":{"enum":BRANCHES},"anchor_id":TEXT,"anchor_reason":TEXT,
    "mode":{"enum":["coarse","fine"]},"schedule":{"type":"array","items":{"type":"number","exclusiveMinimum":0},"minItems":4,"maxItems":4},
    "evidence":TEXT,"prediction":TEXT,"falsifier":TEXT})


def advise(config,teacher,version,round_number,history,domain,skills_dir,folder,final=False):
    if "target" in domain and (domain["target"]!=teacher["target"] or domain["node_id"]!=teacher["node_id"]):
        raise ValueError("Guidance context received another teacher's scale domain")
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    skill_text="\n\n".join(p.read_text(encoding="utf-8") for p in sorted(Path(skills_dir).glob("*.md")))
    anchors={r["id"]:r for r in history}
    seen=[r["schedule"] for r in history]
    common=dict(version=version,round=round_number,teacher={k:teacher[k] for k in ["target","node_id","pic50","heavy_atoms"]},
        domain=domain,history=history,comparison_best=choose(history),no_future_execution=final,
        physical_start="shared R0 state before step 10; action anchor is a separately measured schedule",
        task_memory="only this version's measured history and frozen calibration",
        proposal_contract="exactly one coarse/fine log step in at most two windows; two unique options per inclination")
    common["calibration_knowledge"]={phase:[r for r in read(Path(config["root"])/"calibration"/(phase+"_results.json"))
        if r["target"]==teacher["target"]] for phase in ["coarse","fine","windows","full"]}
    outputs={}
    for branch in BRANCHES:
        own=[r for r in history if r.get("branch")==branch]
        context=dict(common,branch=branch,default_anchor=(own[-1] if own else history[0]))
        if domain.get("policy")==SCALE_POLICY and not final:
            context["default_anchor"],context["default_anchor_reason"]=action_anchor(history,domain,branch)
        schema=object_schema({"analysis":TEXT,"mechanism_uncertainty":TEXT,"proposals":{
            "type":"array","items":ACTION,"minItems":0 if final else 2,"maxItems":0 if final else 2}})
        def validate(result,branch=branch):
            proposed=[]
            for p in result["proposals"]:
                try:
                    if p["branch"]!=branch: raise ValueError("Wrong private inclination")
                    validate_action(p,anchors,domain,seen+proposed)
                    proposed.append(p["schedule"])
                except ValueError as error: raise jsonschema.ValidationError(str(error)) from error
        messages=[{"role":"system","content":skill_text+"\nYou are the private "+branch+" Guidance Agent. Return only the contracted JSON."},
                  {"role":"user","content":json.dumps(context,ensure_ascii=False)}]
        outputs[branch]=call_json(messages,schema,folder/(branch+".json"),response_validator=validate,**config["llm"])
    candidates={}
    for branch,result in outputs.items():
        for index,p in enumerate(result["proposals"]):
            # Transport IDs are qualified so two private agents cannot collide.
            candidates[branch+":"+str(index)]=p
    properties={"analysis":TEXT,"choice_evidence":TEXT}
    if not final:
        properties["selected"]=object_schema({b:{"enum":[k for k in candidates if k.startswith(b+":")]} for b in BRANCHES})
    schema=object_schema(properties)
    def validate_synthesis(result):
        if final: return
        selected=[]
        for branch in BRANCHES:
            p=candidates[result["selected"][branch]]
            try: validate_action(p,anchors,domain,seen+selected)
            except ValueError as error: raise jsonschema.ValidationError(str(error)) from error
            selected.append(p["schedule"])
    synth=call_json([{"role":"system","content":skill_text+"\nYou are the synthesis Agent. Select one existing proposal per inclination; never invent a schedule."},
        {"role":"user","content":json.dumps(dict(common,private=outputs,candidates=candidates),ensure_ascii=False)}],
        schema,folder/"synthesis.json",response_validator=validate_synthesis,**config["llm"])
    return [] if final else [dict(candidates[synth["selected"][branch]],private_context=str(folder/(branch+".json")),
        synthesis_context=str(folder/"synthesis.json")) for branch in BRANCHES]
