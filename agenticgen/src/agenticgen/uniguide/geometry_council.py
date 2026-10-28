"""Bounded, full-name geometry evidence and exact normalized action contracts."""
import json
from pathlib import Path
import jsonschema
from .io import read, frozen, value_sha, sha
from .geometry_condition import DISTANCE, CONTROL_DISTANCE, NEIGHBOR_DISTANCE, MAX_ERROR
from .geometry_protocol import NORMALIZED, EFFECTIVE, MULTIPLIER, BRANCHES, WINDOWS, choose, legal_options, validate_action
from agenticgen.providers.deepseek_api import call_json


def object_schema(properties):
    return dict(type="object",properties=properties,required=list(properties),additionalProperties=False)


def text_schema(limit):return dict(type="string",minLength=1,maxLength=limit)


def parse_output(text):
    def pairs(items):
        out={}
        for key,value in items:
            if key in out:raise json.JSONDecodeError("Duplicate key",text,0)
            out[key]=value
        return out
    def invalid(value):raise json.JSONDecodeError("Nonfinite number",text,0)
    return json.loads(text,object_pairs_hook=pairs,parse_constant=invalid)


def format_contract(schema,diagnostics=None):
    return "Return one complete JSON object, no Markdown, duplicate keys or trailing text. "+json.dumps(
        dict(response_schema=schema,validation_errors=diagnostics),ensure_ascii=False,separators=(",",":"))


def history_view(row):
    keys=["id",NORMALIZED,DISTANCE,CONTROL_DISTANCE,NEIGHBOR_DISTANCE,MAX_ERROR,"attempts","built","connected",
        "unique_smiles","maximum_single_atom_guidance_increment_angstrom","proposal","proposal_sha256"]
    result={k:row[k] for k in keys if k in row}
    if "proposal" in result:
        result["proposal"]={k:v for k,v in result["proposal"].items() if k in
            ("id","branch","anchor_id","updates","evidence","prediction","falsifier")}
    result["sampling_interval_diagnostics"]=[dict(step_bounds=w["steps"],
        coordinate_increment_root_mean_square_magnitude_angstrom=w["increment_vector_rms_A"],
        maximum_single_atom_guidance_increment_angstrom=w["max_atom_increment_A"],
        input_gradient_norm_mean=w["gradient_norm_mean"],native_increment_ratio_mean=w["native_increment_ratio_mean"])
        for w in row.get("window_diagnostics",[])]
    return result


def advise(config,history,domain,folder,skills,failures=(),final=False,provider=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);skills=Path(skills)
    provider=provider or call_json
    anchors={r["id"]:r for r in history};seen=[r[NORMALIZED] for r in history]
    if not history or len(anchors)!=len(history):raise ValueError("Unique completed measured history required")
    common=dict(history=[history_view(r) for r in history],non_anchor_evidence_and_failures=list(failures),
        comparison_best_id=choose(history)["id"],no_future_execution=final,
        condition_sha256=domain["condition_sha256"],teacher_id=domain["teacher_id"],
        normalized_scale_interval=[domain["normalized_minimum"],domain["normalized_maximum"]],
        execution_contract=dict(allowed_integer_steps=[-2,-1,1,2],
            windows=[dict(window_id=i,step_bounds=list(w)) for i,w in enumerate(WINDOWS)],
            allowed_anchor_ids=list(anchors),measured_anchors=[
            dict(anchor_id=r["id"],**{NORMALIZED:r[NORMALIZED]}) for r in history]),
        scale_semantics="actual strength = teacher-calibrated multiplier * 2 ** normalized scale; zero is baseline, negative is weaker",
        history_coverage="all compatible completed measurements in this experiment; repeated fingerprints appear once")
    outputs={}
    def invoke(role,context,schema,validator):
        module="synthesis.md" if role=="synthesis" else "proposal.md"
        messages=[dict(role="system",content="\n\n".join((skills/name).read_text(encoding="utf-8") for name in ["analysis.md",module])+
            "\nYour role is "+role+". Return only the required JSON."),
            dict(role="user",content=json.dumps(context,sort_keys=True,ensure_ascii=False,separators=(",",":"),allow_nan=False))]
        output=folder/(role+".json");claim=folder/(role+".execution_claim.json")
        frozen(folder/(role+".prompt_source.json"),dict(messages=messages,response_schema=schema))
        if not output.exists():
            with claim.open("x",encoding="utf-8") as stream:json.dump(dict(role=role),stream)
        elif not claim.exists():raise ValueError("Cached output lacks execution claim")
        return provider(messages,schema,output,response_validator=validator,response_parser=parse_output,
            format_builder=format_contract,format_policy="geometry-integer-action-json-v2",**config["llm"])
    for branch in BRANCHES:
        options=legal_options(history,domain,branch) if not final else []
        if not final and len(options)<2:raise ValueError("Insufficient untested legal actions; no automatic range expansion")
        action=object_schema(dict(id=text_schema(80),branch=dict(enum=[branch]),anchor_id=dict(enum=list(anchors)),
            anchor_reason=text_schema(200),updates=dict(type="array",minItems=1,maxItems=2,items=object_schema(dict(
                window_id=dict(type="integer",minimum=0,maximum=3),step=dict(type="integer",enum=[-2,-1,1,2])))),
            evidence=text_schema(350),prediction=text_schema(240),falsifier=text_schema(240)))
        schema=object_schema(dict(analysis=text_schema(1200),mechanism_uncertainty=text_schema(350),
            proposals=dict(type="array",items=action,minItems=0 if final else 2,maxItems=0 if final else 2)))
        def check(result,branch=branch):
            proposed=[]
            for p in result["proposals"]:
                try:
                    if p["branch"]!=branch:raise ValueError("Incorrect branch")
                    proposed.append(validate_action(p,anchors,domain,seen+proposed))
                except ValueError as error:raise jsonschema.ValidationError(str(error)) from error
        outputs[branch]=invoke(branch,dict(common,branch=branch,feasible_examples=options),schema,check)
    candidates={b+":"+str(i):p for b,result in outputs.items() for i,p in enumerate(result["proposals"])}
    props=dict(analysis=text_schema(1100),choice_evidence=text_schema(900))
    if not final:props["selected"]=object_schema({b:dict(enum=[k for k in candidates if k.startswith(b+":")]) for b in BRANCHES})
    def check_synthesis(result):
        selected=[]
        for branch in ([] if final else BRANCHES):
            proposal=candidates[result["selected"][branch]]
            try:selected.append(validate_action(proposal,anchors,domain,seen+selected))
            except ValueError as error:raise jsonschema.ValidationError(str(error)) from error
    private={b:{k:v for k,v in answer.items() if k!="proposals"} for b,answer in outputs.items()}
    result=invoke("synthesis",dict(common,candidates=candidates,private=private),object_schema(props),check_synthesis)
    return [] if final else [dict(candidates[result["selected"][b]],private_context=str(folder/(b+".json")),
        synthesis_context=str(folder/"synthesis.json")) for b in BRANCHES]
