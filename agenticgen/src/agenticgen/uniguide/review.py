"""Evidence-triggered GPT review and checkpoint recovery, independent of old versions."""
from pathlib import Path
import numpy as np
import h5py
from .io import read, write, sha, value_sha, frozen
from .protocol import WINDOWS, choose, regressed


def vector_change(old_path,new_path):
    if not Path(old_path).exists() or not Path(new_path).exists(): return {"known":False,"reason":"missing retained increment vectors"}
    with h5py.File(old_path,"r") as a,h5py.File(new_path,"r") as b:
        if "guidance_delta_A" not in a or "guidance_delta_A" not in b: return {"known":False,"reason":"missing vector dataset"}
        x,y=a["guidance_delta_A"][:],b["guidance_delta_A"][:]
        if x.shape!=y.shape or not (np.isfinite(x).all() and np.isfinite(y).all()): return {"known":False,"reason":"incompatible vectors"}
    changes=[]
    for lo,hi in WINDOWS:
        p,q=x[lo:hi+1],y[lo:hi+1]
        rms=lambda v:float(np.sqrt(np.mean(v*v)))
        changes.append(rms(q-p)/max(rms(p),rms(q),1e-8))
    return {"known":True,"relative_change":changes,"all_windows_at_most_5pct":all(c<=.05 for c in changes)}


def teacher_triggers(old,new,old_profiles,new_profiles,vectors):
    reasons=[]
    if abs(new["C"]-old["C"])<=.02 and abs(new["gap"]-old["gap"])<=.02:
        reasons.append("condition_and_affinity_gap_stagnation")
    previous={tuple(s) for s in old_profiles}
    fraction=sum(tuple(s) in previous for s in new_profiles)/len(new_profiles)
    if fraction>=.8: reasons.append("at_least_80pct_profiles_repeated")
    if vectors.get("known") and vectors["all_windows_at_most_5pct"]: reasons.append("all_increment_windows_change_at_most_5pct")
    return {"reasons":reasons,"profile_repeat_fraction":fraction,"vectors":vectors}


def recover_base(completed):
    """A completed recovery-derived version starts a new comparison epoch."""
    if not completed: return "UG_INIT",False
    best=choose([v["overall"] for v in completed])["id"]
    epoch=0
    for i,v in enumerate(completed):
        if v.get("recovery_derived"): epoch=i
    recent=completed[epoch:]
    trigger=len(recent)>=3 and regressed(recent[-3]["overall"],recent[-2]["overall"]) and regressed(recent[-2]["overall"],recent[-1]["overall"])
    return (best if trigger else completed[-1]["id"]),trigger


def build_review(config,version):
    from .quality_queue import snapshot
    root=Path(config["root"]);folder=root/"versions"/version
    current=read(folder/"FINALIZED.json")
    completed=[read(p) for p in sorted((root/"versions").glob("UG*/FINALIZED.json")) if p.parent.name<=version]
    previous=completed[-2] if len(completed)>1 else None
    base,recovery=recover_base(completed)
    triggers={}
    if previous:
        for target,new in current["teachers"].items():
            old=previous["teachers"][target]
            vectors=vector_change(root/old["representative"]/"trace.h5",root/new["representative"]/"trace.h5")
            triggers[target]=teacher_triggers(old,new,old["profiles"],new["profiles"],vectors)
    request=dict(schema="ug-main-review-1",version=version,current=current,previous=previous,
        best=choose([v["overall"] for v in completed]),R0={p.parent.name:read(p) for p in (root/"R0").glob("*/aggregate.json")},
        historical_unguided={p.parent.name:read(p)["matched50"] for p in (root/"baselines").glob("*/unguided.json")},
        strategy_contexts=current["strategy_contexts"],teacher_triggers=triggers,
        quality_diagnostics=snapshot(root,[g['id'] for g in current.get('groups',[])]),
        research_required=any(x["reasons"] for x in triggers.values()),base_checkpoint=base,
        recovery_required=recovery,no_best_rerun=True,next_version=None if version=="UG20" else f"UG{int(version[2:])+1:02d}",
        required_review_chain=["observation","evidence","mechanism_applicability","skill_edit","expected_behavior","next_version_falsifier"])
    write(folder/"AWAITING_GPT_REVIEW.json",request)
    return request


def accept_review(config,version,review_path,skills_dir):
    """The current Codex main agent supplies a review after reading actual evidence."""
    root=Path(config["root"]);folder=root/"versions"/version
    request=read(folder/"AWAITING_GPT_REVIEW.json")
    review=read(review_path)
    if review.get("version")!=version or review.get("base_checkpoint")!=request["base_checkpoint"]:
        raise ValueError("Review version/recovery base differs")
    if review.get("review_request_sha256")!=sha(folder/"AWAITING_GPT_REVIEW.json"):
        raise ValueError("Review is not bound to the finalized evidence")
    if set(review.get("read_strategy_contexts",[]))!=set(request["strategy_contexts"]):
        raise ValueError("All four strategy contexts must be reviewed")
    if len(review.get("patches",[])) not in [1,2,3]: raise ValueError("Update one to three skill modules")
    if request["research_required"] and not review.get("research"):
        raise ValueError("Stagnation requires real literature/code evidence")
    for source in review.get("research",[]):
        if not all(source.get(k) for k in ["url","version_or_commit","accessed_at","applicability","primary_source"]):
            raise ValueError("Research source lacks provenance/applicability")
    base=request["base_checkpoint"]
    base_dir=(root/"UG_INIT/skills") if base=="UG_INIT" else root/"versions"/base/"skills"
    before={p.name:sha(p) for p in base_dir.glob("*.md")}
    after={p.name:sha(p) for p in Path(skills_dir).glob("*.md")}
    changed={k for k in set(before)|set(after) if before.get(k)!=after.get(k)}
    names=[p["module"] for p in review["patches"]]
    if len(set(names))!=len(names) or set(names)!=changed or set(after)!=set(before):
        raise ValueError("Patch manifest must describe every changed skill; module set is fixed")
    for patch in review["patches"]:
        if not all(patch.get(k) for k in request["required_review_chain"]):
            raise ValueError("Incomplete evidence-to-skill mechanism chain")
    checkpoint=dict(review=review,base_checkpoint=base,recovery_derived=request["recovery_required"],
        skill_hashes=after,review_sha256=sha(review_path),next_version=request["next_version"])
    frozen(folder/"REVIEW_ACCEPTED.json",checkpoint)
    return checkpoint
