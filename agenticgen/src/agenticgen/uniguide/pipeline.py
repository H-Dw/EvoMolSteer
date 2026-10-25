"""UG01--UG20 execution; each finalized version yields to the current GPT main agent."""
import shutil
import subprocess
from pathlib import Path
from .io import read,write,sha,value_sha,frozen
from .protocol import BRANCHES,choose
from .prepare import load_model
from .sampler import execute
from .council import advise
from .quality import assess,retention
from .review import build_review


def source_binding():
    package=Path(__file__).resolve().parents[1]
    modules=[*Path(__file__).parent.glob("*.py"),package/"runtime/scoring.py",package/"runtime/rng.py",
             * (package/"providers").glob("*.py")]
    return {str(p.relative_to(package)):sha(p) for p in sorted(set(modules))}


def skill_location():
    return Path(__file__).parent/"skills"


def init_strategy(config):
    root=Path(config["root"])
    source=read(Path(__file__).parent/"strategy_provenance.json")
    if source["source_commit"]!=config["initial_strategy_commit"]: raise ValueError("Wrong initial strategy origin")
    folder=root/"UG_INIT/skills";folder.mkdir(parents=True,exist_ok=True)
    for p in skill_location().glob("*.md"):
        dest=folder/p.name
        if dest.exists() and sha(dest)!=sha(p): raise ValueError("UG_INIT cannot be rewritten")
        if not dest.exists(): shutil.copyfile(p,dest)
    frozen(root/"UG_INIT/manifest.json",dict(id="UG_INIT",strategy_source=source,contains_old_scores=False,
        skills={p.name:sha(p) for p in folder.glob("*.md")}))


def run_r0(config,teachers,domain):
    import torch
    root=Path(config["root"])
    for teacher in teachers:
        folder=root/"R0"/teacher["target"]
        if (folder/"COMPLETE.json").exists():
            assess(folder,teacher)
            continue
        model=load_model(config,teacher)
        execute(model,config,teacher,domain["initial"],folder,"r0")
        assess(folder,teacher)
        del model;torch.cuda.empty_cache()


def freeze_version(config,version,domain):
    root=Path(config["root"]);folder=root/"versions"/version
    index=int(version[2:])
    if not 1<=index<=20: raise ValueError("UG01--UG20 only")
    if index==1:
        base="UG_INIT";recovery=False;parent=None
        sources=root/"UG_INIT/skills"
    else:
        previous=f"UG{index-1:02d}"
        approved=read(root/"versions"/previous/"REVIEW_ACCEPTED.json")
        if approved["next_version"]!=version: raise ValueError("Incorrect review successor")
        base=approved["base_checkpoint"];recovery=approved["recovery_derived"];parent=previous
        sources=skill_location()
        if {p.name:sha(p) for p in sources.glob("*.md")}!=approved["skill_hashes"]:
            raise ValueError("Deployed skills differ from accepted patch")
    skills=folder/"skills";skills.mkdir(parents=True,exist_ok=True)
    for p in sources.glob("*.md"):
        dest=skills/p.name
        if dest.exists() and sha(dest)!=sha(p): raise ValueError("Skills changed within version")
        if not dest.exists(): shutil.copyfile(p,dest)
    repo=Path(__file__).resolve().parents[4]
    commit=subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip()
    manifest=dict(id=version,parent=parent,base_checkpoint=base,recovery_derived=recovery,
        source_binding=source_binding(),code_commit=commit,config_sha256=value_sha(config),domain_sha256=value_sha(domain),
        skills={p.name:sha(p) for p in skills.glob("*.md")},llm=config["llm"],dynamic_memory_reset=True)
    frozen(folder/"manifest.json",manifest)
    return manifest


def run_version(config,version):
    import torch
    root=Path(config["root"])
    acceptance=read(root/"acceptance/COMPLETE.json")
    if acceptance["source_binding"]!=source_binding(): raise ValueError("Implementation changed after remote acceptance")
    domain=read(root/"calibration/domain.json");teachers=read(root/"teachers.json")
    run_r0(config,teachers,domain)
    folder=root/"versions"/version
    if (folder/"FINALIZED.json").exists(): return build_review(config,version)
    manifest=freeze_version(config,version,domain)
    teacher_summaries={};contexts=[];aggregates=[]
    for teacher in teachers:
        target=teacher["target"];troot=folder/target
        r0=read(root/"R0"/target/"aggregate.json")
        history=[r0];groups=[]
        model=load_model(config,teacher)
        for round_number in range(1,6):
            # R0 bootstrap, then latest measured private action anchors; no cross-version task memory.
            proposals=advise(config,teacher,version,round_number,history,domain,folder/"skills",troot/f"advice_R{round_number:02d}")
            for p in proposals:
                if source_binding()!=manifest["source_binding"]: raise ValueError("Executable code changed mid-version")
                out=troot/f"R{round_number:02d}"/p["branch"]
                frozen(out/"proposal.json",p)
                row=execute(model,config,teacher,p["schedule"],out,"formal",resume_path=root/"R0"/target/"checkpoint_010.pt.gz")
                row=dict(row,branch=p["branch"],round=round_number,private_context=p["private_context"],synthesis_context=p["synthesis_context"])
                history.append(row);groups.append(row)
                write(troot/"history.json",history)
            # Nothing about a future R6 is generated.
        advise(config,teacher,version,5,history,domain,folder/"skills",troot/"analysis_R05",final=True)
        for group in groups:
            context_folder=troot/("analysis_R05" if group["round"]==5 else f"advice_R{group['round']+1:02d}")
            group["private_context"]=str(context_folder/(group["branch"]+".json"))
            group["synthesis_context"]=str(context_folder/"synthesis.json")
        for group in groups: assess(root/group["id"],teacher)
        winner=choose(groups)
        retention(root,teacher,groups,winner,troot/"retention")
        contexts.extend([winner["private_context"],winner["synthesis_context"]])
        teacher_summaries[target]=dict(C=winner["C"],gap=winner["gap"],best=winner["best"],
            representative=winner["id"],profiles=[r["schedule"] for r in groups],
            sensitivity={str(b):choose(groups,b)["id"] for b in [.01,.02,.05]},complete=True)
        aggregates.extend(groups)
        del model;torch.cuda.empty_cache()
    if len(aggregates)!=30 or len(contexts)!=4: raise ValueError("Incomplete version")
    overall=dict(id=version,complete=True,**{k:sum(v[k] for v in teacher_summaries.values())/len(teachers) for k in ["C","gap","best"]})
    result=dict(id=version,overall=overall,teachers=teacher_summaries,groups=aggregates,strategy_contexts=contexts,
        manifest_sha256=sha(folder/"manifest.json"),recovery_derived=manifest["recovery_derived"],
        context_hashes={p:sha(p) for p in contexts},quality_and_retention_complete=True)
    frozen(folder/"FINALIZED.json",result)
    return build_review(config,version)
