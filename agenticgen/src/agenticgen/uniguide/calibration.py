"""Four-stage shared scale calibration, at most 5,000 reserved attempts."""
import math
from pathlib import Path
from .io import read, write, frozen
from .protocol import coarse_scales, choose
from .prepare import load_model
from .sampler import execute


def pareto(rows):
    return [r for r in rows if not any(o["C"]<=r["C"] and o["gap"]<=r["gap"] and
        (o["C"]<r["C"] or o["gap"]<r["gap"]) for o in rows)]


def paired(rows,teachers):
    """Equal teacher weights. Missing/failed teachers never improve a shared score."""
    groups={}
    for row in rows:
        if row.get("complete"):
            groups.setdefault(tuple(row["schedule"]),{})[row["target"]]=row
    result=[]
    for schedule,items in groups.items():
        if set(items)!=set(teachers): continue
        result.append(dict(id="shared_"+"_".join(format(x,".8g") for x in schedule),schedule=list(schedule),
            C=sum(r["C"] for r in items.values())/len(items),gap=sum(r["gap"] for r in items.values())/len(items),
            best=sum(r["best"] for r in items.values())/len(items),complete=True,teachers=items))
    return result


def fine_profiles(coarse):
    successful=[r for r in coarse if r.get("complete")]
    if not successful: raise ValueError("No finite calibration evidence")
    indices=set()
    # First cover every nondominated neighborhood, then fill a shared grid.
    front=[]
    for target in sorted({r["target"] for r in successful}):
        front.extend(pareto([r for r in successful if r["target"]==target]))
    centers=[round(2*math.log2(r["schedule"][0])) for r in sorted(front,key=lambda r:(r["C"],r["gap"]))]
    order=[]
    for offset in [0,-1,1,-2,2]:
        for j in centers:
            if j+offset not in order: order.append(j+offset)
    low,high=min(centers)-4,max(centers)+4
    for k in range(33):
        j=round(low+(high-low)*k/32)
        if j not in order: order.append(j)
    old={r["schedule"][0] for r in coarse}
    values=[]
    for j in order:
        v=2.**(j/2)
        if any(math.isclose(v,o,rel_tol=1e-12) for o in old): continue
        values.append([v]*4)
        if len(values)==32: break
    return values


def select_anchors(rows,targets):
    candidates=paired(rows,targets)
    if len(candidates)<3: raise ValueError("At least three paired finite constants required")
    ranked=sorted(candidates,key=lambda r:(r["C"],r["gap"]))
    # Center on response-efficient constants while covering a two-octave range.
    chosen=[choose(candidates)]
    rest=sorted(candidates,key=lambda r:abs(math.log2(r["schedule"][0]/chosen[0]["schedule"][0])-2))
    for r in rest+ranked:
        if r not in chosen: chosen.append(r)
        if len(chosen)==3: break
    return [r["schedule"][0] for r in chosen]


def freeze_domain(full,all_constants,targets):
    constants=[r for r in paired(full,targets) if len(set(r["schedule"]))==1]
    if len(constants)<2: raise ValueError("Insufficient paired full-population constants")
    initial=choose(constants)["schedule"]
    supported=[]
    for target in targets:
        rows=[r for r in all_constants if r["target"]==target and r.get("complete")]
        supported.extend(r["schedule"][0] for r in pareto(rows))
    lo=math.floor(2*math.log2(min(supported)))-2
    hi=math.ceil(2*math.log2(max(supported)))+2
    center=round(2*math.log2(initial[0]))
    if not lo<center<hi: raise ValueError("Calibrated domain lacks actions on both sides")
    return dict(initial=initial,min_index=lo,max_index=hi,lower=2.**(lo/2),upper=2.**(hi/2),
                lattice="2^(j/2)",coarse_log2_step=1.,fine_log2_step=.5,
                derivation="union of teacher nondominated constant intervals plus one octave per side",
                constant_comparison=constants)


def run(config):
    import torch
    root=Path(config["root"]);folder=root/"calibration"
    teachers=read(root/"teachers.json");targets=[t["target"] for t in teachers]
    if (folder/"domain.json").exists(): return read(folder/"domain.json")
    all_rows=[]
    for phase,count in [("coarse",10),("fine",20),("windows",25),("full",50)]:
        path=folder/f"{phase}_profiles.json"
        if path.exists(): profiles=read(path)
        elif phase=="coarse": profiles=[[v]*4 for v in coarse_scales()]
        elif phase=="fine": profiles=fine_profiles(all_rows)
        elif phase=="windows":
            # Fine profiles are exactly on the formal grid; coarse constants are for discovery.
            fine=[r for r in all_rows if r["phase"]=="fine"]
            anchors=select_anchors(fine,targets)
            profiles=[]
            for a in anchors:
                for window in range(4):
                    for sign in [-1,1]:
                        s=[a]*4;s[window]=a*2.**(sign*.5);profiles.append(s)
        else:
            constants=paired([r for r in all_rows if r["phase"]=="fine"],targets)
            windowed=paired([r for r in all_rows if r["phase"]=="windows"],targets)
            profiles=[]
            for candidates in [constants,windowed]:
                # Ten shared response-efficient profiles from each family.
                pending=list(candidates)
                for _ in range(min(10,len(pending))):
                    winner=choose(pending);profiles.append(winner["schedule"]);pending.remove(winner)
            if not constants: raise ValueError("No shared finite fine-scale constant")
        frozen(path,profiles)
        if len(profiles)>{"coarse":26,"fine":32,"windows":24,"full":20}[phase]: raise ValueError("Calibration phase budget exceeded")
        phase_rows=[]
        for teacher in teachers:
            model=load_model(config,teacher)
            for index,schedule in enumerate(profiles):
                out=folder/phase/teacher["target"]/f"p{index:03d}"
                failure=out/"CALIBRATION_FAILURE.json"
                if failure.exists():
                    row=read(failure)
                else:
                    try: row=execute(model,config,teacher,schedule,out,"calibration",count)
                    except FloatingPointError as error:
                        row=dict(id=str(out.relative_to(root)),schedule=schedule,complete=False,error=str(error),attempts=count)
                        write(failure,row)
                        # This profile is failed, with its attempts charged. No weaker retry.
                row=dict(row,target=teacher["target"],phase=phase)
                phase_rows.append(row)
                print({"phase":phase,"target":teacher["target"],"profile":index,"complete":row["complete"]},flush=True)
            del model;torch.cuda.empty_cache()
        write(folder/f"{phase}_results.json",phase_rows)
        all_rows.extend(phase_rows)
    domain=freeze_domain([r for r in all_rows if r["phase"]=="full"],
                         [r for r in all_rows if len(set(r["schedule"]))==1],targets)
    frozen(folder/"domain.json",domain)
    write(folder/"COMPLETE.json",{"profiles":len(all_rows),"domain":domain})
    return domain
