"""Teacher-local empirical multipliers and bounded half-step log scales."""
import math
from pathlib import Path
from itertools import combinations, permutations
from .io import value_sha,sha
from .geometry_condition import DISTANCE, CONTROL_DISTANCE, NEIGHBOR_DISTANCE, MAX_ERROR
from .protocol import WINDOWS, BRANCHES, coarse_scales
from .scale_domain import PHASE_COUNTS, PHASE_LIMITS, GROWTH_LIMIT

POLICY="normalized-internal-distance-guidance-v1"
NORMALIZED="normalized_guidance_scale_schedule"
EFFECTIVE="effective_guidance_scale_schedule"
MULTIPLIER="calibrated_guidance_multiplier"


def source_binding():
    from .pipeline import source_binding as scientific_binding
    result=scientific_binding()
    entry=Path(__file__).resolve().parents[4]/"agenticgen/scripts/run_uniguide_geometry.py"
    result["entrypoint/run_uniguide_geometry.py"]=sha(entry)
    return result


def ticks(schedule):
    if len(schedule)!=4 or any(type(x) not in (int,float) or not math.isfinite(x) or not -10<=x<=10
            or abs(2*x-round(2*x))>1e-10 for x in schedule):
        raise ValueError("Four finite half-integer normalized scales within [-10,10] required")
    return tuple(int(x*2) for x in schedule)


def effective(schedule,multiplier,enabled=True):
    ticks(schedule)
    if type(multiplier) not in (int,float) or not math.isfinite(multiplier) or multiplier<=0:
        raise ValueError("Positive finite calibrated guidance multiplier required")
    if type(enabled) is not bool: raise ValueError("Guidance enabled must be a boolean")
    result=[multiplier*2.**x if enabled else 0. for x in schedule]
    if any(not math.isfinite(x) for x in result):raise ValueError("Effective strength overflow")
    return result


def expand_effective(schedule):
    if len(schedule)!=4 or any(type(x) not in (int,float) or not math.isfinite(x) or x<0 for x in schedule):
        raise ValueError("Invalid effective scale schedule")
    return [next((schedule[w] for w,(lo,hi) in enumerate(WINDOWS) if lo<=step<=hi),0.) for step in range(100)]


def choose(rows):
    if not rows or any(not r.get("complete") or not math.isfinite(r[DISTANCE]) for r in rows):
        raise ValueError("Selection requires complete finite internal-distance measurements")
    return min(rows,key=lambda r:(r[DISTANCE],max(r.get(EFFECTIVE,[0.])),r["id"]))


def displacement(row):
    windows=row["window_diagnostics"]
    if len(windows)!=4: raise ValueError("Missing window displacement evidence")
    out={key:max(w[key] for w in windows) for key in ("increment_vector_rms_A","max_atom_increment_A")}
    if any(not math.isfinite(x) or x<0 for x in out.values()): raise ValueError("Invalid displacement evidence")
    return out


def derive_domain(rows,teacher_id,condition_hash):
    full=[r for r in rows if r["teacher_id"]==teacher_id and r["phase"]=="full"]
    if any(r["condition_sha256"]!=condition_hash or r["attempts"]!=50 for r in full):
        raise ValueError("Incompatible teacher, condition or population in calibration")
    constants=sorted((r for r in full if len(set(r[EFFECTIVE]))==1),key=lambda r:r[EFFECTIVE][0])
    if len({r[EFFECTIVE][0] for r in constants})!=len(constants): raise ValueError("Repeated calibration constant")
    eligible=[];curve=[];previous=None;boundary=None
    for row in constants:
        scale=row[EFFECTIVE][0]
        if not math.isfinite(scale) or scale<=0: raise ValueError("Invalid actual calibration scale")
        entry=dict(source_id=row["id"],effective_scale=scale,complete=row["complete"])
        if not row["complete"]:
            boundary=boundary or dict(source_id=row["id"],reason="failed_constant")
        else:
            absolute=displacement(row);growth={}
            if previous:
                ratio=scale/previous["effective_scale"]
                growth={k:absolute[k]/max(previous["absolute_displacement"][k]*ratio,1e-12) for k in absolute}
                if ratio>2.+1e-10 or max(growth.values())>GROWTH_LIMIT:
                    boundary=boundary or dict(source_id=row["id"],reason="sparse_evidence_or_abrupt_absolute_growth",growth=growth)
            entry.update(absolute_displacement=absolute,normalized_growth=growth)
            if boundary is None: eligible.append(row)
            previous=entry
        entry["eligible"]=row in eligible;curve.append(entry)
    if len(eligible)<3: raise ValueError("At least three contiguous eligible full-population constants required")
    winner=choose(eligible);multiplier=winner[EFFECTIVE][0]
    lower=max(-10.,math.log2(eligible[0][EFFECTIVE][0]/multiplier))
    upper=min(10.,math.log2(eligible[-1][EFFECTIVE][0]/multiplier))
    lower=math.ceil(lower*2-1e-9)/2;upper=math.floor(upper*2+1e-9)/2
    body=dict(policy=POLICY,teacher_id=teacher_id,condition_sha256=condition_hash,
        calibrated_guidance_multiplier=multiplier,normalized_minimum=lower,normalized_maximum=upper,
        initial_normalized_guidance_scale_schedule=[0.]*4,initial_source=winner["id"],
        bootstrap_sources=[r["id"] for r in eligible],source_rows_sha256=value_sha(full),
        constant_comparison=curve,risk_boundary=boundary,stability_guarantee=False,
        ranking="minimum internal-distance root mean square deviation, then smaller effective scale, then source identity")
    return dict(body,domain_sha256=value_sha(body))


def validate_domain(domain,teacher_id=None,condition_hash=None):
    body={k:v for k,v in domain.items() if k!="domain_sha256"}
    if domain.get("policy")!=POLICY or value_sha(body)!=domain.get("domain_sha256"):
        raise ValueError("Invalid normalized scale protocol binding")
    if teacher_id is not None and domain["teacher_id"]!=teacher_id: raise ValueError("Another teacher's multiplier")
    if condition_hash is not None and domain["condition_sha256"]!=condition_hash: raise ValueError("Another condition's multiplier")
    effective([0.]*4,domain[MULTIPLIER])
    if not -10<=domain["normalized_minimum"]<=0<=domain["normalized_maximum"]<=10:
        raise ValueError("Invalid calibrated interval")


def validate_action(action,anchors,domain,seen):
    validate_domain(domain)
    if action["anchor_id"] not in anchors: raise ValueError("Unknown measured action anchor")
    old=ticks(anchors[action["anchor_id"]][NORMALIZED]);new=ticks(action[NORMALIZED])
    if any(not domain["normalized_minimum"]<=x/2<=domain["normalized_maximum"] for x in new):
        raise ValueError("Outside teacher's measured scale interval")
    if action["mode"] not in ("fine","coarse"): raise ValueError("Unknown step mode")
    changes=[b-a for a,b in zip(old,new) if b!=a];size=1 if action["mode"]=="fine" else 2
    if not 1<=len(changes)<=2 or any(abs(x)!=size for x in changes): raise ValueError("Invalid normalized step")
    branch=action["branch"]
    if branch not in BRANCHES or (branch=="increase" and not all(x>0 for x in changes)) or (
            branch=="decrease" and not all(x<0 for x in changes)) or (
            branch=="redistribute" and (len(changes)!=2 or sum(changes)!=0)):
        raise ValueError("Incorrect inclination")
    if new in {ticks(s) for s in seen}: raise ValueError("Configuration already measured or proposed")
    return action[NORMALIZED]


def legal_options(history,domain,branch,limit=2):
    anchors={r["id"]:r for r in history};seen=[r[NORMALIZED] for r in history];options=[]
    for anchor in sorted(history,key=lambda r:(r[DISTANCE],r["id"])):
        for mode,step in [("fine",.5),("coarse",1.)]:
            moves=(list(permutations(range(4),2)) if branch=="redistribute" else
                   [i for count in (1,2) for i in combinations(range(4),count)])
            for indices in moves:
                schedule=list(anchor[NORMALIZED])
                for n,index in enumerate(indices):
                    schedule[index]+=step*(-1 if branch=="decrease" or branch=="redistribute" and n==1 else 1)
                action=dict(anchor_id=anchor["id"],mode=mode,branch=branch,**{NORMALIZED:schedule})
                try: validate_action(action,anchors,domain,seen+[r[NORMALIZED] for r in options])
                except ValueError: continue
                options.append(action)
                if len(options)==limit:return options
    return options


def profiles(rows,teacher_id,phase):
    own=[r for r in rows if r["teacher_id"]==teacher_id and r["complete"]]
    if phase=="coarse":return [[x]*4 for x in coarse_scales()]
    if phase=="fine":
        coarse=sorted((r for r in own if r["phase"]=="coarse"),key=lambda r:(r[DISTANCE],r["id"]))
        if not coarse:raise ValueError("No finite coarse evidence")
        centers=[round(2*math.log2(r[EFFECTIVE][0])) for r in coarse[:3]]
        order=[]
        for offset in [0,-1,1,-2,2]+list(range(-8,9)):
            for center in centers:
                if center+offset not in order:order.append(center+offset)
        old={r[EFFECTIVE][0] for r in coarse}
        return [[2.**(j/2)]*4 for j in order if not any(math.isclose(2.**(j/2),v,rel_tol=1e-12) for v in old)][:PHASE_LIMITS[phase]]
    fine=sorted((r for r in own if r["phase"]=="fine"),key=lambda r:(r[DISTANCE],r["id"]))
    if len(fine)<3:raise ValueError("Insufficient fine calibration")
    if phase=="windows":
        out=[]
        for row in fine[:3]:
            for index in range(4):
                for sign in (-1,1):
                    schedule=list(row[EFFECTIVE]);schedule[index]*=2.**(sign/2);out.append(schedule)
        return out
    if phase=="full":
        windows=sorted((r for r in own if r["phase"]=="windows"),key=lambda r:(r[DISTANCE],r["id"]))
        # Full constants include nearby scales, so eligibility is not decided across sparse gaps.
        best=fine[0][EFFECTIVE][0]
        constants=sorted(fine,key=lambda r:(abs(math.log2(r[EFFECTIVE][0]/best)),r[DISTANCE],r["id"]))[:10]
        return [r[EFFECTIVE] for r in constants+windows[:10]]
    raise ValueError("Unknown calibration phase")
