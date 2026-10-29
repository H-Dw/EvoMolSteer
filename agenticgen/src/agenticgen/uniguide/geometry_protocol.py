"""Frozen teacher/condition/sampler multipliers and executor-owned integer actions."""
import math
from pathlib import Path
from itertools import combinations, permutations
from .io import value_sha,sha
from .geometry_condition import DISTANCE, CONTROL_DISTANCE, NEIGHBOR_DISTANCE, MAX_ERROR
from .protocol import BRANCHES
from .geometry_dynamics import WINDOWS
from .scale_domain import GROWTH_LIMIT
from .geometry_reward import WEIGHTS

POLICY="template-relative-distance-guidance-v6-summed-error"
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
            or x!=round(x) for x in schedule):
        raise ValueError("Four finite integer normalized scales within [-10,10] required")
    return tuple(int(x) for x in schedule)


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
    if not rows or any(not r.get("complete") or not math.isfinite(r[DISTANCE]) or not math.isfinite(primary_score(r)) for r in rows):
        raise ValueError("Selection requires complete finite internal-distance measurements")
    return min(rows,key=lambda r:(primary_score(r),max(r.get(EFFECTIVE,[0.])),r["id"]))


def primary_score(row):
    value=row.get(CONTROL_DISTANCE) if WEIGHTS in row else row[DISTANCE]
    return row[DISTANCE] if value is None else value


def action_weights(action,anchors):
    anchor=anchors[action["anchor_id"]]
    chosen=action.get(WEIGHTS,anchor.get(WEIGHTS))
    if WEIGHTS in action and not action.get("loss_control_id"):
        raise ValueError("Loss weights require a registered control receipt")
    return chosen


def seen_configuration(schedule,loss_weights):
    return {NORMALIZED:schedule,WEIGHTS:loss_weights} if loss_weights is not None else schedule


def displacement(row):
    windows=row["window_diagnostics"]
    if len(windows)!=4: raise ValueError("Missing window displacement evidence")
    # A limiter must not hide a runaway raw gradient from calibration risk checks.
    out={key:max(w.get("unconstrained_"+key,w[key]) for w in windows)
         for key in ("increment_vector_rms_A","max_atom_increment_A")}
    if any(not math.isfinite(x) or x<0 for x in out.values()): raise ValueError("Invalid displacement evidence")
    return out


def derive_domain(rows,teacher_id,condition_hash,sampling_protocol_sha256):
    full=[r for r in rows if r["teacher_id"]==teacher_id and r["phase"]=="full"]
    if any(r["condition_sha256"]!=condition_hash or r["attempts"]!=50 or
           r["compatibility_sha256"]!=sampling_protocol_sha256 for r in full):
        raise ValueError("Incompatible teacher, condition, sampling protocol or population in calibration")
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
    if winner in (eligible[0],eligible[-1]):
        raise ValueError("Calibration winner is on the measured boundary; no automatic range expansion")
    lower=max(-10.,math.log2(eligible[0][EFFECTIVE][0]/multiplier))
    upper=min(10.,math.log2(eligible[-1][EFFECTIVE][0]/multiplier))
    lower=math.ceil(lower-1e-9);upper=math.floor(upper+1e-9)
    body=dict(policy=POLICY,teacher_id=teacher_id,condition_sha256=condition_hash,
        sampling_protocol_sha256=sampling_protocol_sha256,
        calibration_key=value_sha([teacher_id,condition_hash,sampling_protocol_sha256]),
        calibrated_guidance_multiplier=multiplier,normalized_minimum=lower,normalized_maximum=upper,
        initial_normalized_guidance_scale_schedule=[0.]*4,initial_source=winner["id"],
        bootstrap_sources=[r["id"] for r in eligible],source_rows_sha256=value_sha(full),
        constant_comparison=curve,risk_boundary=boundary,stability_guarantee=False,
        ranking="minimum condition-internal atom-pair distance RMSD (all constrained pairs only when internal pairs are absent), then smaller effective scale, then source identity")
    return dict(body,domain_sha256=value_sha(body))


def validate_domain(domain,teacher_id=None,condition_hash=None,sampling_protocol_sha256=None):
    body={k:v for k,v in domain.items() if k!="domain_sha256"}
    if domain.get("policy")!=POLICY or value_sha(body)!=domain.get("domain_sha256"):
        raise ValueError("Invalid normalized scale protocol binding")
    if teacher_id is not None and domain["teacher_id"]!=teacher_id: raise ValueError("Another teacher's multiplier")
    if condition_hash is not None and domain["condition_sha256"]!=condition_hash: raise ValueError("Another condition's multiplier")
    if sampling_protocol_sha256 is not None and domain["sampling_protocol_sha256"]!=sampling_protocol_sha256:
        raise ValueError("Another sampling protocol's multiplier; new calibration required")
    if domain.get("calibration_key")!=value_sha([domain["teacher_id"],domain["condition_sha256"],domain.get("sampling_protocol_sha256")]):
        raise ValueError("Invalid frozen calibration identity")
    effective([0.]*4,domain[MULTIPLIER])
    if not -10<=domain["normalized_minimum"]<=0<=domain["normalized_maximum"]<=10:
        raise ValueError("Invalid calibrated interval")


def validate_action(action,anchors,domain,seen):
    validate_domain(domain)
    if action["anchor_id"] not in anchors: raise ValueError("Unknown measured action anchor")
    if any(k in action for k in (NORMALIZED,EFFECTIVE,MULTIPLIER,"schedule","mode")):
        raise ValueError("Agent must provide integer updates, not executor-owned arrays or strengths")
    old=ticks(anchors[action["anchor_id"]][NORMALIZED]);new=list(old)
    chosen=action_weights(action,anchors)
    if chosen is not None:
        from .geometry_reward import KEYS
        if set(chosen)!=set(KEYS) or any(type(chosen[k]) is not int or chosen[k] not in (0,1) for k in KEYS):
            raise ValueError("Binary loss weights required")
    changed_weights=chosen!=anchors[action["anchor_id"]].get(WEIGHTS)
    updates=action.get("updates")
    if not isinstance(updates,list) or not (0 if changed_weights else 1)<=len(updates)<=2:raise ValueError("Window updates or a receipt-bound loss change required")
    windows=[];changes=[]
    for update in updates:
        if not isinstance(update,dict) or set(update)!={"window_id","step"}:raise ValueError("Invalid update fields")
        window,step=update["window_id"],update["step"]
        if type(window) is not int or not 0<=window<4 or type(step) is not int or step not in (-2,-1,1,2):
            raise ValueError("Window IDs and nonzero step changes must be integers; allowed steps are -2,-1,1,2")
        if window in windows:raise ValueError("Repeated window update")
        windows.append(window);changes.append(step);new[window]+=step
    new=ticks(new)
    if any(not domain["normalized_minimum"]<=x<=domain["normalized_maximum"] for x in new):
        raise ValueError("Outside teacher's measured scale interval")
    if changes and len({abs(x) for x in changes})!=1:raise ValueError("Use the same integer step size for changed windows")
    branch=action["branch"]
    if branch not in BRANCHES or (branch=="increase" and not all(x>0 for x in changes)) or (
            branch=="decrease" and not all(x<0 for x in changes)) or (
            branch=="redistribute" and changes and (len(changes)!=2 or sum(changes)!=0)):
        raise ValueError("Incorrect inclination")
    for prior in seen:
        schedule=prior[NORMALIZED] if isinstance(prior,dict) else prior
        prior_weights=prior.get(WEIGHTS) if isinstance(prior,dict) else None
        if new==ticks(schedule) and (chosen is None or prior_weights==chosen):
            raise ValueError("Configuration already measured or proposed")
    return list(new)


def resolve_action(action,anchors,domain,seen):
    schedule=validate_action(action,anchors,domain,seen)
    return dict(action,**{NORMALIZED:schedule,MULTIPLIER:domain[MULTIPLIER],
        EFFECTIVE:effective(schedule,domain[MULTIPLIER])},guidance_enabled=True,
        **({WEIGHTS:action_weights(action,anchors)} if action_weights(action,anchors) is not None else {}),
        calibration_key=domain["calibration_key"],domain_sha256=domain["domain_sha256"],
        executor_resolved=True,agent_action_sha256=value_sha(action))


def legal_options(history,domain,branch,limit=2):
    anchors={r["id"]:r for r in history};seen=[seen_configuration(r[NORMALIZED],r.get(WEIGHTS)) for r in history];options=[]
    for anchor in sorted(history,key=lambda r:(primary_score(r),r["id"])):
        for step in (1,2):
            moves=(list(permutations(range(4),2)) if branch=="redistribute" else
                   [i for count in (1,2) for i in combinations(range(4),count)])
            for indices in moves:
                action=dict(anchor_id=anchor["id"],branch=branch,updates=[dict(window_id=index,
                    step=step*(-1 if branch=="decrease" or branch=="redistribute" and n==1 else 1)) for n,index in enumerate(indices)])
                try: validate_action(action,anchors,domain,seen+[seen_configuration(validate_action(r,anchors,domain,[]),action_weights(r,anchors)) for r in options])
                except ValueError: continue
                options.append(action)
                if len(options)==limit:return options
    return options
