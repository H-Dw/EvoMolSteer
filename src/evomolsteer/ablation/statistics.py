"""Paired RNG-batch / role-policy uncertainty and explicit pruning decisions."""
import numpy as np
import pandas as pd


def _matrix(frames):
    if not frames:raise ValueError("No measured policy outcomes")
    batches=sorted(frames[0].batch.unique())
    keys=set(map(tuple,frames[0][["batch","slot"]].to_numpy()))
    matrices=[]
    for frame in frames:
        if frame.duplicated(["batch","slot"]).any() or set(map(tuple,frame[["batch","slot"]].to_numpy()))!=keys:
            raise ValueError("Incomplete paired attempt identifiers")
        score=pd.to_numeric(frame.pic50_on_rescore,errors="coerce").to_numpy()
        if not np.isfinite(score).all():raise ValueError("All-attempt score coverage required; missing scores are not zero")
        matrices.append(frame.assign(pic50_on_rescore=score).groupby("batch").pic50_on_rescore.mean().reindex(batches).to_numpy())
    return np.asarray(matrices),batches


def compare(left, right, *, family=1, replicates=20000, seed=42, paired_roles=False):
    """Same molecular RNG batches; shared rewards do not add generation units.

    Analyst role samples are independent between conditions. Designer chains
    can be paired by the identical frozen Analyst input; the caller declares it.
    """
    a,batches=_matrix(left);b,other=_matrix(right)
    if set(map(tuple,left[0][["batch","slot"]].to_numpy())) != set(map(tuple,right[0][["batch","slot"]].to_numpy())):
        raise ValueError("Unmatched paired attempt identifiers")
    if batches!=other or len(batches)<2:raise ValueError("Matched independent RNG batches required")
    if paired_roles and len(a)!=len(b):raise ValueError("Frozen input chain coverage mismatch")
    if family<1 or replicates<1000:raise ValueError("Valid confidence family and bootstrap budget required")
    rng=np.random.default_rng(seed)
    boot=[]
    for begin in range(0,replicates,500):
        n=min(500,replicates-begin)
        bi=rng.integers(0,len(batches),(n,len(batches)))
        ai=rng.integers(0,len(a),(n,len(a)))
        ri=ai if paired_roles else rng.integers(0,len(b),(n,len(b)))
        av=a[ai[:,:,None],bi[:,None,:]].mean((1,2))
        bv=b[ri[:,:,None],bi[:,None,:]].mean((1,2))
        boot.append(av-bv)
    boot=np.concatenate(boot)
    alpha=.05/family
    point=float(a.mean()-b.mean())
    return {"delta_pic50":point,"CI95":np.quantile(boot,[.025,.975]).tolist(),
        "simultaneous_CI":np.quantile(boot,[alpha/2,1-alpha/2]).tolist(),
        "family":family,"independent_generation_batches":len(batches),
        "left_role_chains":len(a),"right_role_chains":len(b),"paired_frozen_Analyst_inputs":paired_roles,
        "batch_deltas":dict(zip(map(str,batches),(a.mean(0)-b.mean(0)).tolist())),
        "bootstrap_replicates":replicates,
        "interpretation":"Conditional role-policy and RNG-batch comparison; repeated numerical programs share biological observations"}


def summarize(frames, elite_threshold):
    # Equal role-chain policy weighting, not a concatenated independent N.
    summaries=[]
    for f in frames:
        valid=f.valid_connected.fillna(False).astype(bool)
        passed=f.pb_fast_pass.fillna(False).astype(bool)
        energy=pd.to_numeric(f.loc[f.energy_status.eq("converged"),"mmff_relief_per_heavy"],errors="coerce").dropna()
        summaries.append({"mean_pic50":float(f.pic50_on_rescore.mean()),"p95_pic50":float(f.pic50_on_rescore.quantile(.95)),
            "valid_rate":float(valid.mean()),"PB_rate":float(passed.mean()),
            "strain_median":float(energy.median()) if len(energy) else None,
            "strain_p90":float(energy.quantile(.9)) if len(energy) else None,
            "energy_coverage":len(energy)/len(f),
            "elite_valid_rate":float((valid & (f.pic50_on_rescore>=elite_threshold)).mean()),
            "valid_unique_graphs":int(f.loc[valid,"smiles"].nunique()) if "smiles" in f else None})
    return {"role_chain_count":len(frames),"attempts_per_chain":len(frames[0]),
        "policy_mean":{k:float(np.mean([s[k] for s in summaries])) if all(s[k] is not None for s in summaries) else None for k in summaries[0]},
        "chains":summaries,"score_coverage":1.,"no_chemical_graph_gate":True}


def quality_envelope(candidate, m2, *, strain_margin=.10, rate_margin=.02):
    c,m=candidate["policy_mean"],m2["policy_mean"]
    reasons=[]
    if c["valid_rate"]<m["valid_rate"]-rate_margin:reasons.append("validity_below_envelope")
    if c["PB_rate"]<m["PB_rate"]-rate_margin:reasons.append("PB_below_envelope")
    for key in ["strain_median","strain_p90"]:
        if c[key] is None or m[key] is None:reasons.append(key+"_unobserved")
        elif c[key]>m[key]*(1+strain_margin):reasons.append(key+"_worse")
    if c["energy_coverage"]<m["energy_coverage"]-rate_margin:reasons.append("energy_coverage_below_envelope")
    return {"passed":not reasons,"reasons":reasons,"scope":"Secondary point-estimate acceptance envelope, not significance of strain effects"}


def pruning_decision(drop_vs_full, drop_vs_m2, envelope, *, margin=.02, valid_chains=True):
    if not valid_chains:return {"adopt":False,"status":"INTERFACE_INCOMPLETE"}
    safe=(drop_vs_full["simultaneous_CI"][0]>-margin and drop_vs_m2["simultaneous_CI"][0]>-margin and envelope["passed"])
    if safe:
        band=drop_vs_full["simultaneous_CI"]
        return {"adopt":True,"status":"CONFIRMED_CONTEXTUAL_REDUNDANCY" if -margin<band[0] and band[1]<margin else "CONFIRMED_NONINFERIOR_PRUNING"}
    return {"adopt":False,"status":"NEGATIVE_OR_INCONCLUSIVE_KEEP_CURRENT"}


def classify_module(full_minus_drop, full_minus_alternative, *, functional_valid, effect=.05):
    if not functional_valid:return "FUNCTIONAL_OR_INTERFACE_BOTTLENECK"
    if full_minus_drop["simultaneous_CI"][0]>effect and full_minus_alternative["simultaneous_CI"][0]>effect:
        return "AFFINITY_NECESSARY_IN_THIS_CONTEXT"
    if full_minus_drop["simultaneous_CI"][1]<0:
        return "NEGATIVE_IN_THIS_CONTEXT"
    if all(-.02<x["simultaneous_CI"][0] and x["simultaneous_CI"][1]<.02 for x in [full_minus_drop,full_minus_alternative]):
        return "PERFORMANCE_EQUIVALENT_IN_THIS_CONTEXT"
    return "INCONCLUSIVE"
