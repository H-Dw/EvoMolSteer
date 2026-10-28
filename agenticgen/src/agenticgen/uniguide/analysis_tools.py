"""Read-only measured evidence, paired statistics and original action validation."""
import math
import re
from pathlib import Path
import numpy as np

from .io import read, frozen, value_sha, sha
from .geometry_condition import DISTANCE, CONTROL_DISTANCE, NEIGHBOR_DISTANCE, MAX_ERROR
from .geometry_protocol import NORMALIZED, MULTIPLIER, validate_action, effective

METRICS = (CONTROL_DISTANCE, NEIGHBOR_DISTANCE, DISTANCE, MAX_ERROR)
STATISTICS_POLICY = dict(id="geometry-descriptive-statistics-v1", trim_fraction_per_tail=.1,
    bootstrap_seed=721904, bootstrap_resamples=1000, interval_level=.95,
    bootstrap_unit="paired_seed_cluster", ranking_change=False, thresholds_invented=False)


class EvidenceIntegrityError(RuntimeError):
    """Source corruption is terminal, never a recoverable model argument error."""


def metric_summary(values):
    finite = [float(x) for x in values if type(x) in (float,int) and math.isfinite(x)]
    if not finite:
        return dict(status="unmeasured", measured=0, expected=len(values))
    ordered = np.sort(finite);cut=int(len(ordered)*STATISTICS_POLICY["trim_fraction_per_tail"])
    middle=ordered[cut:len(ordered)-cut]
    median=float(np.median(ordered))
    return dict(status="complete" if len(finite)==len(values) else "incomplete_descriptive_only",
        measured=len(finite), expected=len(values), arithmetic_mean=float(np.mean(ordered)),
        median=median, symmetric_ten_percent_trimmed_mean=float(np.mean(middle)),
        median_absolute_deviation=float(np.median(np.abs(ordered-median))),
        percentile_90=float(np.quantile(ordered,.9)), percentile_95=float(np.quantile(ordered,.95)),
        maximum=float(ordered[-1]), worst_five_mean=float(np.mean(ordered[-5:])),
        observations_retained=len(middle), unit="angstrom")


def paired_summary(left,right):
    identity=lambda r:(r["logical_seed"],r["logical_slot"])
    a,b={identity(r):r for r in left},{identity(r):r for r in right}
    if len(a)!=len(left) or len(b)!=len(right) or set(a)!=set(b):
        return dict(status="unavailable", reason="Incomplete or repeated paired slot identities")
    keys=sorted(a);seeds=sorted({k[0] for k in keys});result={}
    for metric in METRICS:
        x=[a[k].get(metric) for k in keys];y=[b[k].get(metric) for k in keys]
        if any(type(v) not in (int,float) or not math.isfinite(v) for v in x+y):
            result[metric]=dict(status="unavailable", reason="Missing/nonfinite paired measurement");continue
        x,y=np.asarray(x),np.asarray(y);delta=y-x
        center=lambda v:metric_summary(v.tolist())["symmetric_ten_percent_trimmed_mean"]
        per_seed=[dict(logical_seed=s, aggregation_scope="mean of all paired slots in this seed; not a single molecule",
            paired_slots=sum(k[0]==s for k in keys),
            arithmetic_mean_change=float(np.mean(delta[[k[0]==s for k in keys]]))) for s in seeds]
        rng=np.random.default_rng(STATISTICS_POLICY["bootstrap_seed"])
        clusters={s:[i for i,k in enumerate(keys) if k[0]==s] for s in seeds}
        replicates=[]
        if len(seeds)>=2:
            for _ in range(STATISTICS_POLICY["bootstrap_resamples"]):
                indices=[i for s in rng.choice(seeds,size=len(seeds),replace=True) for i in clusters[s]]
                replicates.append(center(y[indices])-center(x[indices]))
        result[metric]=dict(status="complete", arithmetic_mean_change=float(np.mean(delta)),
            trimmed_mean_change=center(y)-center(x), improved=int(sum(delta<0)),
            worsened=int(sum(delta>0)), unchanged=int(sum(delta==0)), seed_mean_changes=per_seed,
            exploratory_paired_cluster_interval=(np.quantile(replicates,[.025,.975]).tolist() if replicates else None),
            inference="fixed adaptive panel; few clusters; no independent validation or significance claim")
    return dict(status="complete", paired_slots=len(keys), seed_clusters=len(seeds),
        sign="candidate minus reference; negative distance change is improvement", metrics=result)


def object_schema(properties):
    return dict(type="object",properties=properties,required=list(properties),additionalProperties=False)


class EvidenceTools:
    def __init__(self, root, history, domain, folder):
        self.root=Path(root).resolve();self.history={r["id"]:r for r in history}
        self.domain=domain;self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=True)
        self.receipt_ids=set();self.sources={};self.records={};self.paths={}
        for run_id,row in self.history.items():
            path=(self.root/run_id).resolve()
            if not path.is_relative_to(self.root):raise ValueError("Run ID escapes experiment")
            self.paths[run_id]=path
            complete=path/"COMPLETE.json"
            if not complete.exists():
                self.sources[run_id]=dict(status="unavailable",aggregate_snapshot_sha256=value_sha(row));continue
            receipt=read(complete);hashes={}
            for name in ("records.json","aggregate.json","manifest.json","steps.json"):
                if name not in receipt["files"] or not (path/name).is_file():raise ValueError("Incomplete measurement provenance")
                hashes[name]=sha(path/name)
                if hashes[name]!=receipt["files"][name]:raise ValueError("Measured source checksum mismatch")
            actual=read(path/"aggregate.json")
            if any(actual.get(k)!=row.get(k) for k in ("id","attempts","teacher_id","condition_sha256","compatibility_sha256","fingerprint",*METRICS,"window_diagnostics")):
                raise ValueError("History and measured source differ")
            if any(actual.get(k)!=domain.get(k) for k in ("teacher_id","condition_sha256")):
                raise ValueError("Evidence belongs to a different teacher/condition")
            manifest=read(path/"manifest.json")
            identity=manifest["identity"]
            if value_sha(identity["compatibility"])!=row["compatibility_sha256"]:
                raise ValueError("Sampling compatibility source mismatch")
            records=read(path/"records.json")
            if len(records)!=row["attempts"] or len({(r["logical_seed"],r["logical_slot"]) for r in records})!=len(records):
                raise ValueError("Measured population count/identity mismatch")
            self.records[run_id]=records
            self.sources[run_id]=dict(status="available",files=hashes,completion_sha256=sha(complete),
                compatibility_sha256=row["compatibility_sha256"],fingerprint=row["fingerprint"])
        ids=dict(type="string",enum=list(self.history))
        action=object_schema(dict(anchor_id=ids,branch=dict(type="string",enum=["increase","decrease","redistribute"]),
            updates=dict(type="array",minItems=1,maxItems=2,items=object_schema(dict(
                window_id=dict(type="integer",minimum=0,maximum=3),step=dict(type="integer",enum=[-2,-1,1,2]))))))
        specs=[("summarize_population","Read measured full-population geometry, robust center, failures and tail identities.",object_schema(dict(run_id=ids))),
            ("compare_paired_runs","Compare measured candidate to reference on identical seed/slot and sampling protocol; no unmeasured score.",object_schema(dict(reference_id=ids,candidate_id=ids))),
            ("inspect_interval_evidence","Inspect measured windows, counterexample context and tail sample step diagnostics; associations are not causal effects.",object_schema(dict(run_id=ids))),
            ("audit_action_batch","Check original integer action constraints and measured duplicate identity; never sample.",object_schema(dict(actions=dict(type="array",minItems=1,maxItems=6,items=action))))]
        self.definitions=[dict(type="function",function=dict(name=n,description=d,parameters=s)) for n,d,s in specs]
        self.binding=dict(policy=STATISTICS_POLICY,history_sha256=value_sha(history),domain_sha256=value_sha(domain),
            sources=self.sources,implementation_sha256=sha(__file__),action_implementation_sha256=sha(Path(__file__).with_name("geometry_protocol.py")))
        frozen(self.folder/"BINDING.json",self.binding)

    def verify_sources(self):
        for run_id,source in self.sources.items():
            if source["status"]=="available":
                try:
                    changed=(sha(self.paths[run_id]/"COMPLETE.json")!=source["completion_sha256"] or
                        any(sha(self.paths[run_id]/name)!=digest for name,digest in source["files"].items()))
                except OSError as error:
                    raise EvidenceIntegrityError("Verified source no longer readable") from error
                if changed:raise EvidenceIntegrityError("Tool source changed after snapshot")

    def execute(self,name,args):
        if name not in {d["function"]["name"] for d in self.definitions}:raise ValueError("Unregistered tool")
        import jsonschema
        jsonschema.validate(args,next(d["function"]["parameters"] for d in self.definitions if d["function"]["name"]==name))
        self.verify_sources()
        binding=dict(tool=name,arguments=args,registry_sha256=value_sha(self.binding))
        identifier="ev_"+value_sha(binding)[:20];path=self.folder/(identifier+".json")
        if path.exists():
            result=read(path)
            payload={k:v for k,v in result.items() if k!="result_sha256"}
            if result.get("result_sha256")!=value_sha(payload) or result.get("binding")!=binding:
                raise EvidenceIntegrityError("Cached tool receipt changed")
        else:
            method=getattr(self,"_"+name)
            data=method(**args)
            payload=dict(evidence_id=identifier,binding=binding,statistics_policy_id=STATISTICS_POLICY["id"],**data)
            result=dict(payload,result_sha256=value_sha(payload));frozen(path,result)
        self.receipt_ids.add(identifier)
        return result

    def precompute(self):
        return [self.execute("summarize_population",dict(run_id=key)) for key in self.history]

    def observed_calculations(self):
        results=[]
        for identifier in sorted(self.receipt_ids):
            result=read(self.folder/(identifier+".json"))
            if result["binding"]["tool"]!="summarize_population":
                results.append({k:v for k,v in result.items() if k not in ("binding","result_sha256")})
        return results

    def validate_citations(self,texts):
        import jsonschema
        self.verify_sources()
        for text in texts:
            ids=set(re.findall(r"ev_[A-Za-z0-9_]+",text))
            if not ids or not ids.issubset(self.receipt_ids):
                raise jsonschema.ValidationError("Evidence must cite existing calculation receipt IDs; do not invent a receipt")

    def _summarize_population(self,run_id):
        if run_id not in self.records:
            return dict(status="unavailable",run_id=run_id,reason="No verified sample records; history aggregate is not enough for robust statistics")
        rows=self.records[run_id]
        worst=sorted((r for r in rows if type(r.get(CONTROL_DISTANCE)) in (int,float) and math.isfinite(r[CONTROL_DISTANCE])),
            key=lambda r:r[CONTROL_DISTANCE],reverse=True)[:3]
        return dict(status="complete",run_id=run_id,attempts=len(rows),
            metrics={m:metric_summary([r.get(m) for r in rows]) for m in METRICS},
            build_failed=sum(r.get("build_success") is False for r in rows),
            built_disconnected=sum(r.get("build_success") is True and r.get("connected") is False for r in rows),
            unknown_build_status=sum(type(r.get("build_success")) is not bool for r in rows),
            worst_controlled_geometry=[dict(aggregation_scope="single molecule",**{k:r.get(k) for k in
                ("node_id","logical_seed","logical_slot",*METRICS)}) for r in worst],
            quality_scope="No additional PLIP/PoseBusters evaluation; do not infer unselected results",
            ranking="descriptive only; original scientific ranking unchanged")

    def _compare_paired_runs(self,reference_id,candidate_id):
        if any(k not in self.records for k in (reference_id,candidate_id)):
            return dict(status="unavailable",reason="Verified sample records absent")
        a,b=self.history[reference_id],self.history[candidate_id]
        if any(a.get(k)!=b.get(k) or a.get(k) is None for k in ("teacher_id","condition_sha256","compatibility_sha256","attempts")):
            return dict(status="unavailable",reason="Different teacher, condition, sampling protocol or population")
        result=paired_summary(self.records[reference_id],self.records[candidate_id])
        return dict(result,reference_id=reference_id,candidate_id=candidate_id,
            independent_measurement=False if a["fingerprint"]==b["fingerprint"] else "matched existing panel; not held out")

    def _inspect_interval_evidence(self,run_id):
        row=self.history[run_id]
        out=dict(status="complete",run_id=run_id,measured_schedule=row[NORMALIZED],
            windows=row.get("window_diagnostics",[]),causal_attribution="Uncontrolled background differences do not identify a window effect")
        if run_id not in self.records:
            return dict(out,tail_step_diagnostics=dict(status="unavailable"))
        rows=self.records[run_id];steps=read(self.paths[run_id]/"steps.json")
        fields={"max_atom_increment_A":"maximum_single_atom_guidance_increment_angstrom",
            "unconstrained_max_atom_increment_A":"maximum_unconstrained_single_atom_guidance_increment_angstrom",
            "guidance_increment_scale_factor":"guidance_increment_scale_factor"}
        by_sample={}
        for r in rows:
            index=r["tensor_row"];observed=[]
            for item in steps:
                point=dict(step=item["step"])
                for source,name in fields.items():
                    values=item.get(source)
                    if isinstance(values,list) and len(values)==len(rows):point[name]=values[index]
                if len(point)>1:observed.append(point)
            by_sample[index]=observed
        worst=sorted((r for r in rows if type(r.get(CONTROL_DISTANCE)) in (int,float) and math.isfinite(r[CONTROL_DISTANCE])),
                     key=lambda r:r[CONTROL_DISTANCE],reverse=True)[:3]
        maximum=lambda r,key:max((x.get(key,0.) for x in by_sample[r["tensor_row"]]),default=0.)
        applied=fields["max_atom_increment_A"];raw=fields["unconstrained_max_atom_increment_A"]
        displacement=sorted(rows,key=lambda r:maximum(r,applied),reverse=True)[:3]
        unconstrained=sorted(rows,key=lambda r:maximum(r,raw),reverse=True)[:3] if any(raw in p for v in by_sample.values() for p in v) else []
        selected={r["tensor_row"]:r for group in (worst,displacement,unconstrained) for r in group}
        details=[]
        for index,r in selected.items():
            observed=by_sample[index]
            details.append(dict(node_id=r["node_id"],logical_seed=r["logical_seed"],logical_slot=r["logical_slot"],
                selection_reasons=[name for name,group in (("controlled_geometry_tail",worst),
                    ("largest_applied_increment",displacement),("largest_unconstrained_increment",unconstrained)) if r in group],
                status="complete" if observed else "unavailable",largest_increment_steps=sorted(observed,
                    key=lambda d:d.get(applied,0.),reverse=True)[:4],
                largest_unconstrained_increment_steps=sorted((p for p in observed if raw in p),
                    key=lambda d:d[raw],reverse=True)[:4]))
        return dict(out,tail_step_diagnostics=details)

    def _audit_action_batch(self,actions):
        seen=[r[NORMALIZED] for r in self.history.values()];results=[]
        for action in actions:
            try:
                schedule=validate_action(action,self.history,self.domain,seen)
                seen.append(schedule)
                results.append(dict(action=action,status="legal_unmeasured",resolved_normalized_schedule=schedule,
                    resolved_effective_guidance_scale_schedule=effective(schedule,self.domain[MULTIPLIER]),
                    normalized_zero_disables_guidance=False,
                    observed_performance=None,novelty_scope="provided compatible history and this batch; executor must still check global reservation registry"))
            except ValueError as error:
                results.append(dict(action=action,status="rejected",reason=str(error)))
        return dict(status="complete",actions=results,new_molecular_attempts=0)
