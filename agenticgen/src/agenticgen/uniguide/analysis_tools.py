"""Read-only measured evidence, paired statistics and original action validation."""
import math
import re
from pathlib import Path
import numpy as np

from .io import read, frozen, value_sha, sha
from .geometry_condition import DISTANCE, CONTROL_DISTANCE, NEIGHBOR_DISTANCE, MAX_ERROR
from .geometry_protocol import NORMALIZED, MULTIPLIER, validate_action, effective
from .geometry_protocol import seen_configuration,action_weights
from .geometry_reward import WEIGHTS,KEYS,weights,INTERACTION_DISTANCE

METRICS = (CONTROL_DISTANCE, NEIGHBOR_DISTANCE, DISTANCE, MAX_ERROR,INTERACTION_DISTANCE)
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


def paired_summary(left,right,metrics=METRICS):
    identity=lambda r:(r["logical_seed"],r["logical_slot"])
    a,b={identity(r):r for r in left},{identity(r):r for r in right}
    if len(a)!=len(left) or len(b)!=len(right) or set(a)!=set(b):
        return dict(status="unavailable", reason="Incomplete or repeated paired slot identities")
    keys=sorted(a);seeds=sorted({k[0] for k in keys});result={}
    for metric in metrics:
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
    def __init__(self, root, history, domain, folder,condition=None):
        self.root=Path(root).resolve();self.history={r["id"]:r for r in history}
        self.domain=domain;self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=True)
        self.condition=condition
        from .absolute_coordinate import is_absolute, METRICS as coordinate_metrics, RMSD
        self.metrics=coordinate_metrics if condition and is_absolute(condition) else METRICS
        self.primary_metric=RMSD if condition and is_absolute(condition) else CONTROL_DISTANCE
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
            if condition and condition.get("reward_policy") and "trace.h5" in receipt["files"]:
                hashes["trace.h5"]=sha(path/"trace.h5")
                if hashes["trace.h5"]!=receipt["files"]["trace.h5"]:raise ValueError("Reward diagnostic trace changed")
            actual=read(path/"aggregate.json")
            if any(actual.get(k)!=row.get(k) for k in ("id","attempts","teacher_id","condition_sha256","compatibility_sha256","fingerprint",*self.metrics,WEIGHTS,"window_diagnostics")):
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
        if condition and condition.get("reward_policy"):
            from .geometry_condition import validate
            validate(condition)
            if condition["condition_sha256"]!=domain["condition_sha256"]:raise ValueError("Control condition differs from history")
            action["properties"]["updates"]["minItems"]=0
            action["properties"]["loss_control_id"]={"type":"string","pattern":"^ev_[0-9a-f]{20}$"}
            specs.append(("set_guidance_loss_weights",
                "Prepare candidate-local binary loss controls from observed evidence. Returns a receipt ID; only a selected proposal referencing it changes sampling. Never edits skills or starts generation.",
                object_schema(dict(anchor_id=ids,**{k:dict(type="integer",enum=[0,1]) for k in KEYS},
                    evidence_ids=dict(type="array",minItems=1,maxItems=4,uniqueItems=True,
                        items=dict(type="string",pattern="^ev_[0-9a-f]{20}$")),reason=dict(type="string",minLength=1,maxLength=350)))))
        self.definitions=[dict(type="function",function=dict(name=n,description=d,parameters=s)) for n,d,s in specs]
        self.binding=dict(policy=STATISTICS_POLICY,history_sha256=value_sha(history),domain_sha256=value_sha(domain),
            sources=self.sources,implementation_sha256=sha(__file__),action_implementation_sha256=sha(Path(__file__).with_name("geometry_protocol.py")))
        if condition:self.binding["reward_condition_sha256"]=condition["condition_sha256"]
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
        worst=sorted((r for r in rows if type(r.get(self.primary_metric)) in (int,float) and math.isfinite(r[self.primary_metric])),
            key=lambda r:r[self.primary_metric],reverse=True)[:3]
        return dict(status="complete",run_id=run_id,attempts=len(rows),
            metrics={m:metric_summary([r.get(m) for r in rows]) for m in self.metrics},
            build_failed=sum(r.get("build_success") is False for r in rows),
            built_disconnected=sum(r.get("build_success") is True and r.get("connected") is False for r in rows),
            unknown_build_status=sum(type(r.get("build_success")) is not bool for r in rows),
            worst_controlled_geometry=[dict(aggregation_scope="single molecule",**{k:r.get(k) for k in
                ("node_id","logical_seed","logical_slot",*self.metrics)}) for r in worst],
            quality_scope="No additional PLIP/PoseBusters evaluation; do not infer unselected results",
            ranking="descriptive only; original scientific ranking unchanged")

    def _compare_paired_runs(self,reference_id,candidate_id):
        if any(k not in self.records for k in (reference_id,candidate_id)):
            return dict(status="unavailable",reason="Verified sample records absent")
        a,b=self.history[reference_id],self.history[candidate_id]
        if any(a.get(k)!=b.get(k) or a.get(k) is None for k in ("teacher_id","condition_sha256","compatibility_sha256","attempts")):
            return dict(status="unavailable",reason="Different teacher, condition, sampling protocol or population")
        result=paired_summary(self.records[reference_id],self.records[candidate_id],self.metrics)
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
        worst=sorted((r for r in rows if type(r.get(self.primary_metric)) in (int,float) and math.isfinite(r[self.primary_metric])),
                     key=lambda r:r[self.primary_metric],reverse=True)[:3]
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
        return dict(out,tail_step_diagnostics=details,reward_diagnostics=self.reward_diagnostics(run_id))

    def reward_diagnostics(self,run_id):
        if "trace.h5" not in self.sources[run_id].get("files",{}):
            return dict(status="unavailable",reason="No verified reward trace")
        import h5py
        from .geometry_reward import LOSSES
        result=[]
        with h5py.File(self.paths[run_id]/"trace.h5","r") as trace:
            fields=[*LOSSES,"guidance_total_loss",*KEYS]
            present=[k for k in fields if k in trace]
            for window in self.history[run_id].get("window_diagnostics",[]):
                lo,hi=window["steps"];values={}
                for key in present:
                    array=trace[key][lo:hi+1].astype(np.float64)
                    if np.isfinite(array).all():values[key]=dict(mean=float(array.mean()),minimum=float(array.min()),maximum=float(array.max()))
                    else:values[key]=dict(status="unavailable_nonfinite_source")
                result.append(dict(steps=[lo,hi],values=values))
        from .geometry_reward import REWARD_POLICY
        current=bool(self.condition and self.condition.get("reward_policy")==REWARD_POLICY)
        return dict(status="complete" if present else "unavailable",windows=result,
            reward_policy=self.condition.get("reward_policy") if self.condition else None,
            loss_aggregation="half_sum_squared_distance_error" if current else "unknown",
            loss_unit="angstrom_squared" if current else "unknown",
            interpretation="Current components sum half squared distance errors over pairs within each candidate. RMSD metrics retain their mean reduction; pair counts and active components affect loss magnitude. Historical loss units require their original protocol.")

    def _audit_action_batch(self,actions):
        seen=[seen_configuration(r[NORMALIZED],r.get(WEIGHTS)) for r in self.history.values()];results=[]
        for action in actions:
            try:
                action=self.resolve_loss_control(action)
                schedule=validate_action(action,self.history,self.domain,seen)
                seen.append(seen_configuration(schedule,action_weights(action,self.history)))
                results.append(dict(action=action,status="legal_unmeasured",resolved_normalized_schedule=schedule,
                    resolved_effective_guidance_scale_schedule=effective(schedule,self.domain[MULTIPLIER]),
                    normalized_zero_disables_guidance=False,
                    observed_performance=None,novelty_scope="provided compatible history and this batch; executor must still check global reservation registry"))
            except ValueError as error:
                results.append(dict(action=action,status="rejected",reason=str(error)))
        return dict(status="complete",actions=results,new_molecular_attempts=0)

    def _set_guidance_loss_weights(self,anchor_id,evidence_ids,reason,**settings):
        if not self.condition:raise ValueError("Loss control tool not configured")
        chosen=weights(self.condition,settings)
        if not set(evidence_ids)<=self.receipt_ids:raise ValueError("Only observed evidence receipts may justify controls")
        receipts=[read(self.folder/(i+".json")) for i in evidence_ids]
        for receipt in receipts:
            self.execute(receipt["binding"]["tool"],receipt["binding"]["arguments"])
        linked=[r for r in receipts if r.get("status")=="complete" and anchor_id in
            (r.get("run_id"),r.get("candidate_id"),r.get("reference_id"))]
        if not linked:raise ValueError("Complete measured evidence for this anchor required")
        old=weights(self.condition,self.history[anchor_id].get(WEIGHTS))
        if old["condition_outer_weight"]==0 and chosen["condition_outer_weight"]==1:
            primary=CONTROL_DISTANCE if any(p["category"]=="controlled_internal" for p in self.condition["pairs"]) else DISTANCE
            paired=[r for r in linked if r["binding"]["tool"]=="compare_paired_runs" and
                r.get("candidate_id")!=r.get("reference_id") and r.get("independent_measurement") is not False and
                r.get("metrics",{}).get(primary,{}).get("status")=="complete"]
            if not paired:raise ValueError("Outer activation needs an observed nonduplicate paired comparison; explain plateau/diagnostic hypothesis in reason")
        return dict(status="prepared_not_executed",anchor_id=anchor_id,**{WEIGHTS:chosen},
            previous_weights=old,evidence_ids=evidence_ids,reason=reason,
            effect="candidate only after final selection; no global mutation",new_molecular_attempts=0)

    def resolve_loss_control(self,action,allow_saved=False):
        action=dict(action);identifier=action.get("loss_control_id")
        if not identifier:
            if WEIGHTS in action and action[WEIGHTS]!=self.history[action["anchor_id"]].get(WEIGHTS):
                raise ValueError("Unreceipted loss change")
            return action
        if not re.fullmatch(r"ev_[0-9a-f]{20}",identifier) or (not allow_saved and identifier not in self.receipt_ids):
            raise ValueError("Unknown loss control receipt")
        path=self.folder/(identifier+".json")
        if not path.is_file():raise ValueError("Missing loss control receipt")
        receipt=read(path);binding=receipt.get("binding",{})
        if binding.get("tool")!="set_guidance_loss_weights" or receipt.get("anchor_id")!=action["anchor_id"]:
            raise ValueError("Loss control receipt does not belong to this anchor")
        verified=self.execute(binding["tool"],binding["arguments"])
        if verified["evidence_id"]!=identifier or verified["status"]!="prepared_not_executed":raise ValueError("Invalid loss control binding")
        if WEIGHTS in action and action[WEIGHTS]!=verified[WEIGHTS]:raise ValueError("Control weights differ from receipt")
        return dict(action,**{WEIGHTS:weights(self.condition,verified[WEIGHTS])})
