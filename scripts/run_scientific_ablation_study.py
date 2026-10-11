"""Local real-API scientific submodule study; serial remote FLOWR only.

Each proposal is committed and pushed before the inference host pulls it. The
supervisor independently confirms sequential pruning, freezes actual selected
Analyst outputs, and retains M2 if a final candidate fails noninferiority.
"""
import argparse
import collections
import copy
import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex
import sys
import time

import numpy as np
import pandas as pd

from run_m2_skill_api_study import Study, shell
from evomolsteer.ablation.protocol import prepare, validate_activation, validate_prior_policy
from evomolsteer.ablation.agents import call_role, review
from evomolsteer.ablation.compiler import compile_program
from evomolsteer.ablation.diagnostics import run_probes
from evomolsteer.ablation.response import observed_response
from evomolsteer.ablation.statistics import compare, compare_strain, summarize, quality_envelope, pruning_decision, classify_module
from evomolsteer.continuous.m2_skill_api import BASE, REF, effective_signature
from evomolsteer.deepseek_api import LLMInterfaceError
from evomolsteer.io import digest, read_json, write_json


class ScientificStudy(Study):
    def preflight(self):
        activation=validate_activation(self.repo,self.study)
        prior_policy=validate_prior_policy(self.repo,self.study,self.config,self.state)
        work=PurePosixPath(self.config["remote_work"])
        flowr=PurePosixPath(self.config["flowr"])
        if not work.is_relative_to(flowr/"experiments") or not work.name.startswith("evomolsteer_dependency_"):
            raise ValueError("Explicit owned scientific workspace required")
        self.protocol=activation["protocol"]
        statistical_path=self.study/"secondary_statistical_amendment.json"
        statistical=read_json(statistical_path)
        if (statistical["protocol_sha256"]!=activation["protocol_sha256"] or
            statistical["confidence_family_multiplier"]!=3):
            raise ValueError("Unbound secondary endpoint family")
        statistical_sha=digest(statistical_path)
        if self.state.get("secondary_statistical_amendment_sha256",statistical_sha)!=statistical_sha:
            raise ValueError("Statistical amendment changed after activation")
        self.modules=read_json(self.study/"module_registry.json")["modules"]
        with self.ssh() as client:
            check="""import pathlib,json
root=pathlib.Path(REPO)/'configs/experiments'
def indices(x):
 if isinstance(x,dict):
  for k,v in x.items():
   if k in {'batches','batch_indices','screen_batches'} and isinstance(v,list):
    yield from (i for i in v if isinstance(i,int))
   yield from indices(v)
 elif isinstance(x,list):
  for v in x:yield from indices(v)
for p in root.rglob('*.json'):
 try:used=set(indices(json.loads(p.read_text())))
 except (ValueError,UnicodeError):continue
 if used.intersection(RESERVED) and OWN not in str(p):print(p)
""".replace("REPO",repr(self.config["remote_repo"])).replace("RESERVED",repr(set(range(2100,3600)))).replace("OWN",repr(self.config["config_output"]))
            collisions=shell(client,shlex.join([self.config["remote_python"],"-c",check])).strip()
            if collisions:raise ValueError("Unused batch check failed: "+collisions)
            shell(client,"mkdir -p "+shlex.quote(str(work)))
            shell(client,f"test $(df -Pk {shlex.quote(str(work))} | awk 'NR==2 {{print $4}}') -gt 614400")
        self.state.setdefault("conditions",{})
        self.state.setdefault("measured",{})
        self.state.setdefault("matched_controls",{})
        self.state.setdefault("pruning",[])
        self.state.setdefault("api_calls",0)
        self.save(status="ready",error=None,error_type=None,preflight_passed=True,
            protocol_sha256=activation["protocol_sha256"],model="deepseek-flash",
            secondary_statistical_amendment_sha256=statistical_sha,**prior_policy)

    def _budget(self):
        # Count actual successful request receipts, including declared role
        # chains prepared before inference and verified cached/resumed calls.
        self.state["api_calls"]=sum(read_json(p).get("backend")=="real_api"
            for p in self.study.rglob("*.receipt.json"))
        budget=self.protocol["budgets"]
        if self.state["api_calls"]>=budget["maximum_api_role_calls"]:
            raise RuntimeError("Preregistered API budget reached; preserve unresolved tests")
        if len(self.state["program_results"])>=budget["maximum_distinct_inference_cohorts"]:
            raise RuntimeError("Preregistered distinct cohort budget reached")

    def _call(self,case,chain,role,active,**kwargs):
        self._budget()
        output=self.study/"calls"/case/f"chain_{chain:02d}"/(role+".json")
        cached=output.exists() and output.with_suffix(".receipt.json").exists()
        self.save(status="calling_real_api",task=case,active_role=role,active_chain=chain)
        result=call_role(self.repo,self.study,case,chain,role,active,**kwargs)
        if not cached:self.state["api_calls"]+=1
        self.save()
        return result

    def controls(self,batches):
        key=",".join(map(str,batches))
        if key not in self.state["matched_controls"]:
            self._budget()
            sha=effective_signature(read_json(self.repo/BASE))
            result=self.infer(self.repo/BASE,sha,"gradient,unguided",lane="v3",batches=batches)
            self.state["matched_controls"][key]=result
            self.state["measured"][sha+":"+key]=result
            self.save()
        return self.state["matched_controls"][key]

    def measure(self,path,reference,sha,batches):
        controls=self.controls(batches)
        key=sha+":"+",".join(map(str,batches))
        if key not in self.state["measured"]:
            self._budget()
            result=self.infer(path,sha,lane="v3",reference_path=reference,batches=batches)
            for b in batches:
                name=f"gradient/batch_{b:03d}"
                if result["initial_state_signatures"][name]!=controls["initial_state_signatures"][name]:
                    raise ValueError("Candidate and M2 initial state mismatch")
            self.state["measured"][key]=result
            self.save()
        result=self.state["measured"][key]
        destination=self.study/"transport_records"/(result["name"]+".response_vs_M2.json")
        if not destination.exists():write_json(destination,observed_response(self.repo,self.study,result,controls))
        return self.state["measured"][key]

    def frames(self,result,arm="gradient"):
        return pd.read_csv(self.repo/result["outcomes"][arm]["metrics"])

    def condition(self,case,role,active,incumbent,*,lane="G",alternative=None,frozen=None,focal=None,designer_context=None):
        if case in self.state["conditions"]:return self.state["conditions"][case]
        rows=[]
        delivery=None
        if lane=="U":
            path,reference,signature,_=compile_program(self.repo,self.study,
                {"decision":"retain","edit":None},incumbent,lane="U")
            measurement=self.measure(path,reference,signature,self.protocol["batches"]["screen"])
            delivery=dict(measurement["delivery"],effective_signature=signature,
                reference_sha256=digest(reference),source_result=measurement["name"],
                decoded_quality_supplied=False)
        analyst_ids=[m["id"] for m in self.modules if m["role"]=="Analyst"]
        designer_ids=[m["id"] for m in self.modules if m["role"]=="Designer" and m["lane"]=="G"]
        if designer_context is not None:designer_ids=designer_context
        for chain in range(6):
            if role=="Analyst":
                analyst=self._call(case,chain,"Analyst",active,alternative=alternative,incumbent=incumbent,lane=lane)
                designer=self._call(case,chain,"Designer",designer_ids,analyst=analyst,incumbent=incumbent,lane=lane)
            else:
                if frozen is None or len(frozen)!=6:raise ValueError("Designer requires six actual frozen selected Analyst outputs")
                analyst=frozen[chain]
                designer=self._call(case,chain,"Designer",active,alternative=alternative,analyst=analyst,
                    incumbent=incumbent,lane=lane,delivery=delivery)
            folder=self.study/"calls"/case/f"chain_{chain:02d}"
            path,reference,sha,receipt=compile_program(self.repo,self.study,designer,incumbent,lane=lane,folder=folder)
            rows.append({"chain":chain,"program":receipt["program"],"reference":receipt["reference"],"signature":sha,
                "response":(folder/"Designer.json").relative_to(self.repo).as_posix()})
            write_json(self.study/"conditions"/(case+".progress.json"),{"completed_chains":len(rows),"rows":rows})
        functional=None
        try:
            self._budget()
            functional=review(self.study,case,role,focal or active)
            self.state["api_calls"]+=1
        except Exception as error:
            write_json(self.study/"reviews"/(case+".unconfirmed.json"),{"error_type":type(error).__name__,"error":str(error),
                "necessity_claim_allowed":False,"generation_results_preserved":True})
        # Semantic response is measured before a new molecule cohort is launched.
        # A nonresponse remains an explicit interface/functional bottleneck.
        for row in rows:
            row["screen_result"]=self.measure(self.repo/row["program"],self.repo/row["reference"],row["signature"],self.protocol["batches"]["screen"])
        value={"case":case,"role":role,"lane":lane,"active":active,"alternative":alternative,"rows":rows,
            "functional":functional,"valid":len(rows)==6,"distinct_rewards":len({r["signature"] for r in rows})}
        self.state["conditions"][case]=value
        self.save(status="screening",task=case)
        self.write_report()
        self.commit("ablation(scientific): record "+case+" six real role chains")
        return value

    def condition_frames(self,condition,batches):
        return [self.frames(self.measure(self.repo/r["program"],self.repo/r["reference"],r["signature"],batches)) for r in condition["rows"]]

    def contrast(self,left,right,*,paired_roles=False,family=None):
        if family is None:family=3*self.protocol["evaluation"]["confirmation_families"]
        return compare(left,right,family=family,replicates=self.protocol["evaluation"]["bootstrap_replicates"],paired_roles=paired_roles)

    def strain_contrast(self,full,other,*,paired_roles=False,family=None):
        if family is None:family=3*self.protocol["evaluation"]["confirmation_families"]
        return compare_strain(full,other,family=family,
            replicates=self.protocol["evaluation"]["bootstrap_replicates"],paired_roles=paired_roles)

    def sequential(self,phase,role,active,incumbent,*,lane="G",frozen=None,offset=0):
        ids=[m["id"] for m in self.modules if m["role"]==role and m["lane"]==lane]
        screen=self.protocol["batches"]["screen"]
        e=self.protocol["evaluation"]
        for step,module in enumerate(ids):
            saved=next((r for r in self.state["pruning"] if r["phase"]==phase and r["module"]==module),None)
            if saved:
                if saved["decision"]["adopt"]:active=saved["selected_active"]
                continue
            maskhash=hashlib.sha256(",".join(active).encode()).hexdigest()[:8]
            prefix=f"{phase}_{module}_{maskhash}"
            self.save(phase=phase,task=module)
            full=self.condition(prefix+"_full",role,active,incumbent,lane=lane,frozen=frozen,focal=[module])
            dropped=[x for x in active if x!=module]
            drop=self.condition(prefix+"_drop",role,dropped,incumbent,lane=lane,frozen=frozen,focal=[module])
            alt=self.condition(prefix+"_alternative",role,active,incumbent,lane=lane,frozen=frozen,alternative=module,focal=[module])
            cf=self.condition_frames(full,screen);cd=self.condition_frames(drop,screen);ca=self.condition_frames(alt,screen)
            m2=[self.frames(self.controls(screen))]
            ds=self.contrast(cd,cf,paired_roles=role=="Designer",family=1)
            ms=self.contrast(cd,m2,family=1)
            ss=summarize(cd,e["elite_threshold"]);sm=summarize(m2,e["elite_threshold"])
            envelope=quality_envelope(ss,sm)
            normal=full["functional"]
            function_pass=normal is not None and sum(a["score"]==2 for a in normal["assessments"] if a["module"]==module)>=5
            screen_strain=[self.strain_contrast(cf,other,paired_roles=role=="Designer",family=1) for other in [cd,ca]]
            result={"phase":phase,"module":module,"full":full["case"],"drop":drop["case"],"alternative":alt["case"],
                "screen_drop_minus_full":ds,"screen_drop_minus_M2":ms,"function_pass":function_pass,
                "screen_is_confirmation":False,"active_before":active,"confirmation":None,"screen_strain":screen_strain}
            # Screens never establish absence of an effect. Confirm promising
            # pruning, or a sizeable potential necessary effect, prospectively.
            potential=(ds["delta_pic50"]>=-e["M2_noninferiority_margin_pic50"] and
                ms["delta_pic50"]>=-e["M2_noninferiority_margin_pic50"] and envelope["passed"])
            possible_necessary=(-ds["delta_pic50"]>e["necessity_effect_pic50"] and
                self.contrast(cf,ca,paired_roles=role=="Designer",family=1)["delta_pic50"]>e["necessity_effect_pic50"])
            possible_secondary=(all(x["status"]=="AVAILABLE" for x in screen_strain) and
                any(all(x["endpoints"][metric]["relative_worsening"]>.10 for x in screen_strain) for metric in ["median","p90"]))
            if potential or possible_necessary or possible_secondary:
                start=self.protocol["batches"]["confirmation_base"]+(offset+step)*self.protocol["batches"]["confirmation_stride"]
                batches=list(range(start,start+self.protocol["batches"]["confirmation_count"]))
                ff=self.condition_frames(full,batches);fd=self.condition_frames(drop,batches);fa=self.condition_frames(alt,batches)
                mm=[self.frames(self.controls(batches))]
                cdf=self.contrast(fd,ff,paired_roles=role=="Designer")
                cdm=self.contrast(fd,mm)
                env=quality_envelope(summarize(fd,e["elite_threshold"]),summarize(mm,e["elite_threshold"]))
                decision=pruning_decision(cdf,cdm,env,margin=e["M2_noninferiority_margin_pic50"])
                secondary=[self.strain_contrast(ff,other,paired_roles=role=="Designer") for other in [fd,fa]]
                classification=classify_module(self.contrast(ff,fd,paired_roles=role=="Designer"),
                    self.contrast(ff,fa,paired_roles=role=="Designer"),functional_valid=function_pass,
                    effect=e["necessity_effect_pic50"],secondary=secondary)
                if classification=="STRAIN_NECESSARY_IN_THIS_CONTEXT":
                    decision={"adopt":False,"status":"CONFIRMED_STRAIN_BENEFIT_KEEP_CURRENT"}
                result["confirmation"]={"batches":batches,"drop_minus_full":cdf,"drop_minus_M2":cdm,"quality_envelope":env,
                    "classification":classification,"secondary_strain":secondary,"role_designs_frozen_from_screen":True}
                # An ineffective prompt can be retired without claiming that its
                # scientific method was tested when delivery/semantics failed.
                if decision["adopt"] and not function_pass:
                    decision["status"]="CONFIRMED_SAFE_INSTRUCTION_PRUNING_SCIENCE_UNRESOLVED"
            else:
                decision={"adopt":False,"status":"SCREEN_NEGATIVE_OR_UNRESOLVED_KEEP_CURRENT"}
            if decision["adopt"]:active=dropped
            result.update(decision=decision,selected_active=active)
            self.state["pruning"].append(result)
            self.save(status="pruning_confirmed",task=module)
            write_json(self.study/"pruning"/(phase+"_"+module+".json"),result)
            self.write_report()
            self.commit("ablation(scientific): "+decision["status"].lower()+" "+module)
        return active

    def freeze_analyst(self,active):
        path=self.study/"selected_Analyst_pool.json"
        if path.exists():
            pool=read_json(path)
            if pool["active_modules"]!=active:raise ValueError("Selected Analyst pool belongs to different instructions")
            for row in pool["outputs"]:
                if digest(self.repo/row["path"])!=row["sha256"]:raise ValueError("Frozen Analyst output changed")
            return [read_json(self.repo/r["path"]) for r in pool["outputs"]]
        outputs=[]
        for chain in range(6):
            self._call("selected_Analyst_fresh",chain,"Analyst",active,incumbent=read_json(self.repo/BASE))
            p=self.study/"calls/selected_Analyst_fresh"/f"chain_{chain:02d}"/"Analyst.json"
            outputs.append({"path":p.relative_to(self.repo).as_posix(),"sha256":digest(p)})
        write_json(path,{"active_modules":active,"outputs":outputs,"selected_from":"completed sequential Analyst confirmations",
            "current_generation_outcomes_supplied":False})
        self.commit("ablation(scientific): freeze six actual selected Analyst outputs")
        return [read_json(self.repo/r["path"]) for r in outputs]

    def choose_program(self,condition,incumbent,label,batches):
        """Select the modal role recipe; never the lucky best chain."""
        counts=collections.Counter(r["signature"] for r in condition["rows"])
        base_sha=effective_signature({k:v for k,v in incumbent.items() if not k.startswith("_")})
        sha=sorted(counts,key=lambda x:(-counts[x],x!=base_sha,x))[0]
        row=next(r for r in condition["rows"] if r["signature"]==sha)
        candidate=self.measure(self.repo/row["program"],self.repo/row["reference"],sha,batches)
        control=self.controls(batches)
        c=[self.frames(candidate)];m=[self.frames(control)]
        delta=self.contrast(c,m,family=3)
        envelope=quality_envelope(summarize(c,self.protocol["evaluation"]["elite_threshold"]),summarize(m,self.protocol["evaluation"]["elite_threshold"]))
        accepted=delta["simultaneous_CI"][0]>-.02 and envelope["passed"]
        if accepted:
            program=read_json(self.repo/row["program"])
            program["_reference_relative_path"]=row["reference"]
        else:
            program=read_json(self.repo/BASE)
            program["_reference_relative_path"]=REF
        record={"selection":"modal frozen role recipe, tie prefers incumbent", "selected_signature":sha,
            "delta_vs_M2":delta,"envelope":envelope,"accepted":accepted,"batches":batches,
            "fallback":"confirmed matched M2" if not accepted else None,"program":program}
        write_json(self.study/(label+".selection.json"),record)
        self.commit("ablation(scientific): confirm "+label+" and bind selected incumbent")
        return program

    def run(self):
        prepare(self.repo,self.study);self.preflight()
        if not self.state.get("functional_preflight_passed"):
            run_probes(self.repo,self.study)
            self.state["api_calls"]+=6
            self.save(functional_preflight_passed=True)
            self.commit("test(scientific): verify real API sign, missingness and unit response")
        incumbent=read_json(self.repo/BASE)
        self.controls(self.protocol["batches"]["screen"])
        a=[m["id"] for m in self.modules if m["role"]=="Analyst"]
        dg=[m["id"] for m in self.modules if m["role"]=="Designer" and m["lane"]=="G"]
        du=[m["id"] for m in self.modules if m["role"]=="Designer" and m["lane"]=="U"]
        self.condition("A_empty","Analyst",[],incumbent,focal=a)
        a=self.sequential("A", "Analyst", a, incumbent, offset=0)
        frozen=self.freeze_analyst(a)
        self.condition("D-G_empty","Designer",[],incumbent,frozen=frozen,focal=dg)
        dg=self.sequential("D-G", "Designer", dg, incumbent, frozen=frozen, offset=9)
        gcase=self.condition("selected_G_fresh","Designer",dg,incumbent,frozen=frozen,focal=dg or ["D4.1"])
        g=self.choose_program(gcase,incumbent,"G",list(range(2500,2508)))
        # Dose calls retain selected G instructions but can modify only dose.
        combined=dg+du
        self.condition("D-U_empty","Designer",dg,g,lane="U",frozen=frozen,focal=du)
        combined=self.sequential("D-U","Designer",combined,g,lane="U",frozen=frozen,offset=15)
        finalcase=self.condition("selected_U_fresh","Designer",combined,g,lane="U",frozen=frozen,focal=du or ["D3.1"])
        final=self.choose_program(finalcase,g,"final",self.protocol["batches"]["final"])
        self.state["selected_workflow"]={"Analyst":a,"Designer":combined,"program":final,
            "candidate_accepted":read_json(self.study/"final.selection.json")["accepted"]}
        self.final_context(a,combined,final,frozen)
        self.final_report(final)
        self.save(status="complete",phase="complete",task=None,
            no_universal_module_necessity_claim=True)
        self.write_report()
        self.commit("report(scientific): complete sequential submodule study and final comparisons")

    def final_context(self,a,d,incumbent,frozen):
        """Retest retained functions in final context; no screen claim inherited."""
        records=self.state.setdefault("final_context",[])
        for step,module in enumerate(a+d):
            if any(r["module"]==module for r in records):continue
            m=next(m for m in self.modules if m["id"]==module)
            role=m["role"];active=a if role=="Analyst" else d;lane=m["lane"]
            if role=="Analyst":lane="G"
            prefix="final_context_"+module
            full=self.condition(prefix+"_full",role,active,incumbent,lane=lane,frozen=frozen,focal=[module],designer_context=d)
            drop=self.condition(prefix+"_drop",role,[x for x in active if x!=module],incumbent,lane=lane,frozen=frozen,focal=[module],designer_context=d)
            alt=self.condition(prefix+"_alternative",role,active,incumbent,lane=lane,frozen=frozen,alternative=module,focal=[module],designer_context=d)
            batches=list(range(3000+16*step,3008+16*step))
            ff=self.condition_frames(full,batches);fd=self.condition_frames(drop,batches);fa=self.condition_frames(alt,batches)
            passed=full["functional"] is not None and sum(x["score"]==2 for x in full["functional"]["assessments"])>=5
            secondary=[self.strain_contrast(ff,other,paired_roles=role=="Designer") for other in [fd,fa]]
            record={"module":module,"batches":batches,
                "classification":classify_module(self.contrast(ff,fd,paired_roles=role=="Designer"),
                    self.contrast(ff,fa,paired_roles=role=="Designer"),functional_valid=passed,secondary=secondary),
                "full_minus_drop":self.contrast(ff,fd,paired_roles=role=="Designer"),
                "full_minus_alternative":self.contrast(ff,fa,paired_roles=role=="Designer"),
                "secondary_strain":secondary,
                "not_inherited_from_initial_context":True}
            records.append(record);self.save();self.write_report()
            self.commit("ablation(scientific): independently retest final-context "+module)

    def final_report(self,program):
        batches=self.protocol["batches"]["final"]
        path=self.study/"selected_program.json"
        p=copy.deepcopy(program);reference=self.repo/p.pop("_reference_relative_path",REF);write_json(path,p)
        result=self.measure(path,reference,effective_signature(p),batches)
        controls=self.controls(batches)
        c=[self.frames(result)];m=[self.frames(controls)];n=[self.frames(controls,"unguided")]
        e=self.protocol["evaluation"]
        history=pd.read_parquet(self.study/"evaluation_controls/historical_steer.metrics.parquet")
        rng=np.random.default_rng(42)
        cm=c[0].groupby("batch").pic50_on_rescore.mean().to_numpy()
        hm=history.groupby("batch").pic50_on_rescore.mean().to_numpy()
        boot=cm[rng.integers(0,len(cm),(20000,len(cm)))].mean(1)-hm[rng.integers(0,len(hm),(20000,len(hm)))].mean(1)
        vs_native=self.contrast(c,n,family=3);vs_m2=self.contrast(c,m,family=3)
        ci=np.quantile(boot,[.05/6,1-.05/6]).tolist()
        record={"candidate":summarize(c,e["elite_threshold"]),"M2":summarize(m,e["elite_threshold"]),
            "native":summarize(n,e["elite_threshold"]),"historical_Steer":summarize([history],e["elite_threshold"]),
            "candidate_minus_native":vs_native,"candidate_minus_M2":vs_m2,
            "candidate_minus_historical_Steer":{"delta_pic50":float(cm.mean()-hm.mean()),"simultaneous_CI":ci,
                "unpaired":True,"equal_online_budget":False,"not_extreme_path_discovery_proof":True},
            "goals":{"native_superiority":vs_native["simultaneous_CI"][0]>0,
                "M2_noninferiority":vs_m2["simultaneous_CI"][0]>-.02,"historical_mean_superiority":ci[0]>0},
            "learning_and_guidance_support":p["window"],"native_completion":1.,"master_seed":42}
        write_json(self.study/"final_quality.json",record)

    def write_report(self):
        rows=[]
        for r in self.state.get("pruning",[]):
            c=r.get("confirmation") or {}
            rows.append({"phase":r["phase"],"module":r["module"],"function_pass":r["function_pass"],
                "screen_drop_delta":r["screen_drop_minus_full"]["delta_pic50"],"adopted_removal":r["decision"]["adopt"],
                "status":r["decision"]["status"],"confirmation_class":c.get("classification"),
                "confirmation_drop_delta":c.get("drop_minus_full",{}).get("delta_pic50")})
        pd.DataFrame(rows).to_csv(self.study/"submodule_results.csv",index=False)
        write_json(self.study/"study_summary.json",{"status":self.state.get("status"),"model":"deepseek-flash",
            "completed_submodules":len(rows),"total_submodules":17,"api_calls":self.state.get("api_calls",0),
            "independent_inference_cohorts":len(self.state.get("program_results",{})),"results":rows,
            "final_context":self.state.get("final_context",[]),"selected_workflow":self.state.get("selected_workflow"),
            "performance_goal_not_assumed":True,"raw_outputs_retired_after_verification":True})
        lines=["# 科学子模块消融运行状态", "", "模型：真实 DeepSeek API / deepseek-flash。", "",
            f"状态：{self.state.get('status')}；已完成子模块 {len(rows)}/17。", "",
            "|模块|功能验证|确认结论|是否精简|", "|---|---|---|---|"]
        lines += [f"|{r['module']}|{r['function_pass']}|{r['confirmation_class'] or r['status']}|{r['adopted_removal']}|" for r in rows]
        lines += ["", "筛查不等于确认。功能未响应、接口瓶颈和证据不足不表述为科学方法无效。历史 Steer 比较不成对且在线预算不同。", ""]
        (self.study/"status.zh-CN.md").write_text("\n".join(lines),encoding="utf-8")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo",default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--config",required=True)
    args=parser.parse_args()
    study=ScientificStudy(Path(args.repo),read_json(args.config))
    try:study.run()
    except Exception as error:
        study.save(status="failed",error_type=type(error).__name__,error=str(error))
        study.write_report()
        raise
