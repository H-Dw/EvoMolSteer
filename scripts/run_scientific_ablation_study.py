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
import jsonschema

from run_m2_skill_api_study import Study, shell
from evomolsteer.ablation.protocol import prepare, validate_activation, validate_prior_policy, reserved_batches, validate_full_drop_protocol
from evomolsteer.ablation.agents import call_role, review
from evomolsteer.ablation.compiler import compile_program
from evomolsteer.ablation.diagnostics import run_probes
from evomolsteer.ablation.response import observed_response
from evomolsteer.ablation.statistics import compare, compare_strain, summarize, quality_envelope, pruning_decision, classify_module
from evomolsteer.ablation.execution import ExecutionBudgetExceeded, ModuleExecutionError, failure_record
from evomolsteer.continuous.m2_skill_api import BASE, REF, effective_signature
from evomolsteer.deepseek_api import LLMInterfaceError
from evomolsteer.io import digest, read_json, write_json


class ScientificStudy(Study):
    def __init__(self, repo, config):
        super().__init__(repo, config)
        self.state.setdefault("skipped_modules", [])
        self.state.setdefault("stage_failures", [])
        if self.state.get("status") == "failed" and self.state.get("error"):
            self.state.setdefault("failure_before_skip_resume", {
                k:self.state.get(k) for k in ["phase","task","active_role","active_chain","error","error_type","updated_unix"]})

    def preflight(self):
        interface=self.config.get("agent_interface","direct-v4")
        if interface not in {"direct-v4","evidence-tools-v1"}:
            raise ValueError("Unknown scientific agent interface")
        existing=self.state.get("agent_interface","direct-v4" if self.state.get("api_calls",0) else interface)
        if interface!=existing:
            raise ValueError("Agent interface cannot change within an active study; use a separately versioned study")
        self.interface_mode=interface
        self.state["agent_interface"]=interface
        activation=validate_activation(self.repo,self.study)
        error_policy=self.study/"execution_error_policy.json"
        if self.config.get("module_error_policy")!="skip_and_keep_current":
            raise ValueError("Explicit user-authorized module skip policy required")
        policy_sha=digest(error_policy)
        if self.state.get("execution_error_policy_sha256",policy_sha)!=policy_sha:
            raise ValueError("Execution error policy changed after resume")
        prior_policy=validate_prior_policy(self.repo,self.study,self.config,self.state)
        work=PurePosixPath(self.config["remote_work"])
        flowr=PurePosixPath(self.config["flowr"])
        if not work.is_relative_to(flowr/"experiments") or not work.name.startswith("evomolsteer_dependency_"):
            raise ValueError("Explicit owned scientific workspace required")
        self.protocol=activation["protocol"]
        validate_full_drop_protocol(self.protocol)
        if self.protocol.get("agent_interface",interface)!=interface:
            raise ValueError("Agent interface differs from the frozen protocol")
        if activation["plan"]!=self.config.get("plan"):
            raise ValueError("Configured study plan differs from activation")
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
""".replace("REPO",repr(self.config["remote_repo"])).replace("RESERVED",repr(set(reserved_batches(self.protocol)))).replace("OWN",repr(self.config["config_output"]))
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
            execution_error_policy_sha256=policy_sha,
            protocol_sha256=activation["protocol_sha256"],model="deepseek-flash",
            secondary_statistical_amendment_sha256=statistical_sha,**prior_policy)

    def _api_count(self):
        return sum(read_json(p).get("backend")=="real_api"
                   for p in self.study.rglob("*.receipt.json"))

    def _budget(self):
        # Count actual successful request receipts, including declared role
        # chains prepared before inference and verified cached/resumed calls.
        self.state["api_calls"]=self._api_count()
        budget=self.protocol["budgets"]
        if self.state["api_calls"]>=budget["maximum_api_role_calls"]:
            raise ExecutionBudgetExceeded("Preregistered API budget reached; preserve unresolved tests")
        if len(self.state["program_results"])>=budget["maximum_distinct_inference_cohorts"]:
            raise ExecutionBudgetExceeded("Preregistered distinct cohort budget reached")

    def _call(self,case,chain,role,active,**kwargs):
        self._budget()
        self.save(status="calling_real_api",task=case,active_role=role,active_chain=chain)
        try:
            result=call_role(self.repo,self.study,case,chain,role,active,
                             interface_mode=getattr(self,"interface_mode","direct-v4"),**kwargs)
        except (LLMInterfaceError, jsonschema.ValidationError, ValueError) as error:
            raise ModuleExecutionError(case,"role_response",error,role=role,chain=chain) from error
        self.state["api_calls"]=self._api_count()
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

    def condition(self,case,role,active,incumbent,*,lane="G",frozen=None,focal=None,designer_context=None):
        if "alternative" in case or case.endswith("_empty"):
            raise ValueError("Only full/drop scientific conditions and selected-policy calls are permitted")
        if case in self.state["conditions"]:
            value=self.state["conditions"][case]
            failed_review=self.study/"reviews"/(case+".unconfirmed.json")
            if value.get("functional") is None and failed_review.exists():
                report=read_json(failed_review)
                raise ModuleExecutionError(case,"functional_review",ValueError(report["error"]),role="Reviewer")
            return value
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
                analyst=self._call(case,chain,"Analyst",active,incumbent=incumbent,lane=lane)
                designer=self._call(case,chain,"Designer",designer_ids,analyst=analyst,incumbent=incumbent,lane=lane)
            else:
                if frozen is None or len(frozen)!=6:raise ValueError("Designer requires six actual frozen selected Analyst outputs")
                analyst=frozen[chain]
                designer=self._call(case,chain,"Designer",active,analyst=analyst,
                    incumbent=incumbent,lane=lane,delivery=delivery)
            folder=self.study/"calls"/case/f"chain_{chain:02d}"
            path,reference,sha,receipt=compile_program(self.repo,self.study,designer,incumbent,lane=lane,folder=folder)
            rows.append({"chain":chain,"program":receipt["program"],"reference":receipt["reference"],"signature":sha,
                "response":(folder/"Designer.json").relative_to(self.repo).as_posix()})
            write_json(self.study/"conditions"/(case+".progress.json"),{"completed_chains":len(rows),"rows":rows})
        functional=None
        try:
            self._budget()
            functional=review(self.study,case,role,focal or active,
                              interface_mode=getattr(self,"interface_mode","direct-v4"))
            self.state["api_calls"]+=1
        except ExecutionBudgetExceeded:
            raise
        except Exception as error:
            write_json(self.study/"reviews"/(case+".unconfirmed.json"),{"error_type":type(error).__name__,"error":str(error),
                "necessity_claim_allowed":False,"generation_results_preserved":True})
            raise ModuleExecutionError(case,"functional_review",error,role="Reviewer") from error
        # Semantic response is measured before a new molecule cohort is launched.
        # A nonresponse remains an explicit interface/functional bottleneck.
        for row in rows:
            row["screen_result"]=self.measure(self.repo/row["program"],self.repo/row["reference"],row["signature"],self.protocol["batches"]["screen"])
        value={"case":case,"role":role,"lane":lane,"active":active,"rows":rows,
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

    def _module_is_skipped(self,module):
        return any(r["module"]==module for r in self.state.get("skipped_modules",[]))

    def skip_module(self,module,phase,active,error):
        if self._module_is_skipped(module):return
        record=failure_record(module,phase,active,error)
        self.state["skipped_modules"].append(record)
        write_json(self.study/"skipped_modules"/(phase+"_"+module+".json"),record)
        self.save(status="continuing_after_module_error",phase=phase,task=module,error=None,error_type=None)
        self.write_report()
        self.commit("ablation(scientific): skip failed comparison "+module+" without causal attribution")

    def optional_condition(self,case,*args,**kwargs):
        if any(r.get("case")==case for r in self.state.get("stage_failures",[])):return None
        try:return self.condition(case,*args,**kwargs)
        except (ModuleExecutionError,LLMInterfaceError,jsonschema.ValidationError,ValueError) as error:
            if not isinstance(error,ModuleExecutionError):
                error=ModuleExecutionError(case,"common_condition",error)
            record=failure_record("COMMON_STAGE",self.state.get("phase"),[],error)
            self.state["stage_failures"].append(record)
            write_json(self.study/"stage_failures"/(case+".json"),record)
            self.save(status="continuing_after_stage_error",task=case)
            return None

    def optional_selection(self,condition,incumbent,label,batches):
        if condition is None:return incumbent
        try:return self.choose_program(condition,incumbent,label,batches)
        except (ModuleExecutionError,LLMInterfaceError,jsonschema.ValidationError,ValueError) as error:
            record=failure_record("COMMON_STAGE",label+"_selection",[],error)
        except RuntimeError as error:
            if isinstance(error,ExecutionBudgetExceeded) or not str(error).startswith(("FLOWR failed:","Remote driver exited:")):
                raise
            record=failure_record("COMMON_STAGE",label+"_selection",[],error)
        self.state["stage_failures"].append(record)
        write_json(self.study/"stage_failures"/(label+"_selection.json"),record)
        self.save(status="continuing_after_stage_error",task=label)
        return incumbent

    def sequential(self,phase,role,active,incumbent,*,lane="G",frozen=None,offset=0):
        ids=[m["id"] for m in self.modules if m["role"]==role and m["lane"]==lane]
        for step,module in enumerate(ids):
            if self._module_is_skipped(module):continue
            saved=next((r for r in self.state["pruning"] if r["phase"]==phase and r["module"]==module),None)
            if saved:
                if saved["decision"]["adopt"]:active=saved["selected_active"]
                continue
            previous=self.state.get("failure_before_skip_resume",{})
            if previous.get("phase")==phase and str(previous.get("task","")).startswith(phase+"_"+module+"_"):
                error=ModuleExecutionError(previous["task"],"prior_run_failure",ValueError(previous["error"]),
                    role=previous.get("active_role"),chain=previous.get("active_chain"))
                self.skip_module(module,phase,active,error)
                continue
            try:
                active=self._sequential_step(phase,role,active,incumbent,module,step,lane=lane,frozen=frozen,offset=offset)
            except (ModuleExecutionError,LLMInterfaceError,jsonschema.ValidationError,ValueError) as error:
                self.skip_module(module,phase,active,error)
            except RuntimeError as error:
                if isinstance(error,ExecutionBudgetExceeded) or not str(error).startswith(("FLOWR failed:","Remote driver exited:")):
                    raise
                self.skip_module(module,phase,active,ModuleExecutionError(self.state.get("task"),"FLOWR",error))
        return active

    def _sequential_step(self,phase,role,active,incumbent,module,step,*,lane="G",frozen=None,offset=0):
        screen=self.protocol["batches"]["screen"]
        e=self.protocol["evaluation"]
        maskhash=hashlib.sha256(",".join(active).encode()).hexdigest()[:8]
        prefix=f"{phase}_{module}_{maskhash}"
        self.save(phase=phase,task=module)
        full=self.condition(prefix+"_full",role,active,incumbent,lane=lane,frozen=frozen,focal=[module])
        dropped=[x for x in active if x!=module]
        drop=self.condition(prefix+"_drop",role,dropped,incumbent,lane=lane,frozen=frozen,focal=[module])
        cf=self.condition_frames(full,screen);cd=self.condition_frames(drop,screen)
        m2=[self.frames(self.controls(screen))]
        ds=self.contrast(cd,cf,paired_roles=role=="Designer",family=1)
        ms=self.contrast(cd,m2,family=1)
        ss=summarize(cd,e["elite_threshold"]);sm=summarize(m2,e["elite_threshold"])
        envelope=quality_envelope(ss,sm)
        normal=full["functional"]
        function_pass=normal is not None and sum(a["score"]==2 for a in normal["assessments"] if a["module"]==module)>=5
        screen_strain=[self.strain_contrast(cf,cd,paired_roles=role=="Designer",family=1)]
        result={"phase":phase,"module":module,"full":full["case"],"drop":drop["case"],"comparison_policy":"full_drop_only",
            "screen_drop_minus_full":ds,"screen_drop_minus_M2":ms,"function_pass":function_pass,
            "screen_is_confirmation":False,"active_before":active,"confirmation":None,"screen_strain":screen_strain}
        # Screens never establish absence of an effect. Confirm promising
        # pruning, or a sizeable potential necessary effect, prospectively.
        potential=(ds["delta_pic50"]>=-e["M2_noninferiority_margin_pic50"] and
            ms["delta_pic50"]>=-e["M2_noninferiority_margin_pic50"] and envelope["passed"])
        possible_necessary=-ds["delta_pic50"]>e["necessity_effect_pic50"]
        possible_secondary=(all(x["status"]=="AVAILABLE" for x in screen_strain) and
            any(all(x["endpoints"][metric]["relative_worsening"]>.10 for x in screen_strain) for metric in ["median","p90"]))
        if potential or possible_necessary or possible_secondary:
            start=self.protocol["batches"]["confirmation_base"]+(offset+step)*self.protocol["batches"]["confirmation_stride"]
            batches=list(range(start,start+self.protocol["batches"]["confirmation_count"]))
            ff=self.condition_frames(full,batches);fd=self.condition_frames(drop,batches)
            mm=[self.frames(self.controls(batches))]
            cdf=self.contrast(fd,ff,paired_roles=role=="Designer")
            cdm=self.contrast(fd,mm)
            env=quality_envelope(summarize(fd,e["elite_threshold"]),summarize(mm,e["elite_threshold"]))
            decision=pruning_decision(cdf,cdm,env,margin=e["M2_noninferiority_margin_pic50"])
            secondary=[self.strain_contrast(ff,fd,paired_roles=role=="Designer")]
            classification=classify_module(self.contrast(ff,fd,paired_roles=role=="Designer"),
                functional_valid=function_pass,effect=e["necessity_effect_pic50"],secondary=secondary)
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
        """Use only actual validated outputs; declare a common-stage fallback."""
        path=self.study/"selected_Analyst_pool.json"
        if path.exists() and read_json(path).get("fallback"):
            active=read_json(path)["active_modules"]
        try:
            result=self._freeze_analyst(active)
            self.state["frozen_analyst_active"]=active
            self.save()
            return result
        except (ModuleExecutionError,LLMInterfaceError,jsonschema.ValidationError,ValueError) as error:
            from evomolsteer.ablation.agents import full_input_packet,validate_evidence_citations
            packet=full_input_packet(self.study)
            known=set(read_json(self.study/"evidence/index.json")["ids"])|set(packet)|{
                "role","case_id","lane","response_schema","maximum_edits","task_wording",
                "interface_notes","allowed_edits","incumbent","fixed_parameters","formula_registry"}
            candidates=[v for v in self.state["conditions"].values() if v["role"]=="Analyst" and v.get("valid")]
            candidates.sort(key=lambda v:(v["active"]!=active,len(set(v["active"])^set(active)),v["case"]))
            for value in candidates:
                outputs=[]
                try:
                    for row in value["rows"]:
                        p=(self.repo/row["response"]).with_name("Analyst.json")
                        response=read_json(p);validate_evidence_citations(response,known)
                        delivery=read_json(p.with_suffix(".delivery.json"))
                        if getattr(self,"interface_mode","direct-v4")=="evidence-tools-v1":
                            from evomolsteer.agent_workflow.ablation_adapter import analyst_source
                            analyst_source(self.study,value["case"],row["chain"],response)
                        else:
                            receipt=read_json(p.with_suffix(".receipt.json"))
                            if receipt.get("response_sha256")!=digest(p) or receipt.get("backend")!="real_api":
                                raise ValueError("Invalid fallback Analyst receipt")
                        if delivery["active_modules"]!=value["active"] or delivery.get("alternative"):
                            raise ValueError("Fallback pool instructions differ")
                        outputs.append({"path":p.relative_to(self.repo).as_posix(),"sha256":digest(p)})
                except (ValueError,jsonschema.ValidationError):continue
                if len(outputs)!=6:continue
                write_json(path,{"active_modules":value["active"],"outputs":outputs,
                    "selected_from":value["case"],"fallback":True,"error":str(error)[:2000],
                    "not_outputs_of_failed_selected_instructions":True,"current_generation_outcomes_supplied":False})
                self.state["frozen_analyst_active"]=value["active"]
                self.state["stage_failures"].append(failure_record("COMMON_STAGE","Analyst_handoff",active,error))
                self.save();self.commit("ablation(scientific): preserve validated actual Analyst pool after handoff failure")
                return [read_json(self.repo/r["path"]) for r in outputs]
            raise

    def _freeze_analyst(self,active):
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
        prepare(self.repo,self.study,plan=self.config["plan"]);self.preflight()
        if not self.state.get("functional_preflight_passed"):
            run_probes(self.repo,self.study,interface_mode=self.interface_mode)
            self.state["api_calls"]=self._api_count()
            self.save(functional_preflight_passed=True)
            self.commit("test(scientific): verify real API sign, missingness and unit response")
        incumbent=read_json(self.repo/BASE)
        self.controls(self.protocol["batches"]["screen"])
        a=[m["id"] for m in self.modules if m["role"]=="Analyst"]
        dg=[m["id"] for m in self.modules if m["role"]=="Designer" and m["lane"]=="G"]
        du=[m["id"] for m in self.modules if m["role"]=="Designer" and m["lane"]=="U"]
        a=self.sequential("A", "Analyst", a, incumbent, offset=0)
        frozen=self.freeze_analyst(a)
        a=self.state.get("frozen_analyst_active",a)
        dg=self.sequential("D-G", "Designer", dg, incumbent, frozen=frozen, offset=9)
        gcase=self.optional_condition("selected_G_fresh","Designer",dg,incumbent,frozen=frozen,focal=dg or ["D4.1"])
        g=self.optional_selection(gcase,incumbent,"G",self.protocol["batches"]["shape_selection"])
        # Dose calls retain selected G instructions but can modify only dose.
        combined=dg+du
        combined=self.sequential("D-U","Designer",combined,g,lane="U",frozen=frozen,offset=15)
        finalcase=self.optional_condition("selected_U_fresh","Designer",combined,g,lane="U",frozen=frozen,focal=du or ["D3.1"])
        final=self.optional_selection(finalcase,g,"final",self.protocol["batches"]["final"])
        selection_path=self.study/"final.selection.json"
        selection_ok=finalcase is not None and selection_path.exists() and not any(
            r["phase"]=="final_selection" for r in self.state.get("stage_failures",[]))
        self.state["selected_workflow"]={"Analyst":a,"Designer":combined,"program":final,
            "candidate_accepted":read_json(selection_path)["accepted"] if selection_ok else False,
            "final_role_failure_fallback":not selection_ok}
        self.final_context(a,combined,final,frozen)
        self.final_report(final)
        self.save(status="complete_with_skips" if self.state["skipped_modules"] or self.state["stage_failures"] else "complete",phase="complete",task=None,
            no_universal_module_necessity_claim=True)
        self.write_report()
        self.commit("report(scientific): complete sequential submodule study and final comparisons")

    def final_context(self,a,d,incumbent,frozen):
        """Retest retained functions; failed modules stay explicitly unresolved."""
        records=self.state.setdefault("final_context",[])
        for step,module in enumerate(a+d):
            if any(r["module"]==module for r in records):continue
            if self._module_is_skipped(module):
                records.append({"module":module,"classification":"SKIPPED_EXECUTION_ERROR", "causal_claim_supported":False})
                self.save();continue
            try:
                self._final_context_step(a,d,incumbent,frozen,module,step)
            except (ModuleExecutionError,LLMInterfaceError,jsonschema.ValidationError,ValueError) as error:
                self.skip_module(module,"final_context",a if module.startswith("A") else d,error)
                records.append({"module":module,"classification":"SKIPPED_EXECUTION_ERROR","causal_claim_supported":False})
                self.save()
            except RuntimeError as error:
                if isinstance(error,ExecutionBudgetExceeded) or not str(error).startswith(("FLOWR failed:","Remote driver exited:")):
                    raise
                self.skip_module(module,"final_context",a if module.startswith("A") else d,
                    ModuleExecutionError(self.state.get("task"),"FLOWR",error))
                records.append({"module":module,"classification":"SKIPPED_EXECUTION_ERROR","causal_claim_supported":False})
                self.save()

    def _final_context_step(self,a,d,incumbent,frozen,module,step):
        records=self.state.setdefault("final_context",[])
        m=next(m for m in self.modules if m["id"]==module)
        role=m["role"];active=a if role=="Analyst" else d;lane=m["lane"]
        if role=="Analyst":lane="G"
        prefix="final_context_"+module
        full=self.condition(prefix+"_full",role,active,incumbent,lane=lane,frozen=frozen,focal=[module],designer_context=d)
        drop=self.condition(prefix+"_drop",role,[x for x in active if x!=module],incumbent,lane=lane,frozen=frozen,focal=[module],designer_context=d)
        b=self.protocol["batches"]
        start=b["final_removal_base"]+b["confirmation_stride"]*step
        batches=list(range(start,start+b["confirmation_count"]))
        ff=self.condition_frames(full,batches);fd=self.condition_frames(drop,batches)
        passed=full["functional"] is not None and sum(x["score"]==2 for x in full["functional"]["assessments"])>=5
        secondary=[self.strain_contrast(ff,fd,paired_roles=role=="Designer")]
        mm=[self.frames(self.controls(batches))]
        record={"module":module,"batches":batches,"comparison_policy":"full_drop_only",
            "classification":classify_module(self.contrast(ff,fd,paired_roles=role=="Designer"),
                functional_valid=passed,secondary=secondary),
            "full_minus_drop":self.contrast(ff,fd,paired_roles=role=="Designer"),
            "drop_minus_M2":self.contrast(fd,mm),
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
        skipped=self.state.get("skipped_modules",[])
        pd.DataFrame(rows).to_csv(self.study/"submodule_results.csv",index=False)
        pd.DataFrame(skipped).to_csv(self.study/"skipped_modules.csv",index=False)
        processed=len({r["module"] for r in self.state.get("pruning",[])}|{r["module"] for r in skipped})
        write_json(self.study/"study_summary.json",{"status":self.state.get("status"),"model":"deepseek-flash",
            "comparison_policy":"full_drop_only","treatments":["full","drop"],
            "completed_submodules":len(rows),"total_submodules":17,"api_calls":self.state.get("api_calls",0),
            "processed_submodules":processed,"skipped_submodules":len(skipped),"skipped_modules":skipped,
            "stage_failures":self.state.get("stage_failures",[]),
            "independent_inference_cohorts":len(self.state.get("program_results",{})),"results":rows,
            "final_context":self.state.get("final_context",[]),"selected_workflow":self.state.get("selected_workflow"),
            "performance_goal_not_assumed":True,"raw_outputs_retired_after_verification":True})
        lines=["# 科学子模块消融运行状态", "", "模型：真实 DeepSeek API / deepseek-flash。", "",
            f"状态：{self.state.get('status')}；已完成比较 {len(rows)}/17；跳过 {len(skipped)}；已处理 {processed}/17。", "",
            "|模块|功能验证|确认结论|是否精简|", "|---|---|---|---|"]
        lines += [f"|{r['module']}|{r['function_pass']}|{r['confirmation_class'] or r['status']}|{r['adopted_removal']}|" for r in rows]
        lines += [f"|{r['module']}|执行失败|SKIPPED_EXECUTION_ERROR（因果未确认）|False|" for r in skipped]
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
