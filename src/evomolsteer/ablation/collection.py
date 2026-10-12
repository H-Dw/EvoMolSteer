"""Read-only, source-bound collection of an ongoing full/drop study.

Role chains weight policies. They never multiply generated sample counts.
New pooled comparisons are exploratory; original module decisions are copied,
not recomputed, and no active experiment state or input is modified.
"""
import gzip
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from evomolsteer.io import read_json, write_json
from .statistics import compare, summarize


def merge_observations(frames):
    """Deduplicate only identical outcomes of one program/arm/attempt."""
    result = pd.concat(frames, ignore_index=True)
    keys = ["batch", "slot"]
    repeated = result.loc[result.duplicated(keys, keep=False)]
    for _, group in repeated.groupby(keys):
        if len(group.drop_duplicates()) != 1:
            raise ValueError("Conflicting observations for one numerical program and attempt")
    result = result.drop_duplicates(keys).sort_values(keys).reset_index(drop=True)
    scores = pd.to_numeric(result.pic50_on_rescore, errors="coerce")
    if not np.isfinite(scores).all():
        raise ValueError("All-attempt scores are required; never discard or impute failures")
    result["pic50_on_rescore"] = scores
    for name in ["valid_connected", "pb_fast_pass"]:
        values = result[name]
        if not values.dropna().isin([True, False, 0, 1]).all():
            raise ValueError("Explicit boolean quality labels required")
        result[name] = values.fillna(False).astype(bool)
    return result


def quality(frame, threshold):
    value = summarize([frame], threshold)["policy_mean"]
    valid = frame.valid_connected
    return dict(value, attempts=len(frame), independent_batches=frame.batch.nunique(),
                score_coverage=float(np.isfinite(frame.pic50_on_rescore).mean()),
                max_pic50=float(frame.pic50_on_rescore.max()),
                elite_valid_count=int((valid & frame.pic50_on_rescore.ge(threshold)).sum()),
                valid_count=int(valid.sum()), PB_count=int(frame.pb_fast_pass.sum()),
                valid_unique_graphs=int(value["valid_unique_graphs"]))


def historical_difference(candidate, historical, *, replicates=100000, family=3):
    """Independent batch bootstrap, unequal online generation budgets explicit."""
    a = candidate.groupby("batch").pic50_on_rescore.mean().to_numpy()
    b = historical.groupby("batch").pic50_on_rescore.mean().to_numpy()
    rng = np.random.default_rng(42)
    boot = []
    for begin in range(0, replicates, 1000):
        n = min(1000, replicates-begin)
        boot.append(a[rng.integers(0, len(a), (n, len(a)))].mean(1) -
                    b[rng.integers(0, len(b), (n, len(b)))].mean(1))
    boot = np.concatenate(boot)
    return {"delta_pic50": float(a.mean()-b.mean()), "CI95": np.quantile(boot,[.025,.975]).tolist(),
            "simultaneous_CI": np.quantile(boot,[.05/(2*family),1-.05/(2*family)]).tolist(), "family": family,
            "candidate_batches": len(a), "historical_batches": len(b),
            "unpaired": True, "equal_online_budget": False, "exploratory": True,
            "not_independent_final_recipe_confirmation": True}


def tool_outcome(action, observation):
    """A requested tool is not a delivered observation, especially on error."""
    if action.get("action") != "tool":
        return None
    if observation is None:
        return "missing_observation"
    if observation.get("ok") is False:
        return "error"
    return "delivered"


def evidence_coverage(packet, observed):
    """Coverage of supplied summary rows, not a completeness or quality score."""
    result = {}
    for name, table in packet["tables"].items():
        available = {dict(zip(table["columns"], r))["evidence_id"] for r in table["rows"]}
        selected = set(observed.get(name, set()))
        if not selected.issubset(available):
            raise ValueError("Observed evidence is not in the supplied source table")
        result[name] = {"available_rows": len(available), "delivered_unique_rows": len(selected),
                        "fraction": len(selected)/len(available) if available else None}
    return result


def collect(repo, study, output):
    repo=Path(repo).resolve()
    study=Path(study);output=Path(output)
    study=(study if study.is_absolute() else repo/study).resolve()
    output=(output if output.is_absolute() else repo/output).resolve()
    if output == study or output.is_relative_to(study):
        raise ValueError("Collection output must be separate from the active study")
    output.mkdir(parents=True, exist_ok=True)
    # Capture one state, rather than merging completed results with a live glob.
    state_bytes = (study/"run_status.json").read_bytes()
    state = json.loads(state_bytes)
    activation = read_json(study/"activation.json")
    modules = read_json(study/"module_registry.json")["modules"]
    labels = {m["id"]: m["label_zh"] for m in modules}
    threshold = activation["protocol"]["evaluation"]["elite_threshold"]
    inputs = {}
    def tracked(path):
        path = Path(path)
        blob = path.read_bytes()
        inputs[path.relative_to(repo).as_posix()] = hashlib.sha256(blob).hexdigest()
        return blob
    tracked(study/"activation.json"); tracked(study/"module_registry.json")
    packet=json.loads(tracked(study/"evidence/packet.json"))
    curve_source=json.loads(tracked(study/"evidence/continuous_functions.json"))
    feature_catalog={}
    for name, table in packet['tables'].items():
        if 'feature' in table['columns']:
            i=table['columns'].index('feature')
            feature_catalog[name]={row[i] for row in table['rows']}
    observable_map={}
    for table in packet['tables'].values():
        for row in table['rows']:
            record=dict(zip(table['columns'],row))
            observable_map[record['evidence_id']]=str(record.get('feature',''))
    # Preserve exact captured bytes and hash, without copying the large expanded state.
    (output/"state_snapshot.json.gz").write_bytes(gzip.compress(state_bytes,mtime=0))
    cohort_rows, grouped, seen = [], defaultdict(list), set()
    for result in state["program_results"].values():
        if result["name"] in seen: continue
        seen.add(result["name"])
        for arm, outcome in result["outcomes"].items():
            path = repo/outcome["metrics"]
            tracked(path)
            frame = merge_observations([pd.read_csv(path)])
            key = "unguided" if arm == "unguided" else result["effective_signature"]
            grouped[key].append(frame)
            cohort_rows.append({"cohort": result["name"], "program": key,
                "arm": arm, "code_commit": result["code_commit"], **quality(frame,threshold)})
    merged = {k: merge_observations(v) for k,v in grouped.items()}
    history_path = study/"evaluation_controls/historical_steer.metrics.parquet"
    tracked(history_path)
    historical = merge_observations([pd.read_parquet(history_path)])
    summaries, recipe_metadata = [], {}
    from evomolsteer.continuous.m2_skill_api import BASE, effective_signature
    base = read_json(repo/BASE); tracked(repo/BASE)
    base_signature = effective_signature(base)
    for key, frame in merged.items():
        if key == "unguided": label = "无引导"
        else:
            path = study/"programs"/(key+".json")
            program = read_json(path) if path.exists() else base
            if path.exists(): tracked(path)
            label = "冻结 M2" if key == base_signature else (
                "时间教师混合 "+str(program.get("temporal_teacher_blend"))
                if program.get("reward_view")=="endpoint_temporal_pointcloud" else key[:12])
            recipe_metadata[key] = {k:program.get(k) for k in ["reward_view","temporal_teacher_blend",
                "teacher_neighbors","native_rms_ratio","time_ramp_power","window","mixture_temperature",
                "teacher_endpoint_temperature_A2","pointcloud_delta_A"]}
            recipe_metadata[key]["actual_pointcloud_delta_A"] = program.get("pointcloud_delta_A",1.)
        summaries.append({"program":key,"label":label,**quality(frame,threshold)})
    summaries.append({"program":"historical_Steer","label":"历史 Steer",**quality(historical,threshold)})
    comparisons = {}
    gradient_programs=[key for key in merged if key != 'unguided']
    paired_family=max(3,len(gradient_programs)+sum(key != base_signature for key in gradient_programs))
    historical_family=max(3,len(gradient_programs))
    for key, candidate in merged.items():
        if key == "unguided":continue
        for reference in ["unguided", base_signature]:
            if reference == key or reference not in merged:continue
            shared = sorted(set(candidate.batch)&set(merged[reference].batch))
            a=candidate.loc[candidate.batch.isin(shared)];b=merged[reference].loc[merged[reference].batch.isin(shared)]
            if len(shared)<2:continue
            values=compare([a],[b],family=paired_family,replicates=100000)
            comparisons[key+"_minus_"+reference] = dict(values, matched_attempts=len(a),
                candidate_mean=float(a.pic50_on_rescore.mean()), reference_mean=float(b.pic50_on_rescore.mean()),
                exploratory=True, not_independent_final_recipe_confirmation=True)
        comparisons[key+"_minus_historical_Steer"] = historical_difference(candidate,historical,family=historical_family)
    completed, functions = [], []
    for item in state.get("pruning",[]):
        full=state["conditions"][item["full"]];drop=state["conditions"][item["drop"]]
        confirm=item.get("confirmation") or {};delta=confirm.get("drop_minus_full",{})
        ci=delta.get("simultaneous_CI",[None,None])
        row={"module":item["module"],"label":labels[item["module"]],"phase":item["phase"],
             "full_function_pass":item["function_pass"],"full_score2":sum(x["score"]==2 for x in full["functional"]["assessments"]),
             "drop_score2":sum(x["score"]==2 for x in drop["functional"]["assessments"]),
             "drop_minus_full_pic50":delta.get("delta_pic50"),"CI_low":ci[0],"CI_high":ci[1],
             "confirmation_batches":delta.get("independent_generation_batches"),
             "classification":confirm.get("classification"),"adopted_removal":item["decision"]["adopt"],
             "decision":item["decision"]["status"],"full_programs":";".join(x["signature"][:12] for x in full["rows"]),
             "drop_programs":";".join(x["signature"][:12] for x in drop["rows"])}
        completed.append(row)
    for case,condition in state["conditions"].items():
        if not condition.get("functional"):continue
        for a in condition["functional"]["assessments"]:
            functions.append({"case":case,"role":condition["role"],"module":a["module"],
                "chain":a["chain"],"score":a["score"],"quote":a["quote"],"explanation":a["explanation"]})
    tools, coverage, curves, regions = Counter(), defaultdict(set), set(), set()
    tool_outcomes, tool_errors, metadata_reads = defaultdict(Counter), [], Counter()
    decisions = Counter();edits=Counter();completed_workflows=0
    for folder in sorted((study/"calls").glob('*/*/*.tools_v1')):
        if not (folder/"completed.json").exists():continue
        completed_workflows+=1;role=folder.name.split('.')[0]
        for p in folder.glob("step_*.json"):
            if not re.fullmatch(r"step_\d+\.json",p.name):continue
            action=json.loads(tracked(p))
            if action.get("action")=="tool":
                key = role+":"+action["tool"]
                tools[key] += 1
                if action['tool']=='read_metadata':
                    metadata_reads[role+':'+action['arguments']['section']] += 1
                observation_path = p.with_name(p.stem+".observation.json")
                observation = json.loads(tracked(observation_path)) if observation_path.exists() else None
                outcome = tool_outcome(action, observation)
                tool_outcomes[key][outcome] += 1
                if outcome != "delivered":
                    tool_errors.append({"path":p.relative_to(repo).as_posix(), "role":role,
                        "tool":action["tool"], "arguments":action.get("arguments"),
                        "outcome":outcome, "error":(observation or {}).get("error")})
        if role=="Designer":
            path=folder/"Designer.json";response=json.loads(tracked(path))
            decisions[response["decision"]]+=1
            if response["edit"]:edits[response["edit"]["parameter"]+"="+str(response["edit"]["value"])]+=1
        else:
            provenance=json.loads(tracked(folder/"references.json"))
            for identity in provenance["references"].values():
                if identity.get("table"):coverage[identity["table"]].add(identity["source_id"])
                if identity["source_id"].startswith('curve/'):
                    curves.add(identity["source_id"][6:])
                match=re.search(r"landmark_(\d+)_",observable_map.get(identity['source_id'],identity['source_id']))
                if match:regions.add(int(match.group(1)))
    receipts=[];repair_count=0
    for p in study.rglob("*.receipt.json"):
        try:value=read_json(p)
        except (ValueError,OSError):continue
        if value.get("backend")=="real_api":receipts.append(p)
        repair_count+=len(value.get("interface_repairs",[]))
    failures=[]
    # Recover real semantic causes without a new API call or modified decision.
    from evomolsteer.agent_workflow.reviewer import source_cards, extract_review
    for failure in state.get("skipped_modules",[]):
        record=dict(failure)
        if record["error"]=="Object of type Unset is not JSON serializable" and record.get("case"):
            case=record["case"]
            invalid=study/"reviews"/(case+"_Analyst.tools_v1.attempt_1.invalid.txt")
            if invalid.exists():
                try:
                    calls=[read_json(study/"calls"/case/f"chain_{i:02d}"/"Analyst.json") for i in range(6)]
                    extract_review(json.loads(tracked(invalid)),calls,[{"module":record["module"]}],source_cards(calls))
                except Exception as e:record["recovered_semantic_cause"]=type(e).__name__+": "+str(e).splitlines()[0]
        if record["error"]=="Program signature collision":
            record["technical_cause"]="Effective signature inserts default delta=1, while compiler compares full JSON dictionaries. Explicit delta=1 versus absent default share a signature but differ in serialization; this is a no-op canonicalization bug, not a scientific collision."
        failures.append(record)
    tool_report={"completed_role_workflows_at_collection":completed_workflows,
        "tool_actions":dict(tools),"Analyst_distinct_rows_by_table":{k:len(v) for k,v in coverage.items()},
        "tool_outcomes":{k:dict(v) for k,v in tool_outcomes.items()}, "tool_errors":tool_errors,
        "metadata_reads":dict(metadata_reads),
        "Analyst_table_coverage":evidence_coverage(packet,coverage),
        "curve_catalog_audit":{
            "available_curve_features":len(curve_source['functions']),
            "example_curve_features":sorted(curve_source['functions'])[:12],
            "endpoint_features":len(feature_catalog.get('endpoint_effects',set())),
            "endpoint_vs_continuous_same_name_features":len(
                feature_catalog.get('endpoint_effects',set()) & feature_catalog.get('continuous_trends',set())),
            "all_curve_names_in_continuous_trends":set(curve_source['functions']).issubset(
                feature_catalog.get('continuous_trends',set()))},
        "Analyst_curve_features":sorted(curves),"Analyst_landmark_regions":sorted(regions),
        "Designer_decisions":dict(decisions),"Designer_edits":dict(edits),
        "successful_api_receipts_at_collection":len(receipts),"successful_receipt_format_repairs":repair_count,
        "invalid_response_files":len(list(study.rglob('*.invalid.txt'))),
        "workflow_and_request_counts_are_observed_later_than_state_snapshot":True}
    # Include only response audits of cohorts already present in the captured state.
    responses = []
    for name in sorted(seen):
        path = study/"transport_records"/(name+".response_vs_M2.json")
        if not path.exists():
            continue
        raw = json.loads(tracked(path))
        time_curve = raw["whole_window_observed_node_RMS"]
        responses.append({"cohort":name,
            **{k:raw[k] for k in ["same_numerical_program", "mean_forecast_difference_z",
                                  "max_forecast_difference_z", "paired_observed_nodes"]},
            "observed_time_range":[min(time_curve["times"]), max(time_curve["times"])],
            "last_observed_node_RMS_z":time_curve["RMS_z"][-1],
            "top_changed_observables":raw["top_changed_observables"][:3]})
    pd.DataFrame(cohort_rows).to_csv(output/"cohort_quality.csv",index=False)
    pd.DataFrame(summaries).to_csv(output/"program_quality.csv",index=False)
    pd.DataFrame(completed).to_csv(output/"module_results.csv",index=False)
    pd.DataFrame(functions).to_csv(output/"functional_response.csv",index=False)
    for name,value in [("program_comparisons.json",comparisons),("program_recipes.json",recipe_metadata),
                       ("tool_usage.json",tool_report),("execution_failures.json",failures),
                       ("actual_response.json",responses)]:write_json(output/name,value)
    snapshot={"state_sha256":hashlib.sha256(state_bytes).hexdigest(),"state_updated_unix":state["updated_unix"],
        "state_snapshot":"state_snapshot.json.gz",
        "agent_model":activation['model'],"agent_backend":activation['protocol']['agent_backend'],
        "agent_interface":activation['protocol']['agent_interface'],
        "role_chains_per_condition":activation['protocol']['chains'],
        "treatments":activation['protocol']['comparison_policy']['treatments'],
        "collection_time_Asia_Shanghai":datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "study":study.relative_to(repo).as_posix(),"status":state["status"],"phase":state.get("phase"),
        "task":state.get("task"),"completed_comparisons":len(completed),"skipped_comparisons":len(failures),
        "total_submodules":len(modules),"completed_cohorts":len(seen),
        "executed_gradient_programs":len([k for k in merged if k!='unguided']),
        "generated_attempts_deduplicated":sum(len(f) for f in merged.values()),
        "historical_attempts":len(historical),"elite_valid_threshold":threshold,
        "ongoing_study":state["status"] not in {"complete","complete_with_skips"},
        "source_files":inputs,"module_effect_CI_source":"Frozen original decisions, not post-hoc pooled comparisons",
        "role_output_counts_cutoff":"Later collection boundary; role outputs may exist without evaluated conditions"}
    snapshot['Designer_D1.1_confirmation_completed']=any(x['module']=='D1.1' for x in completed)
    write_json(output/"source_manifest.json",snapshot)
    write_report(output,snapshot,summaries,completed,failures,comparisons,tool_report,responses,modules)
    return snapshot


def write_report(output,snapshot,summaries,completed,failures,comparisons,tools,responses,modules):
    def ci(values):return f"[{values[0]:+.4f}, {values[1]:+.4f}]"
    lines=["# M2 完整/删除消融结果收集（阶段快照）","",
        f"收集时间：{snapshot['collection_time_Asia_Shanghai']}。状态：{snapshot['status']}；阶段：{snapshot['phase']}。",
        f"已完成 {len(completed)}/{snapshot['total_submodules']} 个模块的比较，另 {len(failures)} 个因程序异常跳过。研究仍在进行，未完成最终奖励确认或最终上下文复核。","",
        f"已评价 {snapshot['completed_cohorts']} 个独立生成队列，去重后 {snapshot['generated_attempts_deduplicated']:,} 次生成尝试。"
        "同一奖励在六条角色链中重复出现，不算六倍分子样本；历史 Steer 独立列出。","",
        f"实际角色接口：{snapshot['agent_backend']} / {snapshot['agent_model']} / {snapshot['agent_interface']}。"
        f"每条件 {snapshot['role_chains_per_condition']} 条空白上下文角色链，仅比较完整与删除，没有替代条件或温度消融；"
        "旧研究结果不混入本次模块选择。主随机种子 42，按冻结批次索引派生独立初始状态，同一批次程序与对照配对。","",
        "## 已完成生成的质量","",
        "下表为去重后的全部生成尝试；应变为收敛 MMFF 评价的每重原子能量缓解量，报告总体中位数/P90，缺失能量不填零。"
        "该指标不同于各模块确认中独立批次分位数均值。","",
        "|程序|尝试数|平均 pIC50|P95|有效率|PB 通过率|应变中位数|应变 P90|高亲和力有效率|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s in summaries:
        energy=lambda k:f"{s[k]:.4f}" if s[k] is not None else "缺失"
        lines.append(f"|{s['label']}|{s['attempts']}|{s['mean_pic50']:.4f}|{s['p95_pic50']:.4f}|{s['valid_rate']:.2%}|{s['PB_rate']:.2%}|{energy('strain_median')}|{energy('strain_p90')}|{s['elite_valid_rate']:.2%}|")
    lines += ["",f"高亲和力阈值是冻结协议的 {snapshot['elite_valid_threshold']:.6f} pIC50，失败尝试保留在分母。所有已纳入尝试均有模型评分；评分覆盖不等于结构有效。",
              "", "## 配对及历史比较", "",
              "以下为已收集批次的探索性汇总，不替代预注册的独立最终确认。历史比较不成对，在线预算不同；最大值不用于宣称更强性能。"
              "配对比较与历史比较分别按至少三项的比较族校正，不代表整张表及所有模块联合的区间覆盖率。","",
              "|比较|匹配尝试数|平均 pIC50 差值|族内同时区间|族内校正项数|","|---|---:|---:|---|---:|"]
    label={s['program']:s['label'] for s in summaries}
    for key,value in comparisons.items():
        a,b=key.split('_minus_',1)
        lines.append(f"|{label[a]} − {label[b]}|{value.get('matched_attempts','不成对')}|{value['delta_pic50']:+.4f}|{ci(value['simultaneous_CI'])}|{value['family']}|")
    lines += ["","## 已完成的模块删除效果","",
        "差值统一为删除减完整；正值表示删除条件平均分更高。区间来自原实验的跨角色链/生成批次 bootstrap 及 204 个比较族校正。"
        "小差异和跨零区间不证明指引无效；非劣也不等于优效或双向等价。","",
        "|模块|功能|完整/删除功能达标链数|Δ pIC50|同时区间|当前处理|",
        "|---|---|---:|---:|---|---|"]
    for x in completed:
        handling="暂时删除" if x['adopted_removal'] else "保留，尚无明确结论"
        if x['decision']=='CONFIRMED_SAFE_INSTRUCTION_PRUNING_SCIENCE_UNRESOLVED':handling="删除文本冗余，科学作用未检验充分"
        elif x['decision']=='CONFIRMED_NONINFERIOR_PRUNING':handling="当前上下文非劣，暂时删除"
        lines.append(f"|{x['module']}|{x['label']}|{x['full_score2']}/6 → {x['drop_score2']}/6|{x['drop_minus_full_pic50']:+.4f}|{ci([x['CI_low'],x['CI_high']])}|{handling}|")
    lines += ["", "A2.2 的完整指引多数链执行了反例/跨批检查，删除后该功能显著减少；当前确认支持的是删除后的非劣，不能称其有负面效应。",
        "A3.1 的完整条件本身未达到功能要求，且完整/删除全部采用同一 M2 程序；零差值来自共享的真实生成，不是十二次独立验证。",
        "D1.1 完整条件六条链都未获得功能满分，删除条件却全部满分；两者仍输出相同 M2 数值程序。该结果暴露了功能评分与执行奖励之间的断层，不能证明坐标 observable 到公式的映射没有科学价值。",
        "A1.1、A2.1、A4.3 删除后仍执行类似功能，提示指令可能冗余或被其他模块覆盖；现有性能区间仍不足以确认删除安全。","",
        "## 响应到可执行奖励的瓶颈","",
        f"收集边界的 Designer 决定：{dict(tools['Designer_decisions'])}；实际执行的梯度奖励仅 {snapshot['executed_gradient_programs']} 种。",
        "已执行的变化集中在时间教师混合权重 0.5，M2 的静态教师距离奖励仍占主导。混合权重不同于固定温度；温度未消融。",
        "时间混合候选在不同批次有正有负，当前汇总没有显示它优于 M2。部分条件只是改变采用该候选的角色链比例，不能称其设计出一种新的区域奖励。",
        ("Designer 的 D1.1 独立确认状态见模块表。" if snapshot['Designer_D1.1_confirmation_completed'] else
         "Designer 的 D1.1 两条件已经输出，但独立确认尚未完成。")+
        "当前两个条件的设计采用同一数值程序，Reviewer 评分变化说明解释评分不能替代可执行响应和质量证据。",
        "统计趋势符号反转本身不能推出应切换历史教师坐标，也不能将标准化对比曲线当成绝对构象目标。候选需说明奖励如何控制该 observable，并以实测响应检验。","",
        "## Agent 实际读取了什么", "",
        "以下统计来自实际调用记录及主机交付的引用，不根据 Skills 文本推测已完成的功能。观察边界稍晚于生成结果状态快照。"
        "读取比例是去重后的摘要行覆盖，不是信息质量指标；未查询额外表也不等于没有读到继承的 Analyst 摘要。", "",
        "|证据表|可用摘要行|Analyst 实际交付的不同摘要行|覆盖比例|",
        "|---|---:|---:|---:|"]
    for name, values in tools['Analyst_table_coverage'].items():
        fraction=f"{values['fraction']:.1%}" if values['fraction'] is not None else '缺失'
        lines.append(f"|{name}|{values['available_rows']}|{values['delivered_unique_rows']}|{fraction}|")
    retain=tools['Designer_decisions'].get('retain',0)
    total=sum(tools['Designer_decisions'].values())
    retain_rate=f"{retain/total:.1%}" if total else '缺失'
    lines += ["",
        f"Designer 共 {total} 份完成输出，其中 {retain} 份保留原奖励（{retain_rate}）；额外工具调用数为 "
        f"{sum(n for key,n in tools['tool_actions'].items() if key.startswith('Designer:'))}。Designer 接收了 Analyst 的主机提取交接数据，不能将零工具调用解释为没有输入。",
        f"Analyst 的 get_curve 请求 {tools['tool_actions'].get('Analyst:get_curve',0)} 次；实际交付完整曲线 {len(tools['Analyst_curve_features'])} 种。"
        f"其中 {sum(e['tool']=='get_curve' for e in tools['tool_errors'])} 次未能交付，原始请求与错误消息见 tool_usage.json。"
        "它仍读取了部分 continuous_trends 拟合摘要，不能说完全没有进行趋势分析。",
        f"实际可用完整曲线 {tools['curve_catalog_audit']['available_curve_features']} 种；"
        f"endpoint_effects 的 {tools['curve_catalog_audit']['endpoint_features']} 个特征与连续趋势表的同名交集为 "
        f"{tools['curve_catalog_audit']['endpoint_vs_continuous_same_name_features']}。"
        "当前失败请求将端点效应表里的 landmark_00_softmin 名称直接用于曲线查询；"
        "连续曲线实际使用 landmark_00_soft_mass、landmark_00_local_dx 等另一组局部空间 observable。"
        "这属于异构特征目录没有正确衔接，不是缺少整个曲线数据集。应从 continuous_trends 的真实 feature 字段发起查询，并在跨表分析时解释观测量的区别。",
        "区域富集 regional_effects、谱系周转 turnover_accounting、父子增量 increment_effects 没有通过当前工具查询交付；"
        "相比只阅读 endpoint_effects 摘要，Agent 尚未充分建立“即时选择→保留路径→坐标变化→亲和力收益”的证据链。",
        "Reviewer 功能满分主要判断最终文字中的证据和解释，不能保证读取了某张表或完整曲线。"
        "因此，A1.1 删除后仍满分，以及 A4.1/A4.2 的异常，均不能替代对应计算方法的有效性实验。",
        f"metadata 工具读取分布为 {tools['metadata_reads']}。当前没有通过该工具读取完整教师模式、数据限制或 prerequisites；"
        "这进一步限制了教师邻域、谱系证据和局部控制规则的机制解释。接口合同已提供的基本限制与冻结参数仍有效，不能说 Agent 完全不知道约束。",
        "", "## 实际引导是否改变生成", ""]
    changed=[r for r in responses if not r['same_numerical_program']]
    if changed:
        values=[r['mean_forecast_difference_z'] for r in changed]
        lines.append(f"已完成的 {len(changed)} 个时间混合队列与配对 M2 的端点预测几何差异确实非零；"
            f"队列级平均标准化差异为 {min(values):.3f}–{max(values):.3f}。"
            "这里比较的是注入前的模型端点预测摘要，按历史特征尺度标准化，不是每个原子的位移或个体轨迹重建。")
    lines += ["实际变化涉及形状矩、区域距离/占据及原子间距离核等 observable。"
        "这些特征用于响应诊断，当前执行的点云奖励没有对应的独立 motif 或距离核项。"
        "有几何响应但亲和力未优于 M2，说明当前时间混合方向尚无明确增益，不能归咎于引导完全未生效。",
        "", "## 当前两个执行奖励", "",
        "冻结 M2 在模型预测的端点坐标上，计算与教师点云的 Hungarian 匹配均方距离 q，"
        "经伪 Huber 距离代价 rho(q)=delta²[√(1+q/delta²)−1]，再计算 "
        "R=tau·logsumexp(log(pi)−rho/tau)。匹配及教师先验在求导时固定，通过真实 FLOWR 端点坐标 Jacobian 回传到当前坐标。",
        "实际参数为 delta=1 Å、固定 tau=0.5、邻居数 4、原生预测更新相对剂量 0.33、时间 ramp=0；"
        "教师先验的端点距离尺度固定为 4 Å²，评分权重 beta=0。控制支持来自配置的学习窗口（此实验为 0–0.5），之后完成无引导推理。"
        "没有 affinity head 求导或 Steer 重采样。",
        "已执行的新候选 R(t)=0.5·R_static+0.5·R_observed_teacher(t)，混合的是固定教师与同一时钟的历史端点预测教师；"
        "历史端点预测不能当成当时噪声状态 x(t)。其它固定参数不变。0.25 候选已编译但未进入生成，不能作为已完成的性能实验。",
        "", "## 程序异常","", "|模块|阶段|真实原因/已恢复原因|", "|---|---|---|"]
    for f in failures:
        cause=f.get('recovered_semantic_cause',f.get('technical_cause',f['error']))
        lines.append(f"|{f['module']}|{f['stage']}|{cause.replace('|','/')}|")
    lines += ["", "三个 Reviewer 的 Unset 异常发生在格式错误诊断序列化：原始错误包括引用其他链原文和缺失链覆盖。中性格式修复因序列化失败未正常发出。",
        "A4.1 是显式写入默认 delta=1 与省略默认值使用同一有效签名、却被完整 JSON 比较拒绝；不是奖励算法冲突，也不是删除趋势指引导致性能失败。",
        "这些失败不能用于证明对应 Skill 必要或无效，指引已保留。当前收集不修改运行中的 Skills、数值奖励、队列或跳过决定。","",
        "## 剩余任务与后续解释", ""]
    processed={x['module'] for x in completed}|{x['module'] for x in failures}
    pending=[m for m in modules if m['id'] not in processed]
    lines.append("尚未完成比较："+'、'.join(m['id']+'（'+m['label_zh']+'）' for m in pending)+"。")
    lines += ["先让当前完整/删除队列继续；执行异常单独作为接口缺陷，不能加入有效性能比较。"
        "后续应修复 Reviewer 诊断序列化及默认参数归一化，再用相同冻结输入补齐失败条件，避免悄悄改写本次结果。",
        "科学探索应先核实正确名称的曲线及区域/周转/增量证据实际交付，"
        "再验证 Designer 的控制函数是否改变指定 observable，最后以独立批次确认亲和力。"
        "这三个层次分别是工具调用、控制机制和生成质量，不能由文本判分互相替代。",
        "教师权重 beta=0 只表示不在教师先验中使用评分，不能推论选择效应表没有亲和力相关证据。"
        "同时，选择时刻的分数关联不等于已验证的终态亲和力收益；应保留这一差别，提出可否证的坐标控制假设，再交给独立生成评价。",
        "", "## 数据与结论边界","",
        "M2 相比无引导的平均亲和力和应变优势，首先属于冻结奖励本身；尚无充分证据归因于本轮新 Agent 指引。",
        "历史 Steer 的均值、尾部、有效率和应变须分开比较。新队列不等于相同在线预算，也尚没有零样本泛化证据。",
        "工具读取覆盖和 Reviewer 判分保存在 tool_usage.json 与 functional_response.csv；数据可访问不代表所有区域和证据已被角色读取。",
        "本报告仅收集已完成并评价的队列，当前远端运行中的队列不进入质量或样本数统计。源文件校验值与状态快照已保存，原始研究继续运行。",""]
    (output/"report.zh-CN.md").write_text('\n'.join(lines),encoding='utf-8')
