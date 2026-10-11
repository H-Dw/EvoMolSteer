"""Frozen scientific-submodule plans and study activation."""
import copy
import csv
from pathlib import Path

from evomolsteer.continuous.m2_skill_api import BASE, REF
from evomolsteer.io import digest, read_json, write_json
from .modules import MODULES, assemble
from .evidence import build, historical_control

PLAN = "docs/plans/m2_scientific_ablation_20261011_v3"
PARENT = "docs/plans/m2_skill_ablation_20261011_fixed_temperature_v2"


def create_plan(repo):
    repo = Path(repo).resolve()
    out = repo/PLAN
    if (out/"artifact_manifest.json").exists():
        validate_plan(repo)
        return out
    old = read_json(repo/PARENT/"protocol.json")
    protocol = {"schema_version": "scientific-submodule-ablation-3.0", "execution_enabled": True,
        "authorization": "User requested implementation and experimental sequential module pruning",
        "parent_plan": PARENT, "parent_protocol_sha256": digest(repo/PARENT/"protocol.json"),
        "agent_model": "deepseek-flash", "agent_backend": "real_api", "model_fallback": False,
        "chains": 6, "wordings": 2, "fresh_chains_per_wording": 3,
        "baseline": old["execution_baseline"], "temperature_policy": old["temperature_policy"],
        "module_ids": [m["id"] for m in MODULES], "modules": "module_registry.json",
        "phase_order": ["matched controls", "Analyst sequential deletion/alternative",
            "freeze six selected Analyst outputs", "Designer shape deletion/alternative",
            "freeze independently confirmed shape", "Designer dose deletion/alternative",
            "independent final confirmation", "final-context removal validation"],
        "frozen_input_policy": "Same full evidence for every intervention; Designer conditions share six actual selected Analyst outputs",
        "designer_in_analyst_phase": "Frozen full G submodule instructions; no outcome access",
        "lanes": {
            "G": {"maximum_edits": 1, "fixed_dose": "supplied incumbent eta/ramp",
                "allowed": {"teacher_neighbors": [2,4,8], "pointcloud_delta_A": [.5,1.,2.],
                    "region_index": "evidence-bound landmark", "temporal_region_index": "evidence-bound landmark and verified continuous salience",
                    "temporal_teacher_blend": [.25,.5,1.]},
                "teacher_source_edit": "Only registered same-clock static/trajectory mixture from frozen historical sources"},
            "U": {"maximum_edits": 1, "fixed_shape": "confirmed selected G program/reference",
                "allowed": {"native_rms_ratio": [.2,.33,.5], "time_ramp_power": [0.,1.,2.]}}},
        "batches": {"screen": [2100,2101,2102,2103],
            "confirmation_base": 2200, "confirmation_stride": 16, "confirmation_count": 8,
            "final": list(range(2800,2816)), "final_removal_base": 3000,
            "batch_size": 50, "master_seed": 42, "seed_rule": "42+100003*batch",
            "unused_indices_checked_before_execution": True},
        "evaluation": {"primary": "All attempted decoded-final predicted target pIC50; coverage/missingness explicit",
            "M2_noninferiority_margin_pic50": .02, "small_random_fluctuation": "Confirmed paired CI lower > -0.02",
            "native_superiority": "Independent paired CI lower > 0",
            "historical_Steer": "Unpaired frozen reference; mean superiority reported with independent cohort uncertainty, not equal online budget",
            "historical_Steer_mean": 7.651480407714844,
            "historical_Steer_metrics": "docs/experiments/static_combo_20261010/observed_evidence.json",
            "elite_threshold": old["evaluation"]["elite_threshold"],
            "familywise_alpha": .05, "confirmation_families": 108,
            "multiplicity_scope": "54 initial and 54 final-context comparisons; final recipe confirmation is a separately declared family",
            "interval_method": "Crossed role-chain/generation-batch bootstrap, conservative Bonferroni tails; independent confirmation per pruning step",
            "bootstrap_replicates": 100000, "equivalence_band_pic50": [-.02,.02],
            "necessity_effect_pic50": .05, "secondary_strain_relative_margin": .10,
            "PB_rate_margin": .02, "validity_rate_margin": .02,
            "inconclusive_is_not_null": True, "bad_alternative_is_not_necessity": True},
        "pruning": {"screen_is_selection_only": True, "require_six_valid_chains": True,
            "confirm_against": ["current full policy", "matched frozen M2"],
            "functional_nonresponse": "record bottleneck; never claim scientific ineffectiveness",
            "equal_programs": "reuse actual inference; contextual instruction redundancy only",
            "no_silent_incumbent_success": True, "negative_candidate": "archive, leave active mask/incumbent unchanged",
            "final_context_requires_retest": True, "confirmation_outcomes_never_enter_role_inputs": True},
        "budgets": {"maximum_api_role_calls": 1800, "maximum_distinct_inference_cohorts": 240,
            "maximum_pruning_steps": 17, "maximum_final_context_passes": 1,
            "no_unbounded_outcome_driven_retry": True},
        "storage": "Compact source-bound evidence, one temporal reference, float64 summaries; retire owned raw inference after verified local evaluation",
        "protected": ["original Steer", "checkpoints", "old experiment reports", "unrelated tasks"],
        "fixed_correctness_not_research_modules": True,
        "no_claim_before_confirmation": True}
    out.mkdir(parents=True, exist_ok=True)
    write_json(out/"protocol.json", protocol)
    write_json(out/"module_registry.json", {"schema_version": "scientific-submodules-1.0", "modules": MODULES})
    contract = (repo/PARENT/"contracts/shared_contract.md").read_text(encoding="utf-8")
    # The new temporal recipe explicitly replaces the v2 no-new-target limit;
    # all invariant FLOWR/score/temperature requirements remain common.
    (out/"contracts").mkdir(exist_ok=True)
    (out/"contracts/shared_contract.md").write_text(contract +
        "\nRegistered temporal teacher/attention recipes use only supplied historical observed-node coordinates or validated contrast functions. "
        "They are coordinate rewards, not trajectory velocity forces. No current study outcome is available to either role.\n", encoding="utf-8")
    for role in ["Analyst", "Designer"]:
        dest=out/"skills"/role.lower();dest.mkdir(parents=True,exist_ok=True)
        text=(f"---\nname: scientific-{role.lower()}\ndescription: "
            + ("Interpret supplied selection coordinate evidence using independently testable scientific modules." if role=="Analyst" else
               "Design bounded coordinate rewards from supplied scientific hypotheses using independently testable modules.")
            + f"\n---\n\nYou are {role}. Read the external task, supplied evidence and response interface.\n\n"
            + assemble(MODULES,role,[m["id"] for m in MODULES if m["role"]==role])+"\n")
        (dest/"SKILL.md").write_text(text,encoding="utf-8")
    rows=[{"id":"F0_M2","phase":"controls","role":"none","lane":"G","module":"","intervention":"frozen"},
          {"id":"N0_native","phase":"controls","role":"none","lane":"G","module":"","intervention":"none"}]
    for phase,role,lane in [("A","Analyst","G"),("D-G","Designer","G"),("D-U","Designer","U")]:
        rows.append({"id":phase+"_full","phase":phase,"role":role,"lane":lane,"module":"","intervention":"full"})
        rows.append({"id":phase+"_empty","phase":phase,"role":role,"lane":lane,"module":"","intervention":"empty"})
        for m in MODULES:
            if m["role"]==role and m["lane"]==lane:
                for kind in ["drop","alternative"]:
                    rows.append({"id":phase+"_"+kind+"_"+m["id"],"phase":phase,"role":role,"lane":lane,
                        "module":m["id"],"intervention":kind})
    with (out/"conditions.csv").open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write_json(out/"functional_validation.json", {"score": {"0":"absent/wrong", "1":"generic mention", "2":"evidence-bound decision consequence", "NA":"input unavailable"},
        "pass": "5 of 6 full calls with score 2 for the focal function; alternatives assessed separately",
        "drop": "same evidence and non-module-specific response schema; spontaneous function is measured",
        "fixtures": ["actual frozen evidence", "sign-changed diagnostic copy", "missing-input diagnostic copy", "scale-changed diagnostic copy"],
        "fixture_execution": "Bounded real-API diagnostic probes before scientific conditions; quarantined from compilation/inference",
        "fixtures_are_not_generation_evidence": True, "rubric_not_in_role_prompt": True,
        "module_effect_chain": ["instruction delivered","role function","handoff","operative reward","actual injection","decoded final quality"]})
    (out/"plan.zh-CN.md").write_text(PLAN_TEXT,encoding="utf-8")
    files=[p for p in out.rglob("*") if p.is_file()]
    write_json(out/"artifact_manifest.json",{"files":[{"path":p.relative_to(out).as_posix(),"sha256":digest(p)} for p in sorted(files)]})
    return out


def validate_plan(repo):
    root=Path(repo)/PLAN
    for item in read_json(root/"artifact_manifest.json")["files"]:
        if digest(root/item["path"])!=item["sha256"]:
            raise ValueError("Frozen scientific plan changed: "+item["path"])


def prepare(repo, study):
    repo,study=Path(repo).resolve(),Path(study).resolve()
    validate_plan(repo)
    if (study/"activation.json").exists():
        validate_activation(repo,study);return read_json(study/"activation.json")
    p=read_json(repo/PLAN/"protocol.json")
    for name,key in [(BASE,"program_sha256"),(REF,"reference_sha256")]:
        if digest(repo/name)!=p["baseline"][key]:raise ValueError("Frozen M2 changed")
    packet=build(repo,study/"evidence")
    from .observables import build as build_observable_definitions
    definitions=build_observable_definitions(repo,study)
    historical=historical_control(repo,study/"evaluation_controls")
    pcopy=copy.deepcopy(p)
    write_json(study/"module_registry.json",read_json(repo/PLAN/"module_registry.json"))
    activation={"schema_version":"scientific-activation-3.0","plan":PLAN,
        "protocol_sha256":digest(repo/PLAN/"protocol.json"),"protocol":pcopy,
        "packet_sha256":digest(study/"evidence/packet.json"),"screen_batches":p["batches"]["screen"],
        "observable_definitions_sha256":digest(definitions),
        "bound_inputs":packet["source_sha256"],"model":"deepseek-flash","frozen_M2":BASE,
        "module_registry_sha256":digest(study/"module_registry.json"),
        "historical_control_sha256":digest(historical),
        "status":"CONFIGURED","data_prerequisites":packet["prerequisites"]}
    write_json(study/"activation.json",activation)
    return activation


def validate_activation(repo,study):
    repo,study=Path(repo).resolve(),Path(study).resolve();validate_plan(repo)
    a=read_json(study/"activation.json")
    if a["plan"]!=PLAN or a["protocol_sha256"]!=digest(repo/PLAN/"protocol.json") or a["model"]!="deepseek-flash":
        raise ValueError("Scientific activation identity mismatch")
    if digest(study/"evidence/packet.json")!=a["packet_sha256"]:
        raise ValueError("Scientific evidence packet changed")
    if not a.get("observable_definitions_sha256"):
        raise ValueError("Source-bound observable definitions required before scientific calls")
    if digest(study/"evidence/observable_definitions.json")!=a["observable_definitions_sha256"]:
        raise ValueError("Observable definitions changed after activation")
    if a.get("observable_definitions_sha256"):
        for path,sha in read_json(study/"evidence/observable_definitions.json")["source_code_sha256"].items():
            if digest(repo/path)!=sha:raise ValueError("Observable extractor changed: "+path)
    if digest(study/"module_registry.json")!=a["module_registry_sha256"] or digest(study/"evaluation_controls/historical_steer.metrics.parquet")!=a["historical_control_sha256"]:
        raise ValueError("Frozen module registry or historical control changed")
    for path,sha in a["bound_inputs"].items():
        if digest(repo/path)!=sha:raise ValueError("Bound input changed: "+path)
    for name,sha in read_json(study/"evidence/tool_receipt.json")["output_sha256"].items():
        if digest(study/"evidence"/name)!=sha:raise ValueError("Tool output changed: "+name)
    return a


def validate_prior_policy(repo,study,config,state):
    """Bind a declared prior RNG policy before any scientific result is used.

    The immutable scientific plan stays intact. A source-tracked amendment
    identifies a technical input-construction change and excluded warmup data.
    Resuming under another policy would invalidate paired comparisons.
    """
    mode=config.get("prior_seed_mode","legacy_stream")
    if mode not in {"legacy_stream","indexed_batch"}:
        raise ValueError("Unknown prior RNG policy")
    previous=state.get("prior_seed_mode")
    if previous is not None and previous!=mode:
        raise ValueError("Cannot mix prior RNG policies in a scientific study")
    path=Path(study)/"prior_policy_amendment.json"
    if mode=="indexed_batch":
        amendment=read_json(path)
        if (amendment["prior_seed_mode"]!=mode or
            amendment["master_seed"]!=42 or amendment["batch_seed_multiplier"]!=100003 or
            amendment["protocol_sha256"]!=digest(Path(repo)/PLAN/"protocol.json") or
            not amendment["fresh_matched_controls_required"]):
            raise ValueError("Unbound indexed-prior amendment")
        sha=digest(path)
        if state.get("prior_policy_amendment_sha256",sha)!=sha:
            raise ValueError("Prior amendment changed after activation")
        if previous is None and any(state.get(k) for k in ["conditions","measured","matched_controls","program_results"]):
            raise ValueError("Indexed prior cannot adopt unbound earlier results")
    else:
        sha=None
    return {"prior_seed_mode":mode,"prior_policy_amendment_sha256":sha}


PLAN_TEXT = """# M2 科学子模块消融 v3

本版由用户授权实施实验。父版和旧调用不覆盖。保留八个一级工作流模块，拆为九个 Analyst、八个 Designer 科学子模块。实际指令、竞争策略、数据前提和功能评分见 module_registry.json；条件见 conditions.csv。

所有条件读取同一冻结证据。确定性工具重新利用原紧凑批次矩：全窗口连续趋势使用独立批次留出选择低阶曲线，拟合失真时回退到实测节点插值；族系变化以群体差异变化=父子增量差异+turnover residual 做算术分解，残差包括重新分组和父群加权，不宣称因果选择贡献。保留全部弱/零/反向标量效应。不得把最后记录的后代 head 标签当作解码终态亲和力。

先逐项验证 Analyst，再冻结选定 Analyst 的六份新输出，逐项验证 Designer 的形状模块，独立确认奖励形状后再验证两个剂量模块。每次一个子模块，完整/删除/合理竞争方向读取相同数据；语义评审不读取本次性能。删除后自发出现同一功能时记录冗余。参数不生效或同一程序时共享生成，不增加独立分子样本。

v3 公开增加两个可执行时间接口，解决 v2 的时序信息无法落入奖励的问题：观察节点动态教师与 M2 固定教师的解析混合；固定教师的连续区域注意力。二者不增加 FLOWR 或 affinity head 前向。时间变化率绝不作为空间梯度。M2 程序和原教师保持冻结，两种温度仍为原默认值，不调温度。形状阶段固定剂量，剂量阶段固定已确认形状。

随机主种子 42，批次规则保持原协议；100 步，控制支持从学习元数据读取，结束后原生推理到终态。原始 Steer、模型和旧报告保护。每个任务本地提交/push后远端 pull；远端只做推理。仅在传输校验、执行审计、当地结构/能量评价与精简报告保存后删除本研究原始输出。

筛查每个不同程序 200 个尝试。每次拟删除指引必须在新的八批/400 个尝试上确认，相对当前完整策略及冻结 M2 的同时区间下界高于 -0.02 pIC50；匹配的 M2 始终重算或按同一真实凭据复用。最终另用未参与选择的16批/800个尝试，比较无引导、M2和历史Steer。历史Steer不成对、在线搜索预算不同；均值高于参考不能解释为极端高分路径搜索能力更强。

所有模块对比采用角色链和生成批次交叉重采样、族内保守同时区间。明确报告功能未响应、接口瓶颈、数值等价、统计等价、正/负效应和证据不足。宽区间不代表无效；坏的替代方向不单独证明必要性。默认指引只在确认不劣于 M2 后精简；否则归档候选并保留基准。最终上下文再次验证剩余模块。性能目标是均值优于无引导及历史Steer，保持或优于匹配 M2，允许上述小幅波动；这是目标，不是预先宣称已达到的结果。
"""
