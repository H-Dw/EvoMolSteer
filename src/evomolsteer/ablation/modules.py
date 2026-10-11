"""Single-purpose scientific interventions, independent of target and host."""
import hashlib


def _module(identifier, label, instruction, alternative, prerequisite, rubric, lane="G"):
    return {"id": identifier, "parent": identifier.split(".")[0],
            "role": "Analyst" if identifier.startswith("A") else "Designer",
            "label_zh": label, "instruction": instruction,
            "instruction_sha256": hashlib.sha256(instruction.encode()).hexdigest(),
            "alternative_instruction": alternative, "prerequisite": prerequisite,
            "functional_rubric": rubric, "lane": lane, "status": "UNTESTED",
            "drop_changes_evidence": False}


MODULES = [
    _module("A1.1", "即时选择富集",
        "Prioritize signed favorable-versus-comparison coordinate contrasts at the observed selection nodes and over the complete observed window. Bind the population, label and observable; distinguish selection enrichment from an absolute geometric descriptor.",
        "Prioritize distinctive absolute geometry rather than selection enrichment. Describe available comparisons truthfully without using them to rank design directions.",
        "selection_contrasts", ["Actual contrast population and label", "Signed coordinate effect", "Comparison-bound evidence identifier"]),
    _module("A1.2", "后续质量贡献",
        "Compare immediate selection evidence with available descendant survival or quality credit. Let disagreement affect direction priorities. Identify the actual later-label calculation, ancestral coverage and survivor bias. Without decoded terminal labels, do not present last-record credit as final affinity benefit; report the unavailable comparison.",
        "Prioritize immediate selection evidence. Report later labels and their limitations truthfully, but do not let descendant survival or credit change the direction ranking.",
        "descendant_proxy", ["Immediate versus later evidence comparison", "Later-label provenance and missingness", "Priority consequence or explicit unavailable terminal comparison"]),
    _module("A2.1", "不确定性影响排序",
        "Use uncertainty, multiplicity-adjusted support and coverage to rank coordinate hypotheses. Show a concrete priority decision that differs from ranking point estimates alone, or explain why the supplied evidence does not permit one.",
        "Rank by absolute observed effect magnitude. Report uncertainty and adjusted support accurately, but do not use them to change the order.",
        "uncertainty", ["Real uncertainty and support", "Reliability changes priority rather than boilerplate"]),
    _module("A2.2", "跨批重复与反例",
        "Check independent-batch agreement and opposing observations before prioritizing a coordinate rule. Identify which counterexample would weaken a proposed direction, and let reproducibility or counterevidence affect its use.",
        "Prioritize the pooled average effect. State batch disagreement accurately without making reproducibility or opposing observations a selection criterion.",
        "batch_support", ["Actual batch support or opposing evidence", "Counterevidence affects use of a direction"]),
    _module("A3.1", "局部优势超出全局运动",
        "Separate local coordinate information from global pose and shape. Use supplied adjusted or joint diagnostics to decide whether a region adds information beyond common motion; name the extent and surrounding coordinates that should retain freedom. Do not claim an adjustment that the supplied tool did not perform.",
        "Prioritize global pose and whole-cloud shape. Report local observations without recommending selective regional emphasis.",
        "spatial_diagnostics", ["Local versus global distinction supported by actual tool", "Bound region or explicit unresolved localization", "Retained surrounding freedom"]),
    _module("A3.2", "协同空间 motif",
        "Compare coherent joint coordinate patterns with isolated marginal descriptors. Use supplied covariance, intact-conformation or mode diagnostics to identify coupling or redundant evidence; do not average incompatible conformations into an unsupported target.",
        "Prioritize individual marginal descriptors rather than joint patterns. Report correlations truthfully without using coupling or mode structure to choose the design direction.",
        "joint_geometry", ["Actual joint relationship or intact mode evidence", "Coupling affects hypothesis or target representation"]),
    _module("A4.1", "全窗口连续趋势",
        "Inspect the supplied continuous contrast functions across their complete observed support, including sign changes, fit fidelity and rate changes. Compare a constant rule with a time-conditioned rule; use validated fits or observed-node interpolation, without arbitrary time bins or extrapolation. A contrast curve is not an absolute reward target.",
        "Use the whole-window integrated contrast for a stationary rule. Report supplied time variation truthfully without proposing time-conditioned reward information.",
        "continuous_trends", ["Evidence-bound continuous function and support", "Fit fidelity or sign change assessed", "Stationary versus time-conditioned consequence"]),
    _module("A4.2", "谱系变化来源",
        "Compare population-contrast changes with supplied parent-child increment contrasts and their turnover residual. Explain whether the evidence supports within-path change or changing cohort composition, including reclassification and parent weighting. The residual is an accounting diagnostic, not a causal selection effect or atom velocity.",
        "Interpret population contrasts descriptively without using parent-child increments or turnover accounting to choose a coordinate rule.",
        "turnover_accounting", ["Observed edge and population comparison", "Residual semantics and limitations", "Consequence for proposed path rule"]),
    _module("A4.3", "可干预坐标假设",
        "Turn a supported finding into a testable coordinate intervention: observable, intended response, target origin, spatial or temporal scale, and a discriminating failure condition. Separate measurable contrasts from targets; identify which permitted reward recipe could express the hypothesis or why it remains descriptive.",
        "Return descriptive findings and uncertainty without recommending coordinate interventions, reward targets or prospective responses.",
        "selection_contrasts", ["Observable and intended response", "Target and scale origins", "Executable hypothesis or stated interface limitation"]),
    _module("D1.1", "坐标 observable 到公式",
        "Connect an operative reward edit to a specific Analyst hypothesis. Explain how the registered formula changes that coordinate observable and predict a numerical response; do not treat a diagnostic feature as an independent force that the formula does not contain.",
        "Choose a permitted generic coordinate-matching edit without using a specific Analyst contrast to select the formula or parameter.",
        "formula_registry", ["Specific Analyst hypothesis to operative edit", "Correct formula-to-observable relationship", "Numerical response prediction"]),
    _module("D1.2", "静态与时间条件目标",
        "Compare a stationary coordinate target with the permitted time-conditioned alternatives using Analyst's temporal evidence. State whether the proposal changes teacher coordinates, regional attention or only dose. Preserve the shared learned support, and do not interpret a temporal derivative as a spatial gradient.",
        "Use a stationary coordinate target and stationary regional attention. Describe temporal observations without choosing a time-conditioned target; dose edits remain governed by the execution lane.",
        "temporal_recipes", ["Static versus temporal decision", "Target/attention/dose distinction", "Support and source binding"]),
    _module("D2.1", "几何响应与曲率尺度",
        "Use supplied spatial scales and uncertainty to justify a permitted distance-response curvature or spatial emphasis. Distinguish observed scales from exploratory choices and explain the expected response range or direction. Do not infer teacher-mixture behavior from an unrelated intramolecular distance statistic.",
        "Use the supplied incumbent geometric scales rather than evidence-adapted response curvature or spatial emphasis.",
        "geometric_scales", ["Scale source and units", "Correct operative parameter meaning", "Predicted response change"]),
    _module("D2.2", "教师邻域与构象模式",
        "Use intact-teacher dispersion and joint-geometry evidence to compare narrower and broader permitted teacher neighborhoods. Explain mode coverage and competing directions. Neighbor count is not a distance bandwidth; keep the fixed temperatures and supplied teacher provenance.",
        "Use the supplied incumbent teacher neighborhood without using mode coverage or teacher dispersion to change it.",
        "teacher_modes", ["Actual teacher geometry evidence", "Mode/neighborhood decision", "Count distinct from bandwidth"]),
    _module("D3.1", "实际位移剂量",
        "When dose edits are permitted, use actual injection, clipping and backtracking relative to native predictive flow to choose delivered strength. Distinguish a wrong direction from a weak dose; whole-reward rescaling canceled by normalization is not an effective dose edit. In a fixed-dose lane, leave strength unchanged.",
        "When permitted, choose strength from teacher mismatch rather than delivered/native displacement diagnostics. Preserve the lane limits and report actual delivery truthfully.",
        "delivery", ["Delivered versus requested control", "Diagnostic-driven dose decision", "Normalization and lane respected"], "U"),
    _module("D3.2", "窗口内剂量分配",
        "When schedule edits are permitted, compare continuous dose allocation over the full learned support using temporal evidence and delivery coverage. Explain the intended early/late response without changing the support or conflating target motion with control strength. Otherwise retain the fixed schedule.",
        "Retain the supplied constant schedule rather than using temporal evidence to redistribute dose; permitted strength edits remain independent.",
        "delivery", ["Whole-support schedule decision", "Target motion distinct from dose", "Expected coverage or update response"], "U"),
    _module("D4.1", "修改保留与暂缓",
        "Compare the proposal with the supplied incumbent and choose an edit only when evidence provides a defensible expected advantage. Retaining the incumbent or deferring is valid; state how the evidence changes that decision.",
        "Prefer a legal exploratory edit whenever one is available, even without a discriminating expected advantage; report uncertainty truthfully.",
        "formula_registry", ["Evidence affects retain/edit/defer decision", "Concrete incumbent comparison"]),
    _module("D4.2", "反证与周围兼容性",
        "Use opposing evidence to predict a concrete failure response, including unnecessary displacement of surrounding coordinates where supported. Let that prediction affect the design while following the task's affinity-first priority; coordinate graph changes alone are not a failure mechanism.",
        "Choose by expected affinity-related coordinate response without making counterexample or surrounding-compatibility predictions a design criterion; still obey the task contract.",
        "counterevidence", ["Evidence-bound falsifiable failure response", "Counterevidence affects design", "Affinity priority and surrounding freedom"]),
]


def assemble(modules, role, active, alternative=None):
    """Only selected research instructions; no rubrics or treatment labels."""
    known = {m["id"] for m in modules if m["role"] == role}
    if not set(active) <= known or (alternative and alternative not in active):
        raise ValueError("Unknown role module or inactive alternative")
    return "\n\n".join("## " + m["id"] + "\n" + (
        m["alternative_instruction"] if m["id"] == alternative else m["instruction"])
        for m in modules if m["role"] == role and m["id"] in active)
