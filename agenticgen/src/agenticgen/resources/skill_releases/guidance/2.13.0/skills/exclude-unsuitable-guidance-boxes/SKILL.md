---
name: exclude-unsuitable-guidance-boxes
description: Exclude only verified harmful boxes while preserving weak-dose, background-sensitive and unknown opportunities.
metadata:
  version: "2.13.0"
---

独立Selection只寻找不应施加额外条件操作的box。默认保留，证据不足不强行排除；无需证明所有保留窗有益，也不解释生成机制。

读取ReferenceCombo与IncumbentCombo的新实测、同背景NativeOff、LowerRescue、早段增幅及可用的细剂量coverage_probes。负面效应是保留减关闭RMSD；比较必须同教师、同其他窗口、同seed/slot。NativeOff同时关闭梯度与化学，允许窗scale0只关梯度，两者不能混用。

主动寻找挽救和背景反例。一个剂量差不等于整窗有害，旧背景失败不封禁新背景；第一窗缺少低剂量证据时保留。数值发散、工程错误、缺测或一次高档失败不证明全box有害，不能过滤为支持证据。

eligible_box_ids由程序按实际预注册门槛给出，只能选择其子集，允许空集合；不能补字段、扩大集合或要求额外采样。两个单窗关闭有效不保证联合有效，final只能选择confirmed_options中已完整50确认的真实门控。所有初筛、晋级和覆盖探测累计最多500，不因未花完预算而补样。

proposal通过submit_final_response显式提交excluded_box_ids/evidence_ids/reason；final还提交confirmed_profile_id。缺字段不是空集合。说明最多3000字符，保留证据和短决策摘要，不给私有逐步思维，不声称独立验证或全scale证明。

七窗初筛成本更高，读取maximum_excluded与实际eligible_box_ids；500次总额内不足以同时确认两窗时，只确认预算容纳的最强可疑窗。未确认窗保留，不能用六窗51–99的整体证据排除拆分后的任意一段。
