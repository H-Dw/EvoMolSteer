---
name: jump-guidance-selection
description: Select a budgeted diverse portfolio against the actual calibration-rooted global best.
metadata:
  version: "2.11.0"
---

依次核对：标定与global best的真实performance；最佳节点真实父链与回滚坏路径；当前门控及候选完整落点；组合的累计预算及距离。

只选择proposal现存精确ID，不改参数。候选目标是超过全局最佳的机会，不是胜过较差锚点或解释窗口作用。均匀档差、单个邻点失败、局部未改善都不能封禁未知非均匀组合。

若coverage_profile_suggestions中有可执行bold且剩余预算至少50，选中组合必须含一个相对global best距离7–12档、联合改变至少两个允许box的bold。两条以上选中配置两两L1距离至少3档，至少两个box取值不同，不能只选末窗相差一档的近重复。没有强制增加/减小方向配额，未知成绩不能被伪造为正面证据。建议只是未测候选，不是已测收益或稳定保证。

调用审计核对实际可执行性，最多3条。代码才执行采样；你不能补样、提额或恢复禁用box。终局no_future_execution明确hold；有真实预算或有限候选不足时按输入契约报告，不声称全域已穷尽。

通过单独submit_final_response完整提交decision/selected/reason。selected必须显式数组；hold=[]，其他非空、去重、只含真实候选ID。reason最多3000字符；保持原唯一格式修正，不从说明推断控制字段。

当前实验先由Box Selection以负面证据冻结guidance_box_mask。禁用box只运行原生更新，不计算VJP、不额外移动坐标、不化学覆盖；允许box的实际scale=0仍保留化学覆盖。0/99永远不做额外条件操作，实际阶段1–98。只读参照是UniformRef和BestCombo；背景不是scale白名单，已确认完整50结果才是标定根。
