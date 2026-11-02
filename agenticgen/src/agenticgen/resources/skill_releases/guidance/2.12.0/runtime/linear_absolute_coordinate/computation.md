---
name: jump-guidance-evidence-computation
description: Audit all five hypotheses and read background-scoped coverage and observed update diagnostics.
metadata:
  version: "2.12.0"
---

工具只读，不生成、不重新评分、不追加质量检查。summarize_population核对完整50、原始/排名RMSD及构建分母；compare_paired_runs使用同seed/slot的实际配置；audit_action_batch按整组准入模式核对全部最多5方向的累计成本、网格、门控、距离及失败身份。

window_evidence包含程序构建的当前背景覆盖和已存步级观察。只有其他五窗固定、同输入的结果才是该背景的剂量对照；联合变化仅说明完整组合。更新范数比、位移、匹配变化是搜索优先级的线索，不证明窗口有害、无效、线性可加或某个新配置必然改善。不得据此裁剪、缩放或归一化VJP。

确认档只表示已有测量，不是白名单。读取各窗显式网格，首窗细剂量与后窗格距不同；距离采用网格索引，不能再硬算abs(delta)//10。未测、预算不足、失败和超声明域分开；只读建议performance=null，不等于已测成绩。

相同科学身份只引用，失败计费且不删除claim。10/20初筛不能替代完整50；每教师标定含增量最多500，五轮最多25个正式方向/1250次。已提出合法方向必须全部进入本轮执行，不把单项可执行误当整组预算足够。

原生0/99不操作；允许box的零剂量保留化学阶段，禁用box关闭梯度及化学。首窗constant/linear_ramp读取冻结协议，不在分析时切换。只输出公开简短证据摘要，不索取私有思维链，不设计机制/消融任务。证据ID、strict工具、3000字符及原一次修正全部保持。
