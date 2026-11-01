---
name: jump-guidance-evidence-computation
description: Read measured outcomes and executable action costs without inference-mechanism experiments.
metadata:
  version: "2.11.0"
---

工具仅作只读证据与可执行性核验，不采样、不调用模型推理或额外质量检查。summarize_population核对完整人口、原始/排名成绩和构建分母；compare_paired_runs读取同seed/slot已测整配置的差异；audit_action_batch核对跳跃、未知档、真实新增成本和实际准入。

新开放域的确认档不是动作白名单，未测格点不能当成有稳定保证。独立备选与选中组合预算是两个口径：单项executable_now为真，不代表整组预算足够。read-only回执不是采样成功的证明。失败、缺测、未晋级和budget_limited分开；不把10/20候选当作完整50候选成绩，不删除claim或退款。

最优路径复盘只使用真实anchor、已存选择摘要、参数改动及已测结果。证据不足就标未知，不索取隐藏思维链，不编造父节点、窗口贡献或线性响应。步级loss/increment资料留给独立工程审计，当前优化工具不提供窗口机制分析接口。

引用已有测量或真实工具evidence_id，不能缩写或杜撰。严格终结工具的schema、3000字符限制、数组/身份验证、原格式修正与网络预算全部保持。

六窗口新配置为闭区间0–10/11–20/21–30/31–40/41–50/51–99。第0步仅原生转移，第99步保留未覆盖预测；这两步不算VJP、不施加额外梯度，实际更新为1–98步。scale=0是真实关闭该窗额外坐标梯度，不是log2零档；FLOWR原生转移保留；化学inpainting与梯度引导共用1–98步阶段，第0/99步不覆盖。阶段不随scale大小开关，实际scale=0仍保留该阶段的化学条件注入。标定从实际scale=0开始，含失败和晋级增量总计每教师最多500次；具体格点、余额及窗口须读取真实输入。标定零引导仍可成为全局最佳，不因它为0而弃用。

当前实验先由Box Selection以负面证据冻结guidance_box_mask。禁用box只运行原生更新，不计算VJP、不额外移动坐标、不化学覆盖；允许box的实际scale=0仍保留化学覆盖。0/99永远不做额外条件操作，实际阶段1–98。只读参照是UniformRef和BestCombo；背景不是scale白名单，已确认完整50结果才是标定根。
