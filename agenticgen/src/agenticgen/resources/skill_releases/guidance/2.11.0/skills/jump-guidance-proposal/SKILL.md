---
name: jump-guidance-proposal
description: Propose diverse complete scale profiles inside a frozen teacher-specific box gate.
metadata:
  version: "2.11.0"
---

你的任务是探索可能超过已测全局最佳的完整配置，不是理解推理动力学或证明窗口贡献。只给可见短决策摘要，不输出私有逐步思维。

阶段一：读取真实标定根、global best、已存父链及冻结门控。回滚时复盘失败旁支，从真实最佳节点转向不同的新组合；不重新生成最佳。保留允许的未知组合，不把“均匀80差”推成“末窗80不可用”。完整均匀配置的结果不能推广为逐box的独立边界。

阶段二：形成不同完整落点。可考虑递增、递减、交错、早低晚高、多个允许box的重分配。它们只是探索机会，没有机制或方向配额。目标schedule是六个实际整数，0–200、格距10；禁用box必须明确0。target_scale_schedule不是增量、指数或归一化值；不输出updates。最大L1距离12档，focused 1–2、expanded 3–6、bold 7–12，均相对当前global best计算，不能借较差anchor伪装bold。

阶段三：读取coverage_profile_suggestions。这些只是程序生成的未测合法候选，performance=null，不是成绩或稳定保证。它们不穷尽全域；可直接采用或另提更有机会的合法完整组合。若存在可执行bold且预算可覆盖，候选中至少有一条真正联合改变两个允许box的bold，不能整轮都停留在局部小改动。

阶段四：调用audit_action_batch核对门控、全局最佳目标、当前预算、重复和失败身份。最多6个备选、最终最多3个；原始失败配置不能重发。不要因为未知就声称不可用，也不要编造改善幅度。

通过submit_final_response单独提交完整decision/proposals/reason；每条含schema所需optimization_target、target_scale_schedule、exploration_tier及真实证据ID。hold明确proposals=[]；终局no_future_execution不得再提动作。每个说明字段上限3000字符，只有原一次格式修正。

当前实验先由Box Selection以负面证据冻结guidance_box_mask。禁用box只运行原生更新，不计算VJP、不额外移动坐标、不化学覆盖；允许box的实际scale=0仍保留化学覆盖。0/99永远不做额外条件操作，实际阶段1–98。只读参照是UniformRef和BestCombo；背景不是scale白名单，已确认完整50结果才是标定根。
