---
name: jump-guidance-proposal
description: Propose executable complete configurations with a calibration-rooted optimization purpose.
metadata:
  version: "2.9.0"
---

按以下阶段形成候选，输出短决策摘要，不输出逐步私有思维过程。

1. 确定要超过的数值：读取初始标定performance与当前global best。每条方案都针对当前全局最优，不以击败较差锚点或上一轮成绩代替优化目标。
2. 选择新的完整落点：结合已测整配置的成绩、未覆盖区域和最优决策路径，提出仍可能更好的联合调整。说明“这片区域为何值得测”及结果未知，不说明“哪个窗口负责什么机制”。三种exploration_tier只是整数跳跃幅度，方向可以增加、降低或混合，不设机制或方向配额。
3. 识别探索机会：可选择new_joint_region、frontier_jump；回滚时必须选择redirect_from_reviewed_best、从最优节点起步，并避开最近失败路径。与声明锚点距离大不代表与global best距离大，读取审计中的两种距离，避免伪装大跳跃。
4. 审计可执行性：只在declared_grid和整数动作界限内提案；调用audit_action_batch核对executable_now、额外尝试成本、重复与失败身份。备选可单独可执行，但最终组合还须累计预算审计。不能把参数结构合法当作当前可执行。
5. 完整提交：按输入schema显式填写optimization_target，包括真实calibration_source_id、incumbent_source_id、reviewed_path_sha256和固定purpose=seek_potential_global_improvement。用improvement_reason概括潜在更优机会，用uncertainty保留未知，不编造定量收益或成功概率。

禁止以验证窗口贡献、拆分效应、消除混淆、补齐曲线、证明单窗有效或证伪某个解释为选点目的。若一个落点被自己判断为只会更差或基本无改善机会，不因诊断价值而提案。若仍有未知整配置具有潜在收益，可以探索，不需要先证明其机制。

每窗默认一档10，具体边界、步幅和幅度分类读取输入；不采用指数放大，不自行扩域、提额、重采或补样。没有值得且可执行的新路径时允许hold，但不能把局部覆盖说成已穷尽全域。

最终通过submit_final_response单独提交完整decision/proposals/reason，禁止正文替代。hold或no_future_execution时显式proposals=[]，其余至少一项；不输出selected。每个reason/improvement_reason/uncertainty最多3000字符，证据身份逐字复制。收到原唯一格式反馈时提交整个对象，不加重试额度。
