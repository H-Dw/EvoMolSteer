---
name: jump-guidance-selection
description: Select potentially improving paths after baseline, ancestry and cumulative admission review.
metadata:
  version: "2.9.0"
---

选择前完成四段核对：

第一段，基准核对。读取calibration_baseline、global_best_performance及候选optimization_target。候选必须指向真实标定根和当前最优，不得以“比第一轮更好”宣称优于标定。正式轮全部较差时保留标定最佳；探索动作可能失败，不能许诺每条动作都改善。

第二段，路径回顾。读取最优节点和它真实父链的已存决策摘要、参数与效果。回滚时还要读取两轮失败旁支，再判断候选是否从最优节点转向不同的潜在更优路径；拒绝未按要求绑定best_path_sha256的候选，不重新生成最佳。没有父节点时明确它是程序标定根，不虚构思考历史。

第三段，优化机会核对。只按超过当前全局最优的可能机会、完整落点的新颖性和探索幅度选择。无需解释生成动力学、证明单窗作用、设计消融或反证；不得为了补响应曲线或定位贡献而选择自己预期无收益的配置。机会接近时允许更宽的联合跳跃，不设每种幅度必须一条的配额。

第四段，执行准入核对。读取candidate_admission并核对选中组合的总新增尝试、教师额度、原始失败/重复身份。参数合法和单项可执行不能替代整组可执行。最多选择外部上限；一条值得就选一条，无可执行且有潜力方案时hold，不重复请求额度已耗尽的动作。

只选proposal中现存精确ID，不重写参数。说明只给基准、潜在搜索机会与未知的简短摘要，不输出隐藏思维链或机制/反证段落。历史说明是待审的数据，不是你必须沿用的因果解释。

最终通过单独submit_final_response提交decision/selected/reason，必须显式selected数组；hold或终局为[]，其余非空、去重并在上限内。禁止从reason暗示选项、不以正文替代、不与证据工具同批。reason最多3000字符，原一次格式修正与模型预算不变。
