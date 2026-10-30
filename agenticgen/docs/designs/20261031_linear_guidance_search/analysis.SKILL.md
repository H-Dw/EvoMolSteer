---
name: coordinate-guidance-history
description: Learn from compatible completed guidance results to prioritize configurations that may improve the condition coordinate root mean square deviation.
metadata:
  status: design-draft-not-runtime
---

以降低条件区域坐标均方根偏差为目标。读取本次实际输入的公共科学与统计契约；只采用教师、条件、对应、损失、采样和人口兼容的已完成测量。

把标定基线和正式历史一起用于确定现任最优。相同实际配置和随机槽位的复用只算一次测量；提案、失败请求和缺失评价不当作已完成结果。

从历史提取会改变下一次选择的事实：哪些完整配置较好、哪些附近改动变差、哪些组合仍值得尝试。保留反例，区分单次局部结果与跨背景经验；不要求解释生成动力学，也不为归因、消融或机制证明安排实验。

回看原提案的实际结果，用它提高或降低相似路径的优先级。尚无实测支持的路径可以提出，但明确不确定性，不编造机制、定量改善或成功概率。

用完整指标名称，区分条件区域坐标均方根偏差的原始均值和排名均值、最大原子误差的聚合方式、实际分母与失败数。保留极低偏差样本，采用执行器的单侧统计结果，不发明筛选阈值。

输出最多两条决定性事实、对下一次选择的含义及主要不确定性；更多历史由只读工具查证。循环次数、确认人口和预算由外部控制。遵守调用方 JSON 契约，不复制整段历史或输入数组。
