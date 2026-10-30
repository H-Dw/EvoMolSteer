---
name: linear-guidance-evidence-computation
description: Read verified coordinate measurements and audit integer allocations without sampling.
metadata:
  version: "2.6.0"
---

先使用已提供的实测摘要，仅为缺少且会改变选择的证据调用只读工具。使用精确的原测量身份或已读取回执身份。工具不执行采样，也不能为未测配置给出成绩。

区分原始均值、单侧高值过滤后排名均值及其分母；最大条件原子误差的均值不等于全人口最大误差。保持低偏差样本、失败、缺测和不可评价状态。不同人口不直接混排名；配对统计使用相同种子和槽位以及原始保留掩码交集。

只输出调用方要求的完整 JSON，不额外添加数组、机制解释或反证字段。工具调用是中间步骤，最终角色响应必须符合提供的 schema。
