---
name: jump-guidance-evidence-computation
description: Use verified measurements to support optimization decisions without designing mechanism experiments.
metadata:
  version: "2.8.0"
---

只读工具用于核对已完成结果、真实预算和动作合法性，不执行采样，也不为未测配置生成成绩。配对统计是读取已有测量的工具，不要求为其设计新的消融对照。

引用精确原测量身份或已经读取的回执。区分完整人口与小人口、原始均值与单侧高值过滤排名、几何重匹配偏差与初始对应保持偏差。未知、失败和未抽查保持原状态。

三种探索强度由实际整数改动距离核对，方向不决定强度。已有假设的坏结果降低相似路径优先级，但不要求以实验去反证一个解释。输出完整固定 JSON，不追加机制、falsifier 或自定义字段。
