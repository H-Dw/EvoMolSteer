选择第 **2** 轮，以 matched-native 的整体预测亲和力均值变化为首要目标。其 all/unique 增益为 **+0.0804608583 / +0.1116861196**；all 比次名第12轮多 +0.0432626104，比第9轮多 +0.0465045166，且只有第2轮达到本地默认有意义增益阈值 +0.05。第9轮的 MMFF p90 与周围 RMS 诊断更好，但不足以抵消较低的主目标。

|案例|head_change|下一步|
|---|---:|---|
|validation13|+0.0119009209|dose_escalation；若为验证集，只对新 discovery 设计测更大有效剂量，原冻结验证不回调|
|shell4|-0.0068183041|dose_escalation + geometry_target_revision；在 ±0.02 近零带内，但保留轻微负向反证|
|boundary9|+0.0339563417|parent_refinement；保留父方案，分别探索剂量、几何锐度和定位|
|dose2|+0.0804606724|secondary_energy_assessment + parent_refinement；在相近 affinity 下评估能量与有效/唯一产率|

三维特征包括受体框架下区域 centroid、六维中心二阶矩、2/4/6 Å landmark 壳层和1.5/3/5 Å点对分布；完整点云软对应是待注册提案。高低分 pre-selection 空间场仅使用原 Steer 已记录分数与 discovery 全窗口节点。每步新增 affinity 调用为0，head梯度为false；冻结条件对应的坐标导数不等于 endpoint Jacobian。原生图转移保持自由。

数据缺失实际 injection/flow/native RMS、gate、cap/backtracking、奖励及 continuation 响应，也缺 all/valid/unique绝对均值、best valid/PB-fast、unique Top-5、yield和Steer基准；不能把拟议干预写成已证实的因果诊断。第2轮周围 RMS 编码变化 -4.570%、MMFF 编码变化 -1.239%、有效率下降1个百分点均保留为反证。代码默认阈值为meaningful=0.05、flat=0.02，输入未含正式活动阈值；shell4按代码属于flat。行为审计只核对选项和约束，不证明响应文本充分或机制有效。完整点云项尚未注册，线性 predictive-flow 剂量也不支持 cosine schedule。未执行新推理；30轮是预算上限，历史轮号不等于新活动用量。

输入 SHA256：`e469d510d367256d1a5f922ddb5575c915ba55a7431039a2ab95d6977589246f`  
Skill SHA256：`1a4be652ac26a4f4c5662e0f502f9da0c88c50836bfbc027d013910001ce4de3`

