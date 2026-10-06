# 坐标表示缺口与后续可检验假说

2026-10-07；已授权 Analyst/Designer subagent simulation。只读本地发现集与代码，没有读取首4轮结果、修改源码或启动推理。

最有依据的下一分支是：**在预测端点上定义纯几何奖励，通过现有单次 native forward 的真实坐标 VJP 引导**。端点与 noisy proposal 的差异已经实测，但它是不是旧实验弱效的原因，仍须干预验证。

## 分数与几何不是同一种记录

`pic50_on` 与 `predicted_coords` 来自同一次 target forward；head读取pool后的ligand/pocket/interaction隐特征，没有对保存的端点坐标单独重评分。Trace在积分前记录这两项，积分后记录proposal，再重采样。因此应称“伴随端点预测的联合head标签”，不能将其当作已知的端点坐标评分函数。

本地FLOWR两个源文件规范化为LF后的SHA与保留的远端provenance一致；路径/hash详见JSON。

## 已核查的表示差异

下表来自完整候选群体的700个批次×节点汇总（14批、全50节点）；仅展示4节点，未切时间bin。

| score t / proposal state | proposal Rg (Å) | endpoint Rg (Å) | 同槽位二者RMS (Å) |
|---|---:|---:|---:|
| 0 / .01 | .1096 | 3.4943 | 4.3571 |
| .10 / .11 | .3952 | 3.5902 | 4.0102 |
| .25 / .26 | .9231 | 3.5725 | 3.2492 |
| .49 / .50 | 1.7789 | 3.5674 | 2.2101 |

新增端点统计与proposal统计的源轨迹hash、14批、74特征、landmark和50节点完全一致。独立重算effect/p/q/CI的最大误差约1.7e-15。

| 特征 | endpoint高低组z / q | proposal高低组z / q |
|---|---:|---:|
| shape_yy | −.512823 / 1.49e-5 | −.238270 / .3211 |
| pair_kernel_2.5 | +.630662 / 2.19e-5 | −.413814 / .09652 |
| pair_kernel_3.5 | +.539714 / 9.75e-5 | −.240818 / .3211 |

端点47/74项q<.05，proposal原家族0/74。端点65条曲线选常数、8条二次、1条一次；proposal全部常数。各表示使用自己的背景尺度，z大小不能直接解释成Å力。端点14个尺度触及下限，其中10项仍显著，不能只凭q将极小壳核解释为物理机制。

作为后验敏感性，将两表示148项一起BH得到端点40项、proposal2项显著；这没有校正整个历史探索，更不是独立验证。新端点统计也没有消除谱系塌缩、共享oracle或类别/latent混杂。

## 有顺序的假说

1. **端点奖励+真实VJP。**  
   `g_t=J_Y(x_t;fixed cond,categories)^T ∇_Y R_t(Y_t)`，然后原生积分后做有界注入。端点特征和尺度已有实测支持；一次已有forward加backward即可，无额外生产head调用、无head求导。它在旧x_t处是精确VJP，在post-native状态的注入仍是滞后近似。若reward/剂量充分响应而匹配native最终head仍不改善，不能把关联当控制优势。

2. **有支持的局部口袋方向场。**  
   当前landmark只有径向场；可预声明固定受体patch，用局部加权centroid及svec二阶矩区分全局紧缩和局部位置。离线流式保存批次统计与support即可；生产经已验证VJP，不增加网络调用。若全局pose/Rg调整或root-balanced后信号消失，则不能解释为局部优势。壳核仍不等于氢键或能量。

3. **位移与剂量对齐诊断。**  
   从已有current/proposal/endpoint分解预测flow、native随机步、整体平移和内部位移，检查力与这些分量的夹角、去重谱系下一步head变化。先诊断，不凭时间导数指定+x/−y。若actual path/reward改变但head不变，需检验方向/对应/表示；若注入仍极小，按既定亲和力优先协议检验剂量。

4. **条件端点修正代理，仅作低成本替代。**  
   冻结当前proposal P0与anchor A，构造 `T_k=P0+γ_t(Y_teacher,k−A)`，对P与T的soft mixture距离求导。它复用teacher，无新模型；但只是规定的条件坐标目标，**不是FLOWR Jacobian**。只有与absolute-proposal目标在同参考和匹配剂量下比较，才能判断是否值得保留。

## 真正VJP分支的执行要求

`scalar_guidance.live_pullback`已有所需基本接口；不能在现有已detach的forecast上事后恢复Jacobian。需要独立注册derivative_path、端点reward表示、端点尺度及正确的评价数据视图。

生产每步只做已有target forward和坐标backward；一次真实FD的额外±ε forward单独计数，固定cond/类别/对应，逐次恢复相同RNG。VJP完成后释放图并返回detached预测/cond，防止跨步反传。

新family须重新验证100步和final的native/真实求梯度但η0逐字节一致。启用autograd可能改变内核路径，旧family的零对照不能替代；失败不能隐瞒为“仍是单次forward”。

无额外post-native forward时，日志只能记录旧端点reward和一阶估计；不能把R(proposal)冒充实测端点reward变化。评价必须读取predicted_coords，并以对应proposal state<=.5筛选；.49→.50是最后控制步。新适配器的执行就绪结论另见独立endpoint审阅。

前4轮维持冻结；本报告未预测其结果，也未证明真实亲和力、因果优势或未见seed泛化。
