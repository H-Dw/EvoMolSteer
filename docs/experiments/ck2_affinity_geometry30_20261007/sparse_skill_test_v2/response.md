选择实际 **第3轮** 为亲和力父程序：all增益 **+.0618063402**，mean=7.4960811996，valid=97/100，unique增益+.0821218578，best valid=8.1216993332、unique Top5均值=8.0929445267。第5轮all增益 **+.0597522211**，低.0020541191，但unique、valid及MMFF诊断较好，作为相近affinity的secondary比较和VJP剂量对照。historical Steer100 mean/best=7.5103505611/8.3479404449，预算不等且不配对；第3/5轮尚低于该subset benchmark，不推断完整Steer最大值。

这次明确弃用旧 **legendre_effect**：24/24拟合均未通过节点fidelity，RMSE范围.3859133–.9313753，125个field/node强符号冲突。t=.49，pair2.5经验 **+.0425764665** 对旧拟合 **-.4688096495**；shape_yy **+.1632962224** 对 **-.5128231821**；landmark00 occupancy3 **-.0402169430** 对 **+.5449039482**。整窗显著或constant fit不能授权节点方向，旧polynomial roots不是观察到的阶段转换。

新设计 **endpoint_direction / flowr_endpoint_vjp / predicted_endpoint_world_A / linear_node_effect**，原样提交24项node_effect_functions的每项50 values_z及74维binary weights。相邻真实节点间作连续线性插值，在全部节点精确等于独立批mean effect；不编造polynomial，不把time derivative当空间力。pair2.5真实符号相邻变化在0/.01、.40/.41、.42/.43；插值零点是函数推定，未观测精确转折时刻。

24项为2 shape、5 pair、17 landmark；14项floor排除，但全部证据保留。四块[0,1,2,1]按入选字段重新归一化为shape每项1/2、pair2/5、landmark1/17。固定endpoint尺度，系数除以真实stage平均scaled variance后clip[-3,3]。首新field/function比较匹配已测 **R5 VJP η=.3**，不用actual-proposal η=.867充当等效VJP剂量；仍须核对actual RMS/caps。

动态window[0,.5]、50 score节点0–.49对应proposal至.50，time_ramp=0正schedule，窗口后native到1。生产额外head/target forward=0、head梯度false。真实VJP在旧x_t，native后lagged injection；g·actual DeltaX仅为一阶预测，**不是新endpoint reward测量**。

R5已报告zero等价true和真实模型FD通过：analytic3.9074988365，epsilon=.003/.01的相对误差1.85429%/.78809%，一次性额外4 joint forwards；这些是既有R5结果。**新稀疏/node-function程序仍需自己的fullzero与model-FD，尚未推理或激活。**

case动作：R1查送达dose并测eta/caps；R2明确gain后secondary energy；R4 promising父细化；-.1仅是假想负向诊断，改geometry/representation。14独立批是统计单位；关联field不是因果氢键/结合区域，heldout评估冻结设计不回调winner。

Skill SHA256：ce8bb6c6c53ac738c4ee3ae402ff7a0a9b78cc53589092b5aa06150434000222  
Prompt SHA256：c2bd73b248e2761747185c9d71e6406970642dd9198dad37f3a47ce966b1545d  
Input SHA256：061fce4731d40df75d8c808cd6ac021c8a98f0659eb462c7bfbd8dbae949cd61  
Prior SHA256：6394d0089a3e75c5e4246cc4f11bc9e89ad18dd0375ab8432b999e2d72229639

