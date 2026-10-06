选择真实新活动 **第3轮**：all/unique matched-native预测亲和力增益 **+.0618063402/+.0821218578**，all比第2轮高+.0084585667。第2轮unique更高，但首要目标仍为all均值。第3轮有效率下降1个百分点与MMFF median/p90编码变化-.0101476/+.0109374均保留，较低head的第1/4轮物理诊断不替代父选择。

提交 **endpoint_direction / flowr_endpoint_vjp / predicted_endpoint_world_A**，采用 **legendre_effect**。原样复制prior 74维binary weights和24项函数：2 shape、5 pair、17 landmark；19个constant、4个degree2、1个degree1，全部domain[0,.49]。全部14个floor项权重0，完整74项证据仍保留。四块[0,1,2,1]按入选字段和重新归一化，shape每项1/2、pair2/5、landmark1/17；固定endpoint尺度及stage平均scaled variance。

拟合系数变号：pair1.5在t≈.02219/.45460负→正→负；pair2.5在.44098正→负；pair3.5在.43561正→负；pair5在.00139/.41604负→正→负；pair7域内始终负。空间力来自observable坐标梯度与实际FLOWR Jacobian，不来自时导数或新的affinity网络。

沿用实际R3 requested eta=.867、predictive-flow、atom cap=.15Å、path cap=6Å，time_ramp=0。动态window[0,.5]，覆盖50节点0–.49及proposal至.50，随后原生到1；相同requested eta不能证明实际dose相同。R1正负候选相消、R3更高送达dose改善head但motif nonlinear step reward多下降，均保留为反证。

joint latent-head标签不等于endpoint-only affinity评估。head/condition detached，生产额外forward/head calls=0。旧x_t VJP在native步后滞后注入，g·actual DeltaX仅为一阶预测，**不是新endpoint reward测量**。新路径需要重新完整zero/native和实际FLOWR/GPU差分；当前一次性审计两个epsilon×±共4次额外joint forward，单列成本，CPU测试不能替代。

case：R1先查送达dose/cancellation并测试eta/caps；R2明确gain后评估secondary energy；R4保留promising父分支；-.1仅为假想负向诊断，修geometry。**R5依input仍记为running，结果与real GPU VJP/zero证书未知。** 此Agent未推理或激活；主代理独立行为检查、注册/测试/审阅后决定。

Skill SHA256：051991b8dd888fff680d669af6c841f02c09911dd9b947315cb5614d21ec1950  
Prompt SHA256：0a23b64f1e6fa87912bc307654e46b6777a3de94a0d76d2c9e2088555aad7443  
Input SHA256：5be17dfb20242794b69c5bf69c64e26360c03cc07b337b0997f2e1362536fa00  
Prior SHA256：1efe5938daa2570b3dadc6ae0a5562cba039aae756ec68a162deef7d137887a7

