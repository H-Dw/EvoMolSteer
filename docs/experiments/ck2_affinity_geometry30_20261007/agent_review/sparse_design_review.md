# 稀疏端点奖励独立审阅（2026-10-07）

身份：已授权 Analyst/Designer subagent simulation。仅保存本审阅，未改源码、旧45源审阅或literal响应。

**execution_ready=false。独立模块数学与CPU检查通过；实际/评价入口尚未接入，真实新loss未验证。**

74项证据完整保留，选择24项（shape2、pair5、landmark17），排除全部14个floor项。24组系数及[0,.49]域精确复制：19个常数、1个一次、4个二次。literal引用的q、effect、scale、批次一致比例和R3父程序SHA均准确。

块内权重为 B×m/sum(m)，当前总和[0,1,2,1]。这保持系数预算，不保证同梯度或dose。node_contrast先逐批contrast/variance再平均；Legendre模式把全窗拟合effect除以同节点的批次平均variance。二者同时改变时间估计和方差聚合，不只是平滑。

拟合系数存在明确反证：

|字段|t=.49实测high-low z|拟合effect|
|---|---:|---:|
|pair 2.5|+.0425765|−.4688096|
|pair 3.5|+.0551039|−.4900304|
|shape yy|+.1632962|−.5128232|

根的位置计算正确，但whole-window q不验证阶段变号；没有逐节点或根置信区间。LOBO one-SE曲线是整体摘要，实测节点也有噪声。这些冲突须作为控制假说的反证，不能把根称为已发现的阶段优势机制。

4项pytest通过。真实teacher坐标与显式合成非线性端点映射的8例检查覆盖两种模式×4节点：独立reward重算最大误差1.42e−14，VJP FD最大相对误差1.93e−10；主调用1次、head hook0次。这些是CPU检查，不是真实FLOWR测试。

激活前需完成：

- factory及EndpointController目前创建稠密EndpointGeometryReward，同名endpoint_direction会静默忽略新字段/函数。两入口必须显式稀疏分派或拒绝，并同步评价。
- driver绑定新Skill/prior/literal/compiled哈希；细化及冻结保留模式，换家族时处理不兼容字段。
- 当前audit允许错误的response reference hash；真实产物正确、compiler绑定实际参考，所以是合同缺口而非数据污染。补各处reference/prior身份及feature_names顺序校验。
- nearest_standardized_rms仍为74字段未加权距离，须标明不是稀疏目标拟合指标。

literal没有宣称新奖励有效，区分一阶响应及额外4次审计forward。其required_zero提及内部condition/RNG，但当前格式不直接保存这些状态；正式报告只可给记录轨迹/final字节证书及源码控制。沿用R3 requested η=.867不等于相同实际剂量，也不能将未来改善单独归因稀疏化。

该数学分支可保留为探索性候选。等待集成和真实新loss FD/zero；R5其他loss的GPU FD证书不能替代。发现集筛选、拟合和克隆相关性不支持因果优势、泛化或LLM优越性承诺。
