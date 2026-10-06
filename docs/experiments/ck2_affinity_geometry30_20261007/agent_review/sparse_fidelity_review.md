# 稀疏节点忠实度与集成审阅（2026-10-07）

身份：已授权 Analyst/Designer subagent simulation。仅新增本审阅，未修改旧审阅、源码或literal响应。

**execution_ready=true：可启动第6轮真实GPU验证。actual_sparse_GPU_validation_passed=false：新loss尚未获得真实FD或完整zero证书。** 旧R5证书及CPU测试不能替代。

旧反证已落实：24/24 Legendre拟合未通过工程门槛（相对RMSE .3859–.9314、125个字段/节点强符号冲突），当前literal改为linear_node_effect，精确复制24×50=1,200个经验值。runtime与compiler均约束不合格多项式。经验值吻合保证忠实度，不保证其噪声方向会改善亲和力。

两入口已统一：factory返回SparseEndpointReward，真实端点controller调用该factory，离线评价沿用相同公式。一次原生target forward得到端点，纯几何loss通过真实FLOWR坐标VJP求导；head隔断/拒绝hook和生产调用计数保留。导数在旧x_t求值、native步后注入，滞后和离散变化的限制仍在。

新literal为24字段、[0,1,2,1]块权重、η=.3、0–.5窗口，最后.49→.50，此后native至1。第6轮freeze的有效参数逐项不变，三臂native/zero/gradient各100。全窗口指50个原始网格更新；分段线性系数并不意味着runtime支持任意非网格时间。

独立检查：

- 19项针对性pytest通过；根代理另报告全套228通过，未把后者列为本审阅重跑。
- 50节点reward独立重算最大误差7.11e−15；4例真实teacher加合成非线性映射的VJP FD最大相对误差4.71e−11，主调用1次且head hook0次。它们不是FLOWR测试。
- literal audit与保存audit完全一致；重新编译与保存compiled逐字段一致。编译SHA：d086420613851c04641a9d99258b5c1af3dd92dbf382c6b65f878a0d82ba03e7。
- 纯内存执行R6–26全部注册分支，密集切换清除7项稀疏字段，全部50次受控更新。R27–30即使输入极高后验分数仍复用冻结赢家及有效参数。

Node contrast是逐批contrast/variance后平均；经验模式是平均contrast除以平均variance，两者不等价。块内系数总和固定不保证同空间梯度或dose。第6轮因此同时改变字段、归一化和variance aggregation，不能把未来差值单独归因于某一项。

仍须如实限制：dense74残差不是24字段奖励距离；全74诊断仍计算。全分子/landmark奖励经Jacobian可能影响广泛原子。14批内的克隆及相关字段不支持因果优势宣称。内部cond/RNG未直接导出逐字节证书；zero覆盖完整已记录轨迹/final，加源码detach/RNG控制。

第6轮待验证真实新loss FD、完整zero、调用计数、显存/耗时及终态效果。源码和固定输入/响应/编译/audit哈希记录于JSON；不绑定动态round配置或报告。
