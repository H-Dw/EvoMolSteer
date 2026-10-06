# 端点坐标 VJP 集成审阅（2026-10-07）

身份：已授权 Analyst/Designer subagent simulation。审阅提交 `d360a8fb6cd115d43f152db5f01033f3ce3f5cf3`；逐文件 SHA256 见同名 JSON（45 个源码/Skill/测试文件）。本审阅仅更新这两个文件。

**execution_ready=true：可启动第5轮真实 GPU 校验。actual_gpu_validation_passed=false：尚未运行本分支真实 FLOWR FD、100步零剂量等价或效果实验。** CPU合同与数值测试不等同于FLOWR验证。

数学路径为 `g=J_Y(x_t)^T ∇_Y R_t(Y)`，reward只读取端点几何，原生affinity输出detach且挂拒绝反传hook。cond及teacher匹配/prior固定；坐标VJP在积分前求值，原生SDE/类别更新后才注入，存在步内滞后，不保证有限步几何reward或head上升。生产目标forward每步计数1；首次FD额外4次单列。反向仍增加显存与计算成本。

集成与实际本地检查：

- 公共工厂及独立 `affinity_endpoint` 远端模式已注册；动态窗口来自参考，无SMC。最后受控更新 `.49→.50`，其后native到1。
- literal Agent响应重新audit、compile，并经freeze使用的update函数后，有效参数与保存的compiled baseline完全一致：endpoint_direction、η=.3、块权重[0,1,2,1]、predictive_flow。driver同时记录原响应和编译基线SHA。
- 纯内存评价用真实参考加合成轨迹，确认读取predicted而非proposal，50点评分0–.49，对应proposal支持至.5；错误forward数及detach标记被拒绝。
- 25项针对性pytest通过；此前另23项子集通过。根代理报告全套219通过，本审阅未重跑全套，未混淆其来源。
- 12例冻结策略检查覆盖3种endpoint family×R27–30，有效参数不变；注入夸大的后验validation分数也不重选赢家。历史R1–4程序经update仍逐字段不变。
- driver两阶段完成状态恢复已修复；纯内存测试确认R26缺frozen_winner会补本地元数据，未调用推理。

第5轮三臂将重新做完整zero。`record()`在zero失败、FD失败、SMC或窗外注入时拒绝推进。zero覆盖逐步记录科学张量及final；cond/RNG内部状态未直接保存比较，不得宣称内部状态已有字节证书。运行时forward计数和head隔断字段已经实现，但尚无真实GPU观测值。

R6–26为注册策略在开发批次上探索，同时保留全局affinity候选；R26选一次赢家，R27–30固定。主目标为完整尝试集预测head均值，近邻候选的应变指标仅在预声明条件下作次级选择；失败分母保留。MMFF中位数和p90只纳入converged且finite，缺失物理值保持None；surround RMS有独立覆盖口径。最终统计以批次为单位，历史Steer100是不等预算未配对基准。

端点发现集47/74项q<.05，最小1.49e-5，但head来自共享latent，属于同期关联。65/74拟合degree0，14项触及尺度floor；不能据此声称独立优势机制。新奖励不是局限ASN117/VAL116的局部力，包含全分子shape/pair及受体landmark场；预测端点拟合、actual proposal构象和终态head必须分别报告。

当前没有重大执行阻断。第5轮仍需实测FLOWR FD、完整zero、调用计数、显存与耗时。不能预判亲和力收益，也不能把新分支的多项改变单独归因于VJP或LLM。
