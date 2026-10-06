选择历史亲和力父基准 **第2轮**：all/unique matched-native增益 **+.0804608583/+.1116861196**，all比第9轮高+.0465045166。新奖励尚未推理；energy只在保留明确head增益后于相近affinity下评估。

此次设计实际改变为 **endpoint_direction → flowr_endpoint_vjp**。同14独立批/50节点/74特征，endpoint有47项q<.05、proposal为0；pair2.5整窗效应为+.6306622/-.413814。shape_yy效应-.5128232、CI[-.6196424,-.4237735]、q=1.4937e-5。关联来自同次joint FLOWR的pooled-latent head标签，不能视为endpoint-only affinity评估或因果binding机制。

使用已注册74维纯endpoint几何：3个centroid、6个二阶矩、5个点对核、20个landmark×3场。固定endpoint全窗口尺度，floor=1e-5、无量纲variance ridge=.05；R=Σ(phi/s)·clip(mean_batch((hi-lo)/variance),-3,3)·block_weight/block_size。四块权重为 **[0,1,2,1]**，η=.3按predictive-flow RMS校准，time_ramp_power=0。窗口来自输入[0,.5]，50个真实score节点0至.49均正计划支持，proposal至.50；之后原生继续至1。实际位移受caps/backtracking约束，必须测量。

真实导数为 g=J_Y(FLOWR,x_t;detached self-conditioning)^T∇_Y R；预测与condition detached后先做原生随机步，再注入缓存VJP。这是旧状态梯度的滞后控制。g·实际注入只是一阶预测，**不是测量到的post-native endpoint reward改善**。生产每步额外forward/head calls=0、affinity-head gradient=false。

首次真实FLOWR数值审计：epsilon=.003/.01，两组中心差分，**额外4次joint forward**，单列成本并恢复RNG/condition；当前代码要求两个差分均正且至少一个相对误差<.15。切路径后必须重新完整zero/native从0到1验证，包括坐标、categorical状态、condition、RNG及最终输出。此次未运行真实推理或GPU审计。

14项floor全部在landmark。首实验使用已注册四块方向baseline；细feature weights/mask列为未实现的registry扩展：冻结discovery上的q、效应充分性、批次方向稳定性及floor/原单位支持，保持全统计证据，匹配dose对照。不能凭tiny q放大几乎零效应，不能依据原子类型或图筛选。65/74趋势fit为常数，录得逐节点目标不等于显著阶段趋势。

四个case分别保留：validation13有效dose测试但不回调验证winner；shell4目标/表示修订并测dose；boundary9父方案细化；dose2获得明确gain后评估secondary energy。

Skill SHA256：d61ea015a77a20c78196aa895df969233e7d5d98dfa3a6e1129d5c1792f47c3e  
Prompt SHA256：da59c4fae06c5b108aa8d00ba688b03ea516573a6516f16cece7eaead2f861b0  
Input SHA256：a2f0e10d179e7b8ccd2bdb669a6dbb85c5edbb7718a8abbca3d276d8f3180a26
