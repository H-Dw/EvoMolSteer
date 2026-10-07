# 路径探索的代码、数学定义与解释边界

本次按研究报告逐模块实施。1–3轮建立谱系和结果标签；4–17轮进行冻结
对照及逐轴探索；18–20轮在不再调参的条件下验证筛选方案。每个下降的
试验均保留报告，下一次从已评估父方案恢复，再修改一个轴。并没有同时
搜索多个模块，也没有将参数扫描描述成20次独立LLM发现。

## 1. 可回溯的结构身份，而非根据坐标猜祖先

`continuous/path_graph.py`定义两种节点和两种边：

- `CURRENT`：原始step/slot的生成状态；`PROPOSAL`：native更新后的状态。
- `NATIVE`：同slot的current→proposal，时间差为正。
- `COPY`：根据selected_indices连接proposal→下一current，时间差为零。

脚本验证root传播、offspring计数、非选择步的恒等映射及选中坐标逐值复制。
节点使用source_index、step、slot和representation定位原轨迹，坐标不重复
写入图表。14个donor批次得到70,700个节点、70,000条边；两张无压缩
Parquet共2,327,478字节。复制频数用于谱系诊断，不能当作坐标优化速度。

学习支持来自动态window与实际resampled数组，要求proposal时间不超过
window末端。本例学习score时刻是0.00–0.49，控制更新到0.50；0.50 current
作为边界保留。原始Steer另有score=0.50、proposal≈0.51的最后选择。
其终态后代标签因此属于继续Steer的未来，不能称为0.50起native续跑标签。
t=1的结构与评分仅作为结果，不扩展坐标学习或控制窗口。

## 2. 终态尾部信用及未知标签

`continuous/elite_path_credit.py`从最后slot沿全部100步selected_indices
逆向回溯终态祖先。标签来自最终解码后同协议的FLOWR head rescore；
终态结构clock=1，head rescore clock=0.9999，均与中间online预测分开。

阈值从donor有效终态的预定95%分位冻结，得到8.258901977539063 pIC50。
700条donor记录有669条有效、34条elite记录、20种不同elite化学图。
同一祖先内重复终态图取最大分数，克隆计数另存；去重只用于信用和报告，
推理允许化学图自由变化。

被剪掉且没有后代的节点，其未来亲和力是未知，不赋零分。t=0、0.25、
0.49、0.50的可观察祖先分别为17、37、320、443 /700；elite祖先数增加
不能直接解释为几何优势逐渐变强，因为可观察性同时变化。
精简信用表和可观察性曲线共440,113字节。

## 3. 不平均不同路径的坐标库

`continuous/terminal_path_library.py`每个donor选择有效、不同终态图的高分
路径，然后沿精确祖先连接提取各实际学习节点的FLOWR endpoint forecast。
共享当前祖先只存一个teacher，并保留其全部`batch:terminal_slot`路径ID。
坐标在受体世界坐标系中；不对配体独立旋转/平移对齐。

每批预算2得到28条终态路径、23种图、每时刻14–26个teacher；预算3
得到41条路径、35种图、每时刻15–37个teacher。老师分数包含阈值以下
的有效样本，因此两者都不是elite-only库。基准mass先等权批次，再等权
各批当前祖先。top-k局部截断后，仅在选入的teacher上重新归一化。

该模块保留身份和不同路径席位，没有实现低维几何模式聚类、MAP-Elites
变异/交叉、native反事实标签或Analog Markov committor。不能把路径预算
增加解释成已经完成这些算法。当前序列的探索层次明确为有策略标签的
observed Steer-future潜势。

## 4. 两种实际测试的可微奖励

令y为FLOWR对当前X_t预测的endpoint世界坐标，y_j(t)为精确节点teacher。
使用detached endpoint anchor求Hungarian几何对应关系，再选择最近k个
teacher。对应关系与teacher集合在一次导数中冻结；跨调用可切换，因此
全程序是分段光滑，不能声称处处可微。原子类型、键和图身份不进入奖励。

第5轮沿用endpoint_pointcloud公式，只换成终态路径标签与身份。
记q_j为匹配点的平均平方距离、u_j为缓存的终态预测分数、b_j为源mass：

\[
\pi_j=\operatorname{softmax}_j(\log b_j-q_j(\mathrm{anchor})/T+
\beta[u_j-\bar u]),\quad
c_j(y)=\delta^2[\sqrt{1+q_j(y)/\delta^2}-1],
\]
\[
R_{\rm attract}(y,t)=\tau\log\sum_j\pi_j\exp[-c_j(y)/\tau].
\]

pi在导数中冻结，多个teacher以log-sum-exp保留不同模态，避免将所有坐标
平均成一个未观察到的中间目标。该公式仍具有teacher密度吸引。

第6轮新增`generation/path_reward.py`的endpoint_path_value，令C_j=c_j/T：

\[
R_{\rm value}=\tau\left[
\log\sum_j b_j\exp\{\beta(u_j-\bar u)-C_j/\tau\}
-\log\sum_j b_j\exp\{-C_j/\tau\}\right],
\]
\[
\nabla_yR_{\rm value}=-\sum_j(p^u_j-p^0_j)\nabla_yC_j.
\]

减去背景项，使局部u全部相同或只剩一个teacher时梯度为零。对称标签
交换应反转方向；这两条行为及标量有限差分已测试。u是观察到的Steer
后代预测效用，既不是实测亲和力，也不是native成功概率。背景减法也可能
移除有用的几何吸引，因此必须用真实生成比较，而不能因公式更规范就采用。

## 5. 连续性、时间与幅度的逐轴测试

|轮次|单独测试的变化|机制假设|
|---|---|---|
|7|eta 0.33→0.66|区分干预太弱与方向不合适|
|8|k=4→完整可用邻域|检查局部截断是否丢失高值路径；实际教师数仍有限|
|9|beta 2→4|加强终态效用对方向的区分，保持控制剂量不变|
|10|tau 0.5→1|减轻单teacher竞争，测试更平滑的转移|
|11|正时间ramp，power=1|将更多剂量放在较晚、较可观察的节点|
|12|history混合0.5|通过精确路径ID传播上一endpoint的几何posterior|
|13|history混合0.8|检查更强路径承诺是否妨碍发现|
|14|每批路径预算2→3|增加不同有效终态路径覆盖，保持公式和剂量|
|15|核bandwidth 4→2|检查宽邻域是否稀释局部效用对比|
|16|点位移核→74维标准化几何核|检查位置逐原子匹配是否掩盖空间形态信息|
|17|按局部效用差异减弱幅度|防止RMS归一化放大近乎平坦的证据|

每轮父方案见独立plan.json，参数并非沿表顺序全部累加。
history传播由前后frame路径ID交集定义行归一化矩阵，混合源先验与上一
几何posterior；它不使用新head分数，也不是精确FK后验。history缓存不
跨批次，且在中点另做真实FLOWR方向导数检查。

时间ramp为`[0.1+0.9*(t-a)/(b-a)]^power`，a、b来自window，每个学习步
均可生效。第17轮幅度为`beta*sd_p0(u)/sqrt(1+[beta*sd_p0(u)]^2)`；它
是经验效用对比强度，不是统计置信概率，更不是化学图gate。

第16轮74个几何量为质心3、协方差6、原子间径向核5及20个受体landmark
各3个距离/径向壳层特征。名为occupancy3/5的量实际是以3/5 Angstrom为
峰值的平滑径向核，不是球内原子计数。尺度沿用冻结源库，并非新终态标签
条件下的区域效应估计。标准化特征距离的bandwidth/curvature不再是
Angstrom距离；继承的配置键名称不能改变其数学单位。

## 6. 实际FLOWR导数和控制剂量

`generation/endpoint_controller.py`通过同一生产target forward获得endpoint，
仅对坐标标量反传：

\[
g_t=\left(\partial\hat Y_{\theta}/\partial X_t\right)^\top
\nabla_y R_t(y).
\]

这是真实flowr_root模型的VJP，不显式保存完整Jacobian。模型权重冻结，
self-conditioning和离散输入保持当步条件。affinity输出detach，并有禁止
其梯度进入奖励的hook。每步不增加target forward；原有off-target forward
继续存在。一次性有限差分增加4次forward，history核另增加4次，单独记录。

剂量按线性flow位移`Delta_flow=(endpoint-X_t)*dt/(1-t)`的per-atom RMS
校准：requested_RMS=eta*RMS(Delta_flow)*time/contrast_factor，再按梯度
RMS缩放。保留单原子位移上限、累计位移预算与新严重受体碰撞检查。
没有配体pair-distance接受gate、图相等约束或禁止native键变化的控制。

导数在pre-native X_t计算，坐标增量在native随机更新后注入，这是明确的
滞后近似。trace中的first_order_reward_change只是一阶估计，不冒充额外
forward测得的更新后endpoint奖励。窗口外增量严格为零，仍完成至t=1。
FLOWR原生`apply_guidance`入口关闭，EvoMolSteer实施坐标梯度控制，群体
从不选择/复制。真正梯度、100步native续推、初始配对签名和零重采样均审计。

## 7. Agent与工程工作流的响应

Analyst先收到冻结图、信用、可观察性、路径库与对照结果，并收到通用Skill
和本轮仅替换教师源的任务。其结构化输出明确指出早期删失、克隆依赖、
标签时钟及缺失的区域效应证据，没有虚构binding hotspot。Designer随后
收到该结果，只能更换源SHA和目标定义；编译器拒绝同时更改剂量/公式或
引用不匹配的输入。第6轮另由Designer审核单一新公式。

真实subagent模拟保存了字面Skill、prompt、输入/指令/响应SHA及JSON结果。
离线探针确认换源改变坐标梯度；标签交换、平坦效用与路径连接破坏检查
确认运行时响应。参数轮由登记的确定性顺序策略提出，不声称每轮均调用
LLM，也不把指令绑定等同于Skills提升performance的因果消融。
通用Skills没有当前残基、固定分子时段或VPN/GitHub环境指令。

每个远端推理先使用本地测试并提交的代码。推理完成、压缩校验与本地同协议
评估后保留报告，再删除上一生成数据。PoseBusters曾发生跨轮日志流错误，
第8轮精确重放确认50个亲和力与48个应变数值逐项相同，只修复PB结果。
之后评估使用独立进程；评估异常阻止清理与下一轮启动。第16轮列表类型
错误发生在实际生成前，同一配置修复后重试，不能计为性能下降。

慢速传输使用带断点续传、归档校验的terminal-execution view。原始Steer
保持完整；新推理的终态SDF、全部尝试/失败、输入、trace、初始张量和
选择数组精简传输，原轨迹SHA保留。该view用于终态评价，不能还原已清理
的中间坐标。完整/精简审计一致和复制文件逐字节一致已验证。

新增标准入口为build_path_graph.py、build_terminal_path_credit.py、
build_terminal_path_library.py、run_sequential_path_campaign.py、
report_path_campaign.py及finalize_path_campaign.py。具体参数以脚本--help
为准，源数据与输出路径由接口指定。汇总默认拒绝未完成的20轮、未修复的
评估异常、配对初始状态不一致以及两次验证间修改奖励。

50项针对性测试通过；真实推理和独立验证的数值结果另见最终报告。
本文件解释已实施机制与限制，不将未实施的native标定、区域因果挖掘或
几何模式聚类写成完成项。
