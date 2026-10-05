# CK2 单目标坐标优势挖掘与条件梯度模仿

本次从1重新编号，优化上限为30轮。每轮使用主seed42、100个native积分步；两批各50个候选，批次种子为42+100003×batch。原始Steer数据和checkpoint保留。代码先在本地commit，经127.0.0.1:7897代理push到GitHub，再由远端pull后运行FLOWR。远端负责推理和无损导出，本地负责统计、LLM模拟、物理评价与比较。

实际完成5轮开发比较后结束本次campaign；30是上限，未盲目用满。当前表征、奖励和剂量尚未支持中间形状、affinity与物理兼容性的联合改善。第三轮形状接近度最好，代价是较差的终态松弛代理；第五轮周围RMS较好，但实际中间形状和all-head没有收益。这不否定整个方法，只限定本次结论。

最终逐轮数值见[自动汇总](experiments/ck2_coordinate_seed42_20261006/campaign_summary/summary.md)。保存的逐轮证据、配置、失败行和清理审计位于[实验目录](experiments/ck2_coordinate_seed42_20261006/)。

## 科学问题与判定口径

问题是：能否从原始affinity选择中发现局部坐标轨迹优势，用可微奖励从头模仿，使actual x_0.5更接近Steer，同时在无引导继续到1后改善分子质量和周围区域适配？这包含三个不同命题：选择中存在统计特征；奖励能改变实际中间状态；改变能改善最终性质。前两者不自动推出第三者。

0.5主要指标为actual当前坐标到原始700个discovery样本的双向几何匹配距离；采用几何原子指派，类型与键失配单独报告。双向距离同时检查接近参考与参考覆盖，不能以奖励值或预测终点距离替代。最终指标分开报告有效/PB fast子集通过率、唯一分子/骨架、同态FLOWR affinity head、MMFF松弛能/重原子、周围区域松弛RMS。head是共享模型预测；MMFF是内部应变/松弛代理，并非结合自由能。all、valid、unique三个分母均保留，失败不从候选总数中删除。

两个开发批次已多次用于比较，不具有盲验证含义。100个粒子不是100个独立统计重复；报告不据此夸大显著性或宣称未见seed泛化。

## 挖掘逻辑与新增可观测量

实际`resampled`字段决定分析范围，本数据为score t=0,.01,…,.5，共51节点。全部节点统一富集、趋势和窗口效应分析，不切0.1子区间。next-score分析仅有0–.49，因为.5之后不存在窗口内后继；不借.51补齐。

1. 在对齐受体世界坐标系中，计算残基Gaussian区域的xyz质心、散布、预测终点剩余漂移、实际native位移三轴速度及RMS；all与预测NOS通道分开。当前锚定版本40区域×2×11=880特征。
2. 早期current坐标是噪声，不能代表将来结合区域。新增endpoint锚定：由同一步预测终点确定槽位权重，观测actual current/proposal的运动。预先列出的8区域×2×15=240特征。两版本假设集合不同，其BH q不能直接横比。
3. 新增同父节点去重的lag gain：被保留父节点的下一步子代评分先平均，再减当前父评分，调整起始head。同期相关还控制全局质心、散布和NOS数。保留存活偏差、共享oracle及谱系塌缩的限制。
4. 区域特异性分析减去同刻跨区域共同选择效应，另存完整窗口残差和PCA。共同收缩解释all-spread选择变化能量约80.10%，因此显著收缩不能自动称为局部机制。
5. 独立transport诊断只新增8×2×5=80特征：剩余位移RMS、native与剩余位移余弦、有效槽位数、终点核心权重、当前核心权重。它揭示ASN117核心支持始终为0；VAL116/NOS预测锚定核心权重约.7242、有效槽位约1.90，current核心权重却仅约.000169。后者不表示早期已有物理接触。
6. 节点贡献审计保留全窗口求积定义，计算首节点/主导节点贡献与有效节点数，不新增子区间假设检验。ASN117/VAL116 NOS native_y的首节点贡献几乎解释完整净效应，约占绝对贡献总和75.3%/74.2%；VAL116 proposal_x的lag首节点仅占约.366%，两种信号不能混同。
7. 参考构建新增紧凑的selected/background联合均值协方差、selected原始特征值、概率ESS/KL和标准化均值差，便于后续检查经验模仿是否只是复现背景。它们没有自动变成密度比奖励。

每个feature/metric以批次为独立单位，计算完整窗口积分、t置信区间、p及split×metric内BH q；缺失NOS按feature掩码处理，不牵连all通道。lag时间覆盖明确记录，缺失不填零。低阶Legendre函数采用leave-one-batch-out与one-SE选阶，保存解析时间导数。拟合可以违反物理边界，例如产生负spread，故执行奖励使用实测节点参考；时间导数也不能冒充坐标梯度。

原40区域版本adjusted lag没有q<.05结果；endpoint版本16/240通过，但全部64个proposal选择偏移检验未通过。ASN117/VAL116 NOS proposal_x的调整后lag相关约.0330/.0318，ASN117 proposal_spread约−.0452。效应很小，且窗口末端11/14 discovery批已只剩一个根。这些支持可证伪的模仿试验，不支持固定+x、−y或无限压缩的因果规则。

## 与flow matching一致的奖励与剂量

核查远端真实integrator源码后，线性endpoint参数化的预测速度为v=(y−x)/(1−t)，但native SDE增量另含score drift及噪声。g_t=1/(t+.01)，t=0时g=100且dt=.01，score漂移近乎抵消先验坐标。实际native位移并非纯预测flow速度。在噪声坐标和不确定键图上强行计算局部MMFF/接触能会制造证据，因此本次中间控制以坐标为主，物理能量在终态有效图上评价。

条件Gaussian权重由预测endpoint决定，硬NOS掩码固定于本次求导。实际proposal x_(t+dt)给出区域特征z；每个discovery批次按真实选择概率计算同刻mu_b、Sigma_b。14个组件是批次经验代表，不是恢复出的真实分子模态，也不唯一还原原子结构。

混合奖励为：

\[
q_b=\frac{(z-\mu_b)^\top\Sigma_b^{-1}(z-\mu_b)}{D},\qquad
R_t(x)=\tau\log\frac1{14}\sum_b\exp\left[-\frac{\delta^2(\sqrt{1+q_b/\delta^2}-1)}{\tau}\right].
\]

tau=.25，delta=1；协方差采用.1对角收缩与.15 Å特征SD下限。第三/四轮拼ASN117+VAL116/NOS为D=8，第五轮仅VAL116/NOS为D=4。删去相关视图也改变正则化和残差gate，不能称为完全相同数值剂量。

梯度直接对actual proposal坐标求导，不经过预测endpoint锚的模型Jacobian；FLOWR仍执行每一步真实forward、native SDE及类别采样。锚每步刷新，这是一种条件坐标控制，不能宣称对离散类型/键或未来head反传。第一、三轮实际执行了零剂量与native的100步及最终张量一致性检查；第四、五轮没有新增zero臂，沿用第三轮的控制器检查。各轮真实有限差分preflight通过，但只验证记录的测试状态与方向，不代表全部推理状态均做过FD。

引导范围读取学习参考，不硬编码0–.5。score .49→state .5是最后受控更新；score .5所记录proposal .51不能再次注入。之后到1纯native，无粒子克隆/选择/SMC。第四/五轮外部剂量为eta=.05倍`RMS(dt*v)`乘有界残差gate，保持原生积分器不变。记录预测flow、actual native、标定和实际注入RMS。每原子单步cap .025 Å、累计RMS cap1.25 Å、成对距离变化cap .06 Å、新严重碰撞阈值.8 Å、7次回退。可选刚体投影保留为独立消融，本次没有执行。

## 已检验的设计变化

第一轮为ASN117/current/all散布上尾抑制，第二轮为全局散布对照。前者actual x_.5形状距离改善1.68%，全局对照改善2.34%，说明不能把该变化归为独特局部优势。第一轮平方注入全部处于current定义的5 Å核心之外，也促成endpoint归属诊断。

第三轮条件endpoint/NOS联合混合以observed native RMS标定，形状改善2.59%，valid/PB为100/99，valid head7.4558；但MMFF松弛/重原子恶化约4.04%，周围松弛RMS恶化约12.11%。这不支持整体适配假设。

第四轮只改为predictive flow RMS标定。形状改善降至.0778%，两个批次方向相反，generated到reference距离略变差、反向覆盖略改善，不能称为稳定几何收益。周围RMS较native改善约2.89%，MMFF近乎不变，valid/PB为97/97；它们是不同有效结构集合上的描述性代理变化。首受控更新只占平方注入约.8%/1.1%，100个候选的50次窗口更新均有非零注入；弱效果不是只在少数步施加奖励。两批约27.3%/37.1%的平方注入落在endpoint定义的5 Å核心外，周围原子并非完全未受控。它提示第三轮形状收益与native标尺下的控制分布相关，不能用奖励代理改善代替最终成功。

第五轮由已授权Analyst/Designer subagent模拟复核，移除无终点核心支持的ASN117，仅保留VAL116/NOS，沿用第四轮eta、混合架构和flow剂量。参考均值与第三轮VAL子向量逐值一致，最大差0；真实参考由编译器重建并绑定SHA，LLM没有凭文字生成坐标数组。

第五轮actual x_.5双向距离.238067 Å，native为.237427 Å，恶化约.269%；valid/PB97/97，unique71（native69、原Steer24）。all-head7.428219，低于native7.434275；valid-head7.439121略高于native7.437040，但有效分母已改变。MMFF/重原子.548919与native.548773近乎相同；周围松弛RMS.312728 Å比native.327214降低4.43%，仍高于原Steer.202921。50次控制更新均有非零注入，首步仅占总平方量.703%/.957%，之后无注入。两批28.6%/38.7%的注入平方仍在终点5 Å核心外。保留100行候选与385,385字节本地报告，下载39文件包的SHA已校验；随后按计划清理结构与轨迹。

五轮分别检验散布上尾、全局对照、条件NOS混合、flow剂量标尺、删除无核心支持区域。没有足够证据支持继续提高强度作盲调，因此停止。未在heldout批次运行独立验证，未将原先看过的validation汇总包装成盲验证。已授权LLM模拟的最终审阅赞同此限定结论，记录见`final_Analyst_review.simulation.json`。

## 工程接口、存储及复现

详细命令与接口见[坐标挖掘说明](coordinate_affinity_mining_20261006.md)。主要入口：

- `scripts/mine_coordinate_advantages.py`：dataset/campaign/analysis/output；可指定anchor、current/proposal及geometry/transport/joint。
- `scripts/analyze_regional_specificity.py`、`analyze_coordinate_influence.py`：独立结果目录，前者保存共同效应/PCA/导数，后者保存全窗节点贡献摘要。
- `scripts/coordinate_agents.py`：export/import/api/compile；可输入transport与influence补充证据。skills位于`skills/coordinate-analyst`和`skills/coordinate-designer`。严格schema、证据ID、时间范围、程序白名单检查通过后才能编译。
- `scripts/generate_coordinate_flowr.py --flowr-root <程序路径>`：真正FLOWR推理，可指定checkpoint/program/reference。此次使用受保护v2 checkpoint，SHA见执行记录。
- `scripts/summarize_coordinate_campaign.py`：直接读取保存报告，核查seed/窗口/初态匹配/完整100步/无SMC/无越界注入，输出小型CSV/JSON/Markdown，不需恢复原始生成轨迹。

批次float64统计以zstd Parquet保存，不保存逐候选特征、父子边展开、并行分片和复制坐标。canonical当前锚分析约78.5 MB，endpoint约26 MB，独立transport约7.4 MB，节点贡献约.42 MB。区域分析范围不同，不将这些大小视为同定义压缩率。重复旧分析约109 MB已清理；原始轨迹、catalog、参数、source SHA和代码保留，因此能重算。逐轮报告只保存失败包含的候选统计、时间/剂量汇总、配置和结论，生成结构/轨迹/临时结果及下载包在报告校验后删除。

本次新增/修改模块的完整测试为107项通过；报告工具另经四轮和五轮真实保留报告运行验证。
最终在原始生成数据已删除的条件下重建CSV/JSON/Markdown，三个文件逐字节一致。
远端新work目录清理后仅57,182字节，全部新生成数据和归档已消失；原始Steer目录及v2 checkpoint存在。
最终清理、远端状态和重建审计分别为`cleanup_final_remote.json`、`final_remote_state_audit.json`、`final_report_rebuild_audit.json`。

本轮方法允许提出“局部优势轨迹模仿”假设，尚不能证明全局密度匹配或结合因果机制。原生SDE/类别变化、条件槽位的稀疏支持、均值/散布的不可辨识性、存活偏差及少量独立批次共同限制了结论。14个发现批次和700个参考粒子并不能增加本次2个开发批次的独立验证重复数。最关键缺口是selected局部分布是否包含超越背景和全局几何的增量优势信息，并能解释物理兼容性。若当前试验没有联合收益，结论限于当前表征、奖励和剂量，不能推广为整个思路不可行。后续若增加设计，需有明确独立证据与预先定义的比较，而非因30轮额度尚未用满继续盲调。
