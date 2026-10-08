# Analyst实际调用：共同深度区域联合假设

本次读取本目录的字面instructions全文、请求schema、完整payload及供给的四份Skills。实时`regional_workflow.payload`核验所有绑定后，与供给payload完全相同。仅保存原始结构化响应和本报告，未编译、导入响应、运行模型、修改源码或提交。主流程随后执行规范导入。

## 实际工具核验

已执行本轮要求的检查：

```text
.venv/Scripts/python.exe scripts/flowcompat_regional_agent.py --action check --regional-receipt docs/experiments/flowcompat30_20261009/mining/regional_reference_v2/execution_receipt.json --reference configs/experiments/flowcompat30_v1/regional_common_depth_reference.json.gz
```

实际返回值如下，退出码0：

```json
{"passed":true,"reference_sha256":"4349026db4963c55712cf7bd70f7085b1243290cf465f2acd2693420cd7a4b12","receipt_sha256":"04c5710d3b8834f0174f9a2318a62db9e15d4440d0ba8e6c36ba27ff02c5fa34","coverage":{"new_active_teachers":285,"original_eligible_teachers":289,"teachers":1400}}
```

新来源身份是`regional_common_depth`，实际脚本为`synthesize_common_depth_reference.py`，SHA=`cd4d34cb0f8ba37993a7c71ddc49298b458140747d91e74a4a71574350f5eae5`；对应模块SHA=`954f95b96c6ec7173d727633d171bddeeb091c5f0f238fd0203e563efd437d51`。执行receipt绑定72项输入、6项源码、4项输出，执行成功且输入/源码未变。不是旧branch工具产生了这个参考，也不是把混合深度v1结果改名。

同时核验了供给的selection_innovation、flow_compatibility、branch_mutation三个实际receipt与multi_depth真实自执行receipt。参考检查确认原教师真值不变、联合配方匹配、单位atom-RMS、去平移四项不变量。这是来源与几何校验，不是模型实施响应或性能证明。独立特征审计作为绑定文本上下文读取，没有伪装成原工具产物。

## 共同深度的优势与限制

深度2/3/5/8/13的合格区域数分别为3/7/6/4/0，固定发现规则先最大化合格区域数，再比较平均LOO正方向比例，最后选较浅深度，因此选中3。所有入选区域3、6、11、12、16、17、18使用同一观察跨度，降低了混合不同lag时协方差幅度受累积时间影响的问题。选择仍重复使用14个发现批次，不是独立确认，也不证明深度3最大化亲和力。

| 区域 | XYZ均值协方差，×10⁻⁶ Å·pIC50 | 合格分量 | LOO正方向批次 | 平均余弦 |
|---|---|---|---:|---:|
| 3 | (−0.1039,−0.4408,+0.2844) | x、y | 13/14 | 0.6760 |
| 6 | (+0.0731,+0.6225,+0.0659) | y | 10/14 | 0.4373 |
| 11 | (−0.1463,−0.6551,+0.3047) | x、y | 11/14 | 0.4185 |
| 12 | (−0.1330,−0.6553,+0.3241) | y | 13/14 | 0.6792 |
| 16 | (−0.4404,−2.2248,+1.0720) | y | 13/14 | 0.6939 |
| 17 | (+0.2179,+1.1177,−0.0970) | x、y | 11/14 | 0.5223 |
| 18 | (+0.1214,+0.8986,+0.0373) | y | 12/14 | 0.5983 |

完整XYZ保留不显著分量，表内均值并不表示每轴有支持。全部z分量q>0.05；region6仅10/14正方向，接近资格阈值。region16绝对协方差最大，仍可能主导联合场；共同lag没有消除成员密度、方差和区域重叠。LOO是发现集一致性诊断，不提供新p值；批次对、时间行、深度和复制子代不是独立重复。

对应正式ID为`regional_reference/region_XX/depth_3`及`regional_reference/teacher_coverage`。旧分支正式ID`branch_mutation/adjusted_covariance/joint_innovation_RMS_A`给出−3.59935e−5 Å·pIC50、q=0.020284。即时同父端点复制严格相同，但不同立即父分支在共同祖先内仍有变异；初始root坍缩不能取消这类比较。

同深度的`multi_depth/3/…/region_18_internal_displacement_y`原始/调整相关性为+0.163269/+0.152699，协方差+2.36583e−6/+8.98585e−7；调整协方差95%区间[5.29673e−7,1.29367e−6]、q=0.013677。绝对与标准化结果均需要保留，不能只据相关性放大奖励。调整中的化学标签/全局平移是观测诊断，可能移除中介信息，不等于坐标直接因果。

## 时间支持与联合空间解释

请求窗口逐字绑定为[0,0.5]；实际评分至0.49、最后选中状态至0.5。七个区域XYZ拟合均为degree0，观察支持0.03–0.49、时间导数0。0、0.01、0.02没有共同深度3证据，新场缺失，保留原奖励。原来0.02四个合格教师失去新增场，解释了289变为285；这不是学习到的阶段切换，不能补值或创造平滑早期规则。

配方以完整教师T的固定Gaussian成员h合成`bᵢ=Σᵣ(hᵣᵢ−meanᵢhᵣᵢ)Cᵣ(t)`，再取`d=b/RMS(b)`。常数C不意味着教师随时间改变时d也不变。统计成员来自祖先，控制成员来自当前完整教师，这是显式空间转移假设。去平移不去旋转；槽位不等于物理原子。单位RMS归一化丢弃绝对效应尺度，不能把Å·pIC50协方差当Å位移或亲和力空间梯度。

## Designer可检验的边界

本次只有`branch_mixture_regional`与程序`R26_branch_regional`，精确参考SHA见上。只允许修改`branch_mixture.virtual_mass`，范围[0,0.5]；符号、区域权重、控制器等固定。较小α是探索预算，原教师至少保留1−α质量，不能把它叫优化概率。

虚拟位移尺度另取已观察lag2分支对比RMS（Å），以0.2Å封顶；它不是lag3协方差幅度，atom-RMS上限也不是逐原子上限。原lag2 confidence保留资格而不校准新区域方向，不能开平方或归一化制造可靠性。几何、原子类别和化学图均保留原生自由度，不加graph gate、亲和力head梯度、新网络、生产前向或重采样。

假设是：在完整联合背景中，以有限虚拟模式预算试探统计平均方向，可能改善端点路径；目前属于干预假设。必须先验证新增奖励梯度和实际配对窗口轨迹改变、空机制逐位一致、真实FLOWR一次正常forward+VJP与原生续推，再以冻结初态/随机流和实际剂量作正/反/空方向比较。正方向不优于反方向或R26可反驳方向有效性。随后才比较最终亲和力、次级应变与兼容性并做独立确认。

在线标签也参与selector，存活条件与淘汰未来删失仍在。现有数据不证明物理能量、神经attention或head/模块因果贡献；也不能据本次来源检查宣布30轮完成或性能改善。

## 响应交付

请求SHA=`4b51c6dd1ce1fe9b6c855d253196607c388928ffa5b579676957ce2613648fae`，payload SHA=`449eef486eadbdabc26fd772c043feab276588530573a56525df1efbc6ecd198`。

原始响应`Analyst.simulated.raw.json` SHA=`d6938181610aed7b9b908f2054c5bb6696fc087fab771c2a4537d553bbab4312`。已在内存按供给schema验证并检查所有引用ID属于供给集合；规范import尚未执行，交由主流程完成。来源、窗口、指令、注册表、新参考与全部receipt哈希均使用供给绑定，没有手工替换。
