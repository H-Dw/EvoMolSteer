# 动态优质seed划分与空间对比：文献依据和可执行迁移

检索日期：2026-10-08。问题：在已有Steer选择轨迹、连续预测亲和力和谱系条件下，如何识别随生成阶段变化的优质群体与空间区域，以帮助Designer设计坐标奖励？以已独立验证的R26为起点，不新增亲和力梯度、逐步模型推理或神经拟合模型。

本次使用学术研究工作流检索和核验一手论文、作者代码。关键词包含adaptive positive/negative selection、continuous-label contrastive regression、hard negative mining、false negative、dynamic mini-batch。重点覆盖2024–2026，保留2020–2023的相关基础方法。这是定向方法检索，不是声称穷尽所有文献的系统综述。排除仅有转载摘要、无法核实身份或声称分子收益却没有对应实验的数据。

## 已核实的报道和代码

| 方法 | 论文与作者代码 | 可以借鉴什么 | 迁移限制 |
|---|---|---|---|
| FALCON，CVPR 2026，Kim、Shim、Lee | [论文页面](https://openaccess.thecvf.com/content/CVPR2026/html/Kim_FALCON_False-Negative_Aware_Learning_of_Contrastive_Negatives_in_Vision-Language_Alignment_CVPR_2026_paper.html)、[作者代码](https://github.com/ku-dmlab/FALCON) | 根据锚点自适应选择负样本难度，平衡hard和false negatives | 原法在视觉语言任务中训练采样调度器；这里仅借鉴误负例审计，未复现其学习调度器 |
| ConR，ICLR 2024，Keramati、Meng、Evans | [论文](https://arxiv.org/abs/2309.06651)、[作者代码](https://github.com/BorealisAI/ConR) | 关注标签差异大但表征相近的样本，防止稀少高质量样本被多数样本淹没 | 原法训练编码器和回归器；这里采用确定性坐标邻近性，不将坐标相近等同于编码器预测相近 |
| ACCon，AAAI 2025，Zhao等 | [正式报道](https://ojs.aaai.org/index.php/AAAI/article/view/34435)、[全文](https://arxiv.org/html/2501.07045v1) | 连续标签不仅有顺序，还有距离；不应统一对待所有负样本 | 未核实可直接使用的作者公开代码；没有移植球面角度补偿，也不假设亲和力与坐标距离线性对应 |
| Rank-N-Contrast，NeurIPS 2023，Zha等 | [论文](https://proceedings.neurips.cc/paper_files/paper/2023/hash/39e9c5913c970e3e49c2df629daff636-Abstract-Conference.html)、[代码](https://github.com/kaiwenzha/Rank-N-Contrast)、[loss.py](https://raw.githubusercontent.com/kaiwenzha/Rank-N-Contrast/main/loss.py) | 用目标标签的相对排序动态定义对比关系；作者实现根据标签差异构造比较集合 | 本项目不训练连续表征；保存真实评分差和事件相对阈值，不能将方法的回归性能宣称为分子收益 |
| HCL，ICLR 2021，Robinson等 | [作者论文与仓库](https://github.com/joshr17/HCL)、[实际损失代码](https://raw.githubusercontent.com/joshr17/HCL/master/image/main.py) | 使用可控制的负样本难度；重要性重加权强调相似负样本 | 不直接采用其无监督类别先验和编码器训练损失；本项目已有在线评分，使用明确分离的低评分候选 |
| Debiased Contrastive Learning，NeurIPS 2020，Chuang等 | [论文](https://proceedings.neurips.cc/paper/2020/hash/63c3ddcc7b23daa1e42dc41f9a44a873-Abstract.html)、[代码](https://github.com/chingyaoc/DCL) | 随机其余样本可能含真正正例；应明确误负例来源 | 被Steer淘汰不等于原生续推失败；缺失终末标签不赋零分，也不凭空估计真实正例比例 |

核验说明：会议与作者仓库的名称、年份和方法对应。FALCON全文PDF访问返回403，但会议检索记录和作者仓库均可核实；实际读取了[FALCON_utils.py](https://raw.githubusercontent.com/ku-dmlab/FALCON/main/models/FALCON_utils.py)的表征队列和[actor.py](https://raw.githubusercontent.com/ku-dmlab/FALCON/main/models/actor.py)的Beta分位策略网络。ACCon没有伪造一个“官方实现”。上表每项都只是方法依据，分子性能需要下面的独立实验。

## 本项目的具体算法

每个实际选择节点分别排序已观测评分。每个祖先家族总质量相同：粒子权重与该root的候选数成反比。选择满足正负有效权重数至少3的阈值，最大化加权类间方差。阈值两侧留出事件加权IQR比例的模糊带。阈值动态变化，模糊带比例等策略参数则在实验前冻结。该加权Otsu实现是工程选择，不是上述文献提出或证明的分子算法。

正例表示“此节点明确较高在线分数”，负例表示“此节点明确较低在线分数”。二者都不是实测结合标签。模糊样本单独记录。未观测终末后代绝不被解释为负例；本模块的标签直接来自已观测joint-head前向评分。

先做正例与其余明确低评分候选的对比，再单独测试空间难负例。难负例用质心和协方差9个全局坐标量匹配，并保留一部分普通低评分质量；不使用被研究的局部区域特征进行匹配，以免抹掉真实区域差异。原子类型和化学图身份不进入选择规则。

分析模型预测终点的三维坐标，而不是将尚未成形的当前坐标当成熟分子。沿用固定受体参考系的20个几何landmarks，分析softmin距离与中心在3Å、5Å的高斯径向壳层密度；“occupancy3/5”不是半径球内计数。额外保留质心、协方差、距离核等全局诊断，共74项。

全窗口按原始节点汇总，不划分0.1子区间。独立单位是生成批次；谱系权重不产生新的独立重复。对缺乏可识别正负群体的事件记录缺失，统计仅使用观测网格质量并报告覆盖。按批次bootstrap计算区间，使用批次效应检验和BH校正。跨批次方向一致且通过校正的区域才进入奖励候选。

使用0–3阶Legendre函数拟合整个窗口的区域差异。通过留一批次交叉验证及一标准误差规则选择复杂度；解析导数描述群体特征随时间变化率。这个时间导数不是空间力。执行奖励求导仍对坐标场本身求导，再经真实FLOWR endpoint VJP传回当前坐标。

## 分阶段实验与反证

十轮按顺序运行：新R26/无引导对照；轻量区域对比；两档区域权重；难负例；模糊带；正区域吸引替代负密度排斥；冻结后的R26独立对照；冻结候选配对比较；另一组独立验证。每次只改一个模块，前一轮报告落盘、校验、提交后才形成下一轮。具体合同见实验protocol.json。

奖励采用原有R26构象多模态吸引加一个有界区域对比项。若新候选在独立批次未超过历史最优，保留R26，不以训练区富集、一个高分分子或较好筛选批次替代验证。正负区域密度差可遗漏整体相容性，也可能把评分模型偏差带入奖励；这两点需要最终亲和力和构象检查来检验。

本报告及实现由AI辅助整理。保留来源链接、原始轨迹SHA、策略、统计输出、Agent输入/指令/响应绑定和执行commit，供复现和审计。

靶点核验：当前受体输入3PE1对应[人CK2α与CX-4945复合物](https://www.rcsb.org/structure/3PE1)，对照受体6KHF对应[CLK3与CX-4945复合物](https://www.rcsb.org/structure/6KHF)。区域landmark的残基编号从本地精确输入PDB坐标映射，不直接从论文作用机制推断。
