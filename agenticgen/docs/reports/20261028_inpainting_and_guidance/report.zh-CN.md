# 两种inpainting、历史最优Guidance Agent及两层指引汇总

核查时间：2026-10-28 01:10:26（Asia/Shanghai）。数值来自已归档测试，不运行新推理。远端原UG生产目录仍为干净的`8db4d968cd0164b4c7550c8c0cd561b152971776`；UG04为30/30组完整版本，UG05仍为9/30组，未FINALIZED。

**目前化学inpainting更值得继续验证；叠加坐标硬覆盖未提高两教师的平均pIC50，并损害了部分结构质量。最高平均模型评分分别来自6pyy化学/原生、3w54化学/raw999；它们尚不是独立留出验证得到的稳定最优。**

## 统一比较

每组50次、五种子各10槽；平均pIC50与C包含构建失败的终态评分。C是九个PLIP直接互作核心的条件RMSD，不是整分子RMSD；PB为固定10/50抽样PoseBusters，不外推为全量通过率。PLIP列是可评价抽样中的平均教师互作恢复，不是通过率。

| 教师 | 条件与输出 | 平均pIC50 | C（Å） | 可构建/50 | 构建且连通/50 | PB通过/10 | PLIP恢复 |
|---|---|---:|---:|---:|---:|---:|---:|
| 6pyy | 原Agent最优，仅坐标gradient | 4.587738 | 0.583258 | 50 | 50 | 10 | 0.6857 |
| 6pyy | ＋化学inpainting，原生 | **4.769111** | 0.491371 | 44 | 41 | 9 | 0.8143 |
| 6pyy | ＋化学及坐标inpainting，原生 | 4.421605 | **0（硬覆盖）** | 50 | 49 | 7 | 0.8429 |
| 6pyy | ＋化学inpainting，raw999 | 4.662286 | 2.719892 | 50 | 48 | 10 | 0.7714 |
| 6pyy | ＋化学及坐标inpainting，raw999 | 4.338556 | 0.402972 | 50 | 50 | 10 | 0.7143 |
| 3w54 | 原Agent最优，仅坐标gradient | 8.082274 | 0.358047 | 50 | 50 | 10 | 0.8750 |
| 3w54 | ＋化学inpainting，原生 | 8.090530 | 0.250682 | 46 | 45 | 9 | 0.9125 |
| 3w54 | ＋化学及坐标inpainting，原生 | 6.859868 | **0（硬覆盖）** | 48 | 12 | 0 | 0.8750 |
| 3w54 | ＋化学inpainting，raw999 | **8.314264** | 0.207821 | 50 | 50 | 10 | 0.9750 |
| 3w54 | ＋化学及坐标inpainting，raw999 | 7.108307 | 0.484683 | 49 | 47 | 10 | 0.8000 |

两种inpainting均保留原坐标gradient配置。化学通道注入atomics、charges及核心内部bonds；检查点没有杂化预测头，杂化信息只记录。联合组在相同几何映射上额外覆盖核心坐标；原生终态也覆盖，因此其C=0是施加约束的结果。其未覆盖原生预测C分别为0.402140和0.531281 Å。

两次专项分别新增100与300次，合计400次；原Agent控制共100次为历史复用。化学/原生的100次在第二专项继续复用，不能再计作新增实验或独立验证。两次专项使用已选定的Agent四窗口配置，未重新调用LLM优化inpainting，因此不能将处理差异归因于更新skills。

## 历史Agent最优配置

“最优”遵从用户确认的每组50次尝试的平均重评分pIC50最高，在已完成UG结果中核查：

| 教师与node_id | 原组 | W1：10–25 | W2：26–50 | W3：51–75 | W4：76–90 |
|---|---|---:|---:|---:|---:|
| 6pyy_H7S_A_402 / single_b004_final_015 | UG04/R05/redistribute | 181.01933598375618 | 724.0773439350247 | 181.01933598375618 | 64 |
| 3w54_RNB_A_504 / single_b000_final_003 | UG04/R05/decrease | 90.50966799187809 | 181.01933598375618 | 181.01933598375618 | 181.01933598375618 |

这两组恰好也是UG04按既定C–gap规则选中的代表。UG04最佳完整版本的等教师代表均值为C=0.470652、gap=1.288607；这不是全版1500次尝试的均值。原主计划选择逻辑是最小C+0.02 Å带内最小gap，不能概括为从始至终最大化pIC50。UG05只完成第一教师9组，不能替代UG04的完整版本地位。

UG04中途有用户授权的提示修订，不能只看原manifest认定整版使用一套提示。两最优组所在晚期的三份模块文本来源为`3b7ebbc1`，与后续`03a3981e`的三份文本字节一致；6pyy晚期使用v1压缩路由，第二教师使用v2执行锚点契约。文本快照见[UG04晚期skills](best_ug04_late_prompt_skills/)。原第三轮单请求24,576额度偏离亦在原审计中保留。

## 结果如何解释

- 6pyy化学/原生相对原Agent平均pIC50提高0.181373，C降低0.091887 Å，但构建由50降至44，连通由50降至41。平均评分的五种子探索性95%区间跨零，不宜只看分数。构建失败多见核心与外围的剩余价态冲突。
- 3w54化学/raw999相对原Agent平均pIC50提高0.231990，C降低0.150226 Å，50次均可构建且连通。相对化学/原生提高0.223734，五种子探索性95%区间为[−0.141642,+0.589110]，尚未排除无收益；唯一SMILES从27降为25。
- 同为原生输出时，增加坐标覆盖相对化学/原生使pIC50下降0.347506和1.230662。3w54联合/原生抽样中9个键长、9个键角、8个连接性检查不通过，检查可重叠，PB总通过为0/10。坐标更贴合不等于化学及几何更合理。
- 6pyy化学/raw999的种子42槽2最终C=112.242249 Å、pIC50=0且不连通，仍计入人口。其step50最大单原子梯度增量692589.75 Å，数值有限；固定抽样仅取槽0、1，因此该组PB的10/10没有覆盖这例异常。C中位数0.503792 Å不能替代其全量均值2.719892 Å。

raw999组积分网格为0–0.999、100个区间，输出最后循环内覆盖前预测；实际最后坐标/原子网络时间约0.989010。原生组状态到1.0，但额外终态前向约0.999900；未把核心原子单独标为1。所有评分仍统一约0.999900。本比较同时改变网格终点、输出来源及终态覆盖，不是仅比较一个时间常数。

目前不建议把坐标硬替换设为默认。3w54化学/raw999可作为待留出种子复验的候选，6pyy需先处理有限大位移风险；不能根据已用于选择配置的同一批种子宣称泛化改进。全部失败、原始结构及未覆盖终态保留。

## 当前Guidance Agent到底加载哪些skills

最后实际运行的UG05固定在`8db4d968`，使用deepseek-flash、thinking/high、总输出上限16,384。两教师共用同一套三模块，教师身份、数据与锚点通过上下文分别传入，并无两套教师专属通用skills。

| 模块 | 实际职责 | 要求的输出/判断 |
|---|---|---|
| [analysis.md](last_deployed_skills/analysis.md) | 共享证据分析 | 区分C、gap、构建与10槽诊断；应用C容差带；最多两个决定性比较；区分已测事实与机制假设；核对绝对位移 |
| [proposal.md](last_deployed_skills/proposal.md) | 增强/减弱/重分配三个私有倾向 | 每倾向两个合法、未测、不同提案；精确实测锚点；对数格点及方向/窗口约束；证据、预测、反证分别写 |
| [synthesis.md](last_deployed_skills/synthesis.md) | 三倾向综合选择 | 每分支仅选一个已有候选ID；禁止创造/平均schedule；回看已完成动作的预测与实测；最终模式不产生R6 |

实际system构造经过`run_uniguide_prompt_version.py`和两个policy模块：私有角色加载analysis+proposal；综合加载analysis+synthesis。不能因原council代码先拼接三模块，就认定实际HTTP将三角色全文都发给了每个角色。角色后缀说明无损引用记号，v2再提供精确execution_contract和允许anchor_id枚举。

user上下文包括本教师身份、冻结标定、当前版本history、comparison_best、分支default_anchor、四窗口动作约束及已完成proposal证据。v2主要去除数据的重复序列化并路由角色，仍保留完整数值证据，没有将任务缩减为小型候选决策视图。

私有输出固定为`analysis / mechanism_uncertainty / proposals`；每个proposal有`id、branch、anchor_id、anchor_reason、mode、schedule、evidence、prediction、falsifier`。综合输出为`analysis / choice_evidence / selected`，selected含三个分支的已有候选键。最终分析不含未来选择，私有proposals为空。私有analysis目标≤1200字符，综合analysis目标≤1100、choice_evidence≤900；代码另有schema硬字符上限和执行语义校验，拒绝重复键、NaN、Markdown及追加对象。

以上约束最终正文，不能保证内部reasoning留有额度。UG05已在同一请求上两次将16,384全部用于reasoning、正文为空；不宣称精简提示已解决截断。

### 已部署、已修订、设计草案的区别

| 状态 | 内容 | 是否用于已有UG05/上述inpainting采样 |
|---|---|---|
| 最后已部署 | 上述8db4d968三模块及v2路由/锚点约束 | 用于UG05的9组；inpainting仅复用选定scale，不再调用LLM |
| 当前GitHub主线已修订 | proposal新增逐教师域及首轮多scale初始化要求 | 未迁移冻结UG05；6pyy初值181.01933598、3w54为256，来自各自full阶段实测 |
| V3离线设计 | decision_view、代码提供合法action_catalog、固定diagnosis/options/uncertainty等模块、评价结果复用及effort对照 | 尚未部署、未做LLM在线效果验证 |

analysis和synthesis的当前源码与生产字节相同；proposal比生产多出逐教师域/首轮初始化一段。生产与冻结UG05三文件SHA逐项相同，详见[runtime_audit.json](runtime_audit.json)。V3三份草案位于`agenticgen/docs/designs/uniguide_prompt_v3/skills/`，不能称为当前运行skills。

## 主GPT的skills/提示词和优化方式

目前没有UniGuide入口自动加载的独立“主GPT SKILL.md”。主GPT由本对话授权、实施协议的GPT主复盘段、暂停巡检的项目说明及review.py的验收契约共同约束。它是当前对话中进行证据复盘和改写skills的Agent，不是每个采样步骤内的另一个自动API角色。

完整解释见[主GPT项目级指引说明](main_gpt_review_policy.zh-CN.md)。核心要求为：

1. 完整版结束后读AWAITING、30组结果及4份代表策略上下文，比较当前/上一版/最优UG/R0。
2. 由实测结果定位决策能力缺陷，而不是把最优教师答案或scale写进通用skills。
3. 每次编辑1–3个模块，每项写清`观察→证据→机制适用性→具体编辑→预期行为→下版反证条件`。
4. 达到停滞、80%重复或四窗口≤5%微小增量变化条件时，实际检索原始论文/官方代码并记录版本及适用性。
5. 保持科学契约；工程错误另行修复。代码提交、固定部署、accept-review后才开下一版；连续两版下降回到最优兼容skills，UG20后停止。
6. 在下一版同时检验行为和结果，不把文字精简、解析通过、字段齐全或一次成功请求等同于分子效果提升。

最近的UG04复盘实际修改了C带内距离含义、绝对位移诊断、精确锚点整数差分和最终预测回看，并在输入中补充已完成proposal来源。UG05尚未完成且截断复现，因此这些skills修订还没有形成“下一完整版本已验证更好”的结论。accept-review能验证来源绑定与字段完整性，不能自动证明阅读深度、机制正确性或科学收益。

## 原始来源

- `agenticgen/docs/experiments/20261027_chem_inpainting/report.zh-CN.md`及summary.csv。
- `agenticgen/docs/experiments/20261027_joint_endpoint/report.zh-CN.md`、metrics.csv和analysis.json。
- `agenticgen/docs/reports/20261027_UG_summary/all_groups.csv`及报告。
- `agenticgen/docs/plans/20261026_uniguide_flowr_implementation.zh-CN.md:54`。
- `agenticgen/src/agenticgen/uniguide/review.py`、`council.py`及`strategy_provenance.json`。
- `agenticgen/scripts/run_uniguide_prompt_version.py`、`uniguide_prompt_policy.py`、`uniguide_anchor_policy.py`。
- `agenticgen/docs/reviews/20261027_UG04_review.json`、逐教师初始scale报告及V3设计文档。

本次产物：[comparison.csv](comparison.csv)、[summary.json](summary.json)、[生产/UG05 skills及manifest只读快照](runtime_audit.json)。未恢复UG05、未调用LLM或模型采样、未部署任何代码。
