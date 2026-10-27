# 当前主GPT如何优化Guidance Agent

本文件整理已批准的UniGuide项目级操作要求，用于解释当前工作方式；不是历史system prompt的逐字副本，也不是新增并接入执行器的SKILL.md。未改变科学协议、提示词或暂停状态。

## 当前真正的指引来源

1. 本对话的人类任务指令及其后续修订。
2. `agenticgen/docs/plans/20261026_uniguide_flowr_implementation.zh-CN.md`的“GPT主复盘”和部署规则，及execution文档的实际审计。
3. 原`uniguide-flowr`自动化保存的项目操作说明；目前PAUSED，开头的暂停约束覆盖其保留的历史运行说明。
4. `agenticgen/src/agenticgen/uniguide/review.py`构造的AWAITING_GPT_REVIEW与accept_review验收契约。

UniGuide执行器未单独加载一个“主GPT Agent”的SKILL.md，也未在review.py中自动调用另一个GPT API进行主复盘。它构造复盘请求后等待当前对话主Agent实际完成阅读、分析、编辑和提交。

V17的策略来源记录在`uniguide/strategy_provenance.json`：只选择实测行动锚点与最优比较对象分离、私有提案后综合选择、可反驳的机制推理三类思想，排除了旧教师结果、旧scale/裁剪规则和旧版本历史。仓库另有`resources/skills/plip-schedule-review/SKILL.md`，但它不是当前UniGuide主复盘入口加载的独立指令；不能把该文件在当前HEAD的全部旧流程约束当成已绑定UniGuide的提示词。

## 现行要求如何引导一次真实优化

**先读实测结果，再解释决策。** 完整版本须有两教师各5轮、共30组、基础统计和结构保留。读取AWAITING、30组指标以及每教师代表私有分析和对应综合上下文，共4份策略上下文；比较当前、上一版、最优UG与R0。抽样诊断的待完成、未选、构建失败、无法评价及测量不通过分别报告，不等待其余40个未抽样槽位。

**按冻结目标判断改进。** 代表选择使用最小C+0.02 Å带内再比较绝对mean-pIC50 gap，而不是直接最大化pIC50。行动锚点可以与最优比较对象不同。主版本整体比较用两教师代表组等权均值，不是把1500个候选混为总体。后来的inpainting专项按用户明确指定的最高平均pIC50选控制配置，是另一种事后选配置口径。

**把缺陷变成可检验的能力修改。** 从真实决策中找出误读、证据遗漏、机制假设不适用、冗余任务或输出契约问题；每次只更新1–3个现有模块。每个修改必须记录六环链：`observation → evidence → mechanism_applicability → skill_edit → expected_behavior → next_version_falsifier`。下一版需检验预期行为与实际分子结果，不能仅凭文字变短、解释更完整或JSON通过宣布科学改进。

**在规定条件下做真实研究。** 任一教师的相邻版本C和gap变化绝对值均≤0.02，或15套配置至少80%与前版重复，或代表配置四窗口完整增量向量变化均≤5%，触发检索原始论文及官方代码。来源须记录URL、版本/commit、访问时间、适用假设和primary_source。缺失向量是unknown；阈值用于触发研究，不是显著性判断。没有触发且没有实际检索时，不能声称做了新的文献研究。

**保留科学和执行边界。** 常规skills优化不改变loss、VJP、原生适配、窗口、随机流和预算，也不把个别教师最优数组写入通用skills。实现错误单独修复并标记受影响运行。用户后来授权的逐教师首轮标定初始化属于已审计代码修订，不能通过改文件直接迁移冻结UG05；原版本的历史初值和manifest仍应保留。

**用下一完整版本验收。** 本地修改、提交并推送GitHub，空闲边界部署固定提交；accept-review核查请求SHA、4份上下文、1–3个模块的真实变化、六环链和研究来源字段，再允许下一版。连续两个相邻完整主版本下降时，从最优兼容UG skills重新修订，不重跑最优版/R0；UG20后不启动UG21。用户暂停指令优先，目前不执行新版本。

## 代码校验到哪里为止

review.py能检查提交的记录是否绑定请求、列齐四份上下文、模块变化清单是否准确、必需字段是否存在。它不能自行证明主GPT确实理解了文献、选对了机制或使生成更好；这些仍需真实源证据与后续结果检验。它也没有把“完整阅读30组”实现成可证明的模型行为。

最近的真实示例是`agenticgen/docs/reviews/20261027_UG04_review.json`：发现最终模式漏掉预测回看、C带内距离含义混淆和有限大位移诊断不足，分别更新analysis/proposal/synthesis，并增加已完成proposal的来源绑定。UG05随后仍发生16,384-token reasoning耗尽，说明指引可要求简短，但不能保证内部reasoning预算，也不能把复盘完成等同于修复已验证成功。
