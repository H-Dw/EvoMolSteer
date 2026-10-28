# Guidance Agent 序贯实验技能包 1.0.0

日期：2026-10-28，Asia/Shanghai。基于 `74ed430c` 的几何目标指引构建新的标准技能包 `guidance-experiment-optimizer`。版本号属于技能包，不表示启动了新的实验主版本。

## 交付结构与格式

标准入口为 `agenticgen/src/agenticgen/resources/skills/guidance-experiment-optimizer/SKILL.md`，使用 YAML frontmatter，包含合法的小写连字符名称、适用场景描述和 `metadata.version: "1.0.0"`。目录内还包含：

- `references/history-learning.md`：对照、配置身份、预测回看、可更新的历史结论。
- `references/control-diagnostics.md`：可泛化的局部响应、过强干预、状态依赖、区间耦合及稳健性诊断。
- `references/proposal.md`：合法干预选择与固定提案输出模块。
- `references/synthesis.md`：候选组合、互补信息价值与固定选择输出模块。
- `references/sources.md`：原论文、官方代码、访问版本/文件哈希、采纳内容与迁移边界。
- `agents/openai.yaml`：显示名称、简短描述和标准调用提示。
- `scripts/render_runtime.py`：纯文本生成及一致性检查，无网络、模型或采样操作。

现有应用并不会按标准技能目录自动读取引用。生成脚本将共享入口、历史学习和控制知识合并进现有 `uniguide/skills/analysis.md`，将两个角色引用分别生成 `proposal.md`、`synthesis.md`。原有路由仍按“共享分析＋当前角色”注入；文献出处和维护说明不进入每次模型输入。`BUNDLE.json` 记录来源、版本、角色组合和 UTF-8/LF 规范化哈希，既有运行绑定仍对实际注入的 Markdown 字节冻结。

标准技能包是唯一编辑源；三份兼容文件由脚本生成，避免手工维护两套内容。未将该技能安装到全局 Codex 配置。

## 将持续优化落实为证据循环

| 环节 | 必须回答的问题 | 保存到现有输出的位置 |
| --- | --- | --- |
| 历史核对 | 哪些记录同教师、同原子映射、同协议和同随机条件？哪些仅为重复？ | `analysis` 引用决定性记录，不复制全量表格 |
| 假设更新 | 以前的预测得到了支持、被反驳，还是证据不足？此次权重分配结论有何改变？ | `analysis`、`mechanism_uncertainty` |
| 实验选择 | 此候选是利用已测收益，还是区分竞争解释？为何值得消耗预算？ | `evidence` 中写目的、依据和假设 |
| 合法行动 | 使用哪个实际测量锚点，改变哪些区间，是否符合格点、边界、方向和去重规则？ | 原有身份字段、`anchor_reason`、`schedule` |
| 可证伪预测 | 两个几何指标相对原锚点预计如何改变？什么结果将否定预测？ | `prediction`、`falsifier` |
| 实测回看 | 实际执行是否等于注册提案？结果是否改变了先前判断？ | 下次读取 `completed_action_evidence` 后更新分析 |

选择角色同时检查候选集合：不同分支可能抵达相同配置或检验几乎相同的假设。相同科学条件和随机流下的重复评测只算复用证据。候选的名字、父锚点或理由不同，不构成配置新颖性。真正的新随机重复由外部预算授权。

不增加输出字段。私有建议仍为 `analysis`、`mechanism_uncertainty`、`proposals`；每个 proposal 保留现有十个字段。选择仍为 `analysis`、`choice_evidence`、`selected`。分析模式维持原来的空 proposals / 无 selected 契约。新信息模块放入这些既有字符串字段，禁止额外自由正文或 schema 外对象。

## 控制知识如何迁移

技能要求先寻找有效干预与局部响应的证据，再提出调整方向。权重不是坐标位移；它与积分步长、时间系数、梯度、原子匹配和当前状态共同决定实际作用。降低相对增量比、保持数值有限，都不构成稳定性证明。

诊断覆盖：有响应但控制不足、坐标改善伴随局部结构受损、存在时序证据的过冲、对权重变化缺乏响应，以及不同背景下的区间耦合。每种诊断都要求相应观测，并给出可区分解释的合法实验方向。没有轨迹证据时，不依据一个终态指标宣称振荡或过冲；没有系统模型时，不宣称获得阻尼比、稳定裕度或理论安全保证。

增大某一区间权重并减小另一区间权重是联合干预。对数格点上的等幅相反变化不保持总 scale 或总物理位移守恒。也不预先赋予早期/晚期固定功能；分工应从当前教师的匹配历史学习。

目标始终是完整名称的“控制原子坐标均方根偏差”和“控制原子与周围原子的距离偏差”。邻居只属于生成分子内部，区分成键/非成键，使用提供的参考距离或允许区间，不以缩小所有距离替代合理结构。教师之间不平均初始权重，轮次和终止由外部控制器决定。

## 采用与未采用的外部经验

从 LLAMBO 提取历史条件下的候选生成、评测更新与信息价值；从 Eureka 提取分项反馈驱动的假设修正；从 AlphaEvolve 提取评测结果与候选谱系驱动后续选择；从 Coscientist 提取计划、执行和实际证据分离。具体出处及源码哈希见技能的 `references/sources.md`。

这些是本项目的适配，不宣称论文已经验证了本分子任务。没有把 Eureka 的奖励重写指令迁移过来，没有要求模型凭语言构造经过校准的贝叶斯后验，也没有把经验边界当作 SafeOpt 的理论安全集。控制理论知识用于诊断实验，不改写当前科学算法。

## 验证与实际边界

已完成：

1. `skill-creator/scripts/quick_validate.py` 验证通过：标准入口、名称、描述、frontmatter。
2. 相对引用、UTF-8、显示元数据及运行文本检查通过；运行指引没有旧单字母指标、亲和力优化、固定迭代次数或特定教师答案。
3. 渲染后 `--check` 无差异；Windows/Linux 换行得到一致内容与哈希。
4. 新增 7 项技能包检查，加上既有提示策略和执行标识适配检查，共 **42 passed，6.62 秒**。测试覆盖只读漂移检查、无效源文件不产生部分输出、既有文件保留、实际角色加载和无损上下文。
5. 从提交基线源代码叠加本次技能相关文件进行隔离 wheel 构建，**12 个所需资源全部存在且字节一致**。未包含并行出现的其他未提交科学代码修改。

静态语义审阅包括：重复配置不作为独立验证；多区间改动不分拆归因；缺少距离数据保持未知；最终硬覆盖不冒充权重收益；有限极端位移不因均值改善而消失；预测被反驳后要求更新分配假设。这是指引审阅，不是大语言模型行为试验。

每个私有调用的生成指引为 15,571 字符、2,073 个空格分词；选择调用为 14,831 字符、1,965 个空格分词，不等于 provider token 数。新增知识使其比上一份平铺指引更长；通过角色路由和仅保留决定性输出限制冗余，未声称降低过内部 reasoning 消耗或解决截断。

本次没有运行模型请求、数值采样、远端部署或恢复暂停任务。标准技能合格和接口测试通过，不等于已验证实验效果。完整历史查询、规范化配置缓存、距离指标采集、几何排序和全流程轮次参数化仍须执行器真实提供；此技能包不会自动实现它们。本次没有接管或提交工作区并行出现的其他几何执行器/采样器修改。

## 维护命令

在仓库根目录执行：

```powershell
python agenticgen/src/agenticgen/resources/skills/guidance-experiment-optimizer/scripts/render_runtime.py --output-dir agenticgen/src/agenticgen/uniguide/skills
python agenticgen/src/agenticgen/resources/skills/guidance-experiment-optimizer/scripts/render_runtime.py --output-dir agenticgen/src/agenticgen/uniguide/skills --check
python -m pytest -q agenticgen/tests/unit/test_guidance_skill_bundle.py agenticgen/tests/unit/test_uniguide_prompt_policy.py agenticgen/tests/unit/test_uniguide_anchor_policy.py
```

格式校验器位于本机 `C:/Users/Wei/.codex/skills/.system/skill-creator/scripts/quick_validate.py`。不要将生成目标指向冻结的实验目录。真正上线前，需在独立的新实验配置中核对执行器提供的几何指标、历史覆盖及行动契约；该要求不授权恢复原暂停任务。
