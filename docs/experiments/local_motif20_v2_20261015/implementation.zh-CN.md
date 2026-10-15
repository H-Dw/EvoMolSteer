# 本轮可复用接口与执行边界

根目录为 EvoMolSteer。活动配置为 `configs/experiments/local_motif20_v2_20261015/config.json`。学习范围、模型路径、原始/压缩数据路径、生成输出路径、样本数、随机种子、API模型和Skills路径均来自配置。密钥只从环境读取，不写入研究记录。

## 确定性工具

|模块|输入|输出与边界|
|---|---|---|
|`local_motif.conditional_layout`|固定N数的合法布局评分表|单交换条件效应、四角均合法的双交换差分之差；有限布局不是独立重复|
|`local_motif.regional_context`|一个终态候选、支持槽位、原子/口袋/成员表|逐站点N/O几何场及同图/同根对照；不称为作用能|
|`local_motif.geometric_regions`|候选及可比终态对照|扣除整体移动的局部空间场描述性对比；可发现碳骨架区域，非显著性排序|
|`local_motif.regional_trajectories`|压缩轨迹、谱系、局部区域|全学习窗口连续曲线、一次/二次拟合、导数与残差；Parquet曲线与JSON系数分开保存|
|`local_motif.state_controls`|坐标、原子、键、电荷|完整已保存结构状态去克隆，排除焦点及所有副本；不按原子/克隆扩大统计重复数|
|`local_motif.path_targets`|同一候选的真实祖先坐标|锚定教师时刻的二次坐标函数、拟合域与误差；小系数表，不另训模型|
|`local_motif.control_response`|实际guidance审计|梯度、原生流、实际注入、裁剪及匹配切换的精简统计；不保存全Jacobian|
|`local_motif.suffix_probe`|完整可恢复状态及槽位、随机流|恢复prior/current/self-conditioning/times，执行无SMC原生后缀；新子随机流不冒称原历史精确回放|
|`local_motif.hypothesis_ledger`|完整旧研究及本轮记录|程序/区域/实际协议/成败检索，非最近若干条截断|
|`local_motif.actions`|父程序、证据区域、已测数值签名|尚未测过且真正改变执行函数的动作与数值选项|
|`local_motif.semantic_contract`|两角色输出、证据包、编译程序|引用、支持、窗口、参数路径、有效算子及数值一致性校验；不能自动证明科学因果|
|`local_motif.evaluation`|逐尝试终态评分/失败/结构|冻结尾部指标、配对bootstrap、两阶段MMFF及局部形成诊断|
|`local_motif.report`|冻结研究状态、原始程序与结果|重新生成中文报告、逐轮/条件表、元素分解、图及来源哈希|
|`local_motif.quality`|生成结果目录、蛋白PDB|可缓存的标准PoseBusters dock诊断，无化学图身份限制|

## 命令入口

```text
python scripts/run_local_motif_research.py --config configs/experiments/local_motif20_v2_20261015/config.json --prepare-only
python scripts/run_local_motif_research.py --config configs/experiments/local_motif20_v2_20261015/config.json --functional-only
python scripts/run_local_motif_research.py --config configs/experiments/local_motif20_v2_20261015/config.json
python -m evomolsteer.local_motif.report --config configs/experiments/local_motif20_v2_20261015/config.json
python -m evomolsteer.local_motif.quality --folder <生成条件目录> --protein <蛋白PDB> --output <诊断JSON>
```

运行前设置 `PYTHONPATH=src`；真实API和SSH凭据由操作者环境提供。此处环境说明仅在执行文档中，未加入科学Skills。已完成campaign的主入口会直接返回，不重新生成。

## Agent与数值程序的对应

Analyst可以多次查询预编写工具；首次须查询有未测动作的区域，此后可查询其它区域的反证。最终选定动作ID、父程序、局部支持和observable，字段必须与同一个注册动作一致。Designer收到同一完整证据包、父响应、实际算子源码、确切参数变动和可选程序，只能在其中选择或暂缓。

真实API的每次请求/回答与被接受的修订分别保存，`delivery.json`指明采用哪个文件。事实审阅是额外API调用，不是独立分子实验。科学假设、测量、推测与尚未测量的内容分开。`scientific_decisions.json`保留可复现决策依据及否证条件，不记录内部思维链。

选择范围仍由当前注册的区域、算子和可解释数值选项限定；本轮不是开放式任意Python程序搜索，也没有证明LLM比确定性搜索更优。数值动作去重用于避免重复生成，不把修改说明文字当成新奖励。

## 顺序、回滚与数据

每个数值新条件先在本地冻结程序、commit并通过本机VPN push；远端fetch后从指定commit建立运行快照，只执行FLOWR原生推理与终态评分。先完成一个条件并将数据下载验证，再开展下一个条件；没有并行优化多个奖励。

连续两次低于活动参考的预定尾部/均值容忍时，停止沿该失败分支继续微调，恢复活动最佳科学程序。接口正确性和存储修复保留。所有失败程序、历史API失败及原始提交仍可追溯。新程序没有通过冻结确认则不晋升为已验证基线。

S0参考重放与无引导对照成本单列。前两轮后缀实验不能用绝对分数与从头生成比较。重复提案使用结果复用，并从有效实验计数中单独列出。冻结确认使用更大样本和新的引导终态池，不以留出结果继续调参。

结构、边界状态和终态类别数据压缩保留在本地 `data/generated/<campaign>/`；远端逐文件SHA256一致后删除SDF、几何轨迹及类别NPZ，仅留下分数与报告。原始Steer轨迹与checkpoint不删除。生成数据不加入Git；共享证据文本按内容哈希存一份，统计表采用精简Parquet/JSON。
