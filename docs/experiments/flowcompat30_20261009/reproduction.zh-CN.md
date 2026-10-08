# 挖掘、Agent 与真实 FLOWR 推理的复用接口

本文件解释可重复运行的公开步骤。当前活动轮次见 `progress.zh-CN.md`；不要重复运行已经完成的轮次，也不要把本文件当作新实验的性能结论。原始 Steer、checkpoint 和已有冻结输入均需保留。

## 确定性工具

所有入口均有独立输入和输出。历史轨迹通过 `trajectory_source` 读取，支持已转换的紧凑轨迹；不需要展开逐节点宽特征缓存。统计保存批次/时间曲线、全窗口效应、置信区间、q 值、函数与导数、诊断和来源哈希。

|入口|输入|输出|
|---|---|---|
|`scripts/mine_selection_innovation.py`|`--dataset --campaign --reference`|`--output`：复制、再预测创新、空间形变与选择诊断|
|`scripts/mine_branch_mutation.py`|同上|`--output`：共同祖先差异统计；完整教师参考由独立的可选 `--augmented-reference` 保存|
|`scripts/mine_multi_depth_mutation.py`|历史轨迹与原始教师参考；`--depths`、区域宽度和调整项为显式参数|独立统计目录与实际执行 receipt|
|`scripts/synthesize_common_depth_reference.py`|多深度统计、原教师、上游实际 receipt|`--output` 精简规则；`--output-reference` 单份联合教师参考|
|`scripts/summarize_flowcompat_mechanisms.py`|`--input-dataset` 保留的逐轮结果|`--output-dataset` 实际剂量及方向/强度对照|
|`scripts/summarize_flowcompat_module_sensitivity.py`|同上|同模块、同批次/时刻的奖励反传敏感度汇总|
|`scripts/summarize_flowcompat_pair_effects.py`|同上|逐样本配对效应的精简分布统计，避免仅看均值|

实际挖掘命令及输入、源码、输出 SHA 已记录在各 `execution_receipt.json` 的 `command/input_files/code_files/output_files` 中。重建时使用新的输出路径；已有绑定输出被改写后不能继续冒充同一次工具执行。学习时间支持来自参考声明，缺失祖先时刻不会外推成观测。当前区域 XYZ 的全窗口拟合选择常数；没有支持人为分阶段切换的证据。

## Analyst → Designer

入口为 `scripts/flowcompat_agent.py`、`scripts/flowcompat_supplemental_agent.py`、`scripts/flowcompat_regional_agent.py`。三者都是分版本的受约束配置，避免覆盖已经冻结的设计。

1. `export` 导出角色 request，绑定字面 Skills、工具实际 receipt、统计、注册公式和来源文件。
2. `payload` 校验文件并导出实际 messages。模拟后端由 sub-agent 读取这些消息和指定文件，生成结构化响应；不能只向它口头描述一个 Skill 名称。
3. `import` 校验 Analyst 响应。Designer 的新 request 同时绑定完整 Analyst 响应及其 request。
4. `import` 校验 Designer 的公式、参数、证据引用及反证；`compile` 只编译注册公式中的允许参数，输出生成程序。
5. 使用空规则数值对照、条件导数检查、机制诊断与实际轨迹响应验证实施；随后才读取终态亲和力和应变。

真实 API 后端仍通过 `--action api --request <request.json>` 使用同一个消息与响应校验接口。显式环境项是 `EVOMOLSTEER_BASE_URL`、`EVOMOLSTEER_MODEL`、`EVOMOLSTEER_API_KEY`；本次采用实际 sub-agent 模拟，未调用外部付费 API。

Skills 是通用分析约束；分子名称、区域编号、实际时间范围和当前阈值来自输入配置与统计，未写死在 Skills 中。生产环境、网络与 Git 要求属于部署脚本，未写入分析指引。模块输出梯度表示当前奖励的计算敏感度，并不证明 affinity 或 attention 的因果偏好。

## 推理和逐轮部署

单次真实 FLOWR 接口为 `scripts/generate_flowcompat_flowr.py` 或 `scripts/generate_flowcompat_v2_flowr.py`。主要参数是：

```text
--flowr-root <FLOWR程序路径>
--input-dataset <原Steer实验数据集>
--root <新生成数据集>
--checkpoint <模型checkpoint>
--program <已编译奖励JSON> --reference <已绑定教师JSON.gz>
--campaign <新的实验名> --steps 100 --seed 42
--n 100 --batch 50 --batch-indices <两个批次> --arms gradient
--export-terminal
```

窗口从程序与参考一致的 `window` 读取，窗口内复用正常 target 前向并增加一次端点奖励反传，之后原生续推至终点。原生 untarget 诊断前向保留；没有额外生产 target 前向、粒子重采样或 affinity-head 求导。每个梯度批次另有一次有限差分检查，共四次额外 target 前向，单列于 preflight；终态双口袋重评分是评价开销。离散化学输出仍由 FLOWR 预测，不设化学图相等门。

整轮编排由 `scripts/run_flowcompat_campaign.py` 和 `scripts/dispatch_flowcompat_round.py` 完成：本地冻结假设与 commit/push，远端精确 pull，校验前轮已发布报告后清理本项目旧生成文件，执行真实推理，捕获上游四份源码的前后哈希，绑定到生成配置，压缩运输。本地先核验实施，再进行终态评价与报告保留；失败候选恢复 R26。中断后从账本中的下一轮继续，已完成轮次不重新计数。

最终第 25–30 轮只使用冻结候选与六个新批次，期间不调参。`scripts/validate_flowcompat_confirmation.py --repo <仓库> --output <独立审核JSON>` 检查冻结、真实配对、代码与模型证据完整性；`report_flowcompat_campaign.py` 再应用预先设定的效果条件。两者通过也只是同一目标/预测模型内证据，不是生物学亲和力验证。

生成中间数据在校验和发布后按显式白名单删除，留下紧凑报告、代码、参数和来源哈希；这些报告不能独立还原已经退休的测试坐标。历史原始 Steer 始终保留，支持重新挖掘与重建统计。
