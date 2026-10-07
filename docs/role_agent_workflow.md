# 标准Analyst与Designer接口

`scripts/role_agent.py`加载当前活动角色核心，绑定共享证据数据集，并验证实际响应。
它支持subagent模拟和同一HTTP接口。指引、数据和执行配置分别保存；Agent的选择不会被编译器替换。

## 输入数据集

`evidence.json`包含以下字段：

|字段|内容|
|---|---|
|`task`|目标优先级、表示、单位、观察窗口、统计单位和评价标准|
|`execution_contract`|可用导数路径、计算预算、时间支持和执行限制|
|`evidence`|带唯一`evidence_id`的统计、曲线、缺失信息和候选评价记录|
|`program_registry`|候选程序相对路径、程序/参考库SHA256、奖励类型和参数|
|`incumbent_program_id`|已有候选；其身份来自数据，不写入Skills|

Designer还读取同目录`formula_registry.json`（公式、符号、源代码hash等能力定义）以及实际Analyst输出。
标准请求提供编译器接受的标量范围；具体评价阈值应在`task`中由实验方案给出。
参考字段与数学特征由注册程序定义，不由Skill预设分子区域。

## 调用顺序

```text
python scripts/role_agent.py export --role Analyst --evidence <evidence.json> --output <agent_directory>
```

将`Analyst.request.json`的`system_instruction`、`response_schema`及绑定证据交给LLM，
保存`Analyst.response.json`，然后运行：

```text
python scripts/role_agent.py validate --role Analyst --evidence <evidence.json> --output <agent_directory>
python scripts/role_agent.py export --role Designer --evidence <evidence.json> --output <agent_directory>
```

Designer请求绑定Analyst响应SHA256。保存`Designer.response.json`后运行：

```text
python scripts/role_agent.py validate --role Designer --evidence <evidence.json> --output <agent_directory>
python scripts/role_agent.py compile --role Designer --evidence <evidence.json> --output <agent_directory>
```

`api`操作使用同一请求和响应验证器；传输层在调用时加载共享证据，不重复保存完整数据包。
subagent路径应使用独立上下文并保存字面响应。请求hash绑定完整输入；
`instruction_sha256`仅绑定角色处理文本，保留尾部换行并排除共同JSON输出格式后缀。

## 输出与验证

Analyst输出关联证据的观察、假设、反证和限制。Designer输出`retain_existing`、
`modify_existing`或`defer`，并说明程序、标量更新、证据与失败模式。
暂缓时不会产生可执行奖励。其他选择编译为`reward_program.json`，附请求、响应和指引来源hash。

验证覆盖schema、证据引用、真实Analyst绑定、注册源文件、有限参数、参考表示/窗口及导数路径。
验证器不规定最高评分候选或历史最佳数值。新奖励需经过配对生成质量评价后进入实际使用。

## 当前消融范围

实验固定同一发现集、公式注册表和标量更新契约。2×2角色交叉每格重复2次，
完整建议版与8项逐项移除各1次。Designer建议消融绑定同一Analyst结果。
相同可执行程序共享生成结果，保留等价证书；共享结果不计作额外的分子重复实验。
原始统计方法与特征数据保留。该实验不评价未注册任意新公式的原创coding价值。
