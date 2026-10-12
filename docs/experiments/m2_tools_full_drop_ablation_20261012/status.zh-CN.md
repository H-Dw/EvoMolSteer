# 科学子模块消融运行状态

模型：真实 DeepSeek API / deepseek-flash。

状态：continuing_after_module_error；已完成比较 11/17；跳过 7；已处理 17/17。

|模块|功能验证|确认结论|是否精简|
|---|---|---|---|
|A1.1|True|INCONCLUSIVE|False|
|A2.1|True|INCONCLUSIVE|False|
|A2.2|True|INCONCLUSIVE|True|
|A3.1|False|FUNCTIONAL_OR_INTERFACE_BOTTLENECK|True|
|A4.3|True|INCONCLUSIVE|False|
|D1.1|False|FUNCTIONAL_OR_INTERFACE_BOTTLENECK|True|
|D1.2|True|INCONCLUSIVE|False|
|D2.1|True|AGENT_OUTPUT_UNCHANGED_EXECUTION_FUNCTION|True|
|D2.2|True|AGENT_OUTPUT_UNCHANGED_EXECUTION_FUNCTION|True|
|D3.1|True|INCONCLUSIVE|True|
|D3.2|True|AGENT_OUTPUT_UNCHANGED_EXECUTION_FUNCTION|True|
|A1.2|执行失败|SKIPPED_EXECUTION_ERROR（因果未确认）|False|
|A3.2|执行失败|SKIPPED_EXECUTION_ERROR（因果未确认）|False|
|A4.1|执行失败|SKIPPED_EXECUTION_ERROR（因果未确认）|False|
|A4.2|执行失败|SKIPPED_EXECUTION_ERROR（因果未确认）|False|
|D4.1|执行失败|SKIPPED_EXECUTION_ERROR（因果未确认）|False|
|D4.2|执行失败|SKIPPED_EXECUTION_ERROR（因果未确认）|False|
|A2.1|执行失败|SKIPPED_EXECUTION_ERROR（因果未确认）|False|

筛查不等于确认。功能未响应、接口瓶颈和证据不足不表述为科学方法无效。历史 Steer 比较不成对且在线预算不同。

剩余实验固定复用 400 个 M2＋400 个无引导样本。复用记录不增加样本量，最终复核也不是独立留出验证。

- D2.1：本次 Agent 输出未改变执行函数。
- D2.2：本次 Agent 输出未改变执行函数。
- D3.2：本次 Agent 输出未改变执行函数。