# V10 Top1 96K输出额度修复

2026-10-23本次定时任务已通过正常工具调用完成受限SFTP读取；没有审批拒绝。
来源是Top1 R4G2的`turn_00.incomplete_1.receipt.json`，请求及回答哈希均匹配。
provider ID为`41bf77ed-f171-4577-a106-d11c43eca6a6`：输入76377，输出65536，
其中reasoning_tokens为65536；finish_reason=length，content为空（0字节）。
因此这次没有截断的可见JSON可修补。并非输入字节超限、质量或GPU故障。

用户最新指令为“可以直接将接口上限调整至96k”。按96×1024执行：
私有分析/bootstrap上限65536→98304，汇总仍65536，thinking/high、输入512KiB、
timeout600和HTTP并发4均保持。DeepSeek官方[Chat Completions文档](https://api-docs.deepseek.com/api/create-chat-completion/)
在本次查询时允许max_tokens至393216；96K在服务支持范围内。增加额度不能保证
模型不会再次耗尽，也不证明控制策略或亲和力改善；不自动降低推理力度或替换模型。

科学Python变更限于：`group_protocol.py`的私有输出限额、`deepseek_api.py`的
参数上界，以及`restart_failed_teacher.py`的96K事务与已采纳最优轮复盘协议继承。
奖励、FLOWR权重、坐标/梯度/RNG/起点、幅度范围、位移上限和两份skills保持原值。
协议变更追加到新目录，原config/PREPARED/ACCEPTANCE不覆盖。

保留Top2完整五轮500条及原d1163b7e/64K分析来源。Top1原四轮400条仅作失败尝试，
先保存全部prompt/响应/usage及删除哈希清单，在固定代码相关测试、两教师原生
兼容性验收后删除其候选结构与张量，从R1空动态记忆重跑五轮。Top1重跑为96K，
两教师分析预算差异必须披露；失败成本按provider ID单列，不并入正式性能。

新脚本仅执行本次V10修复，不是V11批准。继承原V9→V10真实最优轮策略评估，
重新绑定新commit、验收、保留来源并通过4上下文review-check；full_review_completed=false。
不得恢复旧52会话要求。V10完成后仍需其自己的最优轮评估。

部署方式：本地相关测试与GitHub提交推送；本地固定用途部署客户端让远端fetch
完整commit，从Git对象运行安装脚本及本次修复脚本。代码不通过SSH上传，
状态读取与部署/受控删除/启动是两个独立操作，均遵循正常平台审批。
本文件记录修复方案；是否已部署/删除/重启以部署收据为准。

本地直接相关验证：7个相关测试文件共63项通过；原生common_rng一项明确排除，
留远端已有Torch/DTK环境执行。不以离线测试声称LLM策略有效。定点读取客户端另有
32项边界/来源测试通过，实际定时读取收据在
[诊断来源](../deployments/73cafe51/v10_targeted_diagnosis.json)。
