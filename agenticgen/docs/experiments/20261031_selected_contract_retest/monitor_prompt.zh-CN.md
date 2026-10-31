接续本对话新授权的selected输出契约修复后两项五轮重测，仅处理本专项。先读D:/llm_controlled_generation/EvoMolSteer-uniguide/agenticgen/docs/experiments/20261031_selected_contract_retest/plan.zh-CN.md、launch_audit.json及后续用户指令。只在codex/uniguide-flowr-agent工作；原uniguide-flowr保持PAUSED、35分钟不变，不修改其他自动化、归档分支或向其他聊天发消息。

服务器root@ssh.zzai.scnet.cn:11741，沿用本对话授权凭据和D:/llm_controlled_generation/tmp/uniguide_ops.py的JSON stdin接口，密钥不打印、不写仓库/自动化。API及小写代理由交互bash加载。所有launcher.log，包括api_probe.launcher.log，只留远端，禁止打印或下载原文；core不下载。

固定GPU/CPU提交78e6e1803758f7016a2568b00176eaaa27ef6dc4，目录/root/private_data/uniguide_selected_retest_20261031/runs/selected-78e6e180，审计/root/private_data/uniguide_selected_retest_20261031/audits/selected-78e6e180，结果根/workspace/agenticgen_guidance_selected_retest_20261031，PLAN为该根PLAN.json，SHA256 cb928fd55d658253d79a4ea45ffb8e8272faec7e3735d1ec37889d9ebb0adb09。配置在固定目录-configs，以PLAN/launch_audit的SHA核对；不能用本地后续纯文档HEAD覆盖执行目录。

仅依次运行03_prefix_joint_6pyy、06_five_windows_3w54，均attempt_002：prepare→本提交完整acceptance→独立标定→真实DeepSeek GD01五轮→CPU10/50。前者教师6pyy_H7S_A_402/single_b004_final_015，四窗口10–25/26–50/51–75/76–99，0–9原生坐标/化学inpainting及条件时间1、10起撤去坐标覆盖；后者3w54_RNB_A_504/single_b000_final_003，无联合前缀，五窗口10–20/21–30/31–40/41–50/51–99。各原九个PLIP条件重原子和冻结输入/初态/模型保留，只读引用旧持久inputs；不导入旧λ、标定、回执或预算。source context的attempts.json只供验收只读哈希参照。

旧六任务根/workspace/agenticgen_guidance_comparison5_20261031、固定072e6f78目录和旧FINALIZED只读保留，旧FINALIZED SHA256 9c6fd372b65f1ee05bd9828e41cdb8e48c7b9ff6d8000ff1c2da58c52e83a781。旧3790次=1640标定+2150正式，其中原失败两项360/240次不退款。已完成四项不重跑，原八轮已取消任务不重启，旧根终态不能当本次完成。新两任务各标定350（含100范围）、正式750，新上限700/1500，总新增最多2200；旧+新最多5990。精确重复引用、失败计费、不补样/扩域；新旧同seed/slot是配对复现，不算独立验证。

修复为skills2.8.2+strict_tool_v1：DeepSeek Beta /beta/chat/completions，所有函数strict=true，最终通过单独submit_final_response原生工具参数显式提交。synthesis必填decision/selected/reason，proposal必填decision/proposals/reason；hold/终局必须显式空数组，其他选择是真实候选ID、非空、去重、在上限内。普通正文不作为终结控制，程序不从reason猜选项、不填默认selected。服务端不支持的长度/数组约束列为client_only_checks，但原完整本地schema、3000字符、证据、数组/动作/预算校验全部保持。thinking/high不能强制required/命名工具，使用auto；缺字段/错ID/普通正文只能用原唯一一次完整格式修正，失败保留claim不自动重发、提额或弱化校验。模型deepseek-flash、thinking/high、单响应16384、原网络最多3次，每教师每轮两角色共享30 HTTP/491520输出token。不得跑第六轮、GD02或UG05，analysis_final不是第六轮优化。

上线前本地和远端均310项CPU回归通过（无跳过），新skills4项校验通过。真实工程接口预检两次请求/两次回执、248 completion token，分别显式提交propose选择和hold空数组，无分子、无科学轮次；不能将该探测当正式Agent或GPU验收。新根自己的acceptance/COMPLETE与实际provider REQUEST/RESPONSE、native TOOL接受回执才证明相应阶段通过。

科学契约保持raw999、100区间0–0.999、10–99完整绝对坐标VJP、最后循环未覆盖预测，第99步更新不进入输出；动态逐步/终态Hungarian，同步化学状态注入，只有蛋白决定测量变换。无affinity/相对距离梯度、3Å限幅、向量缩放或RMSD梯度掩码。高值仅完整组按Q3+max(3IQR,1Å)单侧统计排除、至少10观测且最多10%，低值保留；原始/排名RMSD、loss、阈值/排除身份/分母分别记录。NaN/Inf/缺测/工程错误不可过滤成成功。初始对应误差和动态几何覆盖分报，几何匹配不保证化学身份。UG04串行50逻辑人口、固定10行网络/VJP默认stream依次执行，CPU算子8线程；不恢复GPU工作进程池、线程池、额外stream或OOM降档接续。

每次调用固定目录agenticgen/docs/experiments/20261031_selected_contract_retest/inspect_suite.py --plan PLAN.json，保存本根monitoring的Asia/Shanghai快照。核查真实命令/进程状态，全局/workspace/agenticgen_uniguide_flowr_20261026/.executor.lock、suite锁及各任务executor/campaign锁，STATUS/HEARTBEAT、阶段日志、最新steps/progress、attempts、acceptance/COMPLETE、标定域/轮次/失败标记，逐文件核对SOURCE_PRESERVATION（初始23175项，以PLAN为准）、固定commit/源码/配置SHA和旧FINALIZED未变。不能只凭PID、心跳、目录判断健康；nonfinite_isolation是预期故障夹具，不据其文件判全链失败。健康不覆盖代码/skills/manifest/条件/缓存，不重复启动，等待全局锁属于正常排队。

核对每次真实LLM请求全工具strict=true、终结工具参数含所需控制数组，receipt.final_response_mode=strict_tool_v1及原生tool_call_id、TOOL接受/拒绝与HTTP/budget claim一一对应；配置不等于调用。检查每轮checkpoint/history_snapshot/prompt/动作和原始测量。相邻两轮新完整50人口轮最佳连续下降才令下一轮从最佳兼容checkpoint出发并展示坏路径；不使用单调累计最佳，不重采最佳，空测量不伪造下降。探索目标仍是有机会更优的完整路径，不增加机制/消融/反证任务或方向配额。

每任务独立CPU队列检查worker/status/requests、实际命令、worker锁、模块目录/commit/模块SHA及各源组QUALITY_COMPLETE/QUALITY_FAILED。新worker绑定78e6e180，不能启动旧worker。每源组attempts50、checked10、not_selected40、coverage0.2，五种子各slot0/1，构建失败占分母不替换，其余40不补慢查。复用只排实际源组；PoseB内部几何、蛋白距离/冲突、能量及PLIP分开报告，不外推全人口。必要CPU恢复限本专项，先核归属/锁，再加载DTK及本固定目录agenticgen/src和src，隐藏GPU、CPU数值库各1线程。

单任务异常先保存安全日志、源码哈希、配置、账本、SEARCH_STATE/进度、失败请求/claim后跳过，让另一项继续。明确接口/遗漏导入/维度/代码逻辑错误按授权本地修复测试提交推送、新固定目录新尝试；不覆盖健康代码，不盲目重发相同失败会话或清零旧账本，新旧累计完整报告且不得擅自超额。数值发散、真实设备/资源故障、LLM截断、错误证据或在正确传输下再次违反输出契约不能冒充工程错误通过放宽、补样、提额绕过。采样已保存优先仅补后处理。真实SIGKILL/ABRT依据本次日志定位，不套旧结论。

/root/private_data空间有限；新result_snapshots只镜像安全JSON/结构/压缩终态/阶段日志，不含全量HDF5。检查DURABLE_SNAPSHOT/MIRROR_STATE错误、磁盘余量和源/副本SHA；每小时分批下载已COMPLETE、不会再写的HDF5/终态/结构及安全审计到本地本专项目录，保留远端原文，不下载活动HDF5。工程预检launcher日志禁下载。大原始产物不塞Git，提交机器读数/清单/复盘。再次SSH不可达或根不可见时只记录真实证据、通知一次并保留巡检，不伪造快照、按旧PID重启或写FINALIZED。

ALL_JOBS_ATTEMPTED/SUITE_FINISHED后实际读取新两项五轮提示/选择/参数、原始与过滤指标、构建/化学保持和10/50质量，完成真实主GPT复盘与旧同教师结果配对。四项保留2.8.1，本次两项2.8.2，控制指引版本不同须注明；不能冒称六项都在新版本运行，也不把重复固定初态当独立验证。明确部分失败/未完成与旧3790+新增实际次数。下载结果/安全日志、提交推送复盘，FINALIZED绑定实际复盘及源码；仅在路径归属和队列空闲确认后关闭本专项CPU worker、锁释放，通知本对话并只删除当前触发的本专项心跳。纯文档不部署执行目录。

GitHub异常按直连、ghfast.top、gh-proxy.com、ghproxy.net、gh.llkk.cc顺序尝试，全部失败才用已验证且仅含已推送提交的bundle。巡检1小时（60分钟），健康/排队/无实质变化返回DONT_NOTIFY，不额外进度；仅通知完成、实质异常、工程修复或需用户处理。中文UTF-8，自动化创建/更新后JSON ensure_ascii=true读回，无Unicode替换字符。所有记录Asia/Shanghai时间。
