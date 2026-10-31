接续本对话新授权的三策略双教师五轮重测，只处理本专项。先读 D:/llm_controlled_generation/EvoMolSteer-uniguide/agenticgen/docs/experiments/20261031_guidance_comparison5/plan.zh-CN.md、launch_audit.json 和后续用户指令。仅操作 codex/uniguide-flowr-agent；原 uniguide-flowr 保持 PAUSED、35 分钟不变，归档分支不动，不修改其他自动化，不向其他聊天发消息。原八轮任务已取消，本次是新授权新根，不能重启原取消任务。

远端 root@ssh.zzai.scnet.cn:11741，使用本对话已授权凭据及 D:/llm_controlled_generation/tmp/uniguide_ops.py 的 JSON stdin。凭据不打印、不保存仓库或自动化。API 与小写代理来自交互 bash，非交互缺变量不代表服务器未配置；launcher.log 可能含环境，只留远端，不打印、下载原文或提交，core 不下载。

固定 GPU/CPU 代码提交及实际目录、配置 SHA、机器 PLAN、启动进程以 launch_audit.json 为准，不以本地后续纯文档 HEAD 覆盖运行中代码。初始推送提交 072e6f7831bd91b398f7e1cbd65f4725268ec9f4；执行目录 /root/private_data/uniguide_comparison5_20261031/runs/guidance-comparison5-072e6f78，审计 /root/private_data/uniguide_comparison5_20261031/audits/comparison5-072e6f78，结果根 /workspace/agenticgen_guidance_comparison5_20261031，机器计划为该根 PLAN.json。后续工程修复以新审计绑定为准，禁止原位覆盖。

顺序 baseline 两教师→prefix_joint 两教师→five_windows 两教师，每任务新根，prepare→本提交完整 acceptance→独立标定→真实 DeepSeek GD01 五轮→独立 CPU10/50。两个原 teacher/node_id 和各九个 PLIP 条件原子不变；冻结初态、教师、条件和原生 FLOWR 源码的持久快照实际哈希见 INPUT_SNAPSHOT.json。模型为原 FLOWR v2，哈希必须符合配置。本次不导入不可见八轮数据、旧 λ、预算、排名或验收，相同 seed/slot 是配对复现，不计独立验证。inputs/context/attempts.json仅是旧源账本的只读参考，供acceptance核对哈希，不计新预算；新科学attempts仅看各job结果根。每任务标定350（含范围证据最多100）、正式750；六任务上限2100/4500，精确重复引用，失败计费不补样。

说明文字 reason/improvement_reason/uncertainty 每字段3000字符，skills2.8.1；证据 ID/动作/数组校验不放宽。deepseek-flash、thinking/high、单响应16384 token、原网络重试/一次格式修正不变，每教师每轮两角色共享30调用/491520输出token。实际 provider 请求/回执、claims、工具回执才证明调用，不能因配置存在宣称。五轮从外部配置和实际 FINALIZED/ROUND_COMPLETE 核查，不能运行第六轮、GD02或恢复UG05。

baseline 动态逐步及终态 Hungarian，四窗口10–25/26–50/51–75/76–99，坐标梯度与10–99化学状态inpainting；prefix_joint增加索引0–9原生fragment坐标/化学覆盖、条件原子时间1，索引10撤去坐标覆盖和固定时间/位置标记；five_windows基于baseline无联合前缀，10–20/21–30/31–40/41–50/51–99。三者raw999、100区间0–0.999、10–99完整绝对坐标VJP、最后循环未覆盖预测，第99步更新不进入输出。仅蛋白决定变换，无affinity/相对距离项、3Å限幅、向量缩放或RMSD梯度掩码。高值只组级按Q3+max(3IQR,1Å)单侧排除、至少10观测且最多10%，低值保留；原始/排名指标与分母分别保存。动态几何覆盖和初始对应误差分别报告。

UG04串行：逻辑组50，网络/VJP固定10行默认stream依次执行，CPU算子8线程；无GPU子进程池、线程池、额外stream、OOM降档接续，不重新加入这些功能。三种探索幅度是跳跃激进程度，不组织机制/消融/反证分支。相邻两轮新完整50人口的最佳performance连续下降后，下一轮基于最佳兼容checkpoint，实际prompt提供差路径和效果；不能用单调累计最佳判断下降，不能重采最佳。核对实际prompt/动作/回执。

每次用固定执行目录的 agenticgen/docs/experiments/20261031_guidance_comparison5/inspect_suite.py --plan PLAN.json 保存 Asia/Shanghai 快照，核对真实命令、进程状态、全局 /workspace/agenticgen_uniguide_flowr_20261026/.executor.lock、suite锁、各任务executor锁、STATUS/HEARTBEAT、阶段日志、steps/progress、attempts、acceptance/COMPLETE、标定域、实际LLM回执及完成/失败/SUITE标记。逐文件核对 SOURCE_PRESERVATION、固定commit/文件哈希、全部配置SHA；历史不可见路径不能伪造核验。不能只凭PID、心跳或目录判断健康。nonfinite_isolation是预期夹具，不据该文件判正式失败。

单任务异常保存日志、源码哈希、账本、进度、失败请求/claim和配置后跳过，其他任务继续。接口不对齐、遗漏导入、数组维度、代码逻辑缺失导致的工程终止，用户允许本地修复、测试、提交推送、新固定目录验证后用新任务尝试排队；保留原失败与累计账本，不覆盖健康代码、不清零、不盲目重发同一失败请求。采样已保存优先仅补后处理。数值发散、真实设备/资源故障、LLM截断或不存在的证据引用不能冒充接口错误通过弱化schema、提额、补样或删除claim绕过。

检查各任务独立CPU队列的worker.json/status.json/requests/POLICY、真实命令、模块目录/commit/模块哈希、worker锁和QUALITY_COMPLETE/QUALITY_FAILED。每实际源组50，抽查10、未选40、coverage0.2，五种子各slot0/1，构建失败占分母不替换，其余40不补查。复用组只排实际原源组。必要CPU恢复仅限本专项，先查归属/锁，再加载DTK及当前目录agenticgen/src、src，隐藏GPU、CPU数值库各1线程。构建失败、真实不通过、无法评价、工程错误分开，PoseB内部几何/蛋白冲突/能量分别报告，不外推全人口。

/root/private_data持久目录空间有限，DURABLE_SNAPSHOT.json和result_snapshots/MIRROR_STATE.json记录安全JSON/结构/压缩张量/阶段日志镜像；不含全量HDF5、launcher.log或core，不能称完整轨迹备份。检查镜像错误、可用空间和源/副本哈希。每小时下载已完成、不会再写的组/验收产物与安全审计到本地本实验目录，分批归档全部HDF5/终态/轨迹/结构；不下载正在写的HDF5、不删除远端文件。不将大体积原始产物硬塞Git，提交机器读数/清单/实际复盘。远端再次不可达/目录不可见时只记录实际证据、通知一次并保留监控，不伪造快照、按旧PID启动或冒称代码失败。

ALL_JOBS_ATTEMPTED/SUITE_FINISHED后先排查尚有授权可修复工程任务，再实际读取六任务标定、五轮提示/参数/选择、原始/排名RMSD、初始对应/化学保持、结构与10/50质量，按教师配对比较三策略，明确失败/不完整任务及累计消耗，完成真实主GPT复盘并提交推送。纯文档不部署运行目录。实际复盘与源哈希写FINALIZED；仅在本专项路径、进程归属、队列空闲确认后停止本专项CPU worker。通知本对话并仅删除当前触发的本专项heartbeat，不动其他自动化。

GitHub异常按直连、ghfast.top、gh-proxy.com、ghproxy.net、gh.llkk.cc依次，均失败才使用已验证且仅含已推送提交的Git bundle。一小时巡检，健康/排队/无实质变化返回DONT_NOTIFY，不额外进度；仅通知完成、实质异常、工程修复或需用户处理的问题。中文UTF-8，自动化创建/更新后以JSON ensure_ascii=true读回确认无Unicode替换字符。所有操作记录Asia/Shanghai时间。

上线前298f9fd5的远端CPU回归为285通过、1个持久镜像测试失败，未启动科学任务或API。072e6f78修复NFS等长覆盖的size/mtime假阴性，加入内容哈希复核和固定mtime复现测试；保留旧部署审计，不把这次部署预检失败计作科学候选，也不删除旧目录。最终远端回归与启动结果以launch_audit.json为准。
