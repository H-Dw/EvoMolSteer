# 六窗口同步化学条件：双教师标定与五轮线上运行

2026-11-01（Asia/Shanghai），按用户新授权执行。只在codex/uniguide-flowr-agent提交推送；已实现科学代码为d945cf01，最终执行提交包含本部署/巡检入口并以launch_audit.json绑定。旧运行目录、所有失败/claim/账本和归档分支保留，原uniguide-flowr保持PAUSED及35分钟，其他自动化不改，不向其他聊天发消息。

## 两项顺序任务

| 顺序 | 教师/node | 配置 |
| --- | --- | --- |
| 1 | 6pyy_H7S_A_402 / single_b004_final_015 | six_window_sync |
| 2 | 3w54_RNB_A_504 / single_b000_final_003 | six_window_sync |

共用[当前六窗口覆盖配置](../../../configs/uniguide_six_windows_zero_start.json)，各任务新根、prepare→本提交完整acceptance→独立calibrate→真实DeepSeek GD01五轮→独立CPU10/50诊断。只读使用持久inputs中的原九个PLIP条件重原子、冻结蛋白/初态及原FLOWR v2权重；逐文件核对INPUT_SNAPSHOT、模型与原生源码SHA。旧context/attempts.json仅用于验收只读哈希，不计新预算；不导入旧lambda、测量、排名或验收。同seed/slot重测是配对复现，不算独立验证。

六个闭区间为0–10、11–20、21–30、31–40、41–50、51–99。额外坐标梯度及化学状态inpainting共用1–98步；第0步仅原生转移，第99步保留未覆盖预测，两端均不算VJP/不加额外梯度/不覆盖化学。某窗实际scale=0只关闭额外坐标梯度，化学阶段保留。每步/终态Hungarian，条件loss为九原子绝对坐标平方误差和除以2K、完整终态VJP，只有蛋白参与坐标拟合。raw999/100区间不变，无坐标输出覆盖、affinity梯度、相对距离项、3Å限幅、向量缩放或RMSD梯度掩码。上侧异常只按冻结规则在组级统计/排名处理，原始/过滤指标及分母分报。

每教师标定最多500、正式最多750，两项共1000/1500、最多2500新增尝试。标定实际0起点，0–200格距10、最多14个粗筛档/9个完整确认档；10→20→50沿用相同槽位按增量计费，失败不退款、不补样、不扩域。每轮最多3个完整50人口动作，精确重复只引用。全部合法格点未测不等于稳定。默认最优为初始标定，只有严格更好的完整测量才能更新；轮最佳与全局最佳分开，连续下降/两轮未改善的回滚复盘真实父链和失败旁支，实际prompt须绑定标定根、当前最优与路径SHA。

skills2.10.1、strict_tool_v1、DeepSeek Beta原生submit_final_response、deepseek-flash thinking/high、单响应16384、原网络重试3次/唯一格式修正；每教师每轮两角色共享30 HTTP/491520输出token。真实REQUEST/RESPONSE、原生TOOL回执、claims证明实际调用。五轮及analysis_final不等于第六轮，不开GD02、不恢复UG05。UG04串行、50逻辑人口、10行网络/VJP默认stream、CPU算子8线程；无GPU池、线程池、额外stream、OOM降档接续。

## 部署、保存和失败边界

先推送本地代码，再在独立远端目录fetch固定已推送提交；GitHub依次直连、ghfast.top、gh-proxy.com、ghproxy.net、gh.llkk.cc，全失败才使用已验证且仅含已推送提交的bundle。远端CPU回归、冻结输入/模型/原生源码及旧结果哈希核对通过后才启动。实际目录、SHA、PLAN、进程/命令写launch_audit.json及远端READY/LAUNCH。所有launcher.log及轮转副本、core只留远端，不打印或下载；API与小写代理只从交互bash加载，凭据不进入仓库、自动化或日志输出。

科学结果在/workspace/agenticgen_six_window_sync_20261101。代码在/workspace/uniguide_runs/six-window-sync-固定提交，审计和安全镜像在/root/private_data/uniguide_six_window_sync_20261101。持久盘初查仅余约1.95GB，镜像不包含全量HDF5；每小时将已COMPLETE、不会再写的终态/轨迹/结构和安全阶段日志分批下载至本地本专项目录，保留远端原始文件。不将多GB产物写Git。镜像不足只保存真实错误，不删旧文件腾空间。

后台沿用已审计的顺序runner/质量队列；全局锁忙可排队，单任务失败保存日志、配置、源码SHA、SEARCH_STATE、预算、进度、请求/claims后跳过并继续另一教师。不得自动重发失败会话、弱化schema、提额、清零或重采；非有限夹具是预期验收注入，不据其单个失败文件判正式失败。明确接口/遗漏导入/维度/代码逻辑异常可按本对话既有授权在本地修复测试提交，新固定目录/新尝试排队，保留累计账本，不覆盖健康运行代码；数值发散、设备/资源故障、LLM截断或错误证据如实跳过。已保存采样时优先只补后处理。

CPU质量仅每实际源组50中固定抽查10（五seed各slot0/1），构建失败占分母不替换，未选40不补慢查；复用只排实际源组。核对模块目录/commit/哈希、worker锁和质量回执。构建失败、真实不通过、不可评价、工程错误分报，PoseB内部几何、蛋白冲突、能量与PLIP分报；不外推全人口。

## 每小时巡检和收尾

巡检调用固定目录inspect_suite.py --plan PLAN.json，保存Asia/Shanghai快照；核对真实命令/状态、全局/suite/任务executor与campaign锁、阶段安全日志、steps/progress、attempts、验收COMPLETE、标定域、五轮真实provider/工具回执、GLOBAL_BEST、初始根、质量队列及SUITE标记。逐文件核对SOURCE_PRESERVATION、固定提交、运行源码及配置SHA；不能仅凭PID/心跳/目录判健康。完整acceptance须真实包含全100步、零scale和端点/化学相位检查，不能拿CPU测试代替。

完成/终止后实际读取两教师标定、各轮参数/prompt/选择与原始/过滤RMSD、loss、初始对应、pIC50/SA及已有抽样构象质量，明确不完整任务、总尝试和同初态配对限制，提交推送主GPT复盘。仅在队列空闲和进程/路径归属确认后停止本专项CPU worker；FINALIZED绑定真实复盘及源哈希，最后删除本专项心跳。远端不可达时只记录真实证据并保留巡检，不按旧PID重启或伪造完成。中文UTF-8，自动化创建/更新后JSON ensure_ascii=true读回并核对无Unicode替换字符。
