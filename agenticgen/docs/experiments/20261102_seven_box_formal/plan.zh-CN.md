# 七窗口双教师五轮线上实验

本专项获本对话明确授权：在 root@ssh.zzai.scnet.cn:11741 拉取已推送代码，两个原教师分别重新验收、重新标定及Selection，再运行GD01五轮Guidance；后台运行，每2小时巡检。仅操作 `codex/uniguide-flowr-agent`，不修改历史固定目录、其他自动化或其他聊天；原 `uniguide-flowr` 保持PAUSED/35分钟及TOML哈希不变。

## 固定执行与输入

部署脚本 `deploy.py` 提交推送后，用该提交在 `/workspace/uniguide_runs/seven-box-search-<commit前8位>` 独立检出；结果根 `/workspace/agenticgen_seven_box_search_20261102`，审计 `/root/private_data/uniguide_seven_box_search_20261102/audits/seven-box-search-<commit前8位>`。实际commit、路径、配置SHA和PLAN以 `launch_audit.json`、远端READY/LAUNCH/PLAN为准，后续纯文档HEAD不部署。部署中PLAN未生成时只核对部署进程与安全审计，不重复启动。

教师依次为 `6pyy_H7S_A_402/single_b004_final_015`、`3w54_RNB_A_504/single_b000_final_003`，各九个原PLIP条件原子、冻结初态、FLOWR v2模型与源码按原INPUT_SNAPSHOT及实际SHA验证。旧比较实验和Box六窗口结果只读保护；历史参数仅作新预实验背景设计，不导入旧标定、测量、预算、排名或验收。同seed/slot属于配对复现，不是独立验证。

## 科学配置与预算

配置叠加 `uniguide_absolute_coordinate.json` 与 `uniguide_diverse_seven_box_search.json`，skills2.13.0，七窗为0–10/11–20/21–30/31–40/41–50/51–75/76–99。端点0/99仅原生；额外完整绝对坐标VJP及化学inpainting共用1–98，末窗实际76–98。默认首窗constant，首窗0/1/2/5/10/20……400，其他窗0/10……400；单窗网格索引距离43、总283。无坐标覆盖、affinity/相对距离梯度、3Å限幅、向量归一化或RMSD梯度掩码。raw999、100区间、最后循环未覆盖预测不变。

每教师prepare→本固定提交完整GPU acceptance→独立预实验及Selection两阶段→GD01五轮→独立CPU10/50。每轮最多五个假设、所有合法提出方向当轮均执行，每组完整50；正式各1250，预实验各500（33初筛配置及双背景完整确认410，再按剩余预算分配完整负面确认和弱剂量/高段探测）。正常预算最多确认一个负面box；无完整证据保留，不能从旧51–99整体证据排除拆分后的某段。允许box scale0保留化学，禁用box同时关闭梯度和化学。

两教师科学总上限3500=1000预实验+2500正式，失败计费不退款，精确重复只引用不补样，所有修复累计不能清零或擅自超额。完整门控下最优50预实验为默认global best，仅严格更好完整50替换。实际提示检查多样组合探索、真实父链/回滚及逐窗当前背景覆盖，不要求解释窗口机制或提供私有逐步思维。

Selection独立共享30HTTP/491520输出token；Guidance每教师每轮两个角色共享同额。deepseek-flash、thinking/high、strict_tool_v1、单响应16384，原最多3次网络/唯一一次格式修正不变，显式原生submit_final_response控制数组；不能从正文补字段。只运行GD01五轮，不跑第6轮/GD02/UG05，analysis_final不是第6轮优化。配置存在或CPU模拟不能称真实模型/LLM成功。

UG04串行50逻辑人口、10行网络/VJP、默认stream、CPU8线程，不恢复GPU池/线程池/额外stream/OOM降档接续。

## 巡检、失败与收尾

调用本固定目录 `inspect_suite.py --plan PLAN.json`，保存Asia/Shanghai监控快照。核对真实命令/进程状态、全局 `/workspace/agenticgen_uniguide_flowr_20261026/.executor.lock`、suite/任务executor/campaign/CPU锁，STATUS/HEARTBEAT、安全阶段日志、steps/progress、attempts、完整acceptance、SEARCH_STATE/NEGATIVE_EVIDENCE/SELECTION_FROZEN、每轮ROUND_COMPLETE/checkpoint/history_snapshot/实际prompt/provider工具回执及claims、全部配置/源码SHA和SOURCE_PRESERVATION逐文件。nonfinite_isolation是预期夹具；不能凭PID、目录、心跳、SAMPLING_FINISHED或SUITE_FINISHED判完整成功，必须读FINALIZED内部complete及所有hypothesis失败记录。

每实际完整50源组CPU抽10，固定五seed各slot0/1，构建失败占分母不替换，其余40不补慢查；复用仅排实际源组。检查worker/status/requests/POLICY、绑定commit/模块SHA、真实进程与QUALITY_COMPLETE/FAILED。构建失败、真实不通过、不可评价、工程错误分报；PoseB几何、蛋白冲突、能量和PLIP分报，不外推全人口。必要worker恢复仅本专项，先核归属/锁，再加载DTK及固定agenticgen/src和src，隐藏GPU、CPU数值库各1线程。

单教师异常留存日志、源码/配置、账本、进度、失败请求claim后让另一教师继续。明确接口/导入/维度/代码逻辑错误可以本地debug、测试、commit/push，新固定目录新尝试排队；不改健康代码、不清零、不盲目重发、删除claim或跨3500预算。已有采样优先仅补后处理，数值/设备故障、截断、错误证据或正确传输下的LLM契约失败不能冒充工程缺陷绕过。无法在原累计预算内安全接续时保留证据并报告，不自动增额。

连接仅用本对话授权凭据及 `D:/llm_controlled_generation/tmp/uniguide_ops.py` JSON stdin，可通过内存装载凭据的 `seven_box_ops.py` 调用；不输出、提交或写自动化凭据。API和小写代理由交互bash加载，非交互缺变量不代表未配置。所有包含launcher.log的文件及轮转副本、deployment/api_probe启动日志、core只留远端，不打印/下载原文。权重不归档；实验checkpoint/terminal保留。

持久盘有限，核对DURABLE_SNAPSHOT/MIRROR_STATE、容量、源/副本SHA。镜像不含全量HDF5，错误不当采样失败。每次巡检分批收集不可变已COMPLETE的HDF5/终态/结构/安全审计，先查本地容量和64MiB余量，不下载活动HDF5、不删除远端或旧本地文件，不塞大文件进Git。首次新连接/空间异常通知一次，同一阻塞无新证据不重复通知；不可达不伪造快照、按旧PID启动或写FINALIZED。

全部终态后先排查可修复工程任务，再读取两教师五轮、Selection负面证据、原始/排名RMSD、pIC50、标准SA及抽样构象质量，与历史同教师结果做描述性配对比较，说明七窗与六窗配置差异及累计消耗。完成主GPT实际复盘、结果归档、commit/push、源哈希绑定专项FINALIZED；归属明确且队列空闲才停本专项CPU worker。通知本对话并仅删除 `seven-box-guidance-2hourly`。健康/排队/无实质变化返回DONT_NOTIFY，不保持前台轮询。GitHub直连、ghfast.top、gh-proxy.com、ghproxy.net、gh.llkk.cc依次，均失败才使用验证且只含已推送提交的bundle。
