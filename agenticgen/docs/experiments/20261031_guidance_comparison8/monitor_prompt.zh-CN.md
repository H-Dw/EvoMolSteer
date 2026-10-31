接续本对话明确授权的三组双教师八轮实验，只处理本专项。先读 D:/llm_controlled_generation/EvoMolSteer-uniguide/agenticgen/docs/experiments/20261031_guidance_comparison8/plan.zh-CN.md、engineering_repair_protocol.zh-CN.md、launch_audit.json 和后续指令。只在 codex/uniguide-flowr-agent 修改；原 uniguide-flowr 保持 PAUSED、35 分钟不变，归档分支不动，不修改其他自动化，不向其他聊天发消息。

远端 root@ssh.zzai.scnet.cn:11741，使用本对话已授权凭据和 D:/llm_controlled_generation/tmp/uniguide_ops.py 的 JSON stdin 接口。凭据不输出、不写仓库或自动化。API 与小写代理由交互 bash 加载；launcher.log 可能含环境，禁止打印、下载其原文。CPU/GPU 当前固定提交、目录、配置及 PLAN 路径以 launch_audit.json 为准；不要只看本地分支最新文档提交而覆盖运行中代码。

队列按 baseline 两教师→prefix_joint 两教师→five_windows 两教师顺序串行执行。两个 teacher/node_id 与原九个 PLIP 条件原子保持。每任务新根、验收→独立标定→真实 DeepSeek GD01 八轮，最多三动作/轮。科学尝试预算每教师/任务标定350、正式1200，六任务原始上限2100/7200。精确重复引用，失败计费，不补样。修复前后与累计消耗分别统计，不能清零。未调用的配置不能称真实 LLM 回执。

baseline 使用最新动态 Hungarian、四窗口、坐标梯度与化学状态 inpainting；prefix_joint 额外在索引0–9采用原生 FLOWR fragment 坐标/化学覆盖及条件原子时间1，第10步开始禁止坐标覆盖并撤去固定位置/条件时间标记；five_windows 基于 baseline、不启用联合前缀，窗口10–20/21–30/31–40/41–50/51–99。三组均 raw999、100区间、10–99梯度、最后循环未覆盖预测，绝对坐标完整VJP，无 affinity/相对距离项、3Å限幅或 RMSD梯度掩码。上侧异常只组级统计，低值保留。匹配变化后的几何覆盖与初始对应误差分别报告。

skills2.8.0，线性格距10、整数改动、三种跳跃幅度，两个实际角色，无方向配额或机制/消融/反证要求。按每轮新完整50人口配置的最佳performance判断相邻两次下降，不能使用单调累计最优判断。触发后下一轮实际prompt必须显示失败路径及效果，并从最佳兼容checkpoint出发，不重采最佳。检查 ROUND_COMPLETE 的checkpoint、history_snapshot、实际prompt及回执，不以配置存在代替。

每次运行固定目录中的 inspect_suite.py --plan PLAN.json，保存结果根monitoring内带Asia/Shanghai时间的快照。核查真实命令/进程状态、全局与suite锁、各任务executor锁、STATUS/HEARTBEAT、阶段log、progress、attempts、acceptance/COMPLETE、标定域、真实provider回执、任务失败/跳过和SUITE标记。核对SOURCE_PRESERVATION所有历史哈希和执行目录固定提交/文件哈希。不能只凭PID、心跳、目录或单个预期nonfinite故障夹具判定健康/失败。

单任务异常保存日志、源码哈希、账本、组进度、请求/claim及配置后跳过，剩余任务继续。用户已明确授权对接口不对齐、遗漏导入、数组维度或代码逻辑缺失造成的工程终止自主修复：本地修复并测试、提交推送GitHub，部署新固定提交独立目录，验证后以新任务尝试/新结果根排队对应失败实验；保留原失败、claim和账本，不覆盖健康运行代码，不盲目重发同一失败请求。若已保存采样而只剩报告/诊断，优先仅补后处理。数值发散、LLM输出截断、真实设备/资源故障不能当接口错误通过弱化、补样、提额或删除claim绕过；如实跳过并保留证据。修复任务不能取消其他待运行任务，所有重测与相同seed/slot为配对复现，不计独立验证。

独立CPU队列检查worker.json/status.json/requests/POLICY、真实进程、worker锁和QUALITY_COMPLETE/QUALITY_FAILED；模块/哈希绑定当前固定运行目录及launch记录的CPU提交。每组attempts=50、checked_attempts=10、not_selected=40、coverage=0.2，固定五种子各slot0、1，构建失败占分母不替换，其余40不追加慢查。复用组只排原源组。必要CPU恢复限本专项，先查归属/锁，再加载DTK和当前目录agenticgen/src、src，隐藏GPU、CPU库各1线程。构建失败、真实不通过、无法评价、工程错误分开；抽样比例不外推。

ALL_JOBS_ATTEMPTED或SUITE_FINISHED后，先判定是否尚有可修复工程任务，不能把一次失败归档当成已完成授权修复。最终实际读取全部已完成教师标定、八轮参数/选择/提示、原始与排名RMSD、初始对应诊断、化学保持和10/50质量，按教师配对比较三策略，明确不完整任务和积压。下载机器可读结果/安全阶段日志，完成真实主GPT复盘、提交推送；纯文档不部署运行目录。只在路径归属确认、队列空闲后停止本专项CPU worker，写FINALIZED绑定实际复盘，通知本对话并仅删除当前触发的本专项heartbeat。

GitHub异常依次直连、ghfast.top、gh-proxy.com、ghproxy.net、gh.llkk.cc，最后才用已验证且仅含已推送提交的Git bundle。巡检间隔1小时（60分钟）。健康、排队或无实质变化返回DONT_NOTIFY；仅通知完成、实质异常、工程修复或需要用户处理的问题。中文UTF-8，读取/更新自动化后JSON ensure_ascii=true读回，确认无Unicode替换字符。

2026-10-31 06:39工程接续：原序列固定5362952ae2ea43f0f81c2226e9f6c477f9e9bd24，原目录/workspace/uniguide_runs/guidance-comparison8-5362952a，原根/workspace/agenticgen_guidance_comparison8_20261031已SUITE_FINISHED，但专项未完成。baseline6pyy在第1轮综合角色文本长度校验失败，baseline3w54完成1轮/2个正式组后在第2轮提案的格式修正中引用了不存在的证据标识；这两项属于LLM输出契约失败，保留记录，不自动重发、提额、放宽schema或改提示绕过。

已修复两项工程缺陷：联合前缀合并误把prefix_native_times的上下文调用计数2/6作为科学策略差异；五窗口标定排序仍硬性要求四窗口。采样/VJP/RNG及skills不变。新GPU/CPU固定已推送提交e0c3a384b2c54eadcd20b430486d323b78128cbc，目录/workspace/uniguide_runs/guidance-comparison8-repair-e0c3a384，修复根/workspace/agenticgen_guidance_comparison8_20261031_repairs，PLAN为该根PLAN.json。新序列依次仅修复03_prefix_joint_6pyy、04_prefix_joint_3w54、05_five_windows_6pyy、06_five_windows_3w54，各新根attempt_002，prepare→本提交完整acceptance→recover-promotion→calibrate→真实LLM GD01八轮；可等待全局锁，不打断健康任务。初始PID134802仅作定位，必须检查真实状态。不要把后续纯文档提交部署到固定执行目录。

本地和远端278项CPU回归均通过；远端保存产物重建核验通过：两个原失败五十人口视图的终态、所有轨迹数组、records/steps、随机账本均逐位/逐项一致；五窗口分别已保存40/30次，读取并通过真实粗筛排序核验。RECOVERY_PREFLIGHT通过，但不能代替新根真实模型acceptance或RECOVERY_COMPLETE。原联合前缀各150、五窗口40/30，共370次只读保留并在新账本一对一承接；新旧账本是别名视图，不能相加为740次，不能重采这些候选。四修复任务各350标定预算包含承接数，剩余最多1030新增标定；正式各1200不变。原所有失败、claims、配置和原始终态不删。每次检查PROMOTION_RECOVERY_CLAIM/COMPLETE、ORIGINAL_*、RECOVERY_PROVENANCE、源哈希、预算及未测槽位；保留实际采样提交5362952a来源，不冒充e0c3a384重新生成或独立验证。

每次先用原536目录inspect_suite.py检查原PLAN，再用e0c3a384目录inspect_suite.py检查修复PLAN，保存两个monitoring快照；还须检查新根21482个保护文件（数量以PLAN/审计为准）、固定提交和全部配置SHA。原SUITE_FINISHED不能触发停止巡检，需等待修复序列和所有实际源组10/50诊断结束，并排查是否还有用户授权的可修复工程任务。原CPU质量回执绑定5362952a，新修复任务的CPU模块/哈希绑定e0c3a384，不能启动错版本worker。最终报告三策略比较中的baseline不完整、原失败和接续次数，不将两次账本别名重复计数。完成实际复盘和FINALIZED后才删除本专项heartbeat。

2026-10-31 09:20–09:24巡检遇到SSH传输中断：ssh.zzai.scnet.cn解析为42.228.13.144，11741端口拒绝连接（TCP 10061；两次SSH及一次独立TCP检查失败），本机GitHub443连接正常。已通知并询问实例是否重启或SSH地址/端口是否改变，用户后续回复优先。最近真实远端证据仍是08:21–08:25：联合前缀6pyy完成8轮，3w54已通过验收、导入150次并继续标定，五窗口两任务排队；当前进程、锁、预算及新日志均未验证，不能据端口拒绝宣布实验停止或GPU/代码失败。保持本专项心跳ACTIVE、间隔1小时；同一连接故障无新证据时DONT_NOTIFY，不重复通知。连接恢复后先重读两份PLAN、真实命令/锁/claim/账本/失败标记，再决定后续动作，禁止按旧PID重复启动、重采、清零或删除claim；基础设施中断不能冒充接口错误自动重试。SSH不可用期间不能伪造远端快照或新的源哈希核验，也不写实验FINALIZED、不删除本专项心跳。
