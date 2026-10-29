巡检并收尾本对话已授权的“双教师化学inpainting＋原子对误差求和”新损失正式实验。先读取D:/llm_controlled_generation/EvoMolSteer-uniguide/agenticgen/docs/experiments/20261029_chemical_interact_sum/protocol.zh-CN.md、launch_audit.json及本对话后续指令。只操作codex/uniguide-flowr-agent；原uniguide-flowr保持PAUSED、35分钟配置不变，不操作其他任务或自动化，不向其他聊天发送消息。

服务器root@ssh.zzai.scnet.cn:11741，使用本对话授权凭据和D:/llm_controlled_generation/tmp/uniguide_ops.py（JSON stdin）。凭据不得打印、写入仓库或自动化。固定执行目录/workspace/uniguide_runs/chemical-interact-sum-4d7b870d，已推送提交4d7b870d056cc8f5163571f29aae595c4e7c3de0。结果根/workspace/agenticgen_chemical_interact_sum_20261029；外部配置/workspace/uniguide_runs/chemical-interact-sum-4d7b870d-config.json，SHA256 2b9fa88433a769602e691db774df331f6b37043ae099b0740d4066df0e268e45。

本协议template-relative-distance-guidance-v6-summed-error，奖励template-relative-three-component-sum-v2，skills2.3.0。损失为每候选内部各组件的原子对半平方误差求和，不除以原子对数量、模板长度或当前长度；没有normalization或s_t。完整预测终态VJP保留，距离RMSD评价仍为均方根。不得用旧均值/长度归一化条件、λ或排名复用新结果。旧归一化实验130次保留原账本，单独报告，不清零或冒充新测量。新目标标定每教师最多350、共700次，正式最多1500次。

2026-10-29 17:58:00（Asia/Shanghai）已启动一次性后台链acceptance→calibrate→真实LLM run-version GD01→10/50诊断；初始父PID17286。17:59:16核对真实父进程及acceptance子PID17359，三把锁持有，第一教师continuous与resumed已到步骤11，尚无新标定、正式候选或真实LLM回执；远端153项测试通过。PID仅供定位，后续状态以实际核查为准。使用本固定目录agenticgen/docs/experiments/20261029_chemical_interact_formal/inspect_run.py检查并保存本专项monitoring中的Asia/Shanghai快照，核对真实命令/进程、全局/workspace/agenticgen_uniguide_flowr_20261026/.executor.lock、本专项.executor.lock/.campaign.lock、STATUS/HEARTBEAT、阶段日志、最新逐组steps/progress、attempts、完成与失败标记。不能仅凭PID、heartbeat或目录判断健康。

教师保持6pyy_H7S_A_402/single_b004_final_015和3w54_RNB_A_504/single_b000_final_003。各9个原PLIP直接互作配体重原子同时为condition与interact_atoms；内部各36对，外围72/117对，显式配体—靶点44/10对。后者仅在原PLIP记录两端原子集合展开去重，不扩大位点，不当成独立互作事件。原子身份、坐标、化学图、固定生成槽位、靶点对应均已与上一版核对一致；新条件SHA分别d882f6f220e6951cedbe45e7611aae8ef7172d6138834762241353fd6b892468和4f1eda62d9b9ba389dbe19523f89ad8ac66fc95b6387135ed3e8ae2fa92c6d74。初始内部/外围/互作开关1/0/1。

化学状态inpainting开启，仅化学通道；坐标不覆盖。raw999100区间0–0.999，四窗口10–25、26–50、51–75、76–99，返回最后循环覆盖前预测（网络时间约0.98901），不额外终态前向；步骤99更新不进入返回端点。3Å单分子统一倍率限幅从配置读取，仅限制引导增量，保留未限幅证据、原生漂移/噪声与随机流。数值失败账本仍以10行随机流分片为单位，不能冒称逐分子失败隔离。

完整双教师真实模型验收必须先通过：有限差分、无亲和力梯度/额外RNG、100区间独立/合批逐位一致、恢复、OOM降档、3Å压力及非有限梯度注入隔离。工程夹具不计科学候选；nonfinite_isolation内预期故障注入不能仅凭证据文件判定任务失败。任何真实验收失败不得跳过或放宽容差。

新标定为每教师中心1、13格2的−12到12次幂、指数间隔2，各10候选；局部5格20/50人口逐步筛选，同损失下已测槽位复用。边界最优或证据不足停止，不自动扩界或弱化重试。正式5轮、三方向、每组50、双教师30组请求；精确重复配置引用原结果，不重复生成或计为独立验证。Agent使用真实DeepSeek deepseek-flash、thinking/high、16384及既定网络重试/一次格式修正，每教师每轮四角色共享30次调用与491520输出token。核对实际provider请求、回执和claim，配置存在不能当作已调用。不启动GD02，不恢复UG05。

健康运行时禁止覆盖执行目录、源码、manifest、skills或重复启动。若真实进程退出、CAMPAIGN_FAILED、INSUFFICIENT_EVIDENCE等停止，先保存阶段日志、源哈希、账本、组进度及失败请求；只读定位并报告，不删失败/claim、不自动重跑/补样。launcher.log可能含交互式shell环境输出，只留远端，不直接打印或提交原文。GitHub异常按直连、ghfast.top、gh-proxy.com、ghproxy.net、gh.llkk.cc依序尝试，均不可用才使用已验证且仅含已推送提交的SSH Git bundle。

检查本专项quality_queue的worker.json/status.json/requests/POLICY、真实进程和.worker.lock；module_directory、commit及实现哈希必须绑定本固定目录。每正式组attempts=50、checked_attempts=10、not_selected=40、coverage=0.2，五种子各logical_slot0、1；构建失败计入分母、不补样，其余40不追加慢查。区分构建失败、测量不通过、无法评价与工程错误。必要的CPU恢复仅限本专项，先核对进程归属与锁，再加载DTK及本固定目录agenticgen/src、src，屏蔽GPU，各CPU库1线程，不启动历史worker。

SAMPLING_FINISHED后只补后处理。CAMPAIGN_FINISHED后实际读取双教师标定、全部30组请求/复用来源、指标、四角色上下文与实际LLM回执，做真实GPT复盘。按教师报告内部/外围/互作距离、条件区域坐标RMSD、限制器触发、构建/连通/唯一性及10/50质量诊断，不能将未抽查当通过。SOURCE_PRESERVATION的1204个历史文件全部核对不变，保留新旧全部源结构、终态、轨迹及失败账本。下载机器结果与审计至本地本实验docs目录，复盘提交推送GitHub；纯文档不部署运行目录。只在队列空闲且路径/进程归属确认后停止本专项CPU worker，写FINALIZED与实际复盘绑定，通知本对话。完成或终态失败归档后仅删除当前触发的本专项heartbeat，不修改其他自动化。

巡检间隔15分钟。健康、排队或状态未变时保持安静；仅报告完成、实质异常、修复或需用户处理的问题。中文文件UTF-8，自动化读取/更新后用JSON ensure_ascii=true读回，确认无Unicode替换字符。
