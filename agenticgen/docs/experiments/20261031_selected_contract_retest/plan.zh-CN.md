# selected 输出契约修复与两项五轮重测计划

2026-10-31，Asia/Shanghai。用户明确要求修复 LLM 漏填 selected、优化 skills，随后重跑当前未完成实验。本次授权覆盖新的输出接口与指引、新固定提交、新任务尝试；不重放旧失败会话，不覆盖已完成四项或原 FINALIZED。

## 原因与修复

先前072e6f78下，prefix_joint 6pyy在第2轮、five_windows 3w54在第1轮synthesis连续两次省略selected。真实请求含完整schema，说明分别1302/1256和846/813字符、finish_reason=stop，均不是3k或token上限。原接口把schema放在消息文字中并事后检查，模型仍可能返回语法正确但漏字段的JSON。

新指引2.8.2通过DeepSeek Beta严格工具schema提交最终角色对象。终结接口名submit_final_response，必须单独调用；所有字段在arguments中由模型显式给出。thinking/high不支持强制required或命名tool_choice，因此使用auto，并由客户端拒绝普通正文终结；无法合规时有界停止，不能承诺任意模型调用永不失败。

服务端严格要求字段存在。客户端仍执行完整原schema、3000字符、数组数量/唯一性、真实ID、decision与数组关系、整数动作/证据/预算校验。服务端不支持的文字长度与数组数量/唯一性只在wire schema中明确列为client_only_checks，原schema不删除、校验不放宽。未知schema关键字上线前直接拒绝。不得从reason推断或填入selected，也不以新名称重复失败会话。

skills按角色明确必填字段：proposal为decision/proposals/reason，synthesis为decision/selected/reason；先完成控制数组，再写理由。hold、候选为空或终局分析时显式空数组，其他决策非空且只复制现有候选ID。解释不替代数组；修正必须重交完整工具参数。保持跳跃式潜在优化目标，不增加机理/消融/反证分支、方向配额或新采样能力。

官方核查来源：[严格工具参数](https://api-docs.deepseek.com/guides/tool_calls/)、[Chat Completions参数](https://api-docs.deepseek.com/api/create-chat-completion/)、[JSON模式边界](https://api-docs.deepseek.com/guides/json_mode/)。能力是否在实际服务生效，仍以部署时真实请求/回执为准。

## 仅重跑两项

|任务|教师|条件/窗口|新尝试|
|---|---|---|---|
|03_prefix_joint_6pyy|6pyy_H7S_A_402 / single_b004_final_015|索引0–9原生坐标+化学+条件时间1；10起撤去坐标覆盖；四窗口10–25/26–50/51–75/76–99|attempt_002，五轮|
|06_five_windows_3w54|3w54_RNB_A_504 / single_b000_final_003|无联合前缀；五窗口10–20/21–30/31–40/41–50/51–99|attempt_002，五轮|

顺序按上表串行。baseline两教师、prefix_joint 3w54和five_windows 6pyy已经完成，不重跑。旧六任务3790次（1640标定、2150正式）、旧失败claims和所有结构/终态/轨迹、旧FINALIZED完整保留；失败两任务原消耗360和240也不清零。

新任务各自prepare→本提交完整acceptance→独立标定→真实DeepSeek GD01五轮→独立CPU10/50。新根不移植旧λ、标定、测量排名或验收，避免源码绑定兼容检查弱化。固定教师/原九个PLIP直接互作原子、初态、原生FLOWR源码与模型使用已经持久保存的输入，只读共享并按实际快照核对SHA。source context的attempts.json只是acceptance的历史只读参考，不计新科学预算。

每新任务标定最多350（含范围证据最多100）、正式最多750，新增两任务上限700/1500，共2200；加旧实际3790，累计最多5990。各账本分开报告，失败照计、不退款、不补样、不自动扩域。相同seed/slot重测是配对复现，不计独立验证。

## 固定科学与模型契约

UG04串行调度保持：人口50，网络/VJP固定10行、默认stream依次执行，CPU算子8线程；不恢复GPU子进程池、线程池、额外stream、OOM降档或自动接续。raw999、100区间0–0.999、10–99绝对坐标完整VJP、最后循环未覆盖预测不变；step99更新不进入输出。只有蛋白决定变换，动态逐步及终态Hungarian，同步化学状态注入；初始对应误差与动态几何覆盖分别报告。无affinity/相对距离梯度、3Å限幅、向量缩放或RMSD梯度掩码。

高值只按完整组Q3+max(3IQR,1Å)单侧统计排除，至少10观测且最多10%；低值全保留。原始/排名RMSD、loss、阈值、身份和分母分别保存；构建失败、非有限值、缺测、工程错误不能过滤成成功。

DeepSeek deepseek-flash、thinking/high、单响应16384、最多3次既有网络尝试、每角色唯一一次格式修正保持。每教师每轮proposal/synthesis共享30 HTTP调用/491520输出token；最终工具提交也计入实际请求账本，不能额外免计。五轮以外部配置和ROUND_COMPLETE/FINALIZED核查，analysis_final只是终局分析，不是第六轮或GD02。

相邻两轮新完整50人口的轮最佳连续下降后，下一轮从最佳兼容checkpoint出发，真实prompt显示坏路径及效果，不重采最佳。空测量不伪造下降，累计最优不替代相邻performance。

## 上线与真接口预检

先离线测试两次真实缺字段响应的故障注入、原生提交/重放、必填字段/已知ID/唯一性/数组/3000边界、全推理内容回传、单次修正/原预算及旧失败不自动重发，并检查新发布与2.8.1全部文件哈希。随后远端固定新检出运行相应CPU回归，GPU单测在隐藏设备进程中的跳过须单独说明，不代替新任务真实GPU acceptance。

上线前在独立engineering_probe目录，使用两个原失败会话中的真实工具schema，做propose和hold两种明确标记为接口夹具的真实API探测；不执行证据计算、采样或分子评价，不计科学轮次。两个探测共享最多4次HTTP/65536输出token（含网络失败与格式修正）。保留实际Beta请求、严格工具定义、provider回执、显式selected和预算claim；不把配置当调用。失败时先归档并诊断，不启动科学链或盲目重发同一claim。

仅在codex/uniguide-flowr-agent提交推送。远端新固定目录从GitHub拉取该已推送提交；异常依次直连、ghfast.top、gh-proxy.com、ghproxy.net、gh.llkk.cc，均失败才用已经验证且只包含已推送提交的bundle。可只读引用旧Git对象降低空间，但检出、分支及新对象写入均在新目录，旧目录保持072e6f78且干净。

当前/root/private_data约2.48GB可用，部署再实测：执行代码/配置/审计保存在新持久项目目录，原始轨迹在独立/workspace新根；安全JSON/结构/压缩终态镜像到新持久result_snapshots。它不含全量HDF5，不能称完整轨迹备份；每小时归档下载已关闭的HDF5与终态，不下载写入中的文件，不删除远端结果。目录/提交/配置SHA/模型/input/probe绑定以launch_audit.json和远端PLAN为准。

## 巡检、失败与收尾

后台运行并新建本专项1h巡检；旧已完成巡检保持删除，原uniguide-flowr仍PAUSED/35分钟，其他自动化不动，不向其他聊天发消息。每次调用固定目录inspect_suite.py --plan PLAN.json保存Asia/Shanghai快照，查真实命令、进程/全局suite及任务锁、STATUS/HEARTBEAT、阶段log、steps/progress、账本、acceptance、标定、真实严格提交工具回执、ROUND_COMPLETE与失败标记；逐项核对全部SOURCE_PRESERVATION、执行源码、旧FINALIZED和配置SHA，不能只看PID或目录。

每实际源组50，CPU固定五种子slot0/1抽查10，未选40，coverage0.2；构建失败占分母不替换，复用只排真实源组。模块目录/commit/哈希绑定新提交，禁止启动旧worker。PoseB内部几何、蛋白冲突、能量和PLIP分别报告，不外推全人口。

单任务异常保留请求/claim、日志、源码/配置哈希、账本、进度后跳过，让另一项继续。明确工程接口错误允许修复/测试/新提交/新目录/新尝试后接续；不原位覆盖、不重发旧失败会话、不删标记或清零。真实数值发散、资源故障、截断及错误证据不得用弱化/提额/补样绕过。

全部尝试且质量结束后，实际主GPT复盘新五轮参数/prompt/动作/原始与排名指标、对应/化学保持和10/50质量，与原结果按教师配对；明确失败/不完整及新旧累计消耗。下载安全审计/完整闭合产物，提交推送复盘，FINALIZED绑定真实复盘与源码。确认本专项进程归属、队列空闲及锁释放后才关本次CPU worker、删除本次心跳。健康/排队/无实质变化DONT_NOTIFY；仅通知完成、实质异常、修复或用户需处理问题。

SSH root@ssh.zzai.scnet.cn:11741，沿用授权JSON stdin接口与临时凭据；密钥不打印、不写仓库/自动化。API和小写代理由交互bash加载；所有launcher.log（包括API预检启动日志）只留远端，不打印或下载，core不下载。中文UTF-8；自动化读回JSON ensure_ascii=true确认无Unicode替换字符。纯文档提交不部署到运行目录。
