# 历史实测背景的box负面排除与五轮全量线上运行

## 授权与范围

仅codex/uniguide-flowr-agent。两个教师6pyy_H7S_A_402/single_b004_final_015、3w54_RNB_A_504/single_b000_final_003，原九个PLIP条件、模型与冻结初态不变，逐教师逐配置UG04串行50逻辑候选/10行网络与VJP/CPU算子8线程。0/99只用原生端点，实际额外条件操作1–98；无坐标覆盖、affinity/相对距离梯度、限幅、GPU池或OOM降档。

只排除明确有害box，不要求证明保留box有益。禁用同时关闭额外VJP/坐标梯度/化学覆盖；允许box的实际scale=0仍保留化学覆盖。其他自动化不动，原uniguide-flowr保持PAUSED/35分钟，不向其他聊天发送消息。

## 新实测背景

| 教师 | UniformRef | BestCombo | 历史完整50 RMSD |
| --- | --- | --- | --- |
| 6pyy | [50,50,50,50,50,50] | [50,70,70,50,50,70] | 0.503462 / 0.492536 |
| 3w54 | [40,40,40,40,40,40] | [30,40,40,40,60,80] | 0.505152 / 0.408967 |

历史仅设计参照，全部重新生成，不移植旧测量或旧验收。旧0/30/60/90漏50/40；小样本排序不能代替完整人口。旧固定递增20/40/60/80/100/120未实测，不作为主背景。6pyy BestCombo历史构建48/50，失败保留分母；背景不是正式scale白名单。

## 每教师预实验500次封顶

29配置各10候选：NativeControl纯原生、ChemistryOnly全允许零梯度、UniformRef、BestCombo；每背景逐box NativeOff共12；每背景逐box LowerRescue=max(10,原scale-20)共12；UniformRef的第一box提高10共1。其他box/输入/seed-slot均完全相同。

两背景10→20→50按增量各新增40，共80。完整排名较优者为主确认参照。每box检查两个背景的保留与关闭对照、两个LowerRescue及第一box EarlyUpperRescue。所有粗筛对照须同向且达到max(0.01Å,关闭均值2%)，至少3/5 seed同向；任何挽救/等效/冲突证据保留。最多两可疑box按最弱背景标准化负面效果排序，不能按单个最大异常排序。

一可疑box时两背景各晋级关闭配置50；两可疑时在主确认参照各晋级关闭配置50，共最多新增80。完整50原始与过滤差均达到门槛、4/5 seed同向、原始均值的五seed聚类bootstrap(1000次,seed721904)95%区间下界>0才可排除。仅描述适应性固定panel，不称独立显著验证。缺测/工程错误/单次高档发散不作为整box排除证明。

Selection proposal只提交eligible_box_ids子集、证据和说明，允许空列表。两个单独关闭有效不保证联合有效；一个新联合门控再真实50确认，最多500=290+80+80+50。final只从实际完整50且符合负面门槛的门控选择，可退回已确认单box或空集合。最终门控下最佳完整50作默认标定根。未花完不补样。

## 两类Agent

skills2.11.0，Box Selection独立两阶段，共享30 HTTP/491520输出token；Guidance每教师每轮两角色共享同额，固定5轮，仅GD01。deepseek-flash、thinking/high、strict_tool_v1、16384单响应、网络最多3次/唯一格式修正不变。显式原生submit_final_response控制，不从正文补字段，说明3000字符。

Selection仅找负面证据，核查双背景、低档及增幅挽救；均匀80差不推出末窗80不可用，无明显收益/不一致就保留。默认不排除不等于已证明有效；坏输出不能自动当空列表。

Guidance六维绝对scale0–200格距10，禁用box固定0；相对真实global best总距离<=12档，每窗<=12。focused1–2、expanded3–6、bold7–12。存在预算可覆盖的合法bold时每轮至少选一条改变>=2允许box的bold；选中两两距离>=3档且>=2 box取值不同。程序只读覆盖候选不代表已测收益，不按机制/方向设配额。标定根保留，严格更好完整50才替换；回滚复盘真实父链及坏旁支，不重采最佳。

## 构建、提交和上线

本地CPU回归、技能校验、包构建及资源清单通过后commit/push GitHub。远端fetch并锁定真实已推送提交，以新固定目录/新根/新launch_audit绑定；历史归档分支、代码和失败不覆盖。远端CPU通过后本提交完整GPU验收：全100步/50容量/恢复RNG/VJP、零scale、端点、NativeControl独立原生逐位等价及EarlyNative禁用梯度/化学检查。CPU不能代替GPU。然后后台prepare→acceptance→calibrate(含Selection)→GD01五轮→独立CPU10/50。每实际源组抽查固定五seed各slot0/1，失败占分母，40未选不补查，不外推全人口。

root@ssh.zzai.scnet.cn:11741，凭据只用本对话授权及本地uniguide_ops.py JSON stdin。API/小写代理交互bash加载。所有launcher.log及轮转、core只留远端，不打印/下载。GitHub依次直连、ghfast.top、gh-proxy.com、ghproxy.net、gh.llkk.cc；均失败才验证已推送bundle。

## 小时巡检与修复预算

创建本专项box-exclusion-guidance-hourly，1h间隔，同一对话。读取本plan、launch_audit及实际PLAN；若仍部署/CPU预检则检查部署状态，不因结果根未创建判断失败。用固定目录inspect_suite核真实命令、进程/父子、全局/suite/任务/worker锁、日志、progress、attempts、验收、Selection两阶段与真实provider/tool/claims、五轮、质量队列、源码配置SHA及SOURCE_PRESERVATION逐文件。不能只看PID/目录/心跳，nonfinite_isolation为预期夹具。上海时间快照，健康/排队/无变化DONT_NOTIFY。

异常先存安全日志/源码哈希/账本/请求claim/进度，查残留活动采样后定位。接口/导入/维度/代码逻辑缺陷可自主本地修复测试commit/push，新固定目录新尝试；不影响健康教师，原始队列继续，修复排后面。采样已存优先仅补后处理。数值/设备故障、截断、错误证据及正确传输下的LLM契约失败，不放宽/提额/删claim/盲重发绕过。

初始每教师500预实验+750正式，两教师最多2500；每教师另允许一次完整代码修复科学重跑500+750，整个专项最多5000。旧消耗不退款，新旧同seed是配对复现；引用实际产物物理去重。无科学候选产生的部署修复不占科学重跑资格。重跑仍各5轮，不加第六轮。资格/预算耗尽就归档报告，不继续新增采样。

每小时收集已完成不可变HDF5/终态/结构及安全审计到本地，不下载活动HDF5、launcher或core，不把大文件塞Git。持久镜像不含全量轨迹，检查空间错误不删除旧文件。SSH不可达保留巡检，不按旧PID启动或伪造状态。

最终实际读取所有提示、动作、源测量、负面证据与反例、RMSD/pIC50/SA、构建与抽样PLIP/PoseB分项，比较本次标定及历史最佳，报告部分失败/所有实际次数。完成真实复盘commit/push并绑定FINALIZED；队列空闲且归属确认才停本专项worker。完成或终态失败归档后通知并删本专项心跳。自动化读回JSON ensure_ascii=true，无替换字符。
