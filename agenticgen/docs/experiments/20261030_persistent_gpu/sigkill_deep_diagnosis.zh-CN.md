# 3w54第八批SIGKILL：终止证据与UG04代码比对

2026-10-30，Asia/Shanghai。本次实际失败的GPU提交是`399590dd9d723ec3bbfce86fb0f017cf2c6ecb61`，并非此前发生HSA错误的`7670549a`；对照为UG04冻结提交`930db03280c66c3f8a39f131ae01df3202544af4`。最新`e45a70f0`只增加两个CPU恢复校验，没有替换这次GPU执行代码。

**现有证据仍不能确定SIGKILL的发送方。进一步检查不支持把本次终止归因为已经证明的GPU资源增长、引导数值爆炸、内核OOM或验收超时。** 能定位的是此前临时线程/stream调度的确定性资源缺陷，以及当前验收路径和诊断记录的缺口。后两项不能冒充已证明的SIGKILL根因。

本次仅只读核查、保存审计并在CPU上重新读取既有张量；没有加载FLOWR、运行新GPU探针、生成分子、调用LLM、重新启动验收或变更自动化。固定执行目录与历史失败归档均保持原状。

## 终止发生在哪里

验收子进程56811于17:34:22启动，18:41:23以`returncode=-9`退出，运行4020.581710秒。父协调器继续保存了失败回执。因此可确认验收进程收到SIGKILL，不能据此识别发送者。Python规定POSIX负返回码表示信号终止；此处7200秒的`Popen.wait(timeout=...)`没有到期，启动器的该阶段也没有主动kill调用。[Python 3.11 subprocess约定](https://docs.python.org/3.11/library/subprocess.html#subprocess.Popen.returncode)

6pyy完成8/8批，3w54完成7/8批。3w54第八批在积分中断：`steps.json`保存0–89步，HDF5最后写入时间18:41:17；远端确认没有checkpoint_100和terminal。它不属于只需要补后处理的情况。周期保存无法确定被杀瞬间执行的具体指令，也不能把最后保存的89步当作故障必然发生的步号。

重新读取中断批与已完成`batch_005`，全部HDF5数据集的前90步逐位相同。保存范围内的损失为0.172595–1.082931，梯度范数最大0.745156，最大单原子引导增量0.000089586967 Å，最大引导/原生增量比0.000936714234，全部有限。工程夹具实际强度在0.001–0.004之间；它并非4096强度科学测量。保存前缀没有显示大位移、异常高强度或多份梯度叠加失控。

SIGKILL不能被Python捕获或忽略，所以缺少Python异常堆栈、finally收尾或新FAILED文件，不能反推计算没有被强制终止。父进程在子进程之外保存的退出证据才是这里的终态依据。[Linux signal约定](https://man7.org/linux/man-pages/man7/signal.7.html)

## 对OOM、GPU异常与超时的进一步核查

| 检查 | 实际证据 | 能支持的结论 |
|---|---|---|
| 设备OOM/HSA签名 | 安全acceptance.log中HSA错误、KERNEL VMFault、设备OOM和Traceback计数均0 | 没有保存到此前那种SIGABRT故障签名；不证明宿主GPU完全没有事件 |
| 3w54批后显存 | 7批allocated均277325824字节，reserved均19344130048字节 | 已完成批次没有持续增长；reserved约18.02 GiB是缓存保留量，不等于全部活跃张量 |
| 3w54峰值 | 已记录max_memory_allocated均13029203456字节，约12.13 GiB | 这是进程截至各采样点的历史峰值；第八批未持久化的峰值未知 |
| 原生资源 | 15个已完成资源批次中BLAS工作区2、BLASLt 0、ROCr队列事件引用计数2 | 没有重现此前192/292个工作区的增长机制；队列指标不是驱动全部队列枚举 |
| 可见cgroup | memory.max=63350767616字节，memory.events及local的max、oom、oom_kill均0 | 未发现内核记录的该组内存超限/OOM杀进程事件，不支持直接宣称内核OOM |
| 主机累计OOM | /proc/vmstat的oom_kill=283 | 是约39天主机启动期间的累计数，缺少事发前基线，不能把283归于本任务 |
| 运行时限 | 4020.58秒，小于7200秒；可见CPU时间/地址空间限制为unlimited | 启动器的两小时超时未触发；现存进程限制不是已死亡进程的事发快照 |
| 宿主记录 | dmesg权限被拒绝，journal无日志，kern.log/syslog/messages不存在 | 无法读取18:41:23的内核或平台事件，因此发送方仍未定位 |

19:29和此前18:53两次读取的cgroup OOM计数均为0。当前容器PID1于11:54启动，早于验收，未见本次事件造成容器重启。Linux 5.15文档将`oom_kill`定义为该cgroup中被内核OOM killer杀死的进程数；零值是反对“内核OOM已确定”这一解释的证据。用户态资源管理器或平台主动发送SIGKILL不必表现为内核OOM事件，因此还不能排除平台侧资源策略。[与本机5.15内核对应的cgroup文档](https://www.kernel.org/doc/html/v5.15/admin-guide/cgroup-v2.html)

事后主机MemAvailable约395 GiB、容器memory.current约12 GiB，均不能替代事发时内存曲线。现有证据也没有证明手工kill、GPU看门狗、设备重置或平台策略中的任何一项，不能从候选原因中任取一个作为结论。

显存的allocated与reserved分别描述活跃张量占用和缓存分配器管理的保留量；empty_cache只释放可回收缓存，不能用它替代对象生命周期或持续观测。[PyTorch 2.9显存统计约定](https://docs.pytorch.org/docs/2.9/notes/cuda.html#cuda-memory-management)

## UG04五轮与当前执行代码的真实差异

“五轮”是每教师五轮Agent决策、每轮三个配置组，即每教师15个50候选配置组；“完成七批”是连续资源验收中的七个50候选批次。两者不是同一计数单位，不能解释成超过第五次循环后代码必然失效。

| 项目 | UG04 930db032 | 早前故障7670549a | 本次399590dd |
|---|---|---|---|
| 逻辑候选批量 | 50 | 50 | 50 |
| 网络前向/输入VJP形状 | 10行 | 10行 | 10行 |
| GPU调度 | 调用线程串行 | 最多5个单元并发 | workers=1，调用线程串行、固定stream |
| 每次map新建线程池/stream | 无 | 有 | 当前串行分支无 |
| 梯度活动窗口 | 10–90 | 10–99 | 10–99 |
| 输出 | 积分至1及原生终态前向 | raw999 | raw999，最后循环未覆盖预测 |
| 配体对应 | 每步几何匹配 | 第10步匹配后冻结 | 第10步匹配后冻结 |
| 化学状态注入 | 无 | 有 | 有 |
| 本次所走入口 | 正式pipeline | 正式评测 | 工程absolute_acceptance，不含LLM |

远端实际检出和本地Git对象双重核验：`prepare.py`、`native_tools.py`、`random_streams.py`、`batching.py`在930db032和399590dd逐字一致，包含模型加载/冻结、原生插桩和随机流机制。`sampler.Control.forward`中`torch.autograd.grad(value.sum(), x)`调用的AST相同，`condition.guidance_delta`中系数及引导增量两个赋值的AST也相同。新增check回调承担数值检查，没有重写该增量公式。这不等于两种科学协议的完整图或最终分子应当相同；raw999、匹配冻结与化学注入是明确的协议差异。

当前`gpu_execution.TeacherGpuExecutor.map`的workers=1分支只依次调用function，不创建线程池、不申请额外stream、不提交future。固定调用线程还检查stream身份。当前串行路径已经恢复UG04的调度布局。已完成批次保留资源稳定，与该源码路径相符，不能继续把旧的“每步新线程池/轮换stream”当作本次已证明原因。

## 哪些模块确实存在缺陷或缺口

### 1. 早前已证明、当前已替换的资源缺陷

在7670549a中，`chunking.py:9–23`的concurrent_map每次创建ThreadPoolExecutor，每个任务再请求torch.cuda.Stream；`geometry_batch.py`按任务容量派生并发数，使batch50带来最多五份活动网络图和不断重组的handle/stream组合。

此前core实测BLAS工作区map为192/292项；独立矩阵VJP探针中，临时调度在任务结束并empty_cache后仍保留2176 MiB，串行为64 MiB，固定双stream为128 MiB。它证明了调度放大资源保留，尚未证明底层scratch映射失败只有这一原因。细节和原始证据见`../20261030_batch50/kernel_diagnosis.zh-CN.md`。本次只见2项，不能将两类终止合并归因。

### 2. 当前资源验收只记录批后GPU状态，缺少事发时主机证据

定位：`gpu_resources.py:102–111`的snapshot，以及`gpu_stability.py:35–56`的调用位置。

snapshot只记录PyTorch显存、设备余量、线程/stream及具名原生计数，未记录RSS/PSS、VmHWM、cgroup memory.current/events、CPU/IO压力。资源行在采样完成、轨迹比较及GC之后才写入；第八批中途被杀时没有该批资源行。由此不能判断事发前是否存在主机内存上升、瞬时峰值或平台回收。`peak_allocated_bytes`也只是截至上次采样的GPU高水位，不能补出中断批的状态。

这是明确的诊断覆盖缺口，不是内存泄漏已经发生的证明。正确补充方式是由GPU进程之外的父协调器周期保存主机/cgroup与子进程状态，并在每批开始、运行中和结束分别采样；不在每步清缓存制造虚假平台。

### 3. 工程acceptance绕过了正式教师进程隔离入口

定位：`absolute_acceptance.py:73–77,135–137`直接load_model、bind_executor，并在同一个Python/GPU进程内完成两个教师。正式`absolute_runtime.py:211–212`通过`gpu_execution.execution_model`进入`gpu_worker.TeacherProcess`，教师结束后退出该GPU子进程。

因此，这次验收验证了固定执行器和张量运算，却没有端到端覆盖正式系统的教师独立进程创建、跨请求复用和进程退出。两个教师共用验收CUDA上下文也意味着本次3w54不是从干净GPU进程开始。已有GPU平台数据没有显示前一教师资源一直增长，不能直接断言它就是被杀原因；该路径差异仍应消除，避免验收和上线入口不一致。

`gpu_worker.py:65–81`确实存在等待60秒、terminate、再等待15秒后kill的清理路径，但本次acceptance没有调用TeacherProcess，因此它不是此次SIGKILL的已执行来源。不能只搜索到process.kill就归责该模块。

### 4. 部分验收数字只保存在内存，SIGKILL后丢失

定位：`absolute_acceptance.py:78,131–145`，有限差分的误差值先存入numerical/checks，最后才随总COMPLETE写盘。3w54进入第八资源批次，说明此前检查已按控制流通过；但此次精确VJP误差数字没有独立落盘，不能从后续通过轨迹伪造这些数值。

应在每教师每项检查成功后写独立、带源码/配置哈希的回执，最后汇总完整验收。这样中止仍保留已完成证据；它不是跳过剩余验收或转移旧提交的COMPLETE。

## 下一步诊断与修复的优先级

1. 先补父进程侧连续主机/cgroup观测及退出rusage，单项验收及时写盘，确保SIGKILL后仍有现场。SIGKILL本身无法在子进程里安装handler来捕获。
2. 让工程验收走与正式运行相同的教师独立常驻GPU进程，保留50候选、固定10行、并发1和所有数值对照；先验收进程生命周期，再讨论更高并发。
3. 获取SCNet宿主节点在**2026-10-30 18:40:30–18:42:00（Asia/Shanghai）**、容器内PID56811对应的OOM/用户态资源回收、GPU重置和信号审计记录。当前容器没有这些记录的读取权限；本次未联系任何外部人员或修改平台权限。没有事件源记录，就只能报告“SIGKILL来源未确定”。

本次不会通过改损失、恢复3 Å限幅、削弱梯度或放宽一致性验收来处理一个尚未归因的进程级SIGKILL，也未自动重跑。19:35重新核查：固定目录干净、无活动验收进程、两把执行锁空闲、9626个历史保护文件未变、无attempts/calibration/versions；旧FINALIZED和复盘绑定未覆盖。

可复查证据：`evidence/sigkill_deep/source_and_tensor_comparison.json`、`SIGKILL_DEEP_HOST_DIAGNOSIS.json`、`SIGKILL_DEEP_REVERIFICATION.json`。`diagnose_sigkill_sources.py`可在CPU上从既有终态导出目录复算源码与张量对照，不运行模型。
