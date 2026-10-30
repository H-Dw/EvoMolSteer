# GPU内核异常诊断：UG04串行执行与当前并发执行的差异

本轮只读比对UG04冻结提交930db03280c66c3f8a39f131ae01df3202544af4与batch50执行提交7670549a4b8f4e2fbddf3c8da8e6f9c620954cdd，实际检查两个core的栈、scratch池、BLAS工作区容器及当前源码，并在同一远端环境执行三个独立进程的短矩阵工程探针。不加载FLOWR、不生成分子、不调用LLM、不重启正式评测。结论分为已证明的调度资源问题与尚未证明的最底层驱动原因。

## 已确认的直接失败与资源状态

两次中止都落在ROCr的DynamicScratchHandler：error_code=0x1，queue_base=null，状态HSA_STATUS_ERROR_OUT_OF_RESOURCES。需要的scratch为7,864,320字节，即7.5 MiB；字段为48字节/线程、64线程/wave、2560个槽。第二次正式批量50仍复现。

| 证据 | 前次adaptive core.3402 | 本次batch50 core.34057 |
|---|---:|---:|
| scratch虚拟池总长 | 4 GiB | 4 GiB |
| 池内虚拟空闲字节 | 3,616,014,336 | 3,613,392,896 |
| 失败时请求 | 7.5 MiB | 7.5 MiB |
| BLAS工作区map条目 | 292 | 192 |
| 工作区key中不同stream | 33 | 32 |
| 工作区key中不同handle | 11 | 6 |

工作区条目通过GDB的libstdc++容器头计数和红黑树逐节点遍历交叉验证，分别遍历292与192个节点；相应布局由已有调试类型校验，node_count偏移40字节、树节点基部32字节。只读取元数据，未导出core内容或环境。192/292条并不等于192/292个GPU stream。

scratch池仍有约3.37 GiB虚拟空闲，故不能解释为4 GiB池全部用尽。该值不是物理显存余量：二进制反汇编显示实际流程还会调用hsaKmtMapMemoryToGPU，映射失败可释放虚拟块并返回空queue_base。当前core不能保留该已返回调用的准确驱动返回码；因此不能断言究竟是物理显存压力、映射资源、驱动缺陷或别的底层条件。宿主dmesg在容器内读取被拒绝，这一层证据尚缺。

GDB提示可执行文件匹配与reg-xstate警告，原文完整保留；不同core的处理栈、错误字段和本机二进制调用路径一致。日志中巨大的“Available Free mem”数值无效，不能当作显存充足证明。KERNEL VMFault是此次运行时的转储标题，也不能单独证明某个softmax内核非法访问。

本次正式进程没有OOM回执，先连续完成4个50候选批次，第5个批次中断，最后保存到第49步。前次则经历90/80候选OOM后在60候选处中断。故“只有OOM后恢复才会失败”和“仅因batch大于50”均不足以解释两次现场。正式批次保存前缀的loss、梯度及增量全部有限；最后保存步最大增量约0.02415 Å，没有临近中止的数值爆炸证据。

## UG04稳定执行方式发生了什么变化

| 项目 | UG04 / 930db032 | 当前 / 7670549a |
|---|---|---|
| 逻辑候选人口 | 通常50 | 批次上限50 |
| 神经网络前向与VJP形状 | 10行 | 10行 |
| 用户代码的GPU执行 | 10行单元依次处理 | 最多5个10行单元并发 |
| Python线程池 | 采样循环不新建线程池 | 每次concurrent_map新建 |
| 用户创建的stream | 无额外stream调度 | 每个任务请求stream |
| 同时存在的前向/VJP图 | 一次一个10行图 | 最多五个10行图 |
| BLAS handle/stream组合 | 少量且稳定 | 随线程与stream重组增多 |

关键变化在82dc89c1引入，位于chunking.py:9的concurrent_map。该函数在:18为每个任务调用torch.cuda.Stream，在:22为每次执行创建ThreadPoolExecutor。geometry_batch.py:150通过此函数执行VJP，:485又把并行度设为size//10。sampler.py:164仍是完整torch.autograd.grad(value.sum(), x)，而:168将各单元提交并发。因此“物理形状仍为10”只保证算术布局，不保证资源占用与UG04相同。

旧UG04的50个候选不是50个分子同时拥有完整前向/反传图。当前除了父积分状态，还需要多个同时活动的网络图、算子临时张量和更多持久BLAS工作区。相同batch数字不能直接比较峰值资源。

PyTorch的stream不是每次请求都永久增加一个：2.9实现以32个stream循环复用，HIP还特别注明stream创建不轻量。本次日志约33个stream身份与此相符，不能声称线程或stream无限泄漏。[PyTorch stream实现](https://github.com/pytorch/pytorch/blob/v2.9.0/c10/cuda/CUDAStream.cpp)

BLAS handle会在线程退出时归还池，但(handle, stream)对应的workspace另由全局map保留，正常线程结束和empty_cache不会清除这些仍被引用的工作区。当前调度把有限数量的handle与32个stream反复重新组合，扩大常驻工作区集合。[PyTorch 2.9 HIP说明](https://docs.pytorch.org/docs/2.9/notes/hip.html)、[工作区与handle实现](https://github.com/pytorch/pytorch/blob/v2.9.0/aten/src/ATen/cuda/CublasHandlePool.cpp)

若按该构建默认32 MiB工作区估算，192条约6 GiB，292条约9.125 GiB。这是依实现和条目数推算，不能冒称core直接测得的物理驻留量；实际短测试另外给出了可测的保留量。

旧数值验收回执与本次实测均记录torch2.9.0、HIP6.3.26093。旧回执的torch_threads=8是CPU算子线程数，不能解读为8个并发GPU VJP。没有完整的旧驱动二进制指纹，不能据版本号声称两天的整个环境完全相同；下面三个探针则是在同一个当前环境完成，隔离了调度差异。

## 已复现的最小工程测试

同一BW设备、torch2.9.0、HIP6.3.26093，使用512×512矩阵前向和输入VJP，20次循环，每次两个任务。无随机数、分子、采样预算或API；各模式独立进程。表中保留量是在任务已结束、线程池退出且调用empty_cache之后的memory_allocated，扣除了1 MiB常驻测试张量。

| 调度模式 | 使用的stream数 | 保留工作区 | 再清BLAS工作区后的保留量 |
|---|---:|---:|---:|
| 旧式串行 | 1 | 64 MiB | 0 |
| 当前临时线程池/轮换stream | 32 | 2176 MiB | 0 |
| 常驻线程池/固定两个stream | 2 | 128 MiB | 0 |

三者校验值均为134216.9375。当前方式的工作区随前16–18次循环增长，再趋于平台，不能说无限增长；常驻线程与固定stream使保留量下降约94%。仅调用empty_cache没有解除工作区引用；在无活动GPU任务的安全边界调用torch._C._cuda_clearCublasWorkspaces后，保留量回到测试张量基线。这直接验证了工作区保留机制，不依赖分子损失、inpainting或有限性检查。

这个小测试没有复现SIGABRT，也没有验证完整FLOWR的persistent模式正确性。它证明的是当前调度会放大资源常驻量，和core中的多工作区证据一致；尚不能断言这是底层映射失败的唯一原因。

## 建议按以下次序优化

1. **把候选队列容量与GPU并发解耦。** 保持候选batch上限50和10行网络形状，新增独立执行并发策略。先以GPU串行作为稳定基准；之后只开放经过完整验证的1–2个固定GPU执行槽。这样无需改变候选配置、随机流、预算、raw999、10–99步或化学inpainting。
2. **固定资源生命周期。** 每教师创建一次执行器，维护常驻线程及每线程固定stream，跨积分步和组复用，教师结束一次性关闭。不要每一步重建线程池或每个任务轮换新stream；并发受容量和实际资源双重约束。32 MiB默认工作区不应通过盲设0来回避，改变BLAS工作区可能改变选用的算法，须另行数值验收。
3. **明确跨stream张量所有权。** 当前child.synchronize仅等待生产完成。父stream消费子stream分配的张量后，释放前仍须正确record_stream或用事件把消费结束与创建stream衔接；父输入转交子stream也要保证生存期。采样器某些CPU日志同步和局部引用可能已经使具体路径安全，因此这里只能列为需修正的通用接口风险，不能据core宣称已发生UAF。[官方张量stream生命周期约定](https://docs.pytorch.org/docs/2.9/generated/torch.Tensor.record_stream.html)
4. **安全边界回收，异常进程隔离。** 保证所有相关stream完成、任务结束后再清BLAS工作区和unused cache；不每步清理，也不清仍在使用的缓冲。真实OOM后保留回执与账本，通过明确审计的恢复入口在干净GPU进程恢复，不把异常HIP上下文当成健康上下文继续使用。不同教师的GPU进程和容量记录分别管理。SIGABRT仍终止，不自动删claim或重跑。
5. **验收长期资源平台。** 两教师分别比较串行与固定并发：完整100区间、非前缀seed/slot、不同方向和强度、恢复，逐步核对终态/轨迹/梯度/对应/RNG。再连续运行多个工程批次，观察allocated、reserved、设备free、工作区条目和队列数是否达到稳定平台，记录真实余量。一次100步容量通过不等于长期运行稳定。通过后才恢复正式线上运行。

本次优先修复执行器，不能通过恢复3 Å限幅、降低λ、抑制高RMSD候选梯度或回退输出协议来替代资源管理。有限性检查和对应恢复审计继续保留。未实施这些代码改动，正式运行也未重启。

## 仍需补的驱动证据与当前状态

如果固定执行资源后仍出现相同错误，应携带两个core的安全栈、时间、PID、queue错误码、scratch字段、设备BW3000 PCI标识、DTK/HIP版本与工作区曲线向运行环境维护方核查宿主GPU映射失败、页故障、ECC/设备重置等日志。容器内dmesg权限不足，不能臆测宿主错误，也不盲改HSA_NO_SCRATCH_RECLAIM或其他厂商运行时开关。

batch50失败已归档并推送22d54482，FINALIZED绑定实际GPT复盘，空闲CPU worker已停止，15分钟专项巡检已删除；原uniguide-flowr仍PAUSED且35分钟不变。本轮诊断只新增审计和文档，原源码、预算、失败claim和结构保持原状。证据原文位于kernel_diagnosis，core和launcher.log仍只在远端。
