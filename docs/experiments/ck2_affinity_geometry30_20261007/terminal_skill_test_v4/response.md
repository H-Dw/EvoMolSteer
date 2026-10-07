正式 V4 响应保留实测全局 **R7**：all-head 增益 +.2292925739288325、unique +.2288584818143642。R11 all-head −.03905237674713202 回归；R12 +.1390014886856079、unique +.06250335283325459，仍分别比 R7 少 .0902910852432246、.1663551289811096。R12 validity +.02 不能替代主要均值目标。四个诊断案例分别为 R1 剂量探索（先查实际 RMS、FD、zero 和绝对变化）、R2 正增益 refinement/次级 strain 评估、R4 refinement、假设 −.1 的 geometry target revision；假设不写成 GPU 实测。

真实稀疏覆盖为349/700（49.857%）批次节点不足3祖先、531/700（75.857%）不足6祖先；初始存活中位数仅1。新完整 pointcloud teachers 首末为 **17→28**，新旧每批 cap 均为 **2**，原 R7 每节点28老师。早期可用祖先更少是存活限制带来的多样性缩窄；禁止全窗强行构造 terminal credit 高/低组、拟合缺失趋势或给 clone count 加分。

最后记录 joint-latent head 的时间为 **t=.99**，即 row99；manifest 的 stored state t=1 不意味着它是 decoded final-molecule affinity rescore。灭绝的 future affinity 保持 **null**，不是低分或0。row99 上 ancestor map 从 identity 起步，倒序 M_j(d)=selected_indices[j,M_(j+1)(d)]；不能把末行 selection 作用到同一行的 pre-selection 标签。某祖先信用为其可观察末行后代的平均 head score，存活表示存在这类后代。后续真实 selection 可以注释早期 lineage，但窗口外 terminal geometry 不进入老师或生产 guidance。

全部400个 survival 字段/measure 已读取并独立重算联合 BH，最大 q 差0，**无 q<.05**。survival enrichment 最小 q=.12042655325259675；change-rate 最小 q=.3018985343962196。landmark00 soft mass 的 effect+.3650506249191145、12/14 正批次、p=.00077996286627336，调整 q 仍为 .12042655325259675。因此它只是描述性存活关联，不是显著终点优势字段或因果 affinity 特征。等父代内后代、再等批次的统计仍有 root 依赖；重叠区域也不能解释独立机制。

老师保留完整受体坐标点云，以真实 mean terminal-descendant credit 排序。固定坐标 Hungarian RMS≤.001 Å 仅用于去近复制，保留更大 RMS 的候选，不证明独立构象 basin。每批最多2、不补齐、不按后代数量复制。若节点 n 的批次 b 有 m_(n,b) 个老师、共 B_n 个有老师批次，基础每祖先质量为1/(B_n·m_(n,b))，teacher_base_log_weight=−log m_(n,b)；每批基础总质量相等。nearest-K 的截断与离线分数随后改变条件质量，**不能声称条件 prior 仍严格等批次**。

首个比较只替换声明 reference=254d8f42eaaf097bf5fdaa87fa409235f8feb25aee6afe5f35dc2d5faba2d3dd，沿用注册的 R7 endpoint_pointcloud、实际 FLOWR endpoint VJP、η=.3、K4、beta2、teacher temperature4 Å²、τ=.5、δ1 Å、time_ramp_power0 和匹配 cap2。固定 Hungarian 对应及离线条件 prior，沿用 R7 原有坐标距离及归一化 q_k；ρ_k=δ²(√(1+q_k/δ²)−1)，R=τ logsumexp(log π_k−ρ_k/τ)。各 intact prototype 分开保留，不将它们平均为一个坐标云，也不引入200项局部 moment 的 signed loss。R≤0 有有限上界但无有限下界。

只对预测终点坐标求条件梯度，g=J_FLOWRᵀ∇_Y R；self-conditioning、anchor、assignment、选定老师和 priors detached。无 atom-type/graph/pair-distance gate、SMC、重采样/克隆或 head gradient。窗口从本轮输入读取 [0,.5]，覆盖全部50个真实支持节点后 native 到1；不作通用硬编码或任意分段。生产额外 forward/head 调用0；旧 x_t 梯度在原生步后的响应仅是一阶估计。

新 reference GPU 检查尚未执行。一次实际 FLOWR FD 需4次额外 forward，且另需完整100step eta0/native equality；CPU/源代码或旧程序证明不替代。新终点亲和力回归即保留 R7；冻结 discovery winner 后再作未调参的 fresh validation。input 不含绝对均值、best、Top5、失败数或 strain 值，均不得补造，score coverage=1 不代表化学有效率100%。历史 Steer100 是子集，不是完整最大值或配对 heldout 证明。

完整 Skill/input/prompt 字节 SHA 已自行绑定；terminal reference、parent reference、mining source/trajectory 使用 input 精确声明的哈希并标注未越过三文件范围重读外部字节。input=b85ddad2c27a958640d9f009b140bccdadb8ae990fb3bb2f41f8771b434d875a。本次 V4 仅保存 response.json/md，无源码改动、推理或伪造新测量。
