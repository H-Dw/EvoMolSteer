V1 预备响应：保留实测全局 R7（all-head +.2292925739288325、unique +.2288584818143642）。R11 −.03905237674713202 回归，R12 +.1390014886856079 仍低于 R7。案例分别执行剂量探索、正增益 refinement/次级能量评估、promising refinement、假设负值的 target revision；不将假设写成实测。

700 个批次节点中349（49.857%）不足3祖先，531（75.857%）不足6祖先；初始 alive 中位数1，完整 pointcloud teachers 由17到140。禁止在全窗强行分终点 credit 高/低组或填补缺失。最后记录的 joint-head score 位于 t=.99，stored state t=1 不等于 decoded final rescore；灭绝 affinity=null，不是低分或0。末评分行先从 identity 开始，倒序 selected_indices[j] 映射 child(j+1) 至 parent(j)，不能把末行 selection 作用到该行的 pre-selection 标签。窗口外后续选择可以注释历史存活，但窗口外 geometry 不进入老师库或生产 guidance。

完整400项 survival 检验已读并联合重算 BH，最大 q 差0，全部无 q<.05；enrichment 最小 q=.12042655325259675，change-rate 最小 q=.3018985343962196。landmark00 soft mass 虽 effect+.3650506249191145、12/14 正方向，也不能称显著终点优势字段。相同 root 与重叠区域的依赖仍在。

只改变 R7 的老师 reference：endpoint_pointcloud，η=.3、K4、beta2、teacher temperature4Å²、τ=.5、δ1Å、time ramp0；新 ref=0cccd6e34a2ff56328474fe938a1580be3678922705fe45b386c74460cc13a91。用 surviving ancestors 完整空间点云及其平均 terminal-descendant credit，不给 clone count 加权。固定 Hungarian 对应与离线 prior，q=mean_i||Y_i−T_ki||²，ρ=√(1+q)−1，R=.5 logsumexp(log π−ρ/.5)，实际 FLOWR endpoint VJP 回到当前坐标。各 prototype 分开，early diversity 缩窄是明确限制，不能预测亲和力增益。

动态窗口来自 manifest，覆盖全部支持节点后 native 到1；生产额外 forward/head 调用0、无 head gradient/gate/SMC/clone。新 reference 的实际 GPU FD（一次额外4 forwards）及完整100step zero/native equality 均未执行；原生步后只能报告旧 x_t 的一阶估计。Skill/input/prompt 字节 SHA 已绑定，外部 reference/provenance 使用输入提供值。本响应只保留为 v1 预备测试，后续新版另写目录。
