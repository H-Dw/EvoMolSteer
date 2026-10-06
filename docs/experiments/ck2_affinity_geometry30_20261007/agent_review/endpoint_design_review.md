# 端点几何真实VJP：独立执行审阅

2026-10-07；已授权 Analyst/Designer subagent simulation。

**execution_ready = false；单体数学检查通过。** 新端点入口尚未集成公共工厂、端点评价和dispatch。本结论不修改旧geometry审阅，也不改变正在执行的首4轮。当前源哈希与逐项待办见JSON。

## 已验证

- 同14发现批次、同50实际节点、同74特征，端点47项q<.05，最小1.49366e-5。effect/p/q/CI独立重算误差≤1.7e-15；这是联合latent标签的观察关联，不能宣称亲和力因果优势。
- 12项测试通过。真实teacher几何配合**非线性toy端点映射**的24组接口检查全部通过，FD最大相对误差double 4.166e-9、float32 3.967e-4。
- 每例生产predict只调用一次；head hook未被反传；pred/cond/head均脱图；零剂量步严格为0且dtype保持。这不是实际FLOWR/GPU证明。
- 当前唯一运行参考：`configs/experiments/ck2_affinity_geometry30_v1/endpoint_reference.json.gz`；SHA256 `d706173b74a937c5becac08ccbcff187f48e50ca3303d9021667e466fa5fb77c`，2,294,648字节。所有teacher endpoint、score、batch与原库逐值一致；统计表保持原值。

## 梯度和时序合同

`g_t=J_{Yworld}(x_t;fixed cond,categories,pocket)^T ∇_Y R_t(Y_t)`。输入叶节点为native坐标，reward前的scale/COM转换进入正确链式法则；affinity字段逐tensor detach且不传入loss。原生生成backbone的坐标梯度并非affinity-head梯度。

新实现一次已有target forward后计算VJP并释放图，再将detached预测交给native积分器。随后对proposal注入有界g：这是**积分前的真实VJP、积分后的滞后注入**，不保证proposal处reward单调，也不含native SDE或类别转移的梯度。

日志只记录旧端点reward及 `g·actual` 一阶估计；没有额外post-native forward，不能声称实测reward_after。动态窗口要求对应state<=.5，最后.49→.50；之后native到1。

## 调用与数值审计

生产每步保留原有target forward，加一次坐标backward，无额外生产模型/head调用；原有untarget诊断与final corrector另计。一次FD审计用2个epsilon、4次额外forward，单独记录。当前实现已在每个±epsilon前恢复共同RNG，并在末尾恢复审计前状态。

TensorDict affinity脱图和scalar shape/finite检查已修正。若未来前向本身有随机性，还需保存原forward前的RNG以复现解析导数所对应的同一随机实现；当前运行model.eval。

## 尚待完成

1. 公共factory目前不识别endpoint_*，须在R4边界按计划集成。
2. 新endpoint schema的评价必须读predicted_coords，以proposal state_time筛选50节点；当前control_representation='proposal'仅表达时间对齐，不能据此读proposal并冒充端点reward。真实proposal shape独立评价。
3. dispatch、部署来源与新Skill v2行为校验尚未启用。
4. 需完成真实FLOWR FD和100步+final的native/计算VJP但η0严格等价。启用autograd可能改变内核，旧家族零对照不能替代。

元数据缺省差异已修复：新describe现与after_native一致，默认predictive_flow。新增数据式编译器通过动态window绑定测试，并保留当前campaign的剂量caps；工厂/评价/dispatch仍需另外集成。

当前没有亲和力效果结论。新分支具备合理数学基础，但激活之前必须补齐上述集成与真实执行验证；不能以47项统计显著替代这些检查。
