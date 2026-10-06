# 纯坐标亲和力奖励：数学与数据审阅

日期：2026-10-07。身份：已授权 Analyst/Designer subagent simulation。仅本地读代码、发现集统计及 CPU 数值测试；未运行 FLOWR 推理。

**execution_ready = true**：当前参考、固定 25 个活跃槽位、三个已注册奖励和 50 节点对齐未发现未解决的重大执行阻断。此结论不代表 GPU 完整零剂量轨迹等价、亲和力有效性或 30 轮 driver 的选择政策已获验证。

当前唯一运行参考为 `configs/experiments/ck2_affinity_geometry30_v1/geometry_reference.json.gz`，SHA256 为 `7544b0b0ea4cef2abb3f599d988c6b9131c7ae2181f4b866535473de46efab08`。迁移后数据字节未变；最新 manifest SHA 为 `74d6c96af23e86172b9b6199b12c4037800674d757a2863ef1237ca0fd3a830c`。三个必需源码哈希及其余测试依赖均写入配套 JSON。

## 实际数据能支持什么

14 个 discovery 批次、50 个真实评分节点、74 个纯坐标特征。score 0…0.49 的已保存预测 head 为标签，结构目标是对应 proposal 0.01…0.50；没有 .51 或 t=1 标签。每批全窗积分后推断，不能把候选、teacher 或时间点算成独立重复。

- 74 项 BH 检验均未达 q<.05，最小 q=.0965203；15 个未校正 bootstrap CI 不跨零不能替代校正结果。
- 最强描述性信号为 pair_kernel_2.5：高低 score 组差异 −.413814 个尺度单位，CI [−.604587,−.225821]，p=.0013043、q=.0965203；14 批中12批为负。它不是因果优势规则。
- 全部 74 条时间曲线在 LOBO 0–3 次候选中选择常数。运行奖励仍使用每个实际节点，不使用拟合时间导数作空间力。
- 28 个尺度触及 1e-5 数值下限；3 个 landmark occupancy3 在teacher集合中近不变。源码把近零方差特征相关记0，缺乏覆盖标记，因此该0不能解释为真实零效应。
- 跨批最小根数从 t=0 的50降至 .1 的4、.25/.49 的1。28个teacher是14批的经验库，不是28个独立构象模式。

独立重算批次积分、p/q、CI与正式表一致（最大误差均约 1e-14）。四类特征均有真实坐标响应，但并非每个分量都提供有效信息。

## 数学与数值验证

| 检查 | 结果 |
|---|---|
| 已落库单测 | 8 passed |
| 3个奖励×4时刻×2精度，实际teacher FD | 24组全部有限；最大相对误差 double 2.392e-9、float32 2.214e-3 |
| 每组 .001 Å RMS 小步 | 注册reward全部增加，最小增量 3.9115e-5 |
| 74维特征 JVP 对独立 NumPy FD | 4时刻×2teacher；最大标准化绝对误差 1.90e-7，可辨导数最大相对误差 1.37e-7 |
| 原子标签变化 | 三家族reward差为0 |
| 槽位置换 | 同步置换current/anchor后最大误差 3.76e-13 |
| 零剂量、dtype、anchor | CPU零步严格为0，float类型保留，anchor无导数 |
| 时间对齐 | 50节点全部命中；.49→.50最后控制，之后native |

已修复并复核两项实质问题。诊断原按 score<=.5 会请求越界 proposal .51，现按实际 state_time<=.5 处理。点云旧期望cost在双teacher对称例中偏好折中位置；改为
`R=τ logsumexp(log π−C/τ)` 后，teacher处reward −.306058高于中点−.414214，离开中点也增加reward。有限温度仍可能混合邻近teacher；这不是保证完整多模态生成。非法温度/尺度/权重/ramp也已被拒绝。

## 奖励的准确解释

pointcloud先用原生预测终点和Hungarian对应取teacher，再直接比较当前proposal与相应teacher proposal。anchor、对应关系、邻居和先验在导数中固定；这是条件坐标导数，**不是 FLOWR Jacobian**，也不额外调用或反传 affinity head。

74维特征仅作为 pointcloud 的诊断输出；它的梯度来自原始匹配坐标，geometry_block_weights对该family无效。landmark家族是高低score的尺度化距离混合对比，没有Gaussian logdet，不能称为规范化密度比。occupancy3/5实际是径向壳核，不是球内计数或化学相互作用。pointcloud cost、τ为Å²；标准化特征家族τ无量纲。

三个家族均返回全部活跃槽位为core，属于全局控制；noncore=0不能证明周围区域自由。点云诊断键 nearest_standardized_rms 实际单位为Å。默认dose_gate=1，与旧架构受幅度门控抑制的同η剂量不可直接相比。

## 执行边界与解释限制

可以按已授权的探索协议继续；需分别观察实际注入、reward响应和最终head，保留完整真实zero/native控制。当前数值测试只证明坐标公式和导数可执行，不能证明提高亲和力、物理稳定性或LLM因果优势。

教师选择尚用混合单位特征距离定义第二teacher，且紧凑库缺teacher slot/root ID；后期克隆塌缩与共享oracle也限制机制解释。各节点的线性/对比方向可超出经验支持，有限步、caps和native续推不保证单调。这里没有独立能量、氢键、键/电荷或溶剂约束；终态物理表现仍须从有效图独立评价。新30轮driver和GPU推理未纳入本审阅的通过范围。
