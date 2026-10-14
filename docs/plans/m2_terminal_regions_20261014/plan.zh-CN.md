# M2 终态 Top10 教师与 Agent 局部引导实验

## 目的与比较

从冻结 M2 出发，比较教师由终态优秀路径形成后，全局点云损失与 Agent 选择的局部点云损失；各自分别在 0–0.5 和 0.1–0.5 控制，然后原生推理到 1。只有坐标奖励改变，保持模型、积分器、温度和每次控制的 M2 剂量。两个窗口不做总剂量补偿。

原 M2/无引导 400、原 Steer 1000 的结构、评分、应变和失败记录复用。每个新程序用同一八个输入批次、每批 50，共 400；四个程序合计 1600 次新尝试。开发面板已被查看，不能视为新的独立确认。

## 数据与教师

入口 `scripts/prepare_terminal_regions.py` 接收 dataset/campaign/metrics/baseline/config/output；配置是 `configs/experiments/m2_terminal_regions_20261014/analysis.json`。按实际轨迹及完整终态标签的批次集合，以 seed42 和配置的 80%/20% 比例划分。没有默认 `range(14)`。

发现集为 0、2、3、4、5、6、7、9、10、11、12、14、15、16、18、19；验证为 1、8、13、17。根据配置 `elite_selection.top_k=10`，按有效终态预测 pIC50 排序，当前去除重复 SMILES；验证集不选教师。Top10 是预测质量排序，不代表每个分子有统计显著性。其 pIC50 范围约 8.30604–8.70481；没有按应变额外筛选。

教师保持 M2 的末学习节点端点预测表示：从这些终态通过完整 `selected_indices` 回溯，在最后学习 score_time=.49 提取真正祖先的 predicted_coords。它不是终态解码构象，也不是实际含噪 X_.49。10 条终态路径合并成 9 个真实祖先教师；每个教师保留真实 batch/step/slot/root/path_ids。`elite_path_locators.parquet` 保存 500 条学习节点定位信息，坐标留在原始轨迹中。

验证所有时刻的根传播、offspring_count、非选择 identity，以及 current[j+1,d]==proposal[j,selected_indices[j,d]]。不能用同一个 slot 跨时间拼接。

## 数学挖掘与紧凑存储

20 个受体坐标 landmark，各自提取 Gaussian soft mass、三维局部均值、六项协方差，共 200 个坐标特征。全窗逐节点计算：Top10 祖先与有终态的普通祖先对比；以及控制即时评分后的未来终态信用秩相关。未来信用是各祖先可观察后代中每个独特图最大终态分数的均值；灭绝节点保留缺失标签。

根内平衡后以批次推断。一个根中的分叉可参与描述，不将分叉当独立重复；共享早期祖先会导致缺失对比。只积分相邻、同时可观察的边，不跨缺失阶段插值。统计输出包含效应、批次支持、bootstrap CI、p/q、经验趋势及时间变化率。每个 split 对两个统计族的全部 400 项统一 BH；验证统计不进入 Agent 选区。

存储为紧凑的批次宽表、全窗效应、节点可观察性、经验曲线和一个小型教师库，无逐粒子特征缓存。首版要求两个根才计算相关，丢掉了实际收缩到单根的后期分叉，已修订。角色调用读取 mining_v2；正式归档 mining_v4 增加了稀疏路径定位与最终源码/配置绑定，其统计及教师与 v2 的逐字节一致性单独核验。

## 真实 API 工作流

使用现有 `deepseek-flash`，API 温度默认 0；不调用代理 subagent。新增可选 `terminal-region-analyst` / `terminal-region-designer` Skills 与通用 Skills 拼接，环境设置不写入 Skills。

Analyst 使用 read_region_catalogue/get_curve/query_evidence 等受约束工具。先提供全部区域的紧凑统计，再要求实际比较至少三个区域的三条曲线。每个假设必须引用其准确的曲线 key 和真实 handle。Designer 只能从 Analyst 假设涉及的区域中选最多三个，并逐项绑定 hypothesis/evidence；格式由 schema 与确定性校验提取，不执行 LLM 返回的任意代码。

实际 v1 调用读取区域 19 soft_mass、8 local_dz、0 cov_xy。Designer 选择区域 19、半径 4 Å，明确标为 exploratory。最小部分相关 q≈.112，区域 19 soft_mass q≈.199；不能称为校正后显著。其文字中的“q 未按全目录校正”与实际计算不一致，正确口径是已对 400 项统一 BH；原文保留并在报告中纠正。点云匹配也不保证特定 soft_mass 必然增加，需在生成中观察实际响应。

选区后的程序直接编译，未按生成成绩修改选区。保留 literal responses、wire requests、模型回执、证据 hash、工具操作和最终程序 hash；不保存私有推理链。

## 局部奖励

对教师 T_k 的原子，a_ki=Σ_r w_r[max(1−||T_ki−p_r||²/l²,0)]²。未选区外 a=0。

Hungarian 配对与教师邻域选择使用区域加权距离，赋值、空间权重和先验在本次导数中冻结。q_k=Σ_i a_ki||Y_i−T_ki||²/Σ_i a_ki；R=τ logΣ_k π_k exp[−δ²(sqrt(1+q_k/δ²)−1)/τ]。

复用单次 FLOWR forward，对奖励执行真实 endpoint VJP。无 affinity-head 求导、无额外逐步 affinity 调用、无重采样或图锁定。选区外的端点损失直接梯度为零；FLOWR Jacobian 仍可能把区域力传到其它当前坐标，这是模型耦合，需要单独报告。剂量与温度保持 M2：eta .33、邻居 4、tau .5、几何先验温度 4 Å²、beta0、delta 默认1。

程序通过 reference 裁剪控制窗口：完整窗口实际节点0…0.49，共50次；延迟窗口0.10…0.49，共40次。保留 source_learning_window 以区分完整发现证据与控制子窗口。

## 验证、执行与评价

本地数学与谱系测试已通过，包括变动批次集合、配置 Top-k、真实逆索引、缺失边积分、区域外零端点力、有限差分及动态窗口。

因为当前源码与固定对照生成版本不同，先用一个50输入批次运行 M2/无引导/gradient_zero 桥接。必须验证新零剂量与新无引导相等，并且 M2/无引导终态评分与固定控制记录在1e-5内一致，才允许继续复用旧控制。150次桥接是兼容性验证，不是重复生成完整400对照。

代码本地 commit/push 后远端 exact pull，使用 flowr_root_v2.ckpt。远端只推理和记录；完整100步之后导出紧凑评价包，再清理本实验拥有的密集轨迹目录。原 Steer、checkpoint、仓库和其它任务保持受保护。候选失败单独记录并跳过；桥接失败则阻止错误的旧对照复用。

结构、PB、收敛 MMFF 应变均在本地用旧评价协议计算。报告全部尝试的最终预测 pIC50、配对批次 CI、高亲和力产率、独特分子/姿态、有效率及应变覆盖。比较四个新程序对 M2/无引导、局部对全局、延迟对完整，并单列原 Steer 非配对参照；尾部最大值不作唯一判断依据。
