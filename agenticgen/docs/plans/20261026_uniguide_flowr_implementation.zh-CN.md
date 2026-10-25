# UniGuide–FLOWR Guidance Agent 实施协议

此协议替代 scale50 旧计划中的条件、奖励、尺度和主版本规则。新分支为 `codex/uniguide-flowr-agent`，新版本为 UG01–UG20。工程组件可复用；策略只提取 V17 commit `a245eb0ac2ea4a7944d4fd7acfcd3491eaf5dbed` 的行动锚点、私有分析与汇总、可反驳机制分析思想。UG_INIT 不含旧实验分数和教师最优答案。

## 冻结输入与条件

按最终确认，每个 target 采用所列候选中已存重评分最高的一位教师：

| target | node_id | 重原子 | pIC50 | archive |
|---|---|---:|---:|---|
| 6pyy_H7S_A_402 | single_b004_final_015 | 17 | 6.5355939865112305 | targets_0027.tar.gz |
| 3w54_RNB_A_504 | single_b000_final_003 | 22 | 8.71163272857666 | targets_0009.tar.gz |

教师必须按对应批次 SDF 的 node_id 属性唯一提取，核查成员 SHA256、已有重评分、SDF 与 native 的坐标和元素映射，保留 root_slot 与文件 slot 的区别。PLIP 直接报告的配体重原子取并集；环和盐桥使用其直接报告的列表，不添加一跳原子。空 core 或映射失败停止准备。

固定蛋白世界坐标系。在当前模型预测干净终态上，以几何 Hungarian 对 core 与有效生成槽位做注入匹配。离散分配、条件坐标和选择 mask 全部 detach；未匹配位置的 condition map 等于当前预测。损失为 `sum(||xhat1-y||²)/(2K)`，坐标单位 1 Å。仅经当前一次模型前向对当前带噪坐标求导；保留全部有效配体坐标上的 Jacobian，不在反传后只保留 core 梯度。

## FLOWR 适配

使用指定 v2 权重 `/root/private_data/MolSteer/flowr_root/checkpoints/flowr_root_v2.ckpt`，SHA256 为 `f28e863b2b208718f3d3f85f09837c2f907a71123436db6f98a25d2f1858b6a0`。代码位于 `/opt/MolSteer/flowr_root`，源码 hash 在配置中固定。

原生关系是 `v=(xhat1-x)/(1-t)`、`q=(t*v-x)/(1-t+1e-6)`、`b=v+gamma*q`。定义 `q_guided=q-S*g`，其中 g 为坐标损失对当前 native 坐标的梯度，则增量为：

```text
delta_native = -dt*S*((1-t+1e-6)/t + gamma)*g
```

这是对原生 score proxy 的代数适配，条件 likelihood 仍为近似。坐标尺度已进入损失 Jacobian，不额外沿用旧 `/coord_scale²`。原生积分及噪声、离散类别转移和 self-conditioning 均保留，在积分之后加入 delta，不把修正后的 endpoint 再写回 self-conditioning。没有裁剪、梯度单位化、自动缩小尺度或较弱重试。非有限数保存证据并停止受影响执行。

活动 step 标签为 10–25、26–50、51–75、76–90，其他 step 为零。损失函数不换阶段；匹配和条件映射逐步更新。原生最终输出包含额外预测校正，需与 step100 的积分状态分别保存。

## 标定与预算

共同随机数采用五种子 42、100045、200048、300051、400054。正式每种子使用原始前十槽位；原生随机调用保留20行，丢弃的随机填充不是候选。历史 prior 逐字段与历史 HDF5 比对。

远端工程验收发现3w54的50行网络调用超出64 GiB显存，因此逻辑人口仍为50，网络前向、pocket编码及VJP分为固定10行物理批；不足10行时填充并丢弃填充输出。原生积分保持逻辑人口，随机调用仍按每种子20行。每步保存原生随机流摘要，跨配置及不同标定人口核查消耗一致。50行一次调用的失败属于工程验收，未计为科学候选。CPU线程固定为8，缓解原默认257线程造成的小批操作开销。

| 阶段 | 最大尝试数 |
|---|---:|
| 26 个常数 × 两教师 × 10 | 520 |
| 每教师最多32个新增常数 ×20 | 1280 |
| 3共同锚点 ×4窗口 ×2方向 ×25 ×两教师 | 1200 |
| 最多20共同配置 ×50 ×两教师 | 2000 |
| 标定合计 | ≤5000 |
| 共享 R0 | 100 |
| 20版本 ×两教师 ×5轮 ×3方向 ×50 | 30000 |

粗标定覆盖 `{10^k,3*10^k:k=-6..5} ∪ {10^6,50}`。正式初值由全人口共同常数的等教师权重指标选择：以最小 C 加0.02 Å为锚定候选带，再比较平均绝对亲和力差。共同域覆盖两教师常数 C–gap 非支配区域，并各增加一 octave 余量；初值必须有双向合法动作。

正式尺度为 `2^(j/2)`，粗调 ×2 或÷2，细调 ×sqrt(2) 或÷sqrt(2)。最多改两个窗口且单次只用一种步幅。增强只增，减弱只减，重分配恰好一增一减。非法、重复、越界提案拒绝，不替 Agent 四舍五入。失败尝试计入预算，不补样。

每教师三私有 Agent 各提两案，汇总每倾向选一案，不能创造数组。默认行动锚点为本分支最近实测配置；可以显式改选已有实测锚点。全部物理起点固定为共享 R0 的 step10 前状态。第5轮结束仅分析归档，不产生 R6。每版代码、skills 和科学配置冻结，任务记忆每版清零，标定知识可保留。

## GPT 主复盘

每版质量检查、结构保留和 FINALIZED 完成后，当前 Codex 主 Agent 读取30组聚合结果及两教师各自代表私有分析和同轮汇总，共4份上下文。比较当前、上一版、UG 系列最优与 R0。

任一教师满足以下任一项就检索 primary papers 和官方代码：C 与绝对 mean-pIC50 gap 变化同时不超过0.02；15套配置中至少80%与前版重复；代表配置四窗口的完整 guidance 增量向量变化全部不超过5%。向量分母为旧、新 RMS 与1e-8 Å的最大值；缺失是 unknown。阈值用于触发研究，不作为显著性或最低改进门槛。

每次更新1–3个 skills 模块，记录观察、证据、机制适用性、具体编辑、预期行为、下版反证条件。文献需有 URL、版本/commit、访问时间及适用假设。正常策略优化不改 loss、梯度路线、原生适配和尺度域；确认的实现问题单独修复、远端验收并标记受影响运行。

主版本回退按等教师权重 C 判断：ΔC>0.02 为下降，ΔC<-0.02 为改善，在带内再比较平均绝对 gap。连续两个相邻比较下降，则下一版从已有最优 UG checkpoint 的 skills 出发，重新提出1–3项改进，不额外重跑最优版、重算其分数或 R0。只有恢复派生版本实际完成才重置下降序列。UG20 触发时仅报告，不启动 UG21。

## 部署与运行

代码在本地修改、提交并推送 GitHub；远端只执行固定提交。工作目录 `/workspace/EvoMolSteer`，结果目录 `/workspace/agenticgen_uniguide_flowr_20261026`。测试只在指定远端运行，不升级 Torch/DTK。

GitHub 网络异常依次尝试 ghfast.top、gh-proxy.com、ghproxy.net、gh.llkk.cc，每个最多120秒，每次独立 clone，成功即止。检验固定 commit 与 ancestry；镜像不接收 GitHub 凭据。鉴权失败不作为镜像绕过理由。

运行入口：`bash agenticgen/scripts/run_uniguide.sh prepare|init-strategy|acceptance|calibrate|run-version UG01|status`。每版结束写 AWAITING_GPT_REVIEW；主 Agent 完成实际研究及本地 skills 编辑后，提交、部署，通过 `accept-review UG01 review.json` 绑定证据与修改，再启动下一版。DeepSeek 从环境变量读取，不写入仓库。

验收包括固定匹配有限差分、FLOWR score 代数、尺度线性、无裁剪、原生短轨迹零增量等价、step10 精确恢复、随机数和正式批形状、预算和提案契约、停滞向量方向变化以及连续下降回退。没有新的完整零引导科学对照组。保留所有尝试指标、提案、LLM用量、证据、每槽 TOP1 及完整代表50分子；清理不得早于质量与保留清单核验。

## 方法来源

- UniGuide NeurIPS 2024: https://arxiv.org/html/2501.02526v1
- 官方补充代码: https://proceedings.neurips.cc/paper_files/paper/2024/file/faa6276ea12d7afeb3e42b210c86f688-Supplemental-Conference.zip
- Flow matching guidance: https://arxiv.org/html/2502.02150v2
- PyTorch 2.9 autograd: https://docs.pytorch.org/docs/2.9/notes/autograd.html
