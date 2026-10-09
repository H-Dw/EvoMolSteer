# FLOWR.ROOT v2 与 v2.2：文件核查、推理差异与选择建议

日期：2026-10-09。核对官方 README、Zenodo 发布记录及源码，并只读审计现有服务器中两份 checkpoint 的超参数与张量形状。本轮没有运行生成或模型质量对照，也没有切换现有任务模型。

## 1. 当前实验的模型选择

对于论文复现、当前 EvoMolSteer 的 Steer 教师数据和 gradient guidance 对照，继续使用 **`flowr_root_v2.ckpt`**。官方明确将其定位为论文大多数结果对应的模型，推荐用于复现；现有 R26/R11 guidance 记录以及当前 Steer 数据也绑定这个 checkpoint。

**`flowr_root_v2.2.ckpt`** 是官方列出的较新联合结构生成/亲和力模型，适合作为下一组独立对照候选。目前没有找到逐版本、同输入和同采样预算的结果表，能证明它的亲和力、构象合理性、应变都优于 v2。不能仅以“最新”为理由替换当前基线，然后把差异归因于奖励函数。

来源：[官方 Checkpoints 说明](https://github.com/jule-c/flowr_root/blob/739a33c2aab74d074dd24605b9a2e34dc809ae9c/README.md#checkpoints)。

## 2. 实际文件与架构的差异

文件位置为 `/root/private_data/MolSteer/flowr_root/checkpoints/`。核查未下载权重，也未创建模型；直接解析 checkpoint ZIP 中的元数据 pickle，所有原始 pickle globals 均替换为不执行原函数的占位对象（OrderedDict 除外）。全部 state tensor 的形状都成功读取。

| 项目 | `flowr_root_v2.ckpt` | `flowr_root_v2.2.ckpt` |
| --- | --- | --- |
| 官方定位 | 原模型，论文大多数结果，用于复现 | 较新的联合生成/亲和力模型 |
| 配体 invariant width `d_inv`，CLI 对应 `d_model` | 384 | 512 |
| 口袋 invariant width | 256 | 256 |
| 口袋 / 配体 equivariant dimension | 128 / 128 | 128 / 128 |
| 口袋 / 配体层数 | 4 / 12 | 4 / 12 |
| 显式中间坐标/RBF更新配置 | `coord_update_every_n` 未提供，默认 None | `coord_update_every_n=3` |
| 显式中间坐标更新模块的实际层索引（从 0 开始） | 0–10，共 11 层 | 2、5、8，共 3 层 |
| 模型 `gen.*` state 元素总数 | 33,791,289 | 50,281,801 |
| checkpoint 文件字节数 | 677,580,035 | 1,007,216,503 |
| 亲和力预测 head | 有 | 有 |
| self-conditioning | 开启 | 开启 |
| ligand 与 pocket cross-products | 均关闭 | 均关闭 |
| 距离 / RBF / ligand-pocket RBF | 开启 | 开启 |
| 坐标尺度 `coord_scale` | 1.0 | 1.0 |
| 口袋模式 `pocket_noise` | fix | fix |

state 元素总数包含少量 buffer，不是精确的可训练参数统计。v2.2 的模型 state 规模约增加 48.8%。文件均含 optimizer state，因此约 0.68GB / 1.01GB 的文件大小不等于推理权重显存，更不等于全模型峰值显存。

权重形状提供了直接验证：`gen.ligand_dec.inv_emb.atom_emb.0.weight` 从 `[384,271]` 变为 `[512,271]`；口袋对应 embedding 均为 `[256,192]`。两版共有键中有 290 个张量形状不同；v2 还额外包含 96 个中间坐标/RBF模块 state 项，v2.2 没有新增独有 state 键。

两版的口袋编码器 141 个 state tensor 的形状全部相同，元素总数均为 4,279,105。**形状相同不表示学习到的权重数值相同。**

### 身份核验

- v2 SHA-256：`f28e863b2b208718f3d3f85f09837c2f907a71123436db6f98a25d2f1858b6a0`，与既有 EvoMolSteer 实验记录一致。
- v2.2 SHA-256：`b818f41dc12ffb6bc558bb0ad997055581e07cd9e49dcac1b794ed9993c46e4c`。
- v2.2 MD5：`d227b736471c6f6fd90488951571a6fa`，与 [Zenodo 20069589](https://zenodo.org/records/20069589) 发布文件完全一致。

Zenodo 列出 v2.2，并不意味着正文所有结果都使用该版。官方 GitHub 明确区分 v2 的论文复现用途与 v2.2 的较新模型用途。

## 3. 对三维生成最相关的机制变化

v2.2 增大 invariant feature width，使节点表征、前馈网络和生成/评分分支具有更大容量。与此同时，它减少了显式中间坐标投影及 RBF 边刷新模块。

官方 `LigandDecoder` 创建前 11 个中间层时，None 表示每层启用 intermediate update；3 表示在索引 2、5、8 的层启用。最终第 12 层不包含此中间更新模块。更新模块执行坐标残差、重新投影 equivariant features，并刷新距离/RBF边特征。

**这不是“每 3 个 inference 积分步才更新一次分子坐标”，也不是“Steer 每 3 步才重采样”。** 其他层仍执行 equivariant attention 和 feedforward 更新。改变的是网络单次前向内部的几何计算结构。

因此，v2.2 不只是把同一个 v2 多训练若干 epoch 后保存的文件；至少网络宽度和几何更新结构已经改变。其生成终点预测、内部亲和力评分、坐标响应及 endpoint Jacobian 都可能改变。对 guidance 而言，预先从 v2 轨迹拟合的时间趋势、优势区域及最优权重不保证在 v2.2 上维持相同效果。

这些结构差异并不自动证明 v2.2 更平滑、更稳定、应变更低或亲和力更高。它们提供了应当测试的机制假设，不能替代实验。

源码：[层构造](https://github.com/jule-c/flowr_root/blob/739a33c2aab74d074dd24605b9a2e34dc809ae9c/flowr/models/pocket.py)、[中间坐标/RBF更新](https://github.com/jule-c/flowr_root/blob/739a33c2aab74d074dd24605b9a2e34dc809ae9c/flowr/models/pocket_util.py)。

## 4. 对显存与运行方式的影响

两版口袋编码器结构相同；此前 CrossDocked 失败发生在口袋自编码的成对特征拼接处。仅把 v2 换成 v2.2，没有结构性的理由能够消除这一阶段的 OOM。相同 B、口袋原子数和精度下，该阶段的主要张量大小接近。

v2.2 具有更宽的配体表征，但更少的显式中间几何更新，耗时与显存影响不能只按 48.8% 权重增量等比例推算。需分别测量口袋编码、生成前向、guidance反传和终末评价。

正常官方推理的 `load_model` 从 checkpoint 的 `hyper_parameters` 读取 `d_inv`、口袋尺寸和 `coord_update_every_n` 等参数，再构建网络。通常应指定正确 checkpoint 路径，由 loader 读取实际架构；不要在自定义推理入口硬编码旧的 384 或坐标更新频率。

训练/LoRA warm-start 是另一种加载流程，需要构建参数与权重一致。v2.2 对应 `d_model=512`，不能套用默认的 384；两版联合模型的 cross-products 都关闭，不能直接照搬含 cross-products 的结构生成专用模板。完整配置还应以文件元数据和 loader 为准。

坐标更新配置也需要匹配：v2.2 为 `coord_update_every_n=3`；v2 的 None 语义是所有中间层更新，在要求整数的训练 CLI 中可用 `--coord_update_every_n 1` 表达相同模块布局。当前训练 parser 的默认值 3 符合 v2.2，却不符合 v2 的原布局。普通推理 loader 则从 checkpoint 读取此值，无需用训练默认值覆盖。

checkpoint 存储的 integration defaults 均含 100 步、linear、坐标噪声 0，但本项目 controller 可以覆盖 integrator 为 SDE。实际实验应读取最终执行配置，不能只凭 checkpoint 元数据判断当前采样噪声或算法。

源码：[官方模型加载器](https://github.com/jule-c/flowr_root/blob/739a33c2aab74d074dd24605b9a2e34dc809ae9c/flowr/scriptutil.py)。

## 5. 能确认与不能确认的训练差异

两份 metadata 的 dataset 标签都为 `combined_prot-lig`。保存的训练设置有不同学习率（v2 0.0005，v2.2 0.001）、batch-cost（3 与 2）以及 epochs 配置（500 与 100）。这些是训练时存储的信息，不是推理 Steer 的 B。

保存位置分别记录 epoch 42 / global_step 495,214 和 epoch 76 / global_step 349,424。这些计数可能受到训练阶段、批次、恢复和数据版本影响，不能单独证明 v2.2 训练更久、处理更多样本或已经充分收敛。

已检查的官方材料未提供完整逐版本训练数据清单或同协议质量差异表，因而不能确定具体新增/删除了哪些训练数据，也不能给出 v2.2 相对 v2 的亲和力、应变或 PoseBusters 改善百分比。README 对联合模型总体仍描述为未充分收敛，并指出分布外数据存在结构碰撞和亲和力适配问题；“较新联合模型”不能与另一个充分收敛的 SPINDR 结构生成专用 checkpoint 混为一体。

## 6. 与当前核心假设一致的版本验证设计

当前使用 v2 的比较继续固定模型，比较 native、Steer、gradient guidance 三个分支。这样能判断改进来自学习到的坐标奖励，而不是换模型。

如评估 v2.2，建立独立版本组：同输入、同候选预算、同共同竞争 B、同窗口、积分器和 seed 策略，先生成 v2.2 native 与 Steer 教师，再分析其坐标富集和时间变化，重新设计/校准 guidance。迁移旧 v2 奖励也可作为一个明确标记的迁移分支，但不能默认等同于学习 v2.2 教师。

建议最终比较六个主要分支：两个模型版本，各自 native / Steer / guidance。每一版内部的 improvement 使用相同评分器；跨版本时对合并后的终末分子用统一固定评分器评价，或分别用两个 head 交叉重评分并列报告。结构指标、应变计算也固定算法和失败处理。不能只把各版自己的 pIC50 均值直接相减而认定新版更强。

亲和力 head 只用于正常 Steer 评分和终末比较；这一设计不要求在 guidance 每步额外调用亲和力 head 或对其求导。坐标 reward 若使用 endpoint pullback，应使用实际运行版本的生成模型 Jacobian，并重新核验更新强度与轨迹响应。

现阶段的选择建议：**当前论文复现和既有奖励验证用 v2；新版本探索将 v2.2 作为独立对照候选，验证后再决定是否迁移默认模型。**

配套证据：[checkpoint 元数据与形状比较 JSON](flowr_checkpoint_comparison_evidence_20261009.json)。
