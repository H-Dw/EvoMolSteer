# 归一化强度与分子内部距离协议

本协议是独立的 `normalized-internal-distance-guidance-v1`，入口为 `agenticgen/scripts/run_uniguide_geometry.py`，配置示例为 `agenticgen/configs/uniguide_geometry.json`。旧 UG 实现、结果目录、manifest 和失败账本不迁移。新版本使用 GD01–GD20 名称，每次结束等待主 GPT 的真实复盘。此交付不恢复 UG05，不启动标定或正式实验；λ 尚未测定。

## 条件输入

`features_path` 指向已经优化好的分子特征 JSON，无新增 LLM 改分子接口。`atoms` 按输入顺序列出唯一字符串 `id`、`element`、整数 `formal_charge`；`bonds` 列出两个 `atom_ids` 和 `single/double/triple/aromatic`；`controlled_atom_ids` 指定控制集合。示例见 `configs/examples/internal_distance_features.json`。也可用 `geometry_condition.from_molecule` 从 RDKit 分子转换；有 atom-map number 时用其字符串，否则使用补零的原子索引。

可选 `coordinates_angstrom` 与 atoms 顺序一致，是输入三维构象；有构象时不优化、不对齐。只有分子图时加氢后用 ETKDGv3、固定种子 42、单线程生成，去掉生成的氢坐标，保存实际三维几何、RDKit 版本和嵌入参数。嵌入失败退出，没有二维回退。这一用法依据 [RDKit 官方入门文档](https://www.rdkit.org/docs/GettingStartedInPython.html#working-with-3d-molecules)。

可选 `neighborhood_atom_ids` 必须给每个控制原子一个显式邻域列表；否则取分子内两条共价键以内的重原子。控制原子两两配对，并分别连接各自邻域；无自配对、重复对、蛋白原子和额外邻域内部对。可选 `atom_to_generated_slot` 必须是全部输入重原子到有效生成槽位的一对一映射；默认按稳定字符串 ID 排序分配槽位。受控几何不使用 Hungarian 匹配。当前生成背景必须有足够有效槽位；本次不改变原子数先验、不创建新口袋或初始随机流。

条件构建只读取分子特征。`generation_context_root` 仅提供已验证的模型/口袋配置及初始随机状态，可以保留 FLOWR 的口袋背景；其历史复合物坐标和亲和力不进入新损失、标定或排序。条件、参考构象和距离、单位、对应关系内容都参与冻结哈希。新条件必须使用新目录并重新标定。

## 损失、刻度与标定

损失为指定原子对距离误差平方的均值乘 1/2，距离单位为 Å；参考距离不求导，预测距离通过完整模型坐标输出向当前状态反传。原生积分器、100 步、0.2 噪声和随机流保持原样；无坐标替换或化学 inpainting。四窗口为零基步号 10–25、26–50、51–75、76–90；窗口外无引导。

Agent 只给 `normalized_guidance_scale_schedule`。每个元素是 [-10,10] 内的半整数，并在该教师实测区间内；实际值保存为 `effective_guidance_scale_schedule`，计算为 `calibrated_guidance_multiplier * 2 ** normalized_value`。零刻度等于 λ，负值仍为正向但更弱的引导。独立 `guidance_enabled=False` 才关闭；它在工程接口中单独校验，不由 Agent 假借零刻度实现。旧 `schedule` 物理强度字段不能进入新提案。

标定只由显式 `calibrate` 命令启动，沿用每教师最多 26×10、32×20、24×25、20×50 的分阶段预算，两教师总上限 5000；全部候选从原 initials 完整积分，而非复用旧坐标引导检查点。每教师完整人口常数测试中，通过有限性和沿用的绝对位移增长检查的距离误差最低配置给出 λ；完全并列选小强度。至少需要三个连续、相邻强度比不超过 2 的合格常数点。遇到失败/增长边界后不扩大实测区间；证据不足退出。可用区间取实测范围换算结果与 [-10,10] 的交集，向内落到半整数格点。常数 λ 对应的已经完成的 50 个样本直接作为测量基线，不重生成所谓新 R0。

归一化不保证稳定。每组保存所有窗口绝对增量和相对增量，最大有限位移超出已测常数包络十倍时单独标记，保留原始状态，不裁剪、不自动降强度。非有限失败仍计入预算并保留失败状态。实际 λ、可用区间和正式配置均不能在工程验收中虚构。

## 输出、历史与复盘

主指标字段是 `intramolecular_pair_distance_root_mean_square_deviation_angstrom`；另有控制原子内部、控制原子与邻域两类均方根偏差及 `maximum_atom_pair_distance_error_angstrom`。总体排序只用内部距离指标，完全并列按较小实际强度和稳定 ID 确定。构建、连通、唯一性、未评估化学身份/手性，以及质心偏移和刚性对齐诊断独立保存，不用绝对位置或亲和力排名。最大单对误差取整组最大值；构建失败仍有张量距离测量，不补样。

实验指纹包含条件、固定映射、实际强度、模型及实现、原生输出策略、初始状态、种子/槽位和人口。相同指纹复用原 COMPLETE 的逐文件哈希验证结果，REUSED 声明新尝试数 0，不算独立验证。失败或中断 claim 不自动删除/重启。完整历史纳入同一实验的兼容完成测量；小人口、刻度不兼容、关闭引导及失败记录作为非锚点证据保留。条件或采样协议不兼容的历史明确拒绝。

三份 Guidance Agent 指引为 `resources/skills/distance-guidance-{analysis,proposal,synthesis}/SKILL.md`；独立的 `distance-guidance-optimizer/SKILL.md` 供主 GPT 真实复盘。使用完整指标名称、已完成提案的预测/反证、固定 JSON 字段、精确锚点、整数化半步校验、角色分工和文字长度上限。文字长度约束不限制供应商隐藏 reasoning 的实际长度，不能据此宣称彻底消除截断。`rounds` 是外部参数，完成组数按教师数×3×rounds 计算；skills 不写固定轮数。

每版冻结三份指引、实现、配置和历史快照。两教师完成组和逐槽保留后生成 FINALIZED、四份最终策略上下文及 AWAITING_GPT_REVIEW。主 GPT 对照本版/上一版/最优兼容版/测量基线实际阅读，然后更新 1–3 个模块并绑定证据，才能接受复盘。不会自动连跑下一版。

## 命令与工程验收

在独立固定提交目录加载 DTK，设置该目录的 agenticgen/src、src 及现有 FLOWR 依赖到 PYTHONPATH，再运行：

```bash
python agenticgen/scripts/run_uniguide_geometry.py --config /absolute/new_config.json prepare
python agenticgen/scripts/run_uniguide_geometry.py --config /absolute/new_config.json acceptance
```

`prepare` 仅构建/冻结条件；`acceptance` 对两教师做真实有限差分、禁亲和力梯度、随机流和固定映射恢复比较。使用 λ=1 的工程夹具不代表已标定 λ；短轨迹不产生完整候选、不调用 LLM、不写原 attempts。单元测试必须在远端固定提交运行，覆盖距离不变性/梯度/映射、强度转换、严格输出、历史与重复复用、外部循环数和异常状态。

未来经单独授权才执行 `calibrate`、`run-version GD01`。恢复要求 manifest、条件、映射、强度绑定和随机状态相同；失败 claim 需显式审计，不能直接重复启动。当前代码交付及验收的实际进度、固定提交和测量结果应以随附验收证据为准，不将预期检查写成已通过。
