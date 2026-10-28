# 模板相对距离奖励与 Guidance skills 2.2.0 接入

日期：2026-10-29，Asia/Shanghai。科学实现基线为 `511a09b4`；本次从已包含工具会话的 `1f1bb7c4` 继续实现，没有回退 `e3fe9086` 的多轮工具调用系统。

## 检查结论与改动边界

原 geometry 入口已经对照模板计算原子对距离误差，并非绝对坐标误差；但内部与两键邻域混入同一个均方损失，没有三个独立开关。批量执行还持有一份重复的损失公式。旧通用 `sampler.Control` 中的坐标损失仍供历史协议使用，本次新 geometry 入口以 `GeometryControl` 覆盖该方法，不修改历史 UG 结果。

新配置协议为 `template-relative-distance-guidance-v4-binary-losses`，默认加载不可变 release `2.2.0` 的 analysis、proposal、synthesis、computation 四份运行时技能。私有提案加载 analysis + proposal + computation，综合加载 analysis + synthesis + computation。每教师每轮共享最多 30 次 HTTP 模型调用的外部预算继续有效，工具消息可在最终 JSON 前往返多次。外部实验后 RSI 技能独立保存，不进入 Guidance Agent 运行提示。旧 2.0.0、2.1.0 文件不覆盖；2.2.0 带父版本与当前资源哈希。

本次只完成代码、技能和本地验收，没有恢复原 UG05、运行新的分子生成或部署远端执行目录。原实验与自动化状态未改。

## 损失的精确定义

对每个样本，在网络预测终态坐标上计算以下三组距离。参考距离均来自输入 template 的三维结构，单位为埃；不把目标距离设为零，也不做坐标替换。

| 分量 | 原子对 | 默认权重 |
| --- | --- | --- |
| `condition_intra_loss` | 条件区域内所有不重复原子对 | `condition_intra_weight=1` |
| `condition_outer_loss` | 每个条件原子与模板中所有非条件配体原子的组合 | `condition_outer_weight=0` |
| `interact_loss` | 明确指定的配体 interact 原子与对应固定靶点原子 | `interact_weigh=0`，提供非空合法对应关系时为 1 |

每个分量分别按自己的原子对数求平均，避免外围对数量多而自动支配内部目标：

\[
\text{component_loss}=
\frac{1}{2\,\text{pair_count}}\sum_{(i,j)}
\left(\frac{\text{predicted_distance}_{ij}-\text{template_distance}_{ij}}
{\text{normalization_length}_{t}}\right)^2.
\]

空分量为零。总损失为

```text
condition_intra_weight * condition_intra_loss
+ condition_outer_weight * condition_outer_loss
+ interact_weigh * interact_loss
```

代码沿损失的负梯度引导；若用“奖励”表述，奖励等于总损失的负数。三个分量权重仅接受整数 0、1，拒绝布尔值、分数和字符串；它们与四个时间区间的 guidance scale 是两类独立控制量。总梯度仍经过完整模型 VJP 回传到当前坐标，并沿用单候选最大原子增量 3 Å 的限制及原始未限制增量记录。

条件内部/外围损失对配体整体刚体变换不变。互作目标使用固定靶点坐标，因此移动配体而不移动靶点会改变互作损失。空 interact 集合表示未指定互作目标，零损失不代表已满足靶向结合。距离约束也不独立保证互作角度、化学类型、手性或结合活性。

外围范围现在是**所有具有模板对应关系的非条件配体原子**，不再仅限两键邻域。没有模板对应关系的额外生成原子不被虚构参考距离；要约束它们，输入模板及映射必须包含对应原子。新协议要求显式模板三维坐标，不以重新嵌入的构象冒充输入模板。原子与生成槽位映射冻结。

## 归一化方案及依据

默认 `bounded_step_mean_detached`：

1. 模板基准长度是模板全部不重复配体原子对距离的平均值，下界为 1 Å。
2. 当前尺度是**当前积分状态**中全部有效配体原子对距离的平均值；不包含靶点、padding 或对角线，不混合不同候选。
3. 当前尺度限制在模板基准长度至其 4 倍之间，随后停止梯度；用该长度归一化距离误差，再平方。

分母不使用预测终态。原始模型 VJP 保留，只有尺度分支 detach。这防止优化器通过扩大分母降低目标，而非修正距离；模板下界避免早期原子塌缩使梯度骤增，上界避免少数极端距离无限削弱约束。4 倍上界和 1 Å 下界是本次可配置工程默认值，**不是论文证明的最优阈值**；分母可能受离群原子影响，边界与增量限制也不构成稳定性保证。

文献与代码检查：

- [UniGuide，arXiv:2501.02526v1](https://arxiv.org/html/2501.02526v1)使用几何条件损失并经去噪预测回传引导；本次沿用完整反传原则。其对齐坐标目标与本次三项模板距离目标不同，不能把这里的动态均值归一化说成原文方案。
- [AlphaFold2 官方 FAPE 实现](https://github.com/google-deepmind/alphafold/blob/main/alphafold/model/all_atom.py)对局部坐标系误差使用长度尺度和有效项数归一化，并支持误差截断。这里借鉴量纲与数量归一化；没有引入会使大误差区间梯度为零的损失硬截断。
- [RFdiffusion 官方 potentials](https://github.com/RosettaCommons/RFdiffusion/blob/main/rfdiffusion/potentials/potentials.py)的接触势用距离偏移和固定长度尺度构造无量纲变量。它优化接触数量，不等价于精确模板距离匹配。
- [Boltz 官方 DistancePotential](https://github.com/jwohlwend/boltz/blob/main/src/boltz/model/potentials/potentials.py)提供原子对距离及其坐标梯度，配合上下界势约束。这里采用模板距离误差，没有把 flat-bottom 容忍区直接搬入目标。

以上来源支持“明确距离量纲、按有效项数归一化、保持可微坐标约束”的选择；没有直接验证当前 step 平均配体间距对 FLOWR 早期波动的实际收益。为便于后续对照，另外提供 `template_mean`（固定模板尺度）和 `none`（原始半均方距离误差）模式。严格说 `none` 模式的损失单位为 Å²，其余模式无量纲。报告原始埃单位距离均方根偏差、未归一化损失及归一化分母，不通过比较不同尺度下的归一化损失宣称质量改善。

配置、模板与对应原子、归一化策略、源代码哈希参与采样兼容性绑定。新协议必须重新进行每个教师自己的标定；默认搜索中心不是跨教师平均值，也不是已测最优 scale。旧标定、旧缓存不会被混用或改写。

## 输入互作对应关系

在现有 features JSON 的 atoms、bonds、controlled_atom_ids、coordinates_angstrom、atom_to_generated_slot 等字段之外，可加入：

```json
{
  "coordinate_frame": "generation_context_angstrom",
  "target_atoms": [
    {"id": "target_atom_1", "coordinates_angstrom": [0.0, 3.0, 0.0]}
  ],
  "interact_atoms": [
    {"atom_id": "template_atom_1", "target_atom_ids": ["target_atom_1"]}
  ]
}
```

这是格式示例，不是真实教师坐标。配体模板坐标和靶点坐标必须处于生成上下文的同一坐标系；输入需明确声明，代码不会猜测变换。一个 interact 原子可以对应多个靶点原子。未知 ID、重复原子对、非有限坐标或重合的互作参考点会被拒绝。`interact_atoms` 省略或为空时无需靶点原子，互作损失为零。保留用户指定接口名称 `interact_weigh`。

## 外围开关的完整执行链

注册工具 `set_guidance_loss_weights` 接受 anchor_id、三个二值权重、evidence_ids、reason。它返回候选控制回执而非全局配置更新；状态明确为 `prepared_not_executed`。

1. Agent 先读取已有统计回执；缺少决策证据时才调用计算工具。
2. 对内部几何不再明显改善的假设，使用与锚点有关、兼容随机人口的非重复配对比较。工具不凭空设定“停滞”的数值阈值；证据不足必须记为不确定，不能宣称已确认停滞。
3. 请求 `condition_outer_weight=1`，说明预期外围变化以及内部几何退化的否定标准。默认保留内部权重 1；互作权重只在有明确原子对时启用。
4. 提案沿用原角色字段，只额外允许 `loss_control_id` 引用完整控制回执 ID。纯分量切换允许 `updates=[]`，避免强制同时改时间权重而失去对照意义。
5. 综合角色选择已有候选。执行器再次核验回执来源、锚点、条件、权重、合法性与重复配置，再传给单组或混合批量采样。
6. 实际条件损失计算读取候选权重；候选身份、预约账本、manifest、失败记录、结果与历史均包含分量权重。同一时间 scale 但外围开关不同是不同配置，相同完整配置仍复用并禁止重采。

`inspect_interval_evidence` 可读取经哈希核验的 trace，返回三个分量、未归一化值、分母和权重的分窗口诊断；无有效回执时返回不可用。统计/诊断进入当前任务经验库，不修改实验内 skills，不向用户询问。

skills 主要关注条件原子间距离均方根偏差，执行器新协议的最佳配置选择也优先采用该指标，再以较小实际 scale 和来源 ID 打破平局。外围及互作指标分别报告，不能用大数量外围原子对的平均数替代主要指标。仅当模板没有条件内部原子对时，明确退回全部约束原子对指标，不能把缺失内部指标记为零。

## 验收与实际限制

验收包括模板处零损失、刚体不变性、非零模板距离、互作默认权重与梯度方向、空集合、非法权重、当前尺度停止梯度及边界、padding 排除、单组/不同权重混合批次梯度一致、权重身份去重、真实工具消息协议的多步模拟、回执恢复与篡改拒绝、最终角色选择到 runtime executor 的传递，以及旧协议回归。

原始 raw999 积分 0–0.999、100 个区间、最后循环覆盖前输出、随机流、四个时间窗口和增量限制保持现有实现。未来真实模型验收入口也包含两种分量权重的混合批次与独立批次比较；本次未执行 GPU 采样。具体测试结果及源码哈希记录在相邻证据文件 `../evidence/uniguide/20261029_template_relative_reward_validation.json`。模拟 HTTP 回答只验证控制链，不代表真实 LLM 决策质量或分子优化效果。

最终本地验收：六个相关测试模块共 108 项通过（46.62 秒）；五份标准 SKILL.md 格式校验通过；wheel 构建通过，16 份 release 文件及修改后的打包源码逐字节核对。额外覆盖了时间窗口动作耗尽时仍允许进入已注册分量工具阶段，以及主指标配对测量缺失时拒绝激活外围开关。最终动作仍须经过合法性和新颖性校验，不因工具可用而自动补造候选。
