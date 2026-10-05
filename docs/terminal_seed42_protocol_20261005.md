# 局部选择规则与终态兼容性：seed 42 实验协议

本次从第 1 轮重新计数，上限 5 轮，可按反馈提前停止。每轮必须先清理远端上一轮生成数据，并在本地保留报告、失败指标、冻结配置和代码哈希。原始 Steer、原始无引导参考、输入和模型 checkpoints 保留。

## 要检验的假设

原始 Steer 通过候选淘汰和复制改变群体。大量克隆可提高部分评分或保留某些区域，但限制有效独立探索。我们检验：LLM 从选择关联中提出少量局部可接受条件，借助 FLOWR 的坐标 Jacobian 将偏离这些条件的状态推回范围，并让其余自由度继续由生成模型协调；其结果是否在保留局部优势的同时获得更多不同且兼容的终态结构。

这个假设具有可行的控制路径，但目前没有因果保证。选择偏好可能受初始根谱系、全局化学和模型评分共同影响。两个区域距离不能唯一决定奖励，更不能证明获得最优路径。局部奖励也可能通过全模型 Jacobian 影响远端原子。因此，局部目标改善、周围区域兼容性、多样性、终态应变和亲和力预测都要分别观察。

## 奖励及时间

学习参考是原始发现集 0–13 批的同刻 predicted endpoint 表示，选中概率按批次等权。ASN117 与 VAL116 的 N/O/S 归一化 soft-min 距离合为一个二维相关区域。每时刻冻结选中均值、无引导批内协方差及选中分布 50% 质量半径。协方差有显式的数值收缩与下限。该椭圆可能跨越多峰空隙；它是可证伪的可接受区间假设，不能冒称完整分布恢复。

令 `q=(z−μ)^TΣ^-1(z−μ)`、`v=max(sqrt(q/r²)−1,0)`，奖励为 `R=−(sqrt(1+v²)−1)`。椭圆内梯度为零；缺少 N/O/S 支持时记为缺失，零注入，不能视为满足。通过 live FLOWR endpoint 对当前坐标的真实 Jacobian 拉回梯度。模型参数冻结，硬类别掩码 detached，不声称直接优化离散图。

注入幅度由 `η*native_RMS*v/sqrt(1+v²)` 决定，随后执行逐原子上限、累计预算与几何回溯。接近合格边界时实际请求趋于零，避免微小梯度被归一化成固定干预。初始 `η=.10` 是有界探索值；后续内部区间和外部强度分别修改，每轮保留唯一主要变化及剂量审计。

学习窗口由参考文件动态指定。本例是 `[0,.5]`，使用原 100 步网格，最后注入 `.49→.50`，随后原生推理完成到 `t=1`。原生类别抽样和 SDE 保留；全部新组关闭 SMC。窗口后的末端参考仅用于观察局部效应是否保留，不用于施加干预。

主 seed 为 42，各批采样 seed 固定派生为 `42+100003*batch_index`。所有实验臂严格共享初态与派生 seed。新增批次是固定 seed 下的扩展，不等于未见 seed 或跨靶点验证。

## 终态测量

全体候选均进入分母，包括无法解码、无法 sanitise、断图和能量不支持的候选。原始生成 pose 用共同本地 RDKit 与 PoseBusters `dock_fast` 检查化学、键/角/环几何和蛋白碰撞；这是结构子集，不冒称含全部能量测试的 PB-valid。方法来源：[PoseBusters](https://github.com/maabuu/posebusters)。

能量采用同图 MMFF94s 的局部松弛：先仅调整新增氢、固定生成重原子，再释放同图全部原子，计算两者能量差和每重原子差。仅在两阶段都收敛时进入正式能量汇总，所有不支持、失败和不收敛记录保留。原生成坐标从不被评价程序修改。该值反映同图局部应变释放，不是全局最低能、蛋白结合能或自由能。实现依据：[RDKit force-field helpers](https://rdkit.org/docs/source/rdkit.Chem.rdForceFieldHelpers.html)。

预测亲和力同时保存上游值和在相同终态坐标、类别上重新计算的 FLOWR target/off head。它是模型预测，没有独立实测验证，不能单独证明真实结合提升。报告全候选、有效候选和去重候选的分数及数量，避免通过删除失败或复制高分子抬高均值。

自由度代理包括 canonical isomeric SMILES 的独特产率、scaffold 数和去重指纹多样性。去重按首次出现 pose，不能选每个图的最高分 pose。周边区域按配体重原子是否位于 ASN117/VAL116 受体原子 5 Å 内划分，分别记录碰撞和松弛位移；这是空间诊断区域，不是跨不同分子可追踪的固定原子身份。

## 代码和调用

- `scripts/build_local_reference.py`：本地提取连续区域参考。
- `scripts/generate_local_flowr.py`：通过指定 FLOWR 程序路径与 checkpoint 执行局部梯度推理。
- `scripts/scnet_terminal_experiment.sh`：固定主 seed 42、batch=50、100 步、终态解码导出。
- `scripts/check_terminal_round_ready.py`：启动前验证上一轮已删除、报告存在、保护输入存在、轮数不超过 5。
- `scripts/audit_local_execution.py`：本地检查全程完成、窗口后零注入、无重采样、初态配对、每时刻区域变化和非区域干预。
- `scripts/evaluate_terminal.py`：本地终态结构、能量、区域和模型预测指标。
- `scripts/retire_experiment_outputs.py`：按精确绝对路径计划清理，要求报告在删除范围外，保护原始参考和源码。

```text
python scripts/generate_local_flowr.py --flowr-root <FLOWR_DIR> --input-dataset <ORIGINAL_DATASET> --root <NEW_DATASET> --checkpoint <CHECKPOINT> --program <PROGRAM_JSON> --reference <REFERENCE_GZ> --campaign <ROUND> --n 100 --batch 50 --seed 42 --steps 100 --arms gradient --export-terminal
python scripts/audit_local_execution.py --dataset <NEW_DATASET> --campaign <ROUND> --reference <REFERENCE_GZ> --output <REPORT_DIR>
python scripts/evaluate_terminal.py --dataset <NEW_DATASET> --campaign <ROUND> --reference <REFERENCE_GZ> --output <REPORT_DIR> --arms gradient --batches 0,1
```

每轮保留可审阅的证据、公式、参数来源、备选方案和选择理由，不保存私有思维链。所有源代码在本地修改、提交并通过 VPN push，远端 pull 后运行。远端执行模型推理、解码和模型 head 导出；科学评价在本地完成。
