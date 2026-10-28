# 按教师编排候选批：实现与远端验收

最终审计时间：2026-10-28 19:40:14，Asia/Shanghai。

实现固定提交：`bf9603103cecb37e2b390d7c812472c2eda008b9`，已推送 `codex/uniguide-flowr-agent`。远端独立目录为 `/workspace/uniguide_audits/candidate-batch-bf960310`；验收产物为 `/workspace/agenticgen_geometry_acceptance_20261028_bf960310`。本报告及证据的后续文档提交不覆盖该固定执行目录。

## 已实现的执行方式

每个教师单独编排 `candidates_batch.json`，保存每个候选所属配置、随机种子、逻辑槽位、条件绑定、归一化窗口刻度、教师标定倍率及实际强度。预实验在同一筛选阶段汇集不同强度，正式 Agent 在三个方向的提案都通过校验后汇集其候选。不同教师不混批，也不共用容量记录。

首次容量上限为 100。待执行量小于 100 时尝试一次承载全部候选；明确发生设备 OOM 后，按严格小于当前容量的 80、60、40、20、10 阶梯回退，例如 90→80、50→40。容量按教师和设备环境保存，后续优先沿用已确认的容量。三个方向各 50 个候选时，初始编排通常为 100＋50；命中已有槽位则只安排缺少的槽位。

OOM 恢复原外部随机状态，从相同初态重放尚未保存的候选，不调整实验参数，不重复扣除科学预算。完成部分产品后在后处理发生 OOM 会停止，保留产品供审计，不重新采样。非有限数值不进入 OOM 回退流程；失败单元保存异常张量及失败标记，并保持计数。当前最小不可拆分保存单元为 10 个候选，即五个种子各两个固定槽位；单元数值失败时整个单元作为失败保存，不补样。

**候选批容量与物理网络 batch 不相同。** 真实模型测试发现，直接将网络 batch 从 10 扩大到 20，尽管初态、映射及随机流相同，第 0 步网络预测仍出现浮点差异，随后被积分和离散采样放大，终态坐标及化学状态不再一致。因此最终方案固定每个网络／反传单元为 10 个样本，通过独立 GPU stream 并发执行多个单元。100 个候选对应最多十个并发单元，不是一个 100 行网络矩阵运算。未通过的物理 batch 扩大方案及其原始比较证据均已保留。

每个候选仍使用自己的窗口强度、固定映射和原生随机流。反传对逐分子损失求和，不对候选数量取均值。改变同批候选数量不会改变单个候选的引导尺度。恢复检查绑定候选身份、参数和随机状态。

## 远端验证结果及适用范围

本地未运行测试。固定提交在远端运行以下七组单元测试：geometry、geometry_batch、joint_endpoint、chemical_inpainting、prompt_policy、anchor_policy、prompt_version；结果为 **109 passed，8.91 秒**。

| 验证项目 | 6pyy_H7S_A_402 | 3w54_RNB_A_504 | 证据范围 |
|---|---|---|---|
| 两种强度、不同槽位：独立 10＋10 对比并发 20 | 完全一致 | 完全一致 | `82dc89c1`，完整 100 步、终态全部张量、轨迹全部字段、随机状态 |
| 并发 20 的断点恢复 | 完全一致 | 完全一致 | `82dc89c1`，从第 10 步恢复至第 100 步 |
| 90 候选显存探测 | 90 OOM→80 OOM→60 成功 | 90 OOM→80 OOM→60 OOM→40 成功 | `82dc89c1`，12 步前缀与独立执行完全一致 |
| 正常 10＋异常 10 同批执行 | 正常单元完全一致；异常单元保留失败 | 正常单元完全一致；异常单元保留失败 | 最终 `bf960310`，正常单元完整 100 步对照独立参考 |

显存探测的 60／40 是本次设备和短程探测的观测值，不是整个 100 步的预先保证，也不是所有教师或硬件的固定容量。实际运行仍按 OOM 阶梯调整。本次未做吞吐量基准，不宣称已获得某个加速倍数。

完整的 `82dc89c1` 真实模型验收还包含两教师的坐标 VJP、关闭引导、原生随机槽位升级及恢复检查。最终提交与该验收在科学源码绑定中只有 `geometry_batch.py`、`sampler.py` 两个文件不同：变更为失败单元的结构掩码保留、已完成批的显存引用释放，以及诊断溢出的处理。最终提交重新测试这些改动经过的真实混批异常路径；未修改路径通过源码哈希和前次验收回执显式复用，**没有把复用证据描述为全部测试在最终提交重新运行**。细节见 `final_acceptance_bf960310.json` 的 `evidence_reuse`。

验收期间另发现：非常大的有限位移在 float32 范数归约时会溢出为 `inf`，导致诊断 JSON 无法写入。最终只对溢出的诊断项使用 float64 重新归约，采样坐标、梯度和积分公式不变；无法恢复的诊断明确记录为空并附异常说明，原始异常张量保留。不以限幅或减弱强度掩盖失败。触发这一问题的 `886f9572` 验收日志保留。

最终混批异常验证使用两教师各 20 个工程候选，共 40 次工程尝试；其中每教师 10 个正常候选、10 个故意使用极端强度触发数值失败的候选。前次完整混批／恢复对照另有两教师各 60 次轨迹求值，以及原基础验收和短程容量探测。它们均位于独立工程目录，不用于拟合 λ、科学排序或正式验证统计。没有 LLM 调用，没有恢复 UG05，没有启动新的完整标定或正式版本。

最终进程已退出，执行锁空闲，固定 checkout 干净。19 个受保护源文件与验收前逐字节一致；原实验 `attempts.json` 的 SHA256 仍为 `c12fb91e4e2d496f9669449dc4b69ab4e7dac56a68a90af46699e2fd38032410`。原 GPU 主仓库仍为干净的 `8db4d968cd0164b4c7550c8c0cd561b152971776`。

## 复用已有标定并指定初始窗口

新增接口使用新协议的四个归一化整数窗口值，执行器补全并冻结 λ 与实际强度。零刻度表示实际强度为 λ，不能将其当作关闭引导。选定刻度必须位于教师实测范围与 `[-10,10]` 的交集。

```bash
python agenticgen/scripts/run_uniguide_geometry.py --config new_config.json prepare
python agenticgen/scripts/run_uniguide_geometry.py --config new_config.json acceptance
python agenticgen/scripts/run_uniguide_geometry.py --config new_config.json reuse-calibration \
  --sources teacher_sources.json --initial-scales initial_scales.json
python agenticgen/scripts/run_uniguide_geometry.py --config new_config.json run-version GD01
```

`teacher_sources.json` 为教师 ID 到已完成标定结果根目录的映射。`initial_scales.json` 为教师 ID 到四个归一化整数的映射，示例仅说明格式：

```json
{"6pyy_H7S_A_402": [0, 1, 0, -1], "3w54_RNB_A_504": [1, 0, -1, 0]}
```

若当前结果目录已具备完整标定，直接使用 `select-initial --initial-scales initial_scales.json`。首次正式版本开始后不允许改写初始化。启动 Agent 时先测量指定初始化配置的 50 个候选，或引用已有完全相同的测量，然后将结果加入历史；新增候选计入正式预算。此接口不会跳过条件、映射、模型、采样实现、输出策略、随机初态和数值环境的兼容性检查。

导入保留成功及失败证据、源文件哈希与源账本哈希，不改旧结果，不把导入测量计为新采样或独立验证。只有完整标定才可导入；粗筛最优点不能直接作为 λ。当前两教师旧预实验各计入 130 次尝试，共 260 次，均因筛选最优点处于边界而停止，尚无完整 `calibration/domains.json`，因此现在仍不能用它们直接启动正式 Agent。此次没有自动扩界或伪造 λ。

## 证据索引与部署记录

本地证据目录：`agenticgen/docs/evidence/uniguide/20261028_candidate_batch/`。

- `full_acceptance_82dc89c1.json`、两份 `*_full_batch_acceptance.json`：完整真实模型、断点恢复及真实 OOM 探测。
- `final_acceptance_bf960310.json`、两份 `*_final_mixed_failure.json`：最终变更路径的真实验证及证据复用绑定。
- `final_bf960310_tests.log`：109 项远端测试。
- `final_audit.json`、`final_engineering_attempts.json`：终态、原始保护文件、独立工程记账及现有 λ 预实验停止证据。
- `rejected_physical_batch20_*.json`、`diagnostic_overflow_rejected_886f9572.log`：未通过方案及后续修复的原始证据。
- `verify_candidate_batch_bf960310.py`、`audit_candidate_batch_final_20261028.py`：实际使用的验收与归档脚本，哈希与远端回执绑定。

最终部署按用户指定顺序尝试 GitHub 及镜像。本次直连超时，随后 `ghfast.top` 成功，无需继续后面的镜像；之前各次部署的独立失败证据与已验证 bundle 保留。后续继续使用 `ghfast.top`、`gh-proxy.com`、`ghproxy.net`、`gh.llkk.cc` 的顺序。
