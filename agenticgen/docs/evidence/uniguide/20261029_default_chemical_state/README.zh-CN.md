# 默认化学状态注入：工程验收证据

执行代码：`b49adddce93706aca57698e6e9878551b57dab6f`，基于`777bfa36`及其后续运维提交。固定目录`/workspace/uniguide_audits/chemical-default-b49adddc`，结果目录`/workspace/agenticgen_chemical_default_acceptance_20261029_b49adddc_r2`。13:02:29（Asia/Shanghai）已写入远端FINALIZED；这些产物只用于工程验收。

- `deployment.json`：直连、四镜像和Git bundle部署记录；未修改主执行目录。
- `config.json`、`launch.json`：开启化学状态注入的独立配置及启动范围。只运行prepare与acceptance。
- `unit_tests.txt`：120项远端测试通过，37.72秒。
- `acceptance.json`：两教师真实模型VJP、随机流、独立/混合/恢复、容量探测和逐候选3 Å限制验收。
- `6pyy_chemical_condition.json`、`3w54_chemical_condition.json`：条件原子、冻结槽位和实际原生词表编码。该模型使用元素、电荷及键通道；杂化仅在支持的模型中启用。
- `6pyy_output_policy.json`、`3w54_output_policy.json`：只注入化学状态、raw999返回未覆盖预测、无坐标投影和无终态前向。
- `final_audit_script.py`、`final_audit.json`：独立读取保存的终态、checkpoint和轨迹；化学通道等于未覆盖预测、坐标等于原始预测，状态坐标等于投影前引导提案；检查各组化学绑定、3 Å上限、原文件哈希及执行锁。
- `source_preservation.json`：141个旧实验、输入和随机源文件的保护哈希，最终全部一致。
- `finalized.json`：绑定实际验收及独立审计的SHA256。

首次准备阶段将运维审计文件提前写入结果目录，被空目录保护规则拒绝；日志为`prepare_guard_failure.txt`。当时没有模型调用或采样，原目录原状保留，随后在全新r2目录准备。独立审计首次读取混合批checkpoint时误用单组的`geometry_binding`字段，错误保存在`audit_reader_first_error.txt`；按真实格式读取`candidate_binding`后只读核验通过，没有重新采样，也没有更改采样实现。

本次没有拟合λ、开展正式实验、调用LLM或恢复旧UG任务。工程重复轨迹不能当作独立分子验证。采样状态化学注入准确，但未覆盖的最终预测仍可能偏离模板化学类别；逐候选匹配分数在`final_audit.json`中保留，不将其外推为正式生成质量。
