# 完整性审核的枚举上限修正

30 轮真实生成结束后的首次完整审核，仅 `six_batch_pairing_native_runtime_and_upstream` 未通过，原因为 `Underlying FLOWR arguments changed`。原始拒绝保存在 `confirmation_integrity.initial.json`，当时未升级的效果判定保存在 `confirmation.initial.json`。拒绝没有被删除或追改。

真实参数逐键比较显示，除独立输出目录 `save_dir` 外，唯一变化是 `sample_n_molecules_per_target`：探索批次 47/48 为 2450，确认批次 49/50 为 2550，51/52 为 2650，53/54 为 2750。相配 R26 与候选使用相同上限。

实际执行的 `src/evomolsteer/generation/controller.py` 明确设置 `loading_n=(max(batch_indices)+1)*batch`，据此构造可枚举到所需批次的数据加载器。循环在调用 `_generate_selective` 前跳过未指定批次；跳过阶段仍按原生顺序构造和消耗先验，但不执行这些批次的生成模型。各组只生成指定的两个批次、每批50个尝试。执行、终态、初态签名和完整记录的覆盖检查均按这一实际生成量独立进行。

审核器原先只允许输出目录变化，未识别这个由预登记批次编号推导的枚举上限。修正要求每份配置的字段严格等于自身 `batch × (max(batch_indices)+1)`，并核对相配控制/候选的上限一致；只有满足这个规则才从跨不同批次的参数比较中规范化该字段。其他模型、采样、积分及运行参数仍必须一致。不使用简单“忽略字段”的放宽方式。

修正只涉及独立审核与其合成测试；没有改变模型、奖励、推理接口、Skills、历史轨迹、任何轮次的冻结配置或保留结果。亲和力、有效率、应变、置信区间及升级阈值不变，没有新增或重跑生成轮次。之后用新进程对真实六批全部证据重新审核，再重新应用原预登记效果条件；最终是否升级见 `confirmation.json`、`activation.json`（若已升级）与活动配置。
