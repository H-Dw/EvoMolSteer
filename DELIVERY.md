# EvoMolSteer 当前交付

当前工程版本为 **0.3.0**，分析使用 **selection-window v2 compact v3**，有效入口是 [选择窗口报告](results/selection_window_v2_compact_v3/reports/RUN_SUMMARY.md)。原 `main1000_w050` 全程分析及其 LLM 输出已经归为历史记录。

- 数据范围：真实 `resampled` 事件，匹配时间的 unguided 背景；没有后期帧、终态标签或后代结局。
- 计算方法：逐事件富集、差异、选择分解、PCA；保存逐事件和阶段变化率。
- 规则输入：`results/selection_window_v2_compact_v3/agents/evidence_bundle.json` 与 `Analyst.request.json`。
- 主要约束：规则必须对应观测阶段与明确表示；Designer 必须采用数据提供的选择窗口与概率加权原型。
- 可复现入口：[README](README.md)；算法细节：[selection_analysis_v2.md](docs/selection_analysis_v2.md)。

原始下载归档和校验清单仍保留。源码修改前快照位于 `delivery/EvoMolSteer_pre_selection_window_v2_20261004.zip`。

本次新增正式生成模块 `src/evomolsteer/generation/` 与即时存储接口 `src/evomolsteer/storage/streaming.py`、`lifecycle.py`。生成入口为 `scripts/generate_flowr.py`；旧数据转换入口为 `scripts/optimize_trajectories.py`。见 [使用说明](docs/online_generation_storage.md)。远端完成了三组各 2 个分子、每组 100 步的真实 GPU 联调，joint 的五类最终结构张量与同 RNG 原始循环完全一致。
