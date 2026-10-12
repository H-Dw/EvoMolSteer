# 本地接口验证记录（2026-10-13）

本轮只保存探索计划、可选 Skills、请求模板和可执行脚本。真实 LLM API 调用、远端连接、FLOWR 推理、分子生成及真实轨迹的新一轮挖掘均为 **0**。当前配置 `execution_enabled=false`、`enabled_conditions=[]`；准备清单不能当作实验进度或生成结果。

本地检查：`tests/test_temporal_guidance.py` 与既有 Agent workflow、affinity endpoint、cohort contract 回归检查共 **56 项通过**。四份新 Skill 均通过 `skill-creator` 的格式验证。单元检查使用合成坐标、合成预测器和 trace；这不构成真实 FLOWR 的 Jacobian 验证，也不能说明亲和力得到改善。

已检查恒等策略与原 M2 的奖励、梯度及剂量因子一致；标量幅度变化在 RMS 归一化中被抵消，而实际 `dose_gate` 会改变注入。正向、反向和常量混合/曲率使用观测评分时钟，保持离散平均参数一致。动态窗口、来源与引用、数值去重、禁用开关、同端点单次 VJP 和窗口外零注入均有检查。

最终模板目录为 `prepared_final/`，包含 **10 个 Prompt 条件、9 个数值候选**。所有条件都是 `planned_not_called`。先前本地草稿不属于当前配置。全窗口 profile 挖掘、真实 API 角色调用和实际模型/trace 验证均留待后续执行。

当前 Analyst/Designer、冻结 M2、历史 Steer 与固定对照库的生成源码保持原样。可重复使用的对照、随机输入与 checkpoint 通过来源校验；只登记未来 200 个样本初筛、400 个确认的方案，未生成这些样本。配置不自动启动稀有尾部扩样或组合实验。

数值权重的积分只匹配请求比例预算，不保证物理位移相同。未来必须核实实际剂量、方向、教师覆盖和质量。连续权重也不保证教师成员及梯度连续；未保存祖先身份时不能宣称已经复现同一优势路径。

机器可读来源校验见 `verification.json`，单元结果见 `local_unit_checks.xml`。
