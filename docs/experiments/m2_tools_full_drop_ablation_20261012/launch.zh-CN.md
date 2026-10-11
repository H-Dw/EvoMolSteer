# 完整科学子模块消融：统一工具接口启动记录

2026-10-12，原服务器 `root@ksai.scnet.cn:50496` 已拉取并核验代码，完整后台队列已启动。
配置为 `configs/experiments/m2_tools_full_drop_ablation_20261012/runner.json`，冻结计划为
`docs/plans/m2_scientific_ablation_20261012_tools_v5`。此文档是启动快照，不表示完整研究已完成。

实验使用真实 DeepSeek API 的 `deepseek-flash`，依次执行 9 个 Analyst 和 8 个 Designer
子模块的完整/删除比较，每条件六条独立上下文角色链，两种任务表述各三条。
不运行替代条件，不改变默认温度。先 Analyst，再 Designer 形状与剂量，最后独立确认和最终上下文复核。
相同数值程序共享真实生成结果；错误比较暂时跳过，保留对应指引并注明因果未确认。

旧 direct-v4 队列已停止，已有请求、输出和结果保留；新队列不继承其删除决定、跳过状态或生成结果。
新随机批次以主种子 42 建立 M2 与无引导的配对对照；历史 Steer 作为固定独立参照保留。
远端工作目录为 `/root/private_data/MolSteer/flowr_root/experiments/evomolsteer_dependency_tools_full_drop_20261012`。
本地完成 API 调用和数据评价，远端只执行 FLOWR 推理。原始结构在下载校验、评价和精简结果保存后清理，保护原始 Steer 与 checkpoints。

接口新增了以下校验：只允许实际交付的证据句柄；单个严格 JSON 对象；字段显式提取；
Designer 跨条件复用冻结 Analyst 时核验真实输出、引用来源和完成记录；Reviewer 通过原文句柄选证据，由程序提取原文。
任务 ID 不向角色透露“完整/删除”或被删除模块。所有条件使用相同工具与格式约束，不借助格式契约补回删除功能。

启动验证中，81 项相关检查通过；新增任务 ID 与两种表述交付检查后，工具消融专项 8 项检查通过。
前期 14 次真实 API 请求均成功，未触发格式修复。Analyst 实际读取并响应符号翻转、缺失信息。
单位探针的 Designer 没有读取教师尺度，两个保留决定不足以验证单位推理；记录为未实际检验，不能宣称所有科学功能已验证。
详见 `diagnostics/tools_v1/exposure_audit.json` 和每步原始请求、回复、工具观察。

首批配对生成计划为 M2 200 个、无引导 200 个。首次健康检查时，首个梯度批次的 50 个分子已推理至 1.0：
100 积分步中，学习窗口内 50 步均有实际注入，窗口外注入为零，没有 Steer 重采样。
当前尚没有完整子模块性能结论，不能报告新 Skills 已超越 M2 或 Steer。

后续状态读取 `run_status.json`；逐模块比较写入 `submodule_results.csv`、`study_summary.json`、`status.zh-CN.md`，
完整研究结束后写入 `final_quality.json`。完成与仍在运行须以状态文件区分；后台进程 PID 保存在本地测试目录的 `background.pid`。
