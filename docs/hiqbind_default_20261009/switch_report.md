# HiQBind默认配置与旧CrossDocked任务清理

2026-10-09按用户要求将HiQBind设为多target生成入口的默认集合，CrossDocked100保留为显式备选。`scripts/run_steer_targets.py` 与服务器包装脚本采用同一默认配置；`--dataset crossdocked100` 选择备选，`--config` 完整替代默认配置，CLI参数仍可覆盖配置值。两种集合使用独立输出目录。

生成参数不变：100候选槽位、5个独立群体、每群体20粒子、100积分步、seed42、评分/重采样窗口0–0.5，之后无重采样推理至1.0。每10个完成target归档，验证后删除对应生成工作目录；失败槽位与谱系保留。

HiQBind默认输入预定为 `/root/private_data/MolSteer/data/hiqbind_test`，测试系统清单为 `targets.json`。当前远端数据目录只有已核验的CrossDocked测试输入，在检查范围内没有发现可用的HiQBind测试集合。默认入口要求显式测试清单，缺失即在创建输出/启动推理前退出；不会猜测测试成员、硬编码278个系统或自动改用CrossDocked。这里只完成配置切换，尚未启动HiQBind GPU生成。清单需包含每个系统的唯一ID及相对PDB/SDF路径，可沿用 `configs/target_collection.example.json` 的格式。

旧任务身份从远端launch记录、冻结campaign配置及实际进程父子关系共同核验：控制进程365、生成子进程21085、FLOWR worker21088。三者通过正常终止信号退出，无需强制杀进程。停止后才删除 `/opt/MolSteer/generated_datasets/crossdocked100_steer_learning_20261009` 及指向它的旧实验入口链接，包含两份归档、未归档结果、失败任务输出、jobs和进度标记。删除前占用2,701,328,384字节磁盘空间（约2.516GiB），逻辑大小2,694,634,439字节，共1953文件。

停止时旧任务为27个完成（20已归档）、18个失败、1个运行、54个待执行。这些统计是删除前的冻结记录，不是新HiQBind生成结果。模型、输入目录和其他历史实验未作为清理目标。旧每小时首轮健康检查自动化已经处于PAUSED状态。

本地验证：数据集默认选择、显式测试成员范围、缺输入无推理、备选选择、显式config及参数覆盖，并回归并行领取和归档流程，共29项测试通过（42.98秒；CPU测试，不是模型推理）。远端验证结果另存 `remote_validation.json`；清理实测证据见 `old_campaign_cleanup.json`。
