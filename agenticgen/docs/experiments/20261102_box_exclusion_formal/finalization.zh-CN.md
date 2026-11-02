# Box负面排除专项：本地备份与收尾

2026-11-02（Asia/Shanghai）用户恢复D盘空间后，已完成本专项最终覆盖包下载。14:39两个结果包均完成整包SHA、每个成员的字节数/SHA及硬链接目标核验。未展开或执行包内文件。14:43又在远端重新核对归档所绑定的9733个源文件，内容、大小、mtime均未变，源文件缺失数为0；固定科学检出仍干净，所有本专项锁空闲、无所属采样或质量worker进程，两个质量队列各17组完成。

本文件更新[科学复盘](review.zh-CN.md)撰写时的“本地备份待空间恢复”状态。科学复盘及其原始绑定保持不变：提交`6e7f444c2adde5ce0c317e4239a69bd09f468917`，文件SHA256 `57fa74ac974246c0804298e176c40c5fc72212f66520ac5d7fb27947776bde63`。科学代码始终固定`c79627ad35670862694fb6dd59dab7272a9e7c9d`，本次未部署后续文档提交。

## 已核验的本地归档链

| 文件 | 字节数 | 成员/HDF5 | SHA256 |
| --- | --- | --- | --- |
| [首包](downloads/20261102_0534/completed_increment.tar.gz) | 363777660 | 2527 / 151 | ccaa817e95fa80c08ec9174365ce394264230d40c569e09868848b251f50cb39 |
| [最终覆盖增量包](downloads/20261102_0850/completed_increment.tar.gz) | 1334638387 | 7208 / 305 | 68d262a6975fe45f908fc35229c4caed4062c27891b05300ed08f102f46e3f11 |

两包共1698416047字节，均须保留。最终包引用首包中2525个已核验文件；恢复时按首包、最终包顺序使用，不能把最后一个增量包单独当作完整备份。原始科学结果的归档快照在08:51完成；随后巡检与收尾元数据单独保存在本目录及Git记录中。

[累计本地核验索引](downloads/20261102_0850/INDEX_LOCALLY_VERIFIED.json)有9739个逻辑文件条目，包括两个包内文件及4项已核验的旧增量元数据别名；索引SHA256为`75ce5dc01c7d674b634f0293c105afb01dd82b2d94ed66bb28e7d550b153af84`。旧06:26、07:30两个待下载原始包的结果已由最终覆盖包、首包和元数据别名完整覆盖，因此不重复下载。旧包原文仍保留远端，没有将未下载的旧包伪称为本地已核验包。

本地原始备份位于本专项目录`downloads/20261102_0534/`及`downloads/20261102_0850/`。远端最终包仍在`/workspace/uniguide_audits/box-exclusion-final-20261102_0850_archive/completed_increment.tar.gz`；累计索引及本地核验回执已上传该审计目录并读回SHA核对。没有移动结果包至private_data，没有删除任何远端原始结果或旧包，没有把大体积原始包提交Git。

包内不含模型权重、launcher.log及轮转副本或core。实验checkpoint_010.pt.gz、checkpoint_100.pt.gz、terminal.pt.gz按原文件名保留，累计索引中分别130、114、432项；这些是文件条目，不能作为额外科学尝试数。

## 科学与收尾绑定

两教师各完成370次预实验、五轮750次正式尝试，共2240次；34个完整源组共340个固定10/50检查全部结束。此次下载/核验新增生成、模型前向、LLM调用和worker启动均为0。最优RMSD、亲和力预测、标准SA、化学保持及构象分项结论均沿用已推送的真实复盘，没有追加第六轮、补样或重算科学预算。

源核验见[CLOSURE_SOURCE_AUDIT.json](downloads/20261102_0850/CLOSURE_SOURCE_AUDIT.json)，整条本地归档验证见[LOCAL_ARCHIVES_VERIFIED.json](downloads/20261102_0850/LOCAL_ARCHIVES_VERIFIED.json)。专项最终标记及自动化删除分别由[FINALIZED.json](FINALIZED.json)与后续`monitoring/HEARTBEAT_DELETED.json`记录；它们是归档完成后的独立收尾元数据。原`uniguide-flowr`保持PAUSED、35分钟，其他自动化及归档分支不动。
