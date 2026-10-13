# CK2 Scientist–Selector 轨迹检索实验

本实验回溯原始 `main1000_w050/single` 的 1000 条终态及完整 ancestry。这里运行的是隔离的模型描述与分支检索实验，未重新生成分子。

## 运行

在 EvoMolSteer 目录执行：

```powershell
.\.venv\Scripts\python.exe scripts/run_ck2_scientist_selector.py --through-round 50 --workers 3
```

命令自动核对冻结输入摘要并断点续跑；已有合格模型回答不会重新调用。`--prepare-only` 仅准备数据；`--through-round N` 可在完整轮次边界停下。

启动器对同一 study 持有操作系统排他锁，防止并行进程覆盖调用留档。进程退出后锁自动释放，保留的锁文件可以复用。

## 固定协议

- 结构筛选：connected/valid、全部 PoseBusters 检查通过、MMFF converged，且每重原子 MMFF relief 不超过 valid 集合 P75；按终态 CK2 common-rescore pIC50 降序取 TOP30 poses，允许相同分子图。
- Scientist 用 GPT-6 Luna medium，在十个五事件窗口中逐次读取 30 条匿名 teacher 轨迹，提交 300 段私有描述与十个无分支身份的定性假设。提交后的假设不可用后续数据改写。
- Scientist 在同一隔离上下文中每个 turn 只提交一个区间。每条描述的私有起始、中段、结束观测及六状态变化方向均与源数据核对；共享定性演变允许共享措辞，不同演变不能用同一泛化段落替代。仅替换时间编号的假设模板会被拒绝。
- 每个 Selector 用新的 GPT-6 Luna medium 上下文，在 t=0.05…0.50 的 preselection 状态独立选择五个等价 seed 组。全体 seed 的当前摘要均可见；详细数据由内存 broker 提供。仅当前及过去的 online pIC50 可见；终态得分、TOP30 身份、后续存活和 selection probabilities 均隔离。
- 主要 Coverage@k 为 TOP30 中真实祖先组位于前 k 个选择中的比例。补充原始 slot 匹配、唯一祖先 recall、precision、hit 和每个时间点的 oracle ceiling。相同结构与可见历史的 within-batch clone 合并；代表 slot 固定取最小值。
- R00 基线；R01–R25 修改 Scientist，Selector 固定 Q00；R26–R50 固定最佳 Scientist 的版本和实际假设，仅修改 Selector。按十节点宏平均 @5、@3、@1 排序，平局取最早轮次。
- 主流程优化与实际证据/条款应用审计均调用 GPT-6.1 Sol xhigh。模型回答、失败尝试、prompt、输入绑定、工具记录、版本、评分均留档。
- 基线、最佳 Scientist、最佳总体版本做 structure-only 控制；基线与最佳总体做三次独立 Selector 重复（原始选择为第一次）。

## 留档与解读

`protocol.json` 保存输入定位、质量门槛、冻结数据摘要和特征定义；`oracle/ancestry.parquet` 保留全部编号及逐步分支映射。`oracle/` 只供主流程评分。

`versions/` 的每版 SKILL、workflow、诊断、哈希由 Git 保存；`rounds/` 保存实际调用与主流程审计。`run_status.json` 表示执行状态，`report.md` / `report.html` 汇总可核验结果，`iteration_scores.csv` 和 `target_rank_heatmap.*` 展示迭代分数及 TOP30 ×十节点排名。

事实检查未通过的输出保存在 `rounds/Rxx/rejected/Txx/` 和原始 attempts 中，纠错 prompt/workflow 在版本目录的 `repairs/` 留档；失败尝试不进入合格轮次曲线。报告核对请求绑定、回答收据与 Scientist/Selector 的事实审计，`rejected_trials.csv` 单独列出拒绝原因和归档哈希。

早期 current 是生成点云；forecast 是该时点的模型端点预测。距离接触不等于氢键或能量，原始离散键标签不等于经过验证的化学图。Teacher 由同一实验的终态结果回溯选出，因此这些结果只能说明该数据集的回溯检索能力，不能视为前瞻泛化或真实药理效力。

运行产物体积较大，保留在本地。Git 记录代码、协议及技能版本，不提交全部二进制和模型数据包。
