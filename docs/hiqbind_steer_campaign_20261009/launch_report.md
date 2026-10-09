# HiQBind Steer 数据准备与后台运行

数据来源为FLOWR作者的[Zenodo发布记录20069589](https://zenodo.org/records/20069589)，该记录同时链接自概念DOI20069588。使用原始论文对应的 `flowr_root_v2.ckpt`。

`hiqbind__final.tar.gz` 为461,105,394字节，发布MD5 `ad56efdf798c043eacda395bcf182a7a`；`hiqbind__data_prepared.tar.gz` 为3,749,912,378字节，发布MD5 `f125550f99666677a714275aefe82bdb`。两者下载完成后必须匹配发布校验值才用于输入准备。校验下载分片在合并成功后回收，不保存一套重复分片；档案缓存放在节点本地盘，不占私有盘配额。

真实发布包包含独立的生成成员列表：train 31,197、val 77、test 297，共31,571系统。三个列表互不重叠，受限读取器只接受普通列表/字典/字符串元数据，不读取RDKit对象pickle或加载Torch。生成test列表与论文Figure3 Steer源数据的297个系统**逐成员完全相同**，比较记录为 `figure3_membership_comparison.json`。因此当前生成范围使用这297个系统，100槽位/target共29,700个终末候选槽位。

`test_data.csv` 和 `test_data_processed.csv` 各有279行带pK标签记录，辅助 `splits.npz` 的test索引也有279项。不能把这些辅助索引当作全体生成ID的统一全局索引；本次输入选择以发布的 `system_ids_test.pkl` 为准。297与279的差异意味着不能用带标签的评分子集替代完整生成集合，不对缺测亲和力的18个生成系统做无依据删除。论文图注278与当前带标签表279的进一步差异不影响本次已核验的297个Steer生成系统选择。

本地已增加 `scripts/download_verified_archive.py` 与 `scripts/prepare_hiqbind_test.py`。准备器只提取必要成员/评分元数据（约678KB），跳过2.92GB LMDB和RDKit缓存；根据生成test成员提取PDB/SDF原始字节，写入显式 `targets.json`、输入文件SHA256与 `PREPARED.json`，不改变坐标、化学图或补采缺文件。缺失、重名或训练/测试ID重叠时停止。当前默认配置要求297系统；另一个发布版本须显式给出其已核验数量。

生成配置为每target100槽位、5个独立群体×20竞争粒子、100积分步、seed42，Steer评分/重采样仅在0–0.5，之后推理到1.0，不使用reward梯度。每10个完成target打包tar.gz并校验后回收对应生成工作目录，最后不足10个的尾组也归档；进度原子标记支持并行任务跳过已占用、已完成和已失败成员。

部署过程始终先在本地提交并推送代码，再由远端拉取。输入准备及启动的实测记录将在本目录保存；预检、数据下载和进程存活不能冒充真实GPU生成成功。当前报告中的成员核验已经完成，最终启动状态以保存的launch和健康JSON为准。
