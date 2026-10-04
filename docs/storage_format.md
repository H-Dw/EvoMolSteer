# EvoMolSteer 无损节点包 v1

> 本文主体记录原始 **v1 无滤镜格式**。新增的轨迹 **v2** 使用相同的逐位归一化规则，并显式启用 gzip level 6 + byte shuffle；不改动旧 v1 文件。v2 的完整说明和当前分析入口见 [`selection_storage_integration_20261004.md`](selection_storage_integration_20261004.md)。新命令为 `pack-trajectory SOURCE DEST --codec gzip_shuffle`，便携分析输入包由 `scripts/pack_analysis_inputs.py` 构建。

这个格式为一次生成实验保存候选节点、分支索引与几何特征，使用未压缩、连续布局的 HDF5。物理文件按 `campaign / arm / batch` 分区，节点由 `step / slot` 定位。一个文件可以包含全部时间节点；不为每个粒子建小文件或 HDF5 group。

## 安装与入口

在项目环境安装可选依赖：`pip install -e ".[storage]"`。当前实际使用 h5py 3.16.0，精确依赖见 `requirements.lock.txt`。

所有命令均通过 `python -m evomolsteer.storage` 执行。脚本默认拒绝覆盖已有目标。运行以下命令时将 `python` 换成项目 `.venv/Scripts/python.exe`，路径相对于 `EvoMolSteer`。

```text
python -m evomolsteer.storage pack-trajectory SOURCE/trajectory.npz NEW/batch.h5
python -m evomolsteer.storage verify-trajectory NEW/batch.h5
python -m evomolsteer.storage restore-trajectory NEW/batch.h5 NEW/restored_trajectory.npz

python -m evomolsteer.storage pack-features SOURCE/features.parquet NEW/features_v1
python -m evomolsteer.storage verify-features NEW/features_v1 --original SOURCE/features.parquet
python -m evomolsteer.storage restore-features NEW/features_v1 NEW/restored_features.parquet
```

`pack-features --limit-partitions 1` 仅用于试验，其 manifest 标记 `complete=false`，不允许将它误当作完整源文件恢复。正式数据包已位于 `data/optimized/main1000_w050/features_v1`，不需要再次转换。`feature_views/batch_000` 等视图也支持同样的 `verify-features` / `restore-features` 命令。

## 轨迹包

每个原始数组都有 dtype、shape 和包含 dtype/shape 的 SHA-256。数组名顺序也保留。`fields` 包含所有 28 个原始字段，包括选择失败/未获子代的候选。

对于本次文件逐位验证成立的关系：

\[
C_{t+1,b}=P_{t,I_{t,b}},\quad I_{t,b}=\mathrm{selected\_indices}[t,b],
\]

只存 `current` 的初始帧，随后通过上一帧 proposal 与索引恢复。没有成立的字段退回独立存储。坐标、原子、键、电荷分别验证，不根据某一个字段成立就推定其他字段成立。

对称的硬键矩阵在逐位检查后保存上三角和对角线；不对非对称矩阵做强制对称化。完全相同的数组行可共用字典行。稀疏表示只略去全零字节，即正零；负零、非零小概率、NaN 有效载荷均保留。整数索引可换成能容纳全部值的最小整数类型，读取时恢复原 dtype。

```python
from evomolsteer.storage import TrajectoryPackage

with TrajectoryPackage("test/storage_trials/20261004/trajectory/joint_normalized.h5") as p:
    node = p.node(step=30, slot=12)
    coords = node["current_coords"]
    scores = (node["pic50_on"], node["pic50_off"])
    children = node["children_next_slots"]
    all_proposals_at_step = p.read("proposal_coords", step=30)
```

`children_next_slots` 是这一评分步骤选择出的下一代槽位；最后一个积分步骤对应最终选择输出。评分时间与 proposal 所处时间仍按原始 `score_time` / `state_time` / `step_size` 区分。

轨迹包只替代 `trajectory.npz` 的信息。`initial_state.pt.gz`、restart/endpoint 锚点、随机状态、final records、坐标 COM、SDF 及实验配置继续保存在原始实验目录。没有把原本未保存的逐步 soft-bond 概率视作可恢复信息。

## 特征包

每个候选节点的 25 列共享元数据只保存一次；表示相关的字段单独保存。255 个几何特征的完全相同向量共用 `geometry` 的一行，三个表示分别保存 `state_ids`：

\[
X_{r,t,b,:}=U[J_{r,t,b},:],\qquad
\operatorname{valid}(X_{r,t,b,:})=V[J_{r,t,b},:].
\]

去重判定包含有效性掩码与每个有效浮点数的全部位。缺失值与真零不同；`+0.0` 与 `-0.0`、不同 NaN 负载也不合并。原数据中的 900,000 行仍全部存在，只改变引用关系，不能把状态池的行数当作独立样本量。

```python
from evomolsteer.storage import FeaturePackage
from evomolsteer.storage.features import iter_feature_tables

path = "data/optimized/main1000_w050/features_v1/main1000_w050__joint__batch_000.h5"
with FeaturePackage(path) as p:
    # 计算用快速接口。这里 row=30*50+12 对应本实验的 step=30, slot=12。
    values, valid = p.geometry("predicted_endpoint", row=1512)
    feature_names = p.features
    # valid=False 的位置只是占位，计算时必须排除；不要把占位零当作实测值。

    # 完整兼容接口，包括行级评分、谱系与状态标签。
    table = p.node("predicted_endpoint", step=30, slot=12)
    frame = table.to_pandas()

# 无需恢复大文件，按批读取；默认保留原始表中的顺序。
for table in iter_feature_tables(
    "data/optimized/main1000_w050/features_v1",
    representations=["predicted_endpoint"], batches=[0, 1], arms=["joint"],
):
    frame = table.to_pandas()
```

NumPy 接口适合计算，Arrow 接口负责完整表格兼容。逐节点读取并不需要先解压全部轨迹；整表恢复涉及状态展开和字符串重建，未必比原 Parquet 快。现有分析脚本继续读取原 Parquet；新脚本可直接使用上述 API，或导出恢复后的 Parquet 复用旧入口。

## 共享视图与无损范围

20 个 `feature_views/batch_XXX/manifest.json` 仅引用同一套 60 个 HDF5 文件。完整合并表和各 batch 分片可以分别恢复，但无需重复保存数据正文。必须连同相对目录结构一起移动 `features_v1` 和 `feature_views`。

NPZ 恢复保证数组名顺序、dtype、shape、全部数组数据位一致。Parquet 恢复保证 Arrow schema/metadata、行列顺序、有效值及 null 掩码一致。导出文件不会保留原 ZIP/Parquet 的压缩布局、创建信息或容器字节，因此其文件 SHA-256 通常不同。原始容器保留用于归档和来源核查。

所有数组数据集 `compression=None`、`chunks=None`，无 gzip、Zstandard、LZF、shuffle、scale-offset、bit packing 或额外 HDF5 滤镜。字符串/整数引用、常量消除、行去重和稀疏布局属于数据表示的改变；仍有索引重建开销，不能据此声称读取没有 CPU 成本。

## 实现与验证职责

| 模块 | 职责 |
|---|---|
| `storage/arrays.py` | 精确相等、索引类型、常量/行字典/稀疏数组、无滤镜审计 |
| `storage/trajectory.py` | 谱系关系验证、节点读取、NPZ 导出 |
| `storage/features.py` | 元数据归一化、几何池、批量/节点接口、Parquet 导出 |
| `storage/views.py` | 与现有分片逐行核对，建立共享视图 |
| `storage/benchmark.py` | 固定查询、正确性检查、读取时间比较 |
| `scripts/inventory_storage.py` | 原始文件体积与格式清单 |
| `scripts/verify_storage_trial.py` | 全量源文件核对与真实导出回读 |
| `tests/test_storage_lossless.py` | NaN、负零、缺失值、Unicode、非对称键、错误谱系等边界情况 |

本格式针对已完成的、固定形状的 FLOWR 轨迹和具有同质行组的 EvoMolSteer 特征表。混合 batch 的行组、缺失表示、对象数组和不支持的 Arrow 类型会显式拒绝，不默默转换或丢弃字段。未来支持变长原子数时，应以原子/键 offset 表扩展格式版本，不在 v1 中填补假数据。

HDF5 连续布局、切片与可选滤镜的含义依据 [h5py 官方文档](https://docs.h5py.org/en/stable/high/dataset.html)。本次空间与性能结论均来自本地实测，不来自该文档的性能推测。
