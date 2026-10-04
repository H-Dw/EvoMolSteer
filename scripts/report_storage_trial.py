"""Build the measured storage report from retained manifests and benchmarks."""
from pathlib import Path
import json

from evomolsteer.io import digest, write_json

project = Path(__file__).resolve().parents[1]
trial = project / "test/storage_trials/20261004"
optimized = project / "data/optimized/main1000_w050"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def size(path):
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def mb(n):
    return f"{n/1_000_000:,.3f}"


def change(before, after):
    return 100*(1-after/before)


features = read(optimized/"features_v1/manifest.json")
views = read(optimized/"feature_views/views_manifest.json")
bench = read(trial/"benchmark_final.json")
verification = read(trial/"verification/verification.json")
raw = read(trial/"trajectory/manifest.json")
inventory = read(trial/"inventory.json")
norm = [r for r in raw if r["lineage_aliases"]]
main_before = features["source_bytes"]
main_after = size(optimized/"features_v1")
shared_before = main_before+views["source_shard_bytes"]
shared_after = size(optimized)
metrics = {
    "unit": "bytes; MB=1000000 bytes; file lengths, not allocated filesystem clusters",
    "main_feature_table": {"before": main_before, "after_including_manifest": main_after,
                           "saved": main_before-main_after, "reduction_percent": change(main_before, main_after)},
    "main_and_shard_feature_tables": {"before": shared_before, "after_including_manifests_and_views": shared_after,
                                    "saved": shared_before-shared_after, "reduction_percent": change(shared_before, shared_after)},
    "raw_three_batch_trial": {"before_npz": sum(r["source_bytes"] for r in norm),
                              "after_hdf5": sum(r["package_bytes"] for r in norm),
                              "expanded_array_bytes": sum(r["expanded_array_bytes"] for r in norm)},
    "actual_original_bytes_deleted": 0,
    "scope": "Only main1000_w050 feature tables are fully optimized. Three original trajectory batches are a read-performance trial; raw archives and checkpoints remain untouched.",
    "verification": verification,
    "benchmark_source": str((trial/"benchmark_final.json").relative_to(project)),
}
code = list((project/"src/evomolsteer/storage").glob("*.py")) + list((project/"scripts").glob("*storage*.py")) + [project/"tests/test_storage_lossless.py"]
metrics["code_sha256"] = {p.relative_to(project).as_posix(): digest(p) for p in sorted(set(code))}
write_json(trial/"storage_report.json", metrics)

lines = [
    "# EvoMolSteer 本地无损存储优化实测（2026-10-04）", "",
    "已经完成本地优化试验与全量特征转换。适合减容的是几何特征表及其重复分析分片；原始轨迹已经使用 NPZ 压缩，改为未压缩 HDF5 虽能加快节点读取，体积仍增大。因此保留原始轨迹归档，把不压缩的结构去重主要用于分析数据。", "",
    "## 1. 同等信息量的文件体积", "",
    "单位 MB=10⁶ 字节，包含正式数据包的 manifest/视图。下列前两行是两种比较口径，不能相加。", "",
    "| 比较对象 | 原体积 MB | 优化后 MB | 减少 |",
    "|---|---:|---:|---:|",
    f"| 单一主特征表，900,000 行 | {mb(main_before)} | {mb(main_after)} | {change(main_before,main_after):.2f}% |",
    f"| 主特征表 + 原有 20 份同内容分片 | {mb(shared_before)} | {mb(shared_after)} | {change(shared_before,shared_after):.2f}% |", "",
    f"第一种口径减少 **{mb(main_before-main_after)} MB**；第二种口径减少 **{mb(shared_before-shared_after)} MB**。第二种口径的额外节省来自取消正文的重复保存，分片通过共享视图恢复；20 份原分片的全部 900,000 行已与共享包逐行组核对。不是只保留部分分片或删除失败样本。", "",
    "**所有原始文件仍保留。上述数字是等价替换后的体积比较，当前没有删除原文件、没有实际释放这些字节。** 本轮还保存了基准试验包。真实导出验证产生的大型临时副本已在验证成功后移除。", "",
    "## 2. 对原始轨迹的试验结果", "",
    "每个受测批次包含 100 个时间步、50 个候选、25 个原子槽、28 个数组。joint、single、unguided 各测试 batch_000，包含选中和未选中分支。", "",
    "| 组别 | 原 NPZ MB | 原数组展开 MB | 去重后未压缩 HDF5 MB | 相对现有 NPZ |",
    "|---|---:|---:|---:|---:|",
]
for r in norm:
    arm = Path(r["source"]).parents[1].name
    lines.append(f"| {arm} | {mb(r['source_bytes'])} | {mb(r['expanded_array_bytes'])} | {mb(r['package_bytes'])} | 增大 {100*(r['package_bytes']/r['source_bytes']-1):.2f}% |")
raw_before = sum(r["source_bytes"] for r in norm)
raw_after = sum(r["package_bytes"] for r in norm)
lines += ["", f"三个轨迹合计 **{mb(raw_before)} → {mb(raw_after)} MB，增大 {100*(raw_after/raw_before-1):.2f}%**。直接将相同数组写进普通未压缩 HDF5 则为 {mb(sum(r['package_bytes'] for r in raw if not r['lineage_aliases']))} MB。结构去重比直接 HDF5 小得多，但不能把这一改善解释成比原 NPZ 更省磁盘。", "",
          f"全量原始目录中的 71 个 trajectory.npz 合计 {mb(inventory['raw_categories']['trajectory.npz']['bytes'])} MB；没有把三个试验批次的涨幅或速度推定为其余 68 批的实测结果，也没有进行会增大空间的全量轨迹替换。", "",
          "## 3. 聚合与去重算法", "",
          "1. **候选节点与分支分开保存。** 节点保留 step、slot、评分、选择概率、祖先及后代计数，重采样关系使用 selected_indices；物理文件按 campaign/arm/batch 聚合，逻辑上按节点寻址。没有给每个节点建立大量小文件。",
          "2. **当前状态引用上一节点 proposal。** 对每个字段验证 `current[t+1,b] == proposal[t, selected_indices[t,b]]` 的 dtype/shape/全部数据位，再仅保存初始 current 与引用。关系不成立就独立保存该字段。",
          "3. **特征向量按精确内容共享。** 900,000 行、255 个几何特征归一化为 562,609 份向量及映射；25 列共享元数据按 300,000 个候选节点保存一次。三个表示的时间信息、分支身份、失败/缺失标签仍保留。特征相同不表示两个分子或谱系节点相同，更不能减少统计样本行数。",
          "4. **完整表和分片使用同一份正文。** 60 个 HDF5 包支持恢复主表；20 个仅含引用的视图支持分别恢复 batch 分片。",
          "5. **不降低数值精度。** 原 float32 坐标、float16 存档概率、float64 特征均保持原数值位。仅使用整数字典引用、常量消除、精确零的稀疏布局和经验证的对称键矩阵上三角；整数索引读取时恢复原 dtype。", "",
          "HDF5 是容器，压缩是可选滤镜。本实现全部使用连续布局，未启用 gzip、Zstandard、LZF、shuffle、scale-offset 或 bit packing。索引/稀疏表示仍有重建成本。[h5py 官方数据集说明](https://docs.h5py.org/en/stable/high/dataset.html)支持这些格式概念；本报告的体积和速度来自本机实测。", "",
          "## 4. 读取性能", "",
          "本机暖缓存，固定种子 20261004；每组 24 个节点查询 × 3 轮，另有 3 次预热；批量读取 9 轮。表内为中位数，完整 p95 见 benchmark_final.json。未清空操作系统缓存，不代表冷盘或远程网络性能。NPZ/HDF5 每次打开文件；Parquet 保持元数据句柄，以免夸大 Parquet 的开销。", "",
          "| 查询（相同输出语义） | joint 原→新 ms | single 原→新 ms | unguided 原→新 ms |",
          "|---|---:|---:|---:|",
]
comparisons = [
    ("轨迹单节点 9 个数组", bench["raw"], "npz_node", "normalized_hdf5_node"),
    ("轨迹整批 28 个数组", bench["raw"], "npz_full_batch", "normalized_hdf5_full_batch"),
    ("单节点 255 个几何值及缺失掩码，NumPy", bench["features"], "parquet_numpy_geometry_node", "normalized_hdf5_numpy_geometry_node"),
    ("整批一个表示的几何数组，NumPy", bench["features"], "parquet_numpy_geometry_batch", "normalized_hdf5_numpy_geometry_batch"),
    ("单节点 261 列完整兼容输出，Arrow", bench["features"], "parquet_node", "normalized_hdf5_node"),
    ("整批一个表示的完整表，Arrow", bench["features"], "parquet_one_representation", "normalized_hdf5_one_representation"),
]
for label, results, old, new in comparisons:
    values = [f"{r[old]['median_ms']:.3f} → {r[new]['median_ms']:.3f}" for r in results]
    lines.append("| " + " | ".join([label] + values) + " |")
lines += ["", "**读取接口需要按用途选择。** NumPy 几何接口的节点查询约快 7 倍，轨迹节点查询约快 10–13 倍；完整 Arrow 表重建则较慢，主要增加了引用展开与列/字符串对象重建。不能宣称所有读取操作都更快。如果把原 NPZ/Parquet 一次性载入内存，后续切片也远快于每次访问磁盘；基准中已保留这种内存基线。", "",
          "因此推荐：归档继续用现有 NPZ；日常几何运算和 LLM 节点证据抽取使用 HDF5 的直接数组接口；大规模表格扫描可继续用原 Parquet，或一次恢复兼容表后复用。原统计脚本的默认数据入口未改变。", "",
          "## 5. 无损恢复验证", "",
          f"- 特征包：{verification['package_count']} 个文件、{verification['verified_row_groups']} 个原始行组、{verification['verified_rows']:,} 行，全部与原 Parquet 的规范化内容哈希逐组核对。检查 {verification['unfiltered_datasets']:,} 个 HDF5 数据集，全部无滤镜且连续。",
          "- 实际导出完整 Parquet 后重新打开，验证全部行组、schema/metadata、列顺序、行顺序、null 掩码和有效值。大副本验证后已移除；不是只验证内存中几行示例。",
          "- 三个原始轨迹实际导出 NPZ 再回读，每批 28 个数组，共 84 个数组的名字顺序、dtype、shape、全部数值位与原文件一致。",
          "- 自动化测试包括负零、多个 NaN 负载、无穷、空数组、大端浮点、Unicode、null 与空字符串、非对称键、谱系关系不成立时的回退、共享视图与节点随机访问。项目测试全部通过，详见测试记录。", "",
          "这里的无损是**数据信息无损**。NPZ/Parquet 导出的编码布局、压缩容器和文件头可以改变，文件 SHA-256 不保证等于原文件。对原始文件本身需要逐字节保持的用途，应继续保留原归档；本轮已经保留。", "",
          "## 6. 数据和脚本位置", "",
          "- 正式数据：`data/optimized/main1000_w050/features_v1/`，60 个 HDF5 包及 manifest。",
          "- 分片视图：`data/optimized/main1000_w050/feature_views/`。",
          "- 三组轨迹试验：`test/storage_trials/20261004/trajectory/`。",
          "- 量化结果：`test/storage_trials/20261004/storage_report.json`。",
          "- 基准明细：`test/storage_trials/20261004/benchmark_final.json`。",
          "- 无损验证：`test/storage_trials/20261004/verification/verification.json`。",
          "- 格式、API 与完整恢复命令：[storage_format.md](storage_format.md)。",
          "- 核心实现：`src/evomolsteer/storage/`；清单、视图、基准、验证分别在独立脚本中。", "",
          "复现入口（在 EvoMolSteer 目录、项目 Python 环境执行）：", "",
          "```text",
          "python -m evomolsteer.storage verify-features data/optimized/main1000_w050/features_v1 --original results/main1000_w050/features.parquet",
          "python -m evomolsteer.storage restore-features data/optimized/main1000_w050/features_v1 NEW/restored_features.parquet",
          "python -m evomolsteer.storage restore-features data/optimized/main1000_w050/feature_views/batch_000 NEW/batch_000_features.parquet",
          "```", "",
          "本轮没有修改远程生成任务、原始评分、失败对照、LLM 分析规则或奖励程序；没有用降低精度、丢弃低分分支、重算近似坐标换取节省。后续生成可在一个 batch 完成后写入同样的数据包并生成校验 manifest，避免再持久化整套重复分析表。", ""]
(project/"docs/storage_optimization_20261004.md").write_text("\n".join(lines), encoding="utf-8")
print(json.dumps({k:v for k,v in metrics.items() if k in ["main_feature_table", "main_and_shard_feature_tables", "raw_three_batch_trial"]}, indent=2))
