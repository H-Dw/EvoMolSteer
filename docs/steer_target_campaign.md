# 多 target 单目标 Steer 生成与分组 tar.gz 归档

入口 `scripts/run_steer_targets.py` 顺序调用现有 `scripts/generate_steer_learning.py`：发现 target → 写入该 target 的输入清单和生成配置 → 执行 FLOWR.ROOT → 校验并收集完整结果 → 达到配置的 target 数后归档。每个任务只传递 `arms=["single"]`，只优化自己的 target predicted affinity，没有 CK2/CLK3 联合选择或坐标梯度。

## 默认运行

已读取并确认远端 `/root/private_data/MolSteer/data/crossdocked_pocket10` 有 **100 个口袋/配体配对**，输入文件哈希见 `docs/crossdocked_steer_campaign_20261008/remote_inputs.json`。目录形如：

```text
crossdocked_pocket10/
  ABL2_HUMAN_274_551_0/
    4xli_B_rec_4xli_1n1_lig_tt_min_0_pocket10.pdb
    4xli_B_rec_4xli_1n1_lig_tt_min_0.sdf
  ...
```

脚本按文件名配对，每个配对使用去除 `_pocket10.pdb` 后缀的相对路径作为 target ID，按 ID 排序，并要求发现数量为100。同一蛋白目录中的多个复合物分别执行自己的单目标任务，不合并、不遗漏；缺配体、显式清单重复 ID 或路径越界均报错。测试集成员由指定的已准备目录决定；脚本不从训练目录猜测测试集，也不加载未知 pickle。通用数据可提供 `configs/target_collection.example.json` 格式的显式 target 清单。

当前服务器执行入口为：

```bash
# 在远端已拉取更新的 EvoMolSteer 仓库内
bash scripts/scnet_steer_targets.sh --dry-run
bash scripts/scnet_steer_targets.sh
```

该主机包装脚本使用已有 FLOWR/DTK 环境；可通过 `FLOWR_PYTHON`、`FLOWR_ROOT`、`FLOWR_ACTIVATE` 指定主机环境。通用控制器不包含主机认证或 Git 操作。在已经激活的其他运行环境使用：

```bash
python scripts/run_steer_targets.py --config configs/generation_target_collection.example.json
```

服务器配置为 `configs/generation_crossdocked100_steer.json`。默认每个 target 生成 **1000 个终末候选槽位**、50候选/批次、100个积分步，主种子42；真实评分窗口为 `[0,0.5]`，随后无重采样地继续到 t=1。这里默认“每10个”是 **10个 target 任务**，相应共10000个候选槽位，不是仅生成10个分子。分子构建失败和重复后代仍保留。目标与非目标角色在内部接口中别名指向同一口袋；非目标评分不具有真实选择性意义。

## 可配置行为

| 参数 | 默认 | 行为 |
|---|---:|---|
| `--samples` / `--batch-size` | 1000 / 50 | 每 target 的候选数和群体数，要求完整批次 |
| `--window-start` / `--window-end` | 0 / 0.5 | 按评分时间选择，范围传递给现有生成入口 |
| `--steps` / `--seed` | 100 / 42 | 积分步数和每 target 的主随机种子 |
| `--compress` / `--no-compress` | 开启 | 是否创建 tar.gz；关闭时保留可直接读取的工作目录 |
| `--archive-every` | 10 | 累计多少个成功完成的 target 后打包 |
| `--compression-level` | 6 | gzip等级1–9，数组和文件字节不会量化 |
| `--remove-archived-targets` / `--no-remove-archived-targets` | 回收 | 校验归档并持久化定位信息后，是否回收这些已生成工作目录 |
| `--max-targets` | 无限制 | 本次最多执行多少个尚未完成的 target；不改变已冻结的测试集 |
| `--retry-failed` | 关闭 | 为任务级失败创建新 attempt；保留旧失败诊断，不覆盖旧输出 |
| `--flush-archives` | 关闭 | 主动打包当前不足整组的已完成 target |
| `--continue-on-error` | 开启 | 记录失败任务并继续其他 target；最终有失败时退出码为2 |

例子：每5个 target 归档，仅先执行10个尚未完成的任务：

```bash
bash scripts/scnet_steer_targets.sh --archive-every 5 --max-targets 10
```

暂时完全不压缩：

```bash
bash scripts/scnet_steer_targets.sh --no-compress
```

压缩后还保留工作目录：

```bash
bash scripts/scnet_steer_targets.sh --no-remove-archived-targets
```

后者会增加当前占用，不能将压缩文件的缩小比例报告为已经释放的磁盘空间。默认策略回收新生成的、已经归档验证的工作目录；不删除原始 CrossDocked 输入、历史 Steer 结果或 checkpoints。

同一配置重新运行会跳过已完成/归档的 target。`--max-targets` 中途停止后不足10个的工作组暂存，下一次运行继续积累；完成整个队列时会打包最后不足10个的组。任务级失败不计入成功归档组，但目标内部的分子构建失败属于完整结果，应随该 target 一起归档。输入清单、文件哈希或核心配置改变时，要求使用新的输出集合。

## 输出与完整性

```text
<output_dataset>/
  targets_resolved.json
  campaign_state.json
  results_summary.json
  jobs/<target_key>/attempt_001/
    pocket_inputs.json, generation.json, generation.log
  targets/<target_key>/                 # 尚未归档或选择保留的完整数据集
    inputs/
    learning_dataset_manifest.json
    target_result.json
    results/single_w050/...
  archives/
    targets_0000.tar.gz
    targets_0000.tar.gz.json
    targets_0001.tar.gz, ...
```

结果收集先检查所有批次、每个槽位的失败记录、完整谱系和输入/坐标参照校验值，再保存每 target 的原生终末 predicted affinity 与终末再评分的均值/最大值、有效评分覆盖率、构建率及不同 SMILES 数。不同 target 的评分不混为同一个全局亲和力指标。

tar.gz 包含对应 target 的完整已保留目录：窗口内全部成功/淘汰候选、全程谱系、实际终末张量、SDF、评分、失败、输入、词表、坐标参照、配置和程序来源。包内 `STEER_TARGET_ARCHIVE_MANIFEST.json` 登记每个文件的字节数和 SHA-256。打包直接流式写 gzip，不额外保存未压缩 tar；重新读取每个 payload 并校验后才发布。归档校验采用一次顺序解压，避免反复 seek 导致大型 gzip 包重复解压。

回收先持久化归档定位，再验证所有待删除文件与包内清单一致，最后只删除控制器 `targets/` 内已登记且哈希匹配的文件。意外新增文件、修改文件或路径跳转会阻止回收。已验证的归档可在回收中断后继续完成清理。中断的未发布 `.partial` 容器保留为 `.abandoned_*`，不把它误认为完整数据包。

## 恢复单个 target 供 LLM 分析

可以先只验证，再恢复指定 target；原始文件内容逐字节恢复：

```bash
python scripts/verify_steer_target_archive.py \
  --archive /path/to/archives/targets_0000.tar.gz \
  --metadata /path/to/archives/targets_0000.tar.gz.json

python scripts/verify_steer_target_archive.py \
  --archive /path/to/archives/targets_0000.tar.gz \
  --target ABL2_HUMAN_274_551_0/4xli_B_rec_4xli_1n1_lig_tt_min_0 --output /path/to/restored_ABL2

python scripts/read_steer_event.py \
  --batch-directory /path/to/restored_ABL2/results/single_w050/single/batch_000 \
  --step 50 --slot 1 --verify --output /path/to/scoring_event.json
```

恢复目标目录必须尚不存在；归档中的不安全路径、链接、重复成员或不符合清单的内容会被拒绝。恢复到新位置后，原有事件和几何分析接口仍使用相对数据路径与保存的坐标参照。无须把十个 target 的全部轨迹都放入 LLM prompt。

## tar.gz 还能减少多少空间

实测记录见 `docs/crossdocked_steer_campaign_20261008/compression_benchmark.json`。本次使用 **10个不同的真实历史 CK2 单目标生成批次**，每批50候选；通过新存储入口回放，保留窗口、100步谱系、原生终末张量、实际 SDF/评分/失败和输入。数据未人为复制为10份相同轨迹，原文件哈希保持一致。

| 比较 | 大小，十进制MB |
|---|---:|
| 无压缩工作视图，文件长度合计 | 60.006796 |
| 一个十任务 tar.gz，gzip等级6 | 30.954363 |
| 减少 | **29.052433 MB，48.4152%** |
| 十个分别压缩的包，合计 | 30.953608 |

分组包比十个独立包大755字节，差异约0.0024%，可以视为相同压缩效果。因此“每10个打一个包”主要控制归档粒度、文件数和峰值工作空间，不应宣称存在显著的跨 target 共享压缩收益。gzip 的窗口有限，不是跨包的全局去重算法。

如果工作目录本身已经采用 HDF5 内部 gzip，额外收益会下降。此前完整的历史学习工作包实测为61.930929 MB → tar.gz 48.918016 MB，进一步减少21.01%，见 `docs/storage_audit_20261008/report.zh-CN.md`。不能将21.01%与48.42%相加，它们有不同的输入存储状态。

当前48.42%来自同一生物 target 的不同历史批次，事件概率含历史 f16；新 FLOWR 生成保存原始概率精度，CrossDocked 的原子数、构象和概率分布也不同。尚未新生成100个 CrossDocked target，因而没有它们的实测压缩比例。控制器会为每个真实归档组记录 `source_bytes`、`archive_bytes`、`reduction_percent` 和回收状态，可以直接获得该数据集的实际节省，而不是套用这个参考比例。

## 代码接口与验证

发现模块：`evomolsteer.generation.target_catalog.discover_targets`；控制模块：`evomolsteer.generation.steer_campaign.run_campaign`；存储模块：`evomolsteer.storage.target_archive` 提供 `archive_target_directories`、`verify_target_archive`、`restore_target` 和限定输出根目录的回收函数。

新增测试使用 CPU 记录夹具模拟生成脚本调用，验证12个 target 分为10+2、断点续跑、任务失败重试、不开启压缩、逐字节恢复、输入不变和回收保护。真实历史批次另进行了数组逐比特回放与 tar 文件逐字节校验。它们不代表执行了新的 FLOWR GPU 推理。
