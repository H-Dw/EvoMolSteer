# FLOWR.ROOT 生成与即时无损存储

版本：EvoMolSteer 0.3.0，2026-10-04。

生成入口为 `scripts/generate_flowr.py`，批量转换入口为 `scripts/optimize_trajectories.py`。实现位于正式 `src/evomolsteer/generation/` 和 `src/evomolsteer/storage/`，测试目录只保存验证脚本与验证数据。

## 生成过程中的提交与回收

每个积分步先记录 current、模型预测终点、proposal、评分和选择概率，等待重采样决定产生后，一次性提交该步的全部候选及谱系。零子代候选也完整保留。

```text
FLOWR 生成一个积分步
  → 收齐结构、评分、selected_indices、offspring_count
  → 规范化为单步 HDF5，重新打开并逐字段逐字节核验
  → gzip 包装整个容器，再核验包装前后的容器字节一致
  → 发布 step_XXXXXX.h5.gz，提交带 SHA-256 的清单
  → 删除已被替代的临时 HDF5，清空该步原始数组缓存
  → 继续下一个积分步

批次结束
  → 核验全部单步文件，合并并进行跨步去重
  → 重新读取 trajectory.h5，与所有输入数组逐字节比较
  → 记录 consolidated 状态
  → 校验并删除已替代的单步文件
  → 记录 complete 状态
```

新生成过程直接写优化结构，**不会先积累或生成 `trajectory.npz`**。原始轨迹数组在每步成功提交后释放；临时容器按上述协议删除。回收只涉及已有验证替代物的轨迹，初始 RNG、restart 检查点、完整预测锚点和 final 文件继续保留：这些文件包含轨迹编码没有覆盖的恢复状态或完整概率信息。

浮点坐标不量化，也不额外降低概率精度。这里的“无损”是相对于现有控制脚本输出的 28 个轨迹字段：其中原有 `*_probs_f16` 本来就是 float16；不能据此声称所有模型内部概率都保留为 float32。编码保持原字段 dtype、shape、顺序和数组字节。

迁移前后控制脚本的可审阅差异见 [`online_generation_controller_changes.diff`](online_generation_controller_changes.diff)，来源与 SHA-256 见 [`online_generation_controller_origin.json`](online_generation_controller_origin.json)。插桩后循环已与历史验证版本逐行对比一致。

规范化包括：完全相同的行字典、常量、精确零稀疏表示、对称键矩阵上三角、可逆整数缩位，以及逐值确认后的 `current[t+1] = proposal[t, selected_indices[t]]` 引用。最终容器使用无损 gzip + shuffle；不使用有损浮点压缩。

保存的仍是 100 个积分步。**分析范围继续只由实际 `resampled` 事件确定**，不因存储全轨迹而扩大。当前复现默认选择窗口为 `[0,0.5]`，100 个线性积分步对应记录的 51 个选择事件；这不是新增的论文精确参数声明。

## 输入与输出契约

### 调用 FLOWR.ROOT

`FlowrRunConfig` 的 JSON 契约见 [`../schemas/flowr_generation.schema.json`](../schemas/flowr_generation.schema.json)，可修改的示例见 [`../configs/generation_ck2_clk3.example.json`](../configs/generation_ck2_clk3.example.json)。

| 参数 | 含义 |
|---|---|
| `flowr_root` | 实际使用的 FLOWR.ROOT checkout 路径；worker 核验导入来源 |
| `python_executable` | 已具备 FLOWR 依赖及 CUDA/ROCm 的 Python；不会重装或替换 torch |
| `checkpoint` | 模型权重路径，启动清单记录 SHA-256 |
| `input_dataset` | 含 `inputs/` 的数据集根目录，或直接指定输入结构目录 |
| `output_dataset` | 输出数据集根目录；可增加新的 campaign，不能覆盖同名 campaign |
| `campaign` | 本次实验目录名 |
| `samples` / `batch_size` | **每组**分子数 / 每批粒子数 |
| `steps` | 积分步数，默认 100 |
| `window_start` / `window_end` | SMC 选择窗口，默认 0 / 0.5 |
| `arms` | `unguided`、`single`、`joint` 的子集；默认三组 |
| `verify_passive` | 默认 true；joint 首批从同一 RNG 重跑原始循环并比较最终张量 |

此版本将已有 CK2α/CLK3 控制脚本正式模块化。输入仍要求以下同框架对齐文件：

```text
3PE1_protein_aligned.pdb
3PE1_ligand_aligned.sdf
6KHF_protein_aligned.pdb
6KHF_ligand_aligned.sdf
```

它不是任意靶点的自动对齐器。当前适配已验证的 FLOWR checkout 和历史采样兼容开关。运行时检查五个插桩位置唯一，并保存上游和插桩后的实际循环；上游接口改变时明确报错。`apply_guidance=True` 是该上游选择函数的接口开关，执行的是 SMC 重采样，不是坐标 reward 的梯度引导。

在具有 GPU 环境的机器上执行：

```bash
export PYTHONPATH="/path/to/EvoMolSteer/src:${PYTHONPATH:-}"
/path/to/flowr/python /path/to/EvoMolSteer/scripts/generate_flowr.py \
  --flowr-root /path/to/flowr_root \
  --python /path/to/flowr/python \
  --checkpoint /path/to/flowr_root_v2.ckpt \
  --input-dataset /path/to/aligned_inputs_dataset \
  --output-dataset /path/to/generated_dataset \
  --campaign ck2_clk3_streamed \
  --samples 1000 --batch-size 50 --steps 100 \
  --window-start 0 --window-end 0.5 --arms unguided,single,joint
```

也可使用 `python scripts/generate_flowr.py --config configs/generation_ck2_clk3.example.json`；显式 CLI 参数覆盖 JSON。先加 `--dry-run` 可以检查路径和命令，dry run 不创建输出目录。启动器不经过 shell 拼接命令，路径可以含空格。

最低存储依赖是 NumPy、pandas、h5py；已有分析环境可安装 `.[storage]`。FLOWR 环境只需补齐存储依赖并提供 `PYTHONPATH`，不需要在 GPU 环境重新安装整套分析依赖。当前验证使用 h5py 3.16.0；旧版没有 `File.in_memory` 时使用较慢的 BytesIO 回退。

SCNET 当前环境还需先运行：

```bash
source /opt/MolSteer/scripts/scnet/activate_dtk.sh
export LD_LIBRARY_PATH="/opt/miniforge3/envs/molsteer-flowr-dtk/lib:${LD_LIBRARY_PATH:-}"
```

第二行解决当前机器的 `CXXABI_1.3.15` 动态库优先级问题。此次离线安装的 h5py 位于实验目录 `runtime_deps/`，使用时还需把该目录加入 `PYTHONPATH`。该环境调整仅作用于启动进程。

输出结构如下：

```text
output_dataset/
  inputs/                         # 输入口袋与配体
  results/campaign/
    generation_request.json       # 完整配置
    generation_manifest.json      # 运行状态、源码/权重/输入哈希
    generation.log
    runtime.json
    provenance/                   # 原始与插桩循环
    config.json
    frame_batch_000.json
    joint/batch_000/
      trajectory.h5               # 完成后唯一的轨迹结构容器
      trajectory.manifest.json    # 所有步骤的字段/文件哈希和回收状态
      events.json
      initial_state.pt.gz
      restart_step_*.pt.gz
      endpoint_step_*.pt.gz
      final_prediction.pt.gz
      final_records.json
      COMPLETE.json
    COMPLETE.json
```

`.lock` 文件是可复用的操作系统锁文件，进程退出后锁会释放，不代表任务仍在运行。运行中的单步容器位于 `batch_000/trajectory.stages/`；批次成功合并后该目录消失。

### 转换已完成的数据集

```powershell
.venv/Scripts/python.exe scripts/optimize_trajectories.py `
  --input D:/datasets/completed_generation `
  --output D:/datasets/compact_generation `
  --delete-source
```

输入数据集必须有 `results/<campaign>/<arm>/batch_*/trajectory.npz`，对应批次和 campaign 都须有 `COMPLETE.json`。目标使用空目录；重试只允许使用同一次转换的输出目录。输入输出根目录应相同或互不包含。原地转换若不传 `--delete-source` 会暂时保留两份，分析器会拒绝同批 NPZ/HDF5 混用；生产使用独立输出目录或启用已验证删除。

工具先复制上下文并验证全部输出，再按清单删除所指定输入数据集中的已替代轨迹。检查点、口袋和其他文件保留。删除前再次核对原文件 SHA-256；变化、损坏或校验失败都会中止删除。独立输入目录会写入 `trajectory_retirement.json`，指向可继续分析的目标数据集。输出有 `conversion_manifest.json`、逐文件转换收据和重新生成的 `SHA256SUMS`。

单文件也支持同一接口；调用者必须确保生成该文件的进程已经关闭它：

```powershell
.venv/Scripts/python.exe scripts/optimize_trajectories.py `
  --input D:/datasets/closed_batch/trajectory.npz `
  --output D:/datasets/closed_batch/trajectory.h5 --delete-source
```

默认不删除输入。只有显式 `--delete-source` 或 Python `delete_source=True` 才回收已有 NPZ。安装项目后，等价命令是 `evomolsteer-optimize`；无需安装入口时可运行 `python -m evomolsteer.storage.cli`。

### Python 接口与已有分析衔接

```python
from evomolsteer.generation import FlowrRunConfig, launch
from evomolsteer.storage import convert_dataset, convert_trajectory
from evomolsteer.storage import StepTrajectoryWriter, TrajectoryPackage

launch(FlowrRunConfig(
    flowr_root="/path/to/flowr_root",
    python_executable="/path/to/flowr/python",
    checkpoint="/path/to/model.ckpt",
    input_dataset="/path/to/input_dataset",
    output_dataset="/path/to/output_dataset",
))

# 独立生成器也可使用：fields 是单步数值数组字典，不带时间轴。
writer = StepTrajectoryWriter("/path/to/new_batch", expected_steps=100)
for step, fields in enumerate(my_generated_steps):
    writer.append(step, fields)  # 返回时已落盘并逐字节验证，可释放 fields
writer.finalize()               # 合并、核验、自动回收单步文件

with TrajectoryPackage("/path/to/new_batch/trajectory.h5") as data:
    candidate = data.node(step=30, slot=7)
    data.verify()
```

现有 `open_trajectory`、几何提取和四种分析算法都能读取最终 HDF5。新实验需在分析配置中指定新的 `campaign` 及正确的批次划分；要执行当前三组对照分析，生成时应保留三组及匹配先验。

## 异常恢复与容量边界

- 单步写入失败时不会推进步骤清单；已发布但未登记的单步可以用同样的输入重放恢复，输入字节不一致会报错。
- 合并未成功时保留单步文件；`consolidated` 后中断可重新调用 `finalize()`，核验最终容器后继续回收。
- 旧 NPZ 转换可用完全相同的输入输出参数重试，包括“源文件已删除，但完成收据还未更新”的中断。
- 当前生成启动器要求新的 campaign，不自动续跑中断的 GPU 任务。已保存步骤和恢复锚点保留，但继续积分需要显式恢复 RNG/current/self-conditioning；存储事务重试不能代替生成状态恢复。
- 生成时只缓存当前一步，最终合并暂时读入**整批**数组，内存随 `steps × batch_size × atoms` 增长；尚未实现完全流式的跨步合并。
- 每步有同步写入和校验开销。单步容器的元数据会使临时总容量大于一次性压缩的 NPZ；最终合并才得到跨步去重收益。合并瞬间需容纳所有单步文件加一个最终包，不能声称磁盘峰值始终低于 NPZ。
- 文件 fsync、原子发布和操作系统锁覆盖测试中的进程中断。未测试机器断电或存储硬件失效，不将其描述为断电恢复保证。

## 已完成验证

三组真实 B=50 轨迹，各 100 步、28 字段，经过生产写入器逐步回放，全部逐字节一致；300 次提交完成后临时阶段文件已自动回收，原始数据没有改动。最终包总计 14,625,387 字节，原 NPZ 为 21,927,217 字节，减少约 33.30%。逐步提交中位耗时约 0.10–0.11 秒，合并每批约 3.8–4.0 秒；这些是本地回放实测，不是 GPU 生成加速指标。

原 NPZ 与新 HDF5 输入各提取 15,300 行选择窗口几何特征，features、edges、事件、scope 和 catalog 的 SHA-256 全部一致。另用三批真实数据的独立副本执行 CLI `--delete-source`：84 个数组核验通过，副本已删除，历史原文件不变。

详细证据：

- [`../test/online_generation_v1/replayed_final_dataset/storage_replay_verification.json`](../test/online_generation_v1/replayed_final_dataset/storage_replay_verification.json)
- [`../test/online_generation_v1/analysis_input_equivalence.json`](../test/online_generation_v1/analysis_input_equivalence.json)
- [`../test/online_generation_v1/retirement_verification.json`](../test/online_generation_v1/retirement_verification.json)

远端 K100_AI / torch 2.9.0 / HIP 6.3.26093 另完成了**真实新生成**：三组各 2 个分子、每组 100 步。single/joint 均保存 51 次实际重采样，各包含 26 个零子代候选的评分与几何。8,400 个单步字段哈希、所有父子关系和三组匹配先验通过检查，原始 NPZ 未创建，阶段目录已回收。joint 首批从同一随机状态运行原始循环，五类最终结构张量 `torch.equal` 全为 true。该小批测试验证工程正确性，不用于估计多目标优化效力。

远端数据在 `/root/private_data/MolSteer/flowr_root/experiments/evomolsteer_online_20261004/dataset/results/smoke_streamed_v3/`；本地归档为 `test/online_generation_v1/smoke_streamed_v3.zip`，解压数据为 `test/online_generation_v1/remote_dataset/`。GPU 验证记录见 [`../test/online_generation_v1/remote_verification.json`](../test/online_generation_v1/remote_verification.json)。本次本地 43 项测试通过，覆盖现有分析、可逆存储、插桩位置、源文件变化、损坏、中断恢复和删除范围。

下载后已核对归档和三份 HDF5 的 SHA-256，并用现有提取器从新 GPU 数据成功生成 612 行选择窗口特征，见 [`../test/online_generation_v1/remote_analysis_integration.json`](../test/online_generation_v1/remote_analysis_integration.json)。单一批次不足以开展跨批次统计推断，因此这里只执行输入兼容性与几何提取验证。
