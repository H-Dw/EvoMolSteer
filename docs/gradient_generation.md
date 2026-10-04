# 使用选择证据驱动 FLOWR 梯度生成

新增入口 `scripts/generate_gradient_flowr.py` 与原 Steer sampling 入口独立。它复用原生 FLOWR 积分器、解码与即时无损存储，通过两个明确插入点加入当前坐标的 live gradient：目标口袋 forward、原生积分后位移注入。

## 输入与输出

在 FLOWR 的 GPU Python 环境中运行，并把本仓库 `src` 加入 `PYTHONPATH`：

```bash
python scripts/generate_gradient_flowr.py \
  --flowr-root /path/to/flowr_root \
  --input-dataset /path/to/historical_dataset \
  --root /path/to/new_dataset --campaign comparison \
  --checkpoint /path/to/flowr_root_v2.ckpt \
  --program configs/experiments/ck2_single_guidance_v1/reward_program.json \
  --catalog configs/experiments/ck2_single_guidance_v1/reward_catalog.json \
  --n 200 --batch 50 --steps 100 --seed 20261005 \
  --arms unguided,single,gradient_region,gradient_region_compact \
  --strength .05 --max-atom-step-A .025 --live-preflight
```

- `--input-dataset` 接受含 `inputs/` 的数据集或直接的输入目录。本适配器使用历史实验的四份 CK2/CLK3 对齐 PDB/SDF；CLK3 只用于对照评分，奖励不优化它。输入哈希校验防止把冻结坐标系的奖励误用于另一结构。
- `--root` 是输出数据集，`--campaign` 必须是新目录。中断结果保留，重新运行应使用新 campaign，当前不支持静默覆盖或自动续跑。
- `--program` 是受约束的 `live-regional-1.0` 数据文档，允许的计算由可信编译器提供，没有 `eval` 或任意生成代码。`--catalog` 仅含必要特征和受体坐标。
- 两个 gradient arm 都不重采样；`single` 是原单目标重采样对照。`gradient_zero` 用于 `--verify-zero` 检查，执行 live pullback 但不注入位移。
- 当前适配器限定原实验的 100 步、无 inpainting、选择对照窗口 [0,.5]。需要另一采样网格或任务时，须验证相应适配器，而不能把该冻结奖励直接当作通用模型。

输出位于 `root/results/campaign/arm/batch_NNN/`：`trajectory.h5` 保存全部候选、评分、根节点、父节点和选择信息；`guidance_trace.jsonl` 保存奖励、梯度范数、位移及 cap 因子。阶段文件完成无损校验后回收，不产生原始轨迹 NPZ。`initial_state.pt.gz`、restart、endpoint 和 final 原生预测用于审计及恢复上下文；它们不是额外的训练样本。`config.json` 保存执行 commit、checkpoint/奖励/目录哈希；`provenance/` 保存实际插入前后的上游函数。

## 单目标分析与规则冻结

```bash
python scripts/run_pipeline.py --root /path/to/history \
  --output /path/to/single_analysis --config configs/single_target_analysis.json
python scripts/freeze_single_target_reward.py --analysis /path/to/single_analysis \
  --output /path/to/frozen_program
python scripts/summarize_single_hypotheses.py --analysis /path/to/single_analysis \
  --program /path/to/frozen_program/reward_program.json --output /path/to/hypotheses
```

冻结入口是本 CK2 实验的显式假设编译器，只读取 discovery 证据；不是自动为任意靶点寻找最优奖励的系统。第三个入口才读取独立 split 对冻结特征/阶段进行方向复核，不能用这些结果重新选择参数后仍称独立验证。

Analyst 继续使用原 schema 2.0 导出、subagent 或 HTTP 调用和语义校验。原 Designer 2.0 仅支持 saved-endpoint 的双侧窗口，不能冒称支持 live 单侧程序；本实验保存独立 `Designer.live_design.json` 设计审阅，并由确定性冻结脚本提供可执行数据程序。通用 live Designer API 自动编译接口尚未替代原接口。

## 完成后的配对比较

```bash
python scripts/compare_gradient_runs.py --campaign /path/to/new_dataset/results/comparison \
  --output /path/to/comparison_report
```

脚本输出每批、每组、配对结果、冻结几何轨迹和控制审计，拒绝不完整批次或 gradient arm 中的重采样。独立统计单位是批次；四批的精确双侧 sign-flip p 值最低只有 0.125，因此不能据此声称显著改善。生成末态仅用于冻结奖励的实验评估，不进入选择窗口的规则发现。

SCNet 的环境配置已保存为 `scripts/scnet_gradient_experiment.sh`。它从已拉取且 tracked 文件干净的 checkout 执行 `analysis`、`pilot` 或 `comparison`；可通过 `FLOWR_ROOT`、`FLOWR_PYTHON`、`EXPERIMENT_ROOT` 替换路径。远端依赖环境与 h5py 路径见脚本，源代码修改仍应在本地提交和推送后再拉取。
