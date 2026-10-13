# Unguided target集合生成

`scripts/run_unguided_targets.py`独立生成无Steer、无reward gradient guidance的对照集合，复用原有target锁、候选计数、失败记录和无损归档模块。默认HiQBind配置为`configs/generation_hiqbind_unguided.json`。

```bash
PYTHONPATH=src python scripts/run_unguided_targets.py --dry-run
PYTHONPATH=src python scripts/run_unguided_targets.py --config configs/generation_hiqbind_unguided.json
# 当前SCNET运行环境封装：
bash scripts/scnet_unguided_targets.sh
```

可通过`--input-dataset`、`--target-manifest`、`--output-dataset`、`--checkpoint`、`--samples`、`--batch-size`、`--archive-every`和`--config`指定输入、输出及运行预算。默认每target 100候选，5个batch，每batch 20粒子，100积分步，种子42。计数包含构建失败槽位，不通过补生成改变候选数。

## 对照定义

- 原生`_generate_selective`入口原有`assert apply_guidance`；仅unguided归档分支的运行时副本解除该入口断言，调用时显式传`apply_guidance=False`，原有SMC条件分支不会进入。不修改FLOWR源码或积分公式。
- 使用与先前Steer集合一致的SDE设定（坐标噪声0.2）、reference ligand原子数条件、self-conditioning、终点预测/解码和评分流程。不是将SDE同时改成官方另一个入口的默认ODE。
- target/alias的预测评分继续记录，用于后续分析，不影响生成路径。不额外加入奖励函数、梯度或后处理修复。
- [0,0.5]仅作为与Steer数据对齐的观察/存储窗口，全程0–1不发生选择、淘汰或复制。存储格式保留旧`selection_steps`字段以兼容读接口；实际选择须查看`resampled`，本模式要求其全部为false。
- 保存窗口结构、全程身份谱系、完整终点张量、SDF、评分、失败结果及输入/provenance。每一步父索引必须为identity、offspring_count必须全为1；完成核验不通过就不发布finished。
- 同种子及初始prior便于对照，但Steer重采样消耗随机数，随后两种模式的随机路径不会保持逐步相同。

默认`reference_campaign`指定已完成Steer集合。启动前核验target身份/顺序/输入SHA256、候选数、batch大小、积分步、seed、观察窗口、存储策略及实际checkpoint SHA256；任何不一致都会在领取target和启动GPU前报错。生成器源码的历史/当前SHA256均保留在`reference_comparison.json`，不能把配置匹配误当作所有历史代码字节相同。

## 输出与归档

默认新集合：`/opt/MolSteer/generated_datasets/hiqbind_unguided_learning_s100_b20_20261013`，不覆盖Steer集合。

- `generation_progress/`：原子创建的running/finished/error标记，多个同脚本worker自动跳过已领取/完成/失败target。
- `targets/<target>/results/unguided_w050/unguided/batch_*`：工作态batch；全局最终记录位于`results/unguided_w050/final_records.json`。
- `archives/targets_*.tar.gz`：每10个完成target归档一包，逐成员核验后回收对应原始target目录；最后不足10个的剩余target在集合结束时归档。以target为单位，不是10个batch或10个分子。
- `campaign_state.json`、`results_summary.json`、`reference_comparison.json`：定位信息、进度和配对核验。

健康检查为只读：

```bash
PYTHONPATH=src python scripts/check_steer_campaign_health.py --dataset /path/to/new/output --controller-log /path/to/controller.log --pid PID
PYTHONPATH=src python scripts/audit_steer_results.py --dataset /path/to/new/output
```

健康确认要求原生GPU、checkpoint/sampler来源、完整候选数、所有batch的零重采样核验、finished标记且无OOM/任务/归档错误。仅进程存活不能当作完成证明。
