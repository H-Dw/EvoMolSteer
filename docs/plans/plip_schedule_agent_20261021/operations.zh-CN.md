# 四假设、20版本后台运行手册

本文件是当前操作协议。旧逐链/三教师/10轮命令只在Git历史90254155中留档，不得用于新任务。
唯一服务器ssh.zzai.scnet.cn:11762。环境/密钥完整继承交互环境，禁止输出凭据。Python为/opt/miniforge3/envs/molsteer-flowr-dtk/bin/python。

## 准备和运行

先本地测试提交推送，再远端检出固定commit。统一通过 `bash scripts/run_plip_group.sh` 启动：它继承原DTK、LD_LIBRARY_PATH和已审计的依赖覆盖目录，包含h5py与PLIP，不另换科学环境。
Python命令入口为 `scripts/plip_group_pipeline.py`；默认root为plan中的新council结果目录。

1. `prepare --source-root /opt/MolSteer/generated_datasets/plip_schedule_agent_top1_5nn4_r123_i4_s42_20261021`
   仅从旧根引用历史配置、模板、基线及初态，不导入旧R0/优化结果。新根只能prepare一次。
2. `validate`：两教师真实batch400共同随机流及10/26/51/76混合父状态恢复。诊断不计入科学结果，证据保留，结构清理。
3. `launch --version v1`：验收通过后，认证网络预检成功才脱离SSH。完整继承API/代理，不记录值。
4. `status`：结合真实进程、pipeline.json、质量状态及日志判断，不能只凭running。
5. `report --version vN` / `review-pack --version vN`：重建阶段报告或审阅分片，不新增采样。
6. `analyze`：综合所有已运行版本，输出逐假设/整组择优/TOP1的Mean、Best、Top10、质量、历史基线差异、版本配对、共享批次bootstrap、案例和计算/token成本。

工作进程持有worker.lock；两个教师单GPU、CPU/LLM交错。实际部署以launch.json、workers.json及版本deployment.json为准。
失败写operations/failure_*.json并标记attention。成功采样/请求/质量缓存复用；不得覆盖失败证据。
修复必须本地测试、提交、推送再部署。冻结源码指纹变更时必须记录兼容性审计与重新验收，不能直接覆盖PREPARED绕过保护。

## GPT全量复盘及下一版

每版两教师各5轮，4000条优化记录。第5轮分析完成后生成FINALIZED_vN.json，状态awaiting_master_review。
读取skills/plip-schedule-review/SKILL.md，再读versions/vN/review/manifest.json。
逐片覆盖52个会话：2个bootstrap、40个私有组内会话和10个汇总会话。审阅全部独有prompt、工具证据、方案和结果，记录反例及未知问题。
将真实逐会话审阅JSON与复盘文档提交到本地仓库。每条审阅字段：id、evidence_sha256、assessment、decision_update、counterexample_or_limitation、evidence_files。
evidence_files必须引用该会话清单中的实际prompt文件；不要使用自动填充模板冒充实质审阅。
修改通用analyst/synthesizer skills，保留条件、证据、方向试验、反证结构，不携带teacher/slot最优幅度表。
本地提交推送、远端部署相同commit后，写versions/v(N+1)/review_approval.json，字段为：

```json
{
  "previous_version": "vN", "next_version": "vN+1",
  "finalized_sha256": "actual", "manifest_sha256": "actual",
  "review_commit": "deployed 40-character commit",
  "skill_sha256": {"analyst": "actual", "synthesizer": "actual"},
  "review_document": "repository-relative review.md", "review_document_sha256": "actual",
  "coverage": [{"path": "repository-relative reviewed-shard.json", "sha256": "actual"}]
}
```

`review-check --version vN+1`核验全部覆盖、证据、已提交文件与部署commit，再`launch --version vN+1`。
V2–V20均须前一版真实复盘，不允许空文件批准。每版共享R0、清空全部搜索历史；V20结束不再复跑。

## 清理与巡检

每链保留TOP1，四组评价及文档保存完成后校验快照并原子发布，之后仅删除非最佳pt.gz/h5/sdf/pdb等结构。
保留全部JSON/XML/prompt/响应、失败、清理哈希；共享R0和各版本最后100条/教师TOP1不删除。
程序故障时保留输入修复，不用质量失败标记跳过错误。不重新采样旧Unguided/Steer或零引导。
初始每20分钟巡检实际进程、完整轮次、错误、磁盘与质量/清理；稳定后按耗时调整，最高6小时。第5轮结束后及时进行GPT复盘。
只在实质变化/完成/异常通知。V20完整报告交付后删除plip-11762任务。

## 2026-10-22正式部署

- 固定运行目录：`/opt/MolSteer/runtimes/plip_council_7d611a45`；提交`7d611a45b89d8bf92655c2a16fb6476e7f1e1e11`。
- GitHub已推送；远端通过已校验GitHub提交身份的Git bundle fetch建立固定worktree。
- 远端相关测试63项通过，新增完整审阅来源门槛5项通过；无跳过。原生两教师各四个恢复入口共8项通过，坐标及pIC50最大绝对差均0，类别和RNG一致。
- 原生验收运行于902fc9de；至正式7d611a45仅追加脚本、测试及文档，冻结的adaptive源码指纹一致。[验收原件](deployment_20261022/ACCEPTANCE_COMPLETE.json)、[逐项结果](deployment_20261022/recovery_results.json)。
- 启动记录时间1792613726.9582732；初始V1工作PID124465。认证检查HTTP200，工作进程API与代理变量存在；只记录存在性，不保存值。
- `plip-11762`已恢复ACTIVE，初始20分钟；提示已替换为20版本×5轮、52会话复盘和新council根目录。
- 启动时处于新R0生成阶段；本记录不代表V1或20个版本完成。后续以结果根目录实际进度、FINALIZED及部署记录为准。
- 本次前置验收曾修复依赖覆盖缺失及原生TensorDict键遍历，错误与preflight adoption证据保留于operations。修复发生在正式R0之前，未改变科学配置。

## 首次巡检：bootstrap接口修复

首次工作PID124465在两教师200条R0和全部质量评价完成后退出，尚未执行R1。
Agent提出了超过单轮允许变动的调度；本应触发有界纠正，但自定义jsonschema异常中的Unset对象不能序列化，导致错误日志写入失败。
此外，新通用skill要求读取动作边界，而请求未显式提供最多两段、默认±1 tick等已有执行器约束。

- 修复提交：`6e86c4561e5842a89dffeba4a6be15726aa8c717`，目录`/opt/MolSteer/runtimes/plip_council_6e86c456`。
- 仅修改`deepseek_api.py`的错误诊断及`group_agent.py`的已有约束说明；模型、采样、奖励、RNG、质量、保留代码和冻结skills均未改变。
- 本地相关测试33通过、1因无torch跳过，修复事务3通过；远端相关36通过。另一个旧M2 supervisor测试因远端缺paramiko无法导入，该测试本地通过且当前council不使用该旧supervisor，未改变远端环境来运行它。失败及限定范围复测日志都保留。
- 使用专门的`repair_plip_council_bootstrap.py`，先dry-run再应用；仅允许R1前、无已选假设且恰好上述两个文件变化的情况。
- 审计目录：`operations/bootstrap_interface_repair_1792615405866108125/`。完整R0、基线及配置文件哈希前后相同；原生恢复验收因全部科学执行源码未变而保留。新指纹及原绑定、验收记录在journal中逐项保存。
- 原bootstrap全部输入/响应/工具证据归档在原目录的`saved_attempts/bootstrap_interface_repair_1792615405866108125/`；后续review-pack会纳入同一bootstrap逻辑会话，不能忽略失败尝试。
- 续跑初始PID162635，完整继承API及代理。后续以launch/workers及实际完成标记为准，保持20分钟巡检直至完整轮次稳定。
- 第一份失败响应未能写出token收据，其成本未知。综合报告通过`llm_missing_receipts.json`标注这类缺失；记录到的usage不代表完整费用，不能补造缺失收据。
