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
- 续跑已确认两教师bootstrap各完成2次成功API请求，四个假设通过校验；Top1进入R1物理batch400推理，Top2在就绪队列等待GPU。不能据此宣称整轮已提交或V1完成。
- 综合分析脚本另部署于`/opt/MolSteer/runtimes/plip_council_b799f467`（提交`b799f467aed38c89f3c35d09a03defe35e4d9c1a`）；仅新增未知成本标注及文档，不改变运行中的6e86c456推理进程。

## V1完成与分片复盘进度（2026-10-22）

V1两教师各5轮共4000条优化结果、质量评价及清理均已完成；`FINALIZED_v1.json`存在，PID162635正常退出，状态为`awaiting_master_review`。已使用b799f467版本执行综合analyze；不要再次运行prepare、重采样V1或将旧bootstrap错误视为未恢复故障。

首批GPT审阅覆盖26/52会话，全部属于第一位教师（bootstrap、R1–R5私有与汇总）；第二位教师26会话待审阅。逐条实质意见、证据绑定、报告快照和续接索引在`docs/experiments/plip_council_20261022/v1_review/`，先读`progress.json`与`review.zh-CN.md`。`coverage_005.json`目前仅含一条，后续应保留该条并补齐源分片剩余会话。不能把自动生成的来源索引当成实质审阅，也不能只检查覆盖文件存在。

完整证据包在远端`operations/v1_review_evidence.tar.gz`，SHA256为`c5b73ec77fe001d4b162cce2929c28d58dea79104bcfb19db9e77566dc7405f1`。本地`.cache/v1_review_evidence/`已解压，`.cache/read_council_review.py`可显示分片；Windows使用Python UTF-8模式，避免Unicode错误。实际文件及消息哈希必须继续核验。第二位教师从manifest顺序索引26开始，不重复第一位教师已保存的审阅。

当前尚未修改运行skills、批准或启动V2。完成全量52会话及通用技能修订后，再按上文执行提交、推送、固定commit部署、真实review_approval和review-check。保持20分钟自动续接；等待复盘是正常阶段，不是需要重新启动worker的故障。

## V1全量复盘完成与V2技能修订（后续记录）

已完成52/52会话，核对172个请求文件、204个去重消息哈希；coverage_000至010共11片，前十片各5条、末片2条。先前26条实质意见保留，coverage_005已补齐。以最新版progress和完整review.zh-CN.md为准，不再从索引26继续。

本次仅修改通用analyst/synthesizer skills及审阅文档。主要修正B−A方向、实际数组/区间对应、选择性incumbent与固定组的区别、集中Mean收益、互逆动作响应、有效位移与评分分层诊断，以及公共记忆核查。没有修改奖励、科学源码、R0、输入预算、并发或清理策略。

按上文完成本地测试/提交/推送、固定版本部署、全量review-check后才启动V2。实际批准和启动提交/PID记录于远端versions/v2/review_approval.json、deployment.json、根目录launch.json与operations；本地审阅文件里的pending_deployment_gate是提交时状态，不应覆盖后续真实启动记录。V2继续复用新服务器共享R0并清空搜索历史，5轮后再次全量复盘，预算仍为V1–V20。

## V2实际启动（2026-10-22）

- 审阅/技能提交`530de8fc4b1b06c56b46a7875aab64928f81cc45`已推送GitHub，远端GitHub API核对同一提交后以已校验bundle部署到`/opt/MolSteer/runtimes/plip_council_530de8fc`。
- 本地相关测试22通过、1因缺torch跳过；远端同范围23项全部通过。严格review-check返回`approved=True, previous=v1, next=v2, sessions=52`。
- V2工作PID`340996`，启动时间`1792620740.3930507`。认证预检HTTP200，实际进程API/代理变量存在；只核验存在性，不记录值。
- 已核验V2绑定使用新技能，与V1共用相同R0；冻结配置、科学源码及保存的V1/R0完成与分数文件哈希未变。两教师bootstrap的优化历史均为空，初始R0证据单独提供。
- 首次真实推进：两教师bootstrap已完成合法提案，Top1处于R1 generating，Top2处于R1 ready；无新error记录。此状态不代表R1已完成。
- 部署、真实批准、绑定及测试证据见`docs/experiments/plip_council_20261022/v2_deployment/`；远端操作目录为`operations/v2_review_deployment_530de8fc/`。远端master_review_progress_v1已更新为approved_and_v2_started。
- 首次部署核验脚本把bootstrap的优化历史误期望为一条，实际应为空（R0由独立summary提供）；检查已纠正并记录，没有修改科学数据或重启实验。
- `plip-11762`已改为巡检V2，仍每20分钟。首次完整版本耗时可参考V1约42.5分钟，但新工具调用及API时延可能改变总耗时。V2结束后审阅其新52会话，不能复用V1的覆盖批准V3。
- 本段及部署快照属于后续文档提交，不更换运行中的530de8fc，不因文档更新中断正常推理。

## V2完成与部分复盘（2026-10-22后续记录）

V2两教师各5轮、共4000条优化结果及全部质量评价和清理已完成，`FINALIZED_v2.json`存在；原PID340996正常结束，状态为`awaiting_master_review`。已使用原530de8fc部署执行analyze刷新V1/V2综合指标。不要重跑V2、重复prepare或因等待GPT复盘而启动worker。

本批已实质审阅Top1的26/52会话，绑定82个实际请求和99个去重消息哈希。证据、逐条意见及结果对照保存于`docs/experiments/plip_council_20261022/v2_review/`；先读progress.json和review.zh-CN.md。Top2的26会话尚待审阅，从manifest索引26的bootstrap开始。coverage_005当前仅一条Top1最终汇总，后续补齐时保留。

本地`.cache/v2_readonly_evidence`已缓存并核对全部52会话来源，不重复下载；读取器为`.cache/read_v2_review.py`，使用Python UTF-8模式。当前保存的传输前缀审计只用于去重，不算第二位教师的实质审阅。Top1暴露配对对象/方向混淆、G编号与提案谱系混淆、共同恢复偏移被归因为幅度、集中Mean收益被当作噪声及后两轮重复探索等问题；须先完成Top2反例核对再改通用技能。

V2相对V1：Top1逐链TOP1 Mean下降约0.0368，Top2变化约+0.00014；两教师R5整组选优Mean均略低。当前不支持宣称V2技能提高质量。V2十个GPU波次约30.47分钟，完整流程约45.92分钟。仍保留20分钟自动续接全量复盘。

尚未修订skills、写V3批准或启动V3。本次为文档提交，不切换远端部署。完成52/52审阅及必要修订后，再按完整review-check门槛推进下一版本。

## V2全量复盘完成与V3准备

已完成52/52会话、166请求及196去重消息哈希核验，coverage_000至010齐全。Top2剩余26会话已审阅，不能再从索引26重复续接。完整review和validation在docs/experiments/plip_council_20261022/v2_review。

V3仅修订两份通用技能，保持工具接口和全部科学源码一致：显式比较账本、实际数组/来源核对、集中Mean贡献、条件反转、保留TOP1下重复试验的信息价值。未新增工具或改变token预算。提交时状态为complete_review_pending_deployment_gate；以之后真实approval/launch为准。完成本地及远端测试、提交部署与全量review-check才启动V3，不重跑V1/V2、不覆盖PREPARED或ACCEPTANCE。

## V3实际启动（2026-10-22）

V2全量复盘和通用技能已提交推送dea29257cb82b0cf0ba05227e56234b6a55d5961。远端固定目录为/opt/MolSteer/runtimes/plip_council_dea29257；本地22通过、1因无torch跳过，远端23测试通过；review-check核验本版52会话后真实批准并启动V3，PID518238。API认证HTTP200，实际进程继承API和代理。V1/V2/R0及冻结配置哈希前后相同，科学源码未变。

两教师bootstrap历史为空，绑定新技能并复用相同R0。初次真实推进为Top1 R1 generating、Top2 R1 ready，无新错误；这不是整轮完成。部署、审核、进程和磁盘记录在docs/experiments/plip_council_20261022/v3_deployment，远端operations/v3_review_deployment_dea29257。master_review_progress_v2已更新approved_and_v3_started。review/progress中的待部署是提交时快照，不应据此重复批准或启动。

保持20分钟巡检，V3结束再审阅新52会话；不得复用V2覆盖批准V4。后续文档提交不改运行中的固定版本。

## V3分析接口失败的恢复工具

Top1 R1 G1因相对当前anchor的两格变动未获授权而被拒绝；采样和质量已完成，其余成功请求保留。Top2可继续到终态。恢复说明见docs/experiments/plip_council_20261022/v3_recovery/recovery.zh-CN.md。新增scripts/recover_plip_council_analysis.py和resume_plip_council_analysis.py仅改操作状态；不修改版内技能或科学源码。单次后台恢复可等待原worker结束（最长30分钟），核验全部门槛后只重试分析，仍从V3原dea29257运行目录启动。失败请求和额外API成本保留，每教师每版最多3次恢复，不无限重试。以远端operations/council_analysis_retry_*及analysis_resume_*的真实journal核验恢复结果；不能把已排队当恢复成功。

后续巡检已确认恢复成功：analysis_resume_v3_1792649202574674708为resumed，新worker667212仍固定dea29257。Top2五轮质量/汇总/清理全部完成；Top1 R1失败分析重试后已通过并完成保留清理，R2采样完成、进入analyzing（读取时质量标记尚未完成）。恢复事务核对12,605个保护文件哈希，未重复R1采样。记录位于v3_recovery/resumed_verification.json；原worker518238的退出及failure_1792649598336671154.json为已知历史失败，不代表新worker异常。不要重复排队、重跑或因本次文档提交更换运行部署；仍20分钟巡检，V3全量完成后再进入52会话复盘。

再后续巡检发现Top1 R2 G1分析失败，worker667212已退出；400条采样和质量完整，Top2五轮完整。首个响应越过一格动作限制，修复响应引用不存在的tool:1。已用原0ea0341e工具执行第二次有界恢复，新worker732766仍固定dea29257；原科学结果和成功请求复用，失败证据及usage保留。操作为operations/analysis_recovery_round2_v3_20261022，恢复事务council_analysis_retry_v3_1792651058875970356。本次只确认新请求已发出、R2 analyzing，未确认整轮成功。完整收据和诊断在v3_recovery/round2/；Top1本版恢复额度已用2/3，不得无限重试或放宽规则。

## V3完成与Top1完整复盘（后续记录）

V3已完成两教师各五轮4000条结果，质量、汇总、TOP1保留及清理标记完整，FINALIZED_v3.json存在，worker732766正常退出并进入awaiting_master_review。第二次恢复后R2–R5均完成，无需第三次恢复，不重新启动旧worker或重跑V3。已用原dea29257执行analyze；GPU波次累计27.58分钟，首次启动至FINALIZED约97.12分钟，包含两次分析失败等待。

本批GPT完成Top1的26/52会话实质复盘，绑定83个实际请求文件和99个去重消息哈希；42个格式约束及5个额外修复诊断逐项核验。文档、逐条coverage和报告快照见docs/experiments/plip_council_20261022/v3_review。Top2尚待，从manifest索引26继续；coverage_005仅Top1末轮汇总一条，续接保留。全部来源已下载到.cache/v2_readonly_evidence/versions/v3，目录名前缀为历史名称，读取器是.cache/read_v3_review.py，不重复下载或误用v2会话。

V3逐链TOP1 Mean为Top1 4.179882、Top2 4.165585，相对V2仅+0.000089/+0.000601，Top1仍低于V1。已发现非法动作修复后叙述未更新、虚构工具标签错误传播、集中Mean贡献被取消、区间数值错位及R0回放反证未改变汇总结论。先结合Top2反例完成全量审阅再改通用skills；本次不修改技能、科学配置、不写V4批准。保持20分钟自动续接全量复盘，完成52条后按严格门槛部署V4。

## V3全量复盘与V4准备（后续记录）

52/52会话已完成，覆盖163实际请求、190去重消息哈希、79格式前缀及7个修复诊断。11片coverage齐全，Top226条已审阅，不再从索引26续接。详见v3_review的review、validation与progress。两次后台恢复均已成功，不执行第三次恢复或重跑V3。

下一版仅重写通用analyst/synthesizer技能：动作合法性与历史比较分开；集中收益保留Mean；结构证据和指导增量分层；条件斜率与局部峰值区分；已执行清单和反证必须影响下一决策。没有添加工具或更改科学源码，不复制教师、槽位、原子特征规则或最优幅度表。提交时尚待固定部署、远端测试和真实review-check，之后再启动V4；不得仅凭本段当成已启动。

## V4实际启动（2026-10-22）

2eff057f7809d897f373ea2feb6bac0405f078cd已提交推送并固定部署于/opt/MolSteer/runtimes/plip_council_2eff057f。本地22通过、1因无torch跳过；远端23项全通过，严格review-check验证V3全部52会话，真实approval后启动V4，PID843641。源指纹、配置、R0及V1–V3绑定/完成哈希未改变，不覆盖PREPARED/ACCEPTANCE。

初次状态Top1 R1 generating、Top2 R1 ready；各bootstrap两份成功收据、合法四提案，优化历史及公开反思为空。进程命令身份、API代理继承已核验，无新error。证据在docs/experiments/plip_council_20261022/v4_deployment及远端operations/v4_review_deployment_2eff057f。master_review_progress_v3已批准并启动，不再从索引26续接或重做V3复盘。保持20分钟巡检；文档后续提交不更换运行中的2eff057f。

## V4完成与Top1部分复盘（后续记录）

V4两教师各五轮共4000条优化结果、全部质量、汇总、保留和清理均完成，FINALIZED_v4存在，原worker843641正常退出，状态awaiting_master_review。没有新增分析失败，不重启旧worker或重跑V4。使用原2eff057f执行analyze，十个GPU波次30.20分钟、完整48.07分钟；V4无已知缺失usage，V1历史未知费用仍保留。

本批GPT完成Top1的26/52会话实质复盘，绑定74个实际请求与86个去重消息哈希、36个格式约束，无额外修复诊断。逐条coverage和结果快照在docs/experiments/plip_council_20261022/v4_review。Top2另26会话尚待，从manifest索引26继续；coverage_005目前仅一条，补齐时保留。936个来源文件已缓存并核对，.cache/read_v4_review.py使用UTF-8读取，不重复下载或以哈希检查代替实质审阅。

Top1表现出实际同波证据纠正预测、去重、末轮撤回旧R0分数可转移假设等改进，同时仍有配对方向、区间/统计量、集中贡献、动作空间及上下文归因问题。对原始工具再次检查后已纠正复盘自身的区间最大值误判和两个不同数组混淆。须结合Top2反例后再决定技能修订。

V4相对V3逐链TOP1 Mean变化为Top1 +0.0000351、Top2 −0.0005725，R5整组选优Mean均略低，不能宣称技能有效。当前未修订技能、未写V5批准、未启动V5。本次只提交文档和远端进度元数据，不更换2eff057f科学部署。保持20分钟自动续接，完成52条及严格门槛后再推进V5。
