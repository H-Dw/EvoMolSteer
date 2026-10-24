# 当前远端工作目录：后续实验参考保留清单

盘点服务器`root@ssh.zzai.scnet.cn:11762`的`/opt/MolSteer`，时间范围2026-10-24 17:44:23 UTC–2026-10-24 17:45:52 UTC。读取175个目录的限定层级元数据与指定科学JSON，未运行远端命令、未修改/复制/删除远端文件、远端未访问GitHub，也未读取密钥值。下文是信息保留建议，不是新实验或清理授权。

本次实际看到8个generated_datasets直接条目（其中7个结果目录、1个scripts目录）、91个runtimes条目（70个目录、21个文件）、3个bare源缓存。目录自身stat字节不是递归体积，因此只对完整列出直接文件的权重/归档目录给出大小；没有测全工作区总占用或空闲磁盘。

## 必须联合保留的复现最小集合（P0）

不能只复制新实验结果根：它引用外部teacher、prepared protein、模型、环境和Git源。主实验根为：

`/opt/MolSteer/generated_datasets/agenticgen_5nn4_r12_h4_n25_core1hop_fixedstart_s42_20261022`

主结果25版的全部文档与来源应保留；结构层面已按既定策略仅保留50个最终TOP1快照×25槽和共享R0的50槽。其余正式/失败原生结构载荷已清理，不能从后续目录盘点中找回全部25,000条完整SDF/PT/H5轨迹。仍有逐结果条件、评分、质量、节点/端点文档、XML和学习证据可做后续分析。

|路径（相对/opt/MolSteer）|保留内容|后续用途|
|---|---|---|
|`flowr_root/checkpoints/flowr_root_v2.ckpt`|原v2模型权重；现有采样源码、checkpoint hash和版本锁|复现本次结果和比较新控制策略的必要基础|
|`generated_datasets/agenticgen_5nn4_r12_h4_n25_core1hop_fixedstart_s42_20261022`|完整正式结果文档：FINALIZED_v6–v30、reports、versions、binding/search_policy、所有quality/trials/条件/评分/学习/prompt/usage|主结果、控制响应、失败与策略学习的唯一正式证据；后续分析不需重采|
|`generated_datasets/agenticgen_5nn4_r12_h4_n25_core1hop_fixedstart_s42_20261022/shared_r0`|两教师各25条原始完整R0、noise/checkpoints/RNG/streams及COMPLETE哈希|固定起点和共同随机流比较；R0是有指导基线，不可用新采样替换|
|`generated_datasets/agenticgen_5nn4_r12_h4_n25_core1hop_fixedstart_s42_20261022/versions/*/*/incumbents/r05`|50份最终逐槽亲和力TOP1结构快照，25槽/份，共1250结构槽|结构再评价、轨迹传播和奖励分析；与主共享调度评价分开|
|`generated_datasets/agenticgen_5nn4_r12_h4_n25_core1hop_fixedstart_s42_20261022/teachers.json`|teacher SDF/PDB路径、原子/键映射、原核心及一跳区域、source_terminal/source_sdf哈希、pIC50|定义模板区域和相同蛋白框架；保留原分子身份/坐标来源|
|`generated_datasets/plip_guidance_5nn4_5ost_s42_20261020/teachers`|两目标原teacher.sdf、模板定义与PLIP相互作用资料|teacher来源独立于当前运行目录，必须与主结果联合备份|
|`generated_datasets/analysis_hiqbind_examples_20261020/examples`|相同prepared protein、输入靶标/配体与历史样例证据|teacher及基线引用该目录的target_protein.pdb，不能只备份新结果根|
|`generated_datasets/plip_guidance_5nn4_5ost_s42_20261020/baseline_profiles_graph_v2`|修正后的unguided/steer终态complex.pdb、native映射、PLIP XML/interaction/quality；配套baselines/*_records.json及baseline_source_records.json|本次100条基线的构象/亲和力比较及未来条件、结构和接触分析|
|`generated_datasets/agenticgen_5nn4_r12_h4_n25_core1hop_fixedstart_s42_20261022/protocols`|全部旧/新config、PREPARED、原生验收引用、active pointer和授权/兼容链|解释预算扩展、混合教师来源、接口修复和验收复用边界；不能只保留最新config|
|`generated_datasets/agenticgen_5nn4_r12_h4_n25_core1hop_fixedstart_s42_20261022/operations`|失败事务journal/deletion_manifest、failure_audit、aborted_resource_usage、旧binding/deployment/保护hash|区分失败与正式结果，复核1638次调用成本，保留实际故障和修复来源|
|`operations/agenticgen_deploy_*`|固定源码安装、传输、相关验证、审批、启动和完成收据；优先原混合来源及V17/V25/V30|复现与溯源；科学Python/原生验证和传输测试不能相互冒充|
|`runtimes/agenticgen_a245eb0a`|V17最好历史运行代码与skills、依赖链接；配套V25/5401fb41与V30/d823ddf5固定源|保留历史最好兼容基底及最终适配差异；其他历史commit由Git对象与来源索引归档|
|`envs/agenticgen_deps_20261022`|PLIP3.0.1、lxml、h5py外部依赖与版本元数据；Flowr环境及DTK/Torch锁|原生成/验收环境能否运行依赖外部路径；runtime源码目录本身不是完整环境|

当前最重要的源码锚点：V17 `a245eb0a`、V25 `5401fb41`、最终V30 `d823ddf5`；混合教师来源V7/V10/V11/V21的旧commit亦须保留。Git完整40位来源、代码/skills/hash见原最终source_manifest。当前根目录中的旧脚本与辅助部署脚本不能自动视为最新科学源；重用应以固定runtime和真实binding为准。

原环境应保持Torch/DTK/模型版本锁和运行时对外部依赖的路径关系。无需把鉴权值放入资料包；保留变量名、脱敏配置和授权来源即可。移动或归档目录前要考虑PDB/SDF引用、.pth、软链接及外部包位置。

## 高价值历史参考（P1）

|路径（相对/opt/MolSteer）|保留内容|后续用途/已见文件量|
|---|---|---|
|`generated_datasets/hiqbind_unguided_learning_s100_b20/archives`|30个历史target tar.gz与各自json清单/hash，配套results_summary/campaign_state/targets_resolved|后续跨目标实验与未筛选基线的原始参照，不要只保留5nn4优秀样例；直接文件4.480GiB|
|`generated_datasets/hiqbind_steer_learning_s100_b20_w050/archives`|30个历史target tar.gz与各自json清单/hash，配套results_summary/campaign_state/targets_resolved|后续跨目标实验与未筛选基线的原始参照，不要只保留5nn4优秀样例；直接文件3.253GiB|
|`datasets/hiqbind_test_zenodo_20069589`|测试靶标PDB/SDF与原数据划分、下载来源|后续独立目标与结构环境输入；直接文件0.169GiB|
|`generated_datasets/plip_guidance_5nn4_5ost_s42_20261020`|旧guidance runs、initials、settings、source_audit、acceptance/gradient/numerical证据、teacher_analysis和图表|控制函数和数值路径的机制参考；与V6–V30协议分开使用|
|`generated_datasets/plip_schedule_council_5nn4_r12_h4_b100_crn_s42_20261022`|V1–V5旧协议、完整报告和失败/恢复证据|对照协议演变与控制策略局限；不与新协议合并排名|
|`generated_datasets/plip_schedule_agent_top1_5nn4_r123_i4_s42_20261021`|更早单教师/分支实验、HANDOFF/PREFLIGHT及retained|保留早期机制和路径对照；记录科学条件差异|
|`generated_datasets/analysis_hiqbind_examples_20261020`|历史100条臂比较、采样曲线、区域分析、source audit和paper-source证据|用于解释teacher为何被选及历史steer/unguided差异|
|`knowledge`|既有控制假设、状态schema和函数表；源码中实际三份skills及每版hash|重用合理机制假设与格式协议；经验仍须由新实验验证|
|`flowr_root/checkpoints/flowr_root_v2.2.ckpt`|另一个已存在v2.2 checkpoint|未来独立模型对照资源；不是本次运行模型；直接文件0.938GiB|

5nn4与5ost原teacher/PLIP资料可支持条件定义与机制复查；它们不构成已授权新方向。更早V1–V5与本轮人口/协议不同，应分开解释。

历史基线画像应优先保留`baseline_profiles_graph_v2`。初版`baselines/.../unguided`有部分error.json，修正版中相应分子有完整complex.pdb、interactions与质量结果。旧错误证据可留作方法修复来源，但不能选用错误画像覆盖修正版。此次两历史臂各100条的结构读取完整，保留原子映射与protein_sha256才能重新核对蛋白固定框架。

## 可以重建或后续选择性归档的内容（P2）

|路径|实际价值|后续整理条件|
|---|---|---|
|`source_cache`|三个bare Git对象缓存ghfast_8a07160a/929ca474/d823ddf5；网络受限下便于复原对象；来源hash/commit索引比每份重复checkout更重要|rebuildable cache; consider consolidation only after independent backup/verification|
|`runtimes`|当前91个条目（70个目录、21个文件）中的重复源码、依赖安装缓存与构建副本；保留对应Git commit和关键源码/依赖链接后才可规划整理|no deletion now; exact disk size not measured|
|`dataset_cache/hiqbind_zenodo_20069589`|原下载tar.gz、inspection与split_metadata；下载日志/pid通常可重建；原tar可作离线备份，已验证有其他备份时可选择保留一份；划分/来源不得丢|do not delete automatically|
|`outputs`|8个test_*目录、.pytest_cache及旧测试临时输出；已纳入正式验收/修复收据的内容保留；其他临时测试载荷后续可评估重建|not assumed safe to delete without checking links and sole copies|

P2不是立即删除清单。不同目录可能是唯一副本或被当前环境引用；本次没有验证全部硬链接、软链接和内容等价关系，也没有删除任何文件。全部元数据与保留优先级见[remote_retention_inventory.json](remote_retention_inventory.json)。

## 建议的后续参考资料包

可按用途组织为五份逻辑资料：①模型/环境/固定源码；②teacher/蛋白/模板与基线；③R0和正式条件/评分/质量记录及最终结构；④策略/skills演进与版本结果；⑤失败事务和provider-ID费用证据。资料包应有统一相对路径索引和hash清单，保留原协议边界。这里只列组成，没有在远端创建新包或执行迁移。
