# CK2优质初态对齐复测与远端容错执行

日期：2026-10-17。这里的checkpoint是已冻结的奖励、教师和控制程序，不是重新训练的FLOWR权重。

## 固定设计

按原始单目标Steer全部1000个终态的`pic50_on_rescore`排序，分数相同依batch、final slot排序。前5%、10%、20%分别为50、100、200个席位，对应11、14、18个不同初始根节点。保留原始结构失败，不额外做SMILES或root去重。

每个席位从其真实`root_slot`对应完整t=0状态恢复，包括坐标、原子、键、电荷、mask和口袋条件。同根不同席位共用初态，以NumPy SeedSequence([42,batch,root_slot,final_slot])派生不同续生成流；同一席位跨方法共用该流。整数seed不用于重建初态。三个阈值集合复用重叠结果，不构成独立重复试验。

正式方法：无引导、M2、Top10全局0–0.5、S3、恒定λ=.5/S2、C3、恒定λ=.25、C2、W1、W3、S1、旧线性衰减、Top10全局.1–.5、Top10局部.1–.5、N/C autoresearch R11、local motif R15_v2、ancestor R02_v3。共17×200=3400次完整推理。冻结每项原有教师、点云支持、奖励数学、剂量、裁剪和窗口，不重新调用LLM，不把旧B50/B24或不同cohort重复计为方法。λ是奖励混合比例，温度保持历史默认值。

统一B1原生线性Euler、100步至1、SDE、坐标噪声.2、类别噪声1、关闭类别Euler guard、最终预测校正开启。每个策略在自己的历史窗口生效，之后原生续推。禁用SMC与affinity梯度。保留M2原有全局原子步长裁剪和新严重受体碰撞回溯；不增加化学图一致性gate。后期局部控制器保持原实现。保存真实FLOWR VJP执行与窗口覆盖审计。

## 工程接口

准备：`python -m evomolsteer.elite_alignment.prepare --repo .`，生成冻结配置、200席位谱系/评分/随机流索引和18个无损初态文件。

队列：`scripts/run_elite_initial_alignment_queue.py --repo REPO --config CONFIG --work OUTPUT`。支持`--dry-run`、`--status`、默认断点恢复、显式`--retry-failed`。实际worker为`evomolsteer.elite_alignment.worker`。每条件加载一次模型，逐席位提交产物；前四个优先席位为排名1、50、100、200，计入正式预算。

运行环境仅出现在部署配置，不写入skills。程序和教师SHA256写入冻结配置；模型SHA256核对为f28e863b2b208718f3d3f85f09837c2f907a71123436db6f98a25d2f1858b6a0。远端代码必须先从GitHub拉取已推送commit；队列启动后不再依赖本机SSH、VPN、LLM API或实时拉取。

## 容错规则

普通单样本异常保存完整失败原因后继续。崩溃、OOM或超时记录当前席位，最多重建一次条件worker，跳过已有完成/失败席位继续；再次崩溃将该条件剩余项标为skipped_condition_error，执行其他条件。启动前奖励/教师校验失败仅跳过对应条件。M2或无引导失败不阻止其余策略，但报告缺失配对比较。

初始化及单样本默认超时各1800秒。父进程使用独立进程组管理worker，避免超时留下孤儿。独占锁阻止重复派发；每份结果在状态、结构和审计文件校验后原子提交。终态区分complete、failed和skipped_condition_error，队列区分complete、complete_with_errors、blocked_environment和interrupted。失败不补抽、不填0、不从计划分母删除。全局checkpoint/GPU/磁盘异常保留进度后阻断，环境恢复再续跑。重试显式保留旧失败证据。

## 存储和汇总

本地：`data/generated/elite_initial_alignment_20261017`。

远端：`/root/private_data/MolSteer/flowr_root/experiments/evomolsteer_elite_initial_alignment_20261017`。

初态按root去重，教师原文件共享；每席位保留终态SDF、.5和终态坐标、类别状态、评分/质量/能量、精简逐步控制及失败记录。不展开全程255特征表，不修改原始Steer，不删除历史受保护输入。临时文件只可在本次工作目录内清理。

每个集合报告全尝试评分均值及覆盖、根等权均值、全部评分Best/有效连通Best/严格基本结构Best、≥8.258901977539063及>8.704806327819824的有效分子产率、历史对应终态分数、失败、能量和应变覆盖。与新配对M2和无引导比较；旧400对照不直接复用。置信区间按根聚类，三个嵌套集合和多方法的边际区间不作为确认性多重比较显著证据。PB作为单独质量检查，未计算不得冒称通过。

## 验收与部署

测试实际候选数量/初态映射、原始控制器数值等价、λ=.5与S2值/梯度等价、结果校验、重复锁、真实子进程异常及超时后执行后继、断点恢复。GPU首例对原始native与零控制进行逐张量一致性验证。先本地commit/push，再远端拉取与SHA验证，后台脱离终端执行。监控至首批实际获得终态结构、评分和完成receipt后结束交互，队列继续运行。

本实验是从已知高分路径起点进行的回溯研究；不是从随机新起点泛化或等搜索预算优于Steer的证明。
