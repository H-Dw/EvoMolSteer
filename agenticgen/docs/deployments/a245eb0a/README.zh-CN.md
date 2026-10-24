# V17固定部署与启动

固定提交`a245eb0ac2ea4a7944d4fd7acfcd3491eaf5dbed`，runtime `/opt/MolSteer/runtimes/agenticgen_a245eb0a`。
原GitHub官方API逐commit/tree/blob及获取器SHA校验完成，限定部署子树、复用对象缓存，固定安装成功；没有SSH源码上传。
安装前后80项旧协议/R0/历史结果关键hash保持，科学Python指纹兼容，复用既有原生验收。

V16自身最优轮评估真实通过：两教师均R5G1，2教师4上下文，full_review_completed=false。
本次仅三份通用skills及文档变化，两端没有自动化测试；tests_receipt为not_run_skills_documents_only、passed=null，不能写成测试通过。
V16相对V15等权gap增加0.000434699，连续退步1，历史最好仍V13；本次尚未触发回滚。
V17分开条件Å容差、功能pIC50排序及预测区间的用途；按实测曲率/反转/条件余量决定步长，并识别跨背景共享的幅度外推风险。
保留小但真实的带内功能收益，不强制二分、网格或新配额，不注入教师数组。
能力假设、反证标准及skill哈希见[本版适配](../../experiments/v16_strategy_review/selective_adaptation.json)。

U1、coarse0.2、q0.1、max_range_probes0、max_changed_intervals2以及模型/奖励/采样/RNG/位移/预算不变。
双方96K私有/bootstrap、64K汇总、精确短引用/私有最多两次中性修正，从空动态历史开始。
worker 2764399，启动1792791877.9126267；核验1792791884.0330298存活且命令身份匹配，无新failure。
当时两教师pipeline尚未落盘，属于初始bootstrap阶段，不能称首轮完成。
不重复安装/批准/启动，不为后续文档同步切换科学源；35分钟仅对活动实验正常巡检。

五份远端JSON逐字节SHA核验见source_manifest.json。远端收据`/opt/MolSteer/operations/agenticgen_deploy_a245eb0a`。
V16结果、有限评估和成本见[评估](../../experiments/v16_strategy_review/review.zh-CN.md)。
