# V3部署与首轮启动记录

V2的52个会话已全量审阅，166个实际请求与196个去重消息哈希通过核验。两份通用技能和完整复盘材料提交为 `dea29257cb82b0cf0ba05227e56234b6a55d5961`，已推送GitHub；远端核验GitHub提交身份和bundle后，固定部署到 `/opt/MolSteer/runtimes/plip_council_dea29257`。

本地相关测试22通过、1因缺torch跳过；远端同范围23项全部通过。真实review-check返回52会话通过，批准绑定V2的FINALIZED、manifest、11个coverage分片、审阅文档、新技能和部署提交。没有替换模型、修改采样源码或工具接口，没有覆盖PREPARED/ACCEPTANCE，也没有重新采样R0/V1/V2。

V3工作PID **518238**；认证检查HTTP200，实际进程继承API和代理变量，只记录存在性。启动时间1792646901.0987399，worker登记1792646919.5812051。版本绑定与V2共享同一R0、配置和源码指纹；两位教师bootstrap的优化历史和公共反思均为空，新技能哈希一致。

首轮真实状态：Top1 R1 generating，Top2 R1 ready，两教师bootstrap均完成两次成功API请求并产出合法假设；没有新error记录。此记录不代表R1已完成。进程实际参数已核验。磁盘空闲1,300,892,917,760字节（约1.18 TiB），当前没有存储不足迹象。独立GPU查询工具nvidia-smi不可用，不据此报告显存数字；最初查询器因此退出，修正后磁盘和进程检查通过，实验进程没有受影响。

远端操作记录：`operations/v3_review_deployment_dea29257/`；`operations/master_review_progress_v2.json`已标记approved_and_v3_started。本目录保存journal、测试、review-check、approval、binding、launch和首轮快照。V2 review/progress.json中的pending_deployment_gate是部署前提交快照，不代表仍待批准。

V3继续两教师、四假设同batch400、5轮，完整预算仍V1–V20。保留每链TOP1及全部分析文档，质量完成后清理非最佳结构。保持20分钟巡检；参考V2完整流程约45.92分钟，不含之后GPT全量复盘。首轮稳定后可按实际耗时调整；V3完成后必须重新审阅其52会话才能批准V4。

本部署记录属于后续文档提交，不更换正在运行的dea29257，也不为同步文档重启。
