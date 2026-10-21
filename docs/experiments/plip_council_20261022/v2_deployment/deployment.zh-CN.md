# V2 已批准并启动

服务器 ssh.zzai.scnet.cn:11762，结果根仍为 `/opt/MolSteer/generated_datasets/plip_schedule_council_5nn4_r12_h4_b100_crn_s42_20261022`。

V1 的 52 个会话已全量审阅，修订后的技能和审阅材料提交为 `530de8fc4b1b06c56b46a7875aab64928f81cc45`。远端固定部署目录 `/opt/MolSteer/runtimes/plip_council_530de8fc`；GitHub 提交身份、bundle、技能和审阅哈希均已核验。

本地测试 22 通过、1 因无 torch 跳过；远端 23 项全部通过。真实 review-check 验证 52 个会话，批准文件绑定 V1 FINALIZED、manifest、11 个覆盖分片、完整审阅文档、两份技能和固定提交。

V2 工作进程 PID 为 340996。实际进程继承 API 和代理环境，认证预检 HTTP200。V2 已绑定新技能，R0 和冻结配置保持一致，两个 bootstrap 优化历史为空。首次检查已看到 Top1 进入 R1 推理、Top2 就绪等待 GPU，无新错误。以后以实际 pipeline、完成/清理标记和进程为准。

本目录保留 `journal.json`、`review_approval.json` 和 `binding.json` 原件。完整远端测试与门槛日志保留在 `operations/v2_review_deployment_530de8fc/`。V1 审阅目录中的 pending_deployment_gate 是批准前的提交快照，实际状态已由本次操作推进。

继续每20分钟检查首轮推进。每版仍为两教师、四假设、每假设100候选、五轮，质量评价后仅保留逐链TOP1结构与全部分析。模型、奖励、幅度边界、随机设置和清理代码没有修改。V2同预算结果尚未形成，不能预先宣称新技能提高了pIC50。
