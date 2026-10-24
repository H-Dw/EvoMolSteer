# V30固定部署与启动

科学与传输提交`d823ddf53be9e38b2a26853a10a849410d24ad2d`，runtime `/opt/MolSteer/runtimes/agenticgen_d823ddf5`。首个授权ghfast镜像成功，其他三个未尝试；Git pack一次clone后本地导入，固定commit/祖先/来源SHA及部署blob完整性核验，无逐对象REST API。安装前后135项历史关键hash保持。

传输前置检查按相同三个源码hash及原Junit核验复用：reused=True、fresh_tests_run=False、passed=True，原Junit SHA a63403748fdf00a862e2e9d2a5d24b855ce541175ef4a7b95d2e5a597b4e6a3f。这不是新跑测试。
相对V29科学提交只有三份skills和文档变化，两端未跑单元测试、无LLM离线回放；正常tests_receipt状态not_run_skills_documents_only、passed=null，不称测试通过。科学Python兼容，显式复用原生验收，无prepare/native迁移或新原生采样。

V29有限review-check为Top1 R5G2、Top2 R5G1私有/同轮汇总，2教师4上下文、full_review_completed=false。V29较V28等权gap增加0.000668530，连续退步1、历史最好V17，无需回滚，目标未达成。
V30沿929ca474适配基底细化局部合法性与范围配额、精确反向与新分段/背景迁移，U1及完整科学政策保持；不是V17字节一致恢复。详见[有限评估](../../experiments/v29_strategy_review/review.zh-CN.md)。

worker3745725，启动1792855742.5293188；实际核验1792855753.293922存活、命令身份匹配，无launch后新failure。初始状态不代表首轮完成；不重复启动或为文档同步切源。
六份远端JSON逐字节SHA核验见source_manifest；远端`/opt/MolSteer/operations/agenticgen_deploy_d823ddf5`。

V30是最后一版。35分钟巡检保持；V30真实FINALIZED且质量/保留/清理完成，完成自身有限评估并交付原方向完整报告，commit/push保存最终代码/配置/skills及来源后删除agenticgen定时任务。不运行V31，不恢复已撤回后续方向。
