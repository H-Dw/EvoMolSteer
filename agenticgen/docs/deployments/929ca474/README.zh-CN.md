# V29固定部署与启动

科学与传输提交`929ca474d065d00ff6f6969a194273245cf88325`，runtime `/opt/MolSteer/runtimes/agenticgen_929ca474`。首个授权ghfast镜像成功，其他三个未尝试；Git pack一次clone后本地导入，固定commit/祖先/来源SHA及部署blob完整性核验，无逐对象REST API。安装前后131项历史关键hash保持。

传输前置检查按相同三个源码hash及原Junit核验复用：reused=True、fresh_tests_run=False、passed=True，原Junit SHA a63403748fdf00a862e2e9d2a5d24b855ce541175ef4a7b95d2e5a597b4e6a3f。这不是新跑传输前置测试。
相对V28科学提交，已纳入Git部署脚本和对应测试改动，正常部署门槛仅远端test_reviewed_deploy.py实际通过，Junit SHA cf797e837571d1e88c80afbad399cc45a75dfe3ba1136a588ca0f51b0bf207b3。本地无自动化测试，无LLM离线回放，无全套测试。科学Python兼容，显式复用原生验收，无prepare/native迁移或新原生采样。

V28有限review-check为Top1 R5G2、Top2 R5G1私有/同轮汇总，2教师4上下文、full_review_completed=false。V28较V27等权gap减少0.000102887，连续退步0、历史最好V17，无需回滚，目标未达成。
V29沿d7d35b13适配基底细化跨背景成本/收益配对与差分残差识别，U1及完整科学政策保持；不是V17字节一致恢复。详见[有限评估](../../experiments/v28_strategy_review/review.zh-CN.md)。

worker3678906，启动1792851526.8652952；实际核验1792851541.0439606存活、命令身份匹配，无launch后新failure。初始状态不代表首轮完成；不重复启动或为文档同步切源。
六份远端JSON逐字节SHA核验见source_manifest；远端`/opt/MolSteer/operations/agenticgen_deploy_929ca474`。

当前仅LLM协助gradient guidance，35分钟巡检保持。V30真实FINALIZED且质量/保留/清理完成，完整报告交付并commit/push保存最终代码/配置/skills及来源后删除agenticgen定时任务。
