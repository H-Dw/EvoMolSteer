# V10固定部署与启动

科学运行提交：`d1163b7eb873e7305189973670341950018bd1bf`。
运行目录：`/opt/MolSteer/runtimes/agenticgen_d1163b7e`。

本地仅改skills和复盘文档，未运行本地测试。已提交推送GitHub，远端fetch同一完整提交后安装。复用已接受的科学源码/配置/原生验收；没有重新prepare、迁移协议、删除或重跑旧结果。安装保护的43项旧协议、R0和完成版本关键hash前后相同。

远端现有部署测试159项通过、无失败/跳过。V9最优轮review-check通过：2教师、4个策略上下文，full_review_completed=false。这些检查验证部署和门槛，不证明skills有效。

V10启动1792740063.1825883，worker1912935。启动核验1792740105.8528693确认存活、命令身份、API环境及三个代理变量完整继承，没有新增failure。该检查时pipeline尚未生成，属bootstrap启动阶段，不是首轮完成；不要重复启动。后续以launch/workers/pipeline和实际完成标记为准。

完整无凭据收据见[startup_and_receipts.json](startup_and_receipts.json)，本地下载SHA256为`8743428cd26727eab86a61d012643050086dd14863dc4015d901fe98684e6519`。远端来源目录为`/opt/MolSteer/operations/agenticgen_deploy_d1163b7e`。

使用原SSH控制台正常重连。此前失效会话与历史审批拒绝不等同于科学故障；其拒绝根因未确定。未改平台权限、未通过替代通道规避。后续文档提交不更换本次固定科学运行源。

巡检35分钟。V10 FINALIZED后只进行自己的最优轮评估，结合聚合performance决定V11。范围扩展最早可在该边界考虑，必须有边界响应/实际位移/未解决假设的证据；不是自动扩大。连续两次退步则遵守回滚门槛。
