# V16待部署：GitHub API额度等待

科学目标`64ede898387316fcbd2ee6022016e4a66234dc52`已提交推送，V15有限策略评估完成，但V16尚未启动。
本次正常工具及SSH执行获准；固定GitHub源码获取返回HTTP403 rate limit exceeded，
无获取完成、安装、测试、批准或launch收据。SFTP确认V16目录不存在、V15 worker退出。
V15完整结果保留；不重复prepare/恢复/重跑，不改变冻结协议。

一次GET /rate_limit在1792785642.562293返回HTTP200：
未认证core limit60、used60、remaining0、reset1792786340，Retry-After缺省。
这是源码下载请求额度问题，不是上传文件过大或Codex审批拒绝。
原失败请求的响应头未取回；此reset来源是这一次额度查询，不将它冒称原请求header。
保留代理/TLS与Git对象缓存，不循环查询、换出口或制造新科学commit。

最早恢复时间为2026-10-23T20:13:20+00:00（北京时间2026-10-24T04:13:20+08:00），取响应reset后60秒安全余量。
未到时不发远端请求；保持35分钟自动续接，不持续等待。到时先执行：

```powershell
python -X utf8 agenticgen/scripts/deploy_reviewed_version.py --version v16 --commit 64ede898387316fcbd2ee6022016e4a66234dc52 --inspect-only --output .cache/v16_64ede898_inspection.json
```

确认无后续安装/绑定/活动进程/launch后，去掉--inspect-only对同一提交做一次有界正常部署。
已有部分安装先按固定收据处理；已有目标版本不重复批准启动。再次限流保留真实响应和缓存按时续接。
本次只改skills/文档，两端无单元测试；source/完成/锁/review-check尚需真实远端通过。
正常恢复不得使用旧失败教师修复入口，不归档删除任何候选。

V15仅Top1 R4G4、Top2 R5G3四上下文/40组评估，full_review_completed=false；
等权gap较V14减少0.000018497，连续退步0，历史最好V13。V16两项改写和U1保留见
[本版评估](../../experiments/v15_strategy_review/review.zh-CN.md)。V30后补充计划继续休眠。
