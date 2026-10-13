# 先前异常退出实验组的重试完成与远端复核

W10_analyst_extension、W11_both_extensions、C00_canonical 已在前一轮 v3 后台重试中全部完成。本次请求到来时队列已经结束，status=complete，failures=[]。

| 条件 | 完整角色链 | 真实 API action | 格式错误 / 科学引用拒绝 | 众数奖励 |
|---|---:|---:|---:|---|
| W10_analyst_extension | 6 | 18 | 0 / 0 | W2_middle_peak |
| W11_both_extensions | 6 | 18 | 0 / 0 | W2_middle_peak |
| C00_canonical | 6 | 18 | 0 / 0 | C1_robust_to_base |

本次已连接 root@ksai.scnet.cn:50496，通过本机 VPN 转发拉取到源码 commit 8c6840a7。随后在远端实际重新执行了三组完整性审计，核验真实 API 凭证、原始回答、曲线来源、角色交接与奖励编译签名。三份远端审计结果均与本地逐项相同，已下载保存于 remote_completion_audits/。

本次执行的是远端复核，没有再次提交已完成的角色调用，也没有额外进行 FLOWR 推理。先前后台重试总计 18 条完整角色链、54 个真实 API action；使用同一批次输入和已有数值配方，复用原来的 12 个数值 cohort 和冻结 M2/无引导/Steer 对照。原始失败与非法回答保留。当前没有待补跑的上述三组条件。

可复核入口：scripts/check_temporal_role_condition.py；配置：configs/experiments/temporal_guidance_evidence_v3_20261013/runner.json。机器记录见 retry_completion_20261013.json 和 remote_retry_verification_20261013.json。
