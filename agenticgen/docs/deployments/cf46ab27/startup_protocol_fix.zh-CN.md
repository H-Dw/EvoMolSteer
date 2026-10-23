# V11 普通启动入口对修复协议的兼容

2026-10-23：`cf46ab27` 已从官方 GitHub 对象安装；9 项获取器检查通过，53 项历史关键哈希保持一致。启动入口在写测试、批准或 launch 收据之前报 `KeyError: strategy_review_operation`，未生成 V11 版本目录或运行结果。

原因：V10 的已完成教师修复按设计将原 `strategy_review_operation` 移至历史字段，并以 `teacher_restart_operation` 绑定新的原生验收和保护清单。普通版本启动脚本仍无条件调用旧复盘迁移验证器；现有 `group_review.review_check` 已正确支持当前修复协议。

修复只在普通启动脚本按当前协议选择现有验证器，要求唯一有效事务及 `best_round_strategy/full_review_prohibited`，修复协议仍执行其原生验收、科学源码、历史哈希和复盘范围继承检查；后续新版本的有限 review-check 保持执行。批准记录保存实际 active_protocol_gate。没有改科学 Python、skills、配置或原生模型，没有重新迁移协议、删除候选、改 pipeline 或重跑 V10。新提交用于修复真实入口错误，不是为网络重试制造科学变体。

本地不运行自动化测试。远端固定环境仅执行 start/deploy/fetch 三个实际变化接口的直接相关检查，包括教师修复协议分派、原门槛失败传播、缺失/歧义协议和全量复盘模式拒绝。实际远端结果以部署收据为准。
