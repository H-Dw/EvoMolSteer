# 后台启动记录

2026-10-28 17:34:34（Asia/Shanghai）已在 `root@ssh.zzai.scnet.cn:11741` 启动两教师 λ 预实验队列。

- 结果根目录：`/workspace/agenticgen_lambda_preexperiment_20261028`。
- 执行代码：独立固定目录 `/workspace/uniguide_audits/distance-system-da1b535a`，实现提交 `da1b535ad0f71fb9e37e37773d6d0c58234dff4e`。
- 后台协调入口：`run_background.py`，由已推送 `95aac8ee` 提交提供，远端与本地 SHA256 一致。
- 后台主进程定位 PID 3362；17:35:08 通过真实命令、子进程和两个执行锁确认启动成功。PID 仅为本次定位记录，不代表后续持续健康。
- 启动确认时，6pyy 正在执行独立配置的工程验收，子进程 PID 3435；通过后进入 λ 标定，再处理 3w54。此时尚未产生预实验采样尝试。
- 每教师最大新增 350 次，共 700 次；失败不补样、不自动重试。每教师保存独立结果，单方失败后队列继续处理另一方。

运行状态见根目录 `status.json`，后台总日志为 `run.log`，各教师有独立 acceptance 和 calibrate 日志。结束时写入 `SUMMARY.json`；全部成功写入 `COMPLETE.json`，有失败写入 `FINISHED_WITH_FAILURES.json`。λ 位于各教师子目录的 `calibration/domains.json`，只有标定成功后才存在。

本轮只做启动确认；不建立对话监控，不等待预实验完成，不恢复 UG05，不启动正式 Guidance Agent。原有自动化未修改。机器可读启动证据为同目录 `MANIFEST.json`、`LAUNCH.json` 和 `STARTUP_CONFIRMED.json`。
