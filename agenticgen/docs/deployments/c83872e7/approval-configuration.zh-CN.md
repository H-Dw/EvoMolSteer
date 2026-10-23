# 2026-10-23：自动审核配置与代码发布方式

**后续用户指令：改用本地Full Access。** 用户在界面切换后又明确要求持久化该设置；实测当前文件已是`approval_policy="never"`、`sandbox_mode="danger-full-access"`。`approvals_reviewer="auto_review"`保留，但在never模式不触发自动审核。当前会话执行权限也为Full Access。旧定时任务此后已通过应用工具成功删除。下面的自动审核和删除被拒记录保留为先前过程，不代表当前仍被阻断；Full Access依然不能覆盖受管组织或服务端策略。

用户明确要求持久启用自动审核，并要求文件、代码更新通过原 GitHub 仓库提交和远端拉取完成。

已按照 [OpenAI Docs](https://learn.chatgpt.com/docs/sandboxing/auto-review) 核对并更新本机 `C:/Users/Wei/.codex/config.toml`：

```toml
approval_policy = "on-request"
approvals_reviewer = "auto_review"
sandbox_mode = "workspace-write"
```

写入前后经 TOML 解析对比，仅以上三个设置改变；其他设置保留，没有复制或输出凭据。Codex 本地配置加载成功。自动审核决定具体请求是否获准，受管策略仍优先；`never` 不会自动批准待审批操作。

现有 SSH 通道已成功传输之前准备的安装辅助脚本并执行安装。科学代码实际从 GitHub fetch，固定为 `c83872e7795aba2e8a6ff88d296443972f64a648`。首次 fetch 出现 GnuTLS 连接中断，原流程重试后安装成功；该错误是网络传输错误，不是审批拒绝或科学失败。

后续不再直接通过 SSH 上传代码或复盘进度文件。统一流程为本地修改、测试、commit、push，远端从原仓库 fetch，再检出指定提交、测试和启动。SSH 用于执行这些命令；仍须遵循工具实际审批结果，不能用 Git 路径规避拒绝。

本次更新定时任务提示的应用工具返回 `MCP tool call requires approval, but approval policy is never`，更新未成功。它与已经通过的 SSH 执行入口不同；持久配置不能保证当前应用会话的全部入口即时重新加载。当前聊天的权限控制需采用 **Approve for me**，受管设置若禁止该模式则由管理方处理。不直接改写自动任务文件绕过工具审批。已有十分钟间隔保持；后续实际部署状态以本目录新收据和 operations 当前段为准。
