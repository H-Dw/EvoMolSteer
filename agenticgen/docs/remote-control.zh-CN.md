# 受限巡检与实验控制

2026-10-23：用户要求采用更合理的调用方式并验证。新增本地客户端
`agenticgen/scripts/read_remote_status.py`，将日常状态读取与部署、启动分开。

## 只读巡检

```powershell
python -X utf8 agenticgen/scripts/read_remote_status.py --version v10 --output .cache/v10_remote_status.json
```

只连接既定的11762服务器并验证已知主机指纹。密码在交互终端无回显输入，
不作为命令行参数、环境变量、输出或文件保存。客户端使用SFTP读取固定结果目录
内允许的JSON来源及launch中PID对应的`cmdline/status`，不提供远端shell、上传、
删除、进程控制、自定义主机或自定义远端路径。服务端仍是既定root账户，不能将
客户端接口限制描述为服务器账户已降权。

日常读取包括两个教师的五轮采样、质量、汇总、保留和清理标记；FINALIZED中的
报告/manifest哈希、1000条预算及五轮计数；pipeline完成状态；对应进程身份；
当前启动后的operations目录failure记录名称和时间。权限错误不能解释为文件缺失或进程
退出，FINALIZED或launch在读取时改变不能宣称完成。缺FINALIZED不自动恢复采样。

仅pipeline明确失败时额外读取该教师该轮的analyst_failures摘要，限制长度并遮蔽
URL及凭据形式文本，不扩展到其他轮或全量会话。该接口不读取凭据环境、原始日志
内容或递归遍历失败归档。磁盘空闲值没有SFTP
接口时明确记为不可用；实际故障的日志和磁盘诊断属于另一次明确范围的操作。
状态收据不替代下一版本的策略评估、review-check或部署门槛。

## 部署与启动

当前V10短引用修复使用`--citation-repair`选择固定的restart_v10_citations入口，
已部署c86706b5并启动，不再重复运行。日常仍只用read_remote_status。
新增`--interface-diagnostics`仅对已退出worker的当前失败组读取invalid receipt及
hash绑定回答，按实际error_type选择，不读取其他会话；原始回答只存本地证据。
本次读取和部署均通过正常工具权限，不能据此推断后续操作永久免审批。

V10定点诊断可加`--incomplete-diagnostics`，只对pipeline当前失败轮、
analyst_failures中Incomplete JSON的组读取严格命名的收据、原始回答及对应请求。
校验回答/请求hash和max_tokens，原文仅进入本地证据文件，不打印到终端；
不递归读取成功组或其他轮，不获取共享prompt全文。

本次96K修复使用`deploy_v10_repair.py --commit <完整SHA> --output <本地收据>`。
这是另一个有写操作的固定用途部署入口：远端GitHub fetch、固定安装及已测试
Top1归档/删除/五轮重启，明确保留Top2。它不是只读巡检，也不是免审批路径；
不接受任意远端命令、主机或路径，工具拒绝时停止，不换接口继续执行。

代码继续本地修改、检查差异、commit/push，再由远端GitHub fetch固定提交。
本任务不再运行本地自动化测试；必要兼容性验证仅在远端按改动范围执行。
部署与`start_reviewed_version.py`启动分开，使用既有完成状态、锁、固定提交及
review-check。只读客户端不能触发这些操作，也不能自动重启、删除或重跑版本。
本次客户端在本地运行，不更换V10科学运行来源，不在GPU环境安装Paramiko。
本地需已有Paramiko；当前验证环境为5.0.0，不自动升级依赖。

## 权限和验证边界

这是缩小操作能力范围，不是免审批通道。实际执行仍交给正常Codex工具策略；
若新只读请求被拒绝，停止依赖该请求的远端动作，不改工具、编码或传输继续执行。
旧通用控制台的发送成功不证明此接口或下一次定时运行一定获准。不要为
`python`、`ssh`或任意shell设置宽泛自动允许规则。

本次仅运行与新增客户端直接相关的25项测试，全部通过。科学代码、skills、奖励、
模型、采样和冻结协议均未变；测试不证明Guidance性能提高。真实读取结果另存
部署目录收据，在获得返回值前不宣称V10完成或权限问题已彻底解决。
