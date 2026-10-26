# UniGuide–FLOWR 实施与启动记录

记录时点：2026-10-26 03:21（Asia/Shanghai）。这是启动快照，后续进度以远端产物为准。

实现位于独立分支 `codex/uniguide-flowr-agent`。本地工作树为 `D:\llm_controlled_generation\EvoMolSteer-uniguide`，原工作树已有的修改未改动。科学协议见 [实施协议](20261026_uniguide_flowr_implementation.zh-CN.md)，机器可读验收及启动证据见 [远端快照](../evidence/uniguide/20261026_remote_acceptance_and_launch.json)。

## 已实现

- 按准确 `node_id` 提取最终确认的两位教师；核查 SDF 成员哈希、已有分数及 native 槽位映射。两位教师分别含 17、22 个重原子，各有 9 个 PLIP 直接互作重原子，未扩展一跳邻居。
- 使用指定 `flowr_root_v2.ckpt`，已上传到指定路径并验证 SHA256。未使用服务器原有的其他版本权重。
- 逐步构造 detached Hungarian condition map，在干净终态预测上计算坐标损失，通过当前前向对带噪输入坐标求梯度，再按原生 FLOWR score/velocity 关系注入坐标增量。模型参数冻结，类别采样与 self-conditioning 保留原生路径。
- 实现四阶段共同尺度标定、三私有 Guidance Agent 与汇总选择、UG01–UG20 版本快照、当前 GPT 主 Agent 复盘接口、停滞研究触发和连续两次下降后的最优 UG skills 回退。
- 记录尝试预算、输入和输出哈希、逐步梯度及增量向量、随机流、质量检查和结构保留清单。数值失败保留证据并计入预算，没有自动减弱尺度或补样。

## 远端验证

执行环境为用户指定服务器的 `/workspace/EvoMolSteer`。仅在远端运行测试，使用现有 Torch 2.9.0 / DTK 环境。补充依赖 h5py 3.11.0 隔离安装于 `/workspace/uniguide_deps`，未升级 Torch 或 DTK。

14 项测试已通过：验收入口运行 11 项；随后新增的 3 项版本冻结、UG20 终止和 Agent 提案选择契约测试单独在同一远端环境通过。三项新增测试使用替身 provider 验证接口约束，实际 DeepSeek API 尚未通过连通性和凭据验证。

两位教师的真实模型验收均通过：

| 检查 | 6pyy | 3w54 |
|---|---|---|
| 无引导增量的原生短轨迹等价 | 状态、self-conditioning、时间、随机流完全一致 | 完全一致 |
| step10 前恢复并继续到 step12 | 上述四项完全一致 | 完全一致 |
| 真实输入梯度方向有限差分相对误差 | 0.27%、1.09% | 0.55%、0.91% |
| 模型参数冻结 | 通过 | 通过 |
| 固定物理批内分组一致性 | 通过 | 通过 |

3w54 的原生 50 行网络调用曾在工程验收中超出显存。因此保持逻辑人口与原生随机数协议，改为固定 10 行网络前向、pocket 编码和 VJP；不足 10 行时填充并丢弃填充输出。比较固定物理批与拆分逻辑批后结果一致，没有放宽数值容差。原始失败日志均保留。

当前科学引擎提交为 `24b863c671a4e1f9862754d23cd91cb3e046a599`。新增契约测试提交为 `78eb5f64636ebf7a6f39a6d2776dba95b25fc81f`，未改变科学引擎。远端在标定期间保持引擎提交固定。

## 实验状态与继续入口

结果根目录为 `/workspace/agenticgen_uniguide_flowr_20261026`。标定进程已启动；记录时点已完成 17 个粗标定配置（170 个候选），下一个配置运行中，已预留 180 次尝试。R0 与正式 UG 版本尚未运行。这些是粗标定结果，尚不能据此确定共同初值或报告优化收益。

查看状态与日志：

```bash
cd /workspace/EvoMolSteer
bash agenticgen/scripts/run_uniguide.sh status
tail -n 30 /workspace/agenticgen_uniguide_flowr_20261026/calibration_pipeline.log
```

标定完成后，满足 API 条件才启动正式版本：

```bash
bash agenticgen/scripts/run_uniguide.sh run-version UG01
```

每版完成后写入 `AWAITING_GPT_REVIEW.json`。当前 Codex 主 Agent 应实际读取全部指定证据，必要时研究文献和代码，在本地更新 skills 后提交并部署，再用 `accept-review` 绑定修改和证据。此接口不会伪造 GPT 复盘或自行声称完成文献检索。

启动快照时，非交互 SSH 进程未读取到 `DEEPSEEK_API_KEY`，无凭据的 API 连通性检查超时。后续核查已确认：服务器早已配置密钥和代理，原先将进程未继承变量表述为服务器缺少配置不准确。具体原因与修复见下节。没有创建新的本地 API 代理，标定始终独立运行。

GitHub 直连及四个指定 mirror 均依序超时。代码先在本地成功提交和推送 GitHub，再通过 SSH 传输已推送提交的 Git bundle，远端执行 `git bundle verify` 并检出固定提交。凭据未交给镜像，也未写入仓库。

## 证据路径

- 当前 GPU 验收：`acceptance/runs/1792955229428978455/`
- 新增契约测试：`acceptance/additional_contract_tests.log`
- 已验证教师与初始状态：`teachers.json`、`TEACHERS_PREPARED.json`、`INITIALS_VERIFIED.json`
- 标定完整结果及数值失败：`calibration/`
- 尝试预算：`attempts.json`
- 静态启动快照：`implementation_status_snapshot.json`

上述路径均相对于远端结果根目录。保留所有现有产物，没有执行结构清理。

## SSH 环境加载问题已定位

实际连接端点为 `root@ssh.zzai.scnet.cn:11741`，服务器主机名为 `crdnotebook-2114384248860155906-daweihuang-76949`。外部端口经服务端转发，服务器内部 `SSH_CONNECTION` 的目标端口为 22，不能据此判断连接了另一台服务器。

`/root/.bashrc` 第 6 行为 `[ -z "$PS1" ] && return`；密钥的 export 位于第 120、129 行。Paramiko `exec_command` 创建非交互 shell，普通 `bash -c` 与 `bash -lc` 都没有绕过该判断。因此执行进程既没有继承 `DEEPSEEK_API_KEY`，也没有继承已有的小写 `http_proxy`、`https_proxy`。仅改为 login shell 不足以解决问题。

实测 `bash -ic` 与 `bash -lic` 均能读取密钥；在交互式 shell 中再加载 DTK 环境后，密钥和小写代理变量仍然存在。使用该环境对 `https://api.deepseek.com/models` 发起带鉴权请求，返回 HTTP 200，模型列表包含 `deepseek-flash`。检查只输出是否存在、HTTP 状态及模型可用性，没有输出密钥或代理凭据。

启动脚本现已在 `run-version` 缺少密钥时，通过一次交互式 shell 加载现有配置并重新执行自身；有显式传入的密钥时保留调用方环境。如果加载后仍缺少密钥则明确失败，避免递归重试。此变更只涉及启动环境，不改科学引擎或正在运行的标定。
