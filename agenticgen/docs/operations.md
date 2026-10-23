# Deployment and continuation

2026-10-24用户最新策略要求：**回滚后继续选择性吸收经验模块，允许主Agent结合实际结果重写Guidance指引**。
已在恢复后的V10 skills基础上融合两项候选模块：机制解释直接决定下一合法动作、有效背景上的时间维持/释放配合。
保留带内功能排序和及时利用，不整套照搬文献、不增加全量审阅或诊断清单。主Agent可据实保留、删减、合并和改写。
详见[选择性模块说明](strategy/20261024-selective-modules-after-rollback.zh-CN.md)。三份skills/文档已准备，未部署。
V13仍绑定ab91ba50，版内冻结；本次不重启/重跑。V13完整后完成自身有限评估，再决定下一正常版本如何采用。
下方“未加入新Guidance条款”描述V13已启动的历史恢复状态，不是禁止后续版本学习文献。
不冒称适配后的skills为字节一致恢复，不静默绕过门槛；科学协议、35分钟巡检保持。仅skills/文档，无两端测试。

2026-10-24最新：**V12完整完成并有限评估，V13按规则恢复V10策略后已真实启动**。
V12两教师R5G4代表Mean4.212099457/4.209410229；相对V11等权gap增加0.000889187，
连续退步2触发历史最好V10恢复。V13固定`ab91ba5094aaec0f26ce4723e9296000ac08611c`，worker2418839，启动1792774928.0182397。
官方API获取/固定安装成功，64项历史关键hash保持；仅两份直接相关远端检查通过，无本地测试。
真实V12→V13有限review-check及原强制回滚门槛通过，full_review_completed=false。
运行时两份skills逐字节恢复V10，完整范围/控制U1/粗尺度0.2/q0.1保持历史绑定，
保留当前96K短引用/中性修正兼容接口；主Agent机制框架保留，未加入新Guidance条款。
包内科学Python不变，复用原生兼容性，不重新prepare/native/删除候选；初始核验不代表首轮完成。
V12健康完整运行33.072分钟，正式GPU8.866分钟，58份API输入3343472/输出714851，无作废尝试。
见[实际部署](deployments/ab91ba50/README.zh-CN.md)和[有限评估](experiments/v12_strategy_review/review.zh-CN.md)。
V6–V12均不重跑；下方V13待部署、V12运行等均为历史，不重复启动V13，35分钟活动巡检。

2026-10-24最新：**V12已FINALIZED并完成有限评估，V13必须恢复历史最好V10**。
两教师各五轮1000正式记录和质量/汇总/保留/清理齐全，无新故障；原e640a40e analyze成功。
只读两教师R5G4私有/汇总四上下文与40组聚合表现，full_review_completed=false。
Top1/Top2代表Mean4.212099457/4.209410229，较V11一升一降；等权gap增加0.000889187，
连续退步2，触发既定回滚。V13恢复V10绑定analyst/synthesizer及U1/粗尺度0.2/q0.1完整策略，
保留已接受的96K短引用/中性接口修正。主Agent有限机制复盘框架保留，不叠加新Guidance条款。
V12耗时33.072分钟、正式GPU8.866分钟；58份API输入3343472/输出714851，无本版作废尝试。
见[本版有限评估](experiments/v12_strategy_review/review.zh-CN.md)。正常启动器补齐真实rollback绑定，
尚待固定提交部署和两份直接相关远端检查；没有本地自动化测试，尚未批准/启动V13。
V6–V12均不得重跑/prepare；下方V12启动和更早待部署段落为历史。35分钟巡检保持。

2026-10-23最新：**V11完整结束并完成有限评估，V12已真实批准启动**。
V12固定`e640a40e31a6f3885cc884c48a45a855e2f707eb`，worker2338101，启动1792770456.370391，runtime `/opt/MolSteer/runtimes/agenticgen_e640a40e`。
官方GitHub API获取/固定安装成功，60项旧协议/R0/结果关键hash保持；仅skills/文档变化，
本地和远端单元测试均未运行，科学Python指纹/原生兼容性及有限review-check实际通过。
V11评估只读Top1 R5G2、Top2 R4G2私有/汇总，共4上下文与40聚合组，full_review_completed=false。
V11相对V10等权gap增加0.000091705，连续退步1、不回滚；失败费用已纳入，目标仍未达成。
V12应用用户文献框架的三项机制/时间作用/预算能力，补充背景响应及自身动作域，U1/科学设置保持。
初始核验不代表首轮完成，不重复启动或为文档同步更换科学源。见[实际部署](deployments/e640a40e/README.zh-CN.md)。
V6–V11不prepare/重跑；下方待V12部署或V11运行段落均为历史。35分钟正常活动巡检。

2026-10-23最新：**V11已FINALIZED，有限策略评估完成，V12待固定部署**。
两教师各五轮1000正式结果及质量/汇总/保留/清理完整，worker2281628退出、无新failure；
原07fa7423 analyze及完成/report/manifest绑定通过。只读Top1 R5G2、Top2 R4G2私有与汇总四上下文，
结合40组聚合结果，full_review_completed=false。见[本版策略评估](experiments/v11_strategy_review/review.zh-CN.md)。
Top1/Top2代表Mean4.212057199/4.211230860，较V10一降一升；等权gap增加0.000091705，
连续退步1、不回滚。目标未达成。正式61+作废33=94份API，共5135525输入/1044510输出tokens，
作废GPU另304.469秒，失败分数不混正式结果。最后成功重跑24.099分钟，总84.981分钟含故障等待。
V12保留用户框架的三项机制/时间作用/预算假设，并按本版证据补充背景响应与提案自身合法域；
保持U1与全部科学协议。只改skills/文档，无本地测试/LLM回放；尚未批准或启动V12。
V6–V11不得重跑/prepare；下面V11启动和准备记录均为历史，不再调用任何失败修复事务。

2026-10-23策略续接：用户指定框架已转为三份 skills 的通用调控推理，
见[策略说明](strategy/20261023-guidance-mechanism.zh-CN.md)。这是下一边界的准备，未部署或批准 V12。
V11 仍固定07fa7423，不因本地skills提交重启或切换科学源。完成后按更新的主 Agent skill
做V11自己的最优轮有限评估，结合结果/回滚决定下一版；不恢复52会话流程、不增加本地测试。

2026-10-23最新：V11 Top1接口失败已按既有授权修复并完整五轮重跑；Top2已完成五轮保留。
固定`07fa7423a64c497c8550fda99a040f3437ae3eec`，worker2281628，runtime `/opt/MolSteer/runtimes/agenticgen_07fa7423`。
首答重复调度，修正答JSON结构错误，两答均stop且未耗尽96K；私有中性修正额度1→2，完整字段提示增强，
不改变冻结skills或科学设置、不手改方案。四份相关远端检查及两教师原生验收通过，无本地测试。
39218项保护hash一致，500失败候选归档后删除，Top2原cea3611c来源/结果不变；失败费用单列。
真实V10→V11有限review-check通过并已launch，核验存活且无新failure，尚未确认新Top1首轮完成。
见[修复部署](deployments/07fa7423/README.zh-CN.md)。不重复修复事务、不重跑Top2或V6–V10，
旧failure_1792763949081064684已处理；35分钟仅检查当前活动worker，完成V11后再评估V12。
下方cea3611c运行、403/超时等待和V10修复等段落均为历史快照。

2026-10-23最新：GitHub API限流已在一次有界恢复中解除，**V11已批准并启动**。
固定`cea3611c7c07b47868038a20a71877706e845d97`，worker2175969，启动1792762086.6861448，runtime `/opt/MolSteer/runtimes/agenticgen_cea3611c`。
额度诊断真实返回未认证60次/小时、剩余51；不是上传文件过大。本次只再下载20个blob，
GitHub对象/安装/三份相关远端检查及有限review-check（2教师4上下文，full_review_completed=false）通过，
53项历史关键hash保持。未运行本地测试，原科学Python不变、复用原生验收；V11保持U1。
启动核验存活、命令身份匹配、无新failure，尚未确认首轮完成。不重复安装/批准/启动。
见[实际部署](deployments/cea3611c/README.zh-CN.md)及[限流原因/恢复策略](deployments/cea3611c/github_rate_limit.zh-CN.md)。
35分钟仅检查活动V11；下面403等待、待cf46/cea部署段落为历史快照，不再执行其重试命令。

2026-10-23最新实际状态：V11尚未启动，固定待部署目标改为`cea3611c7c07b47868038a20a71877706e845d97`。
限定部署子树的获取修复已验证成功：cf46ab27获取/安装完成，9项获取器远端检查通过，53项旧hash保持。
随后发现普通启动入口不兼容V10当前teacher_restart协议，报KeyError；此真实代码问题已在cea3611c修复，
只分派到现有验证器，不改科学Python/skills/配置或原生协议。V10有限评估已完成，V11保持U1。
新目标首次获取返回GitHub API403 rate limit exceeded，无安装/测试/批准/launch收据。
无reset响应头，保守等至2026-10-23T13:11:14.042253+00:00再按同一目标inspect-only及一次有界恢复，
期间不重复请求/换代理、不重做已完成版本。不是审批拒绝。本地没有运行自动化测试。
见[当前恢复计划](deployments/cea3611c/README.zh-CN.md)；以下待部署/运行段落均为历史快照。

The second bounded cf46ab27 acquisition also timed out. Fixed endpoint HEADs returned
HTTP200 with rate capacity; a20s recursive-root GET transferred only370029 bytes.
The fetcher now verifies ancestor trees and recurses only the three deployment
subtrees (gzip enabled), refusing uncached unrelated trees. A separately pinned
fetcher-commit can repair transport while keeping scientific target cf46ab27 unchanged.
Its committed standalone checks run remotely before acquisition; no local tests.
See [evidence and scope](deployments/cf46ab27/fetch_repair.zh-CN.md). Await actual receipts.

Latest deployment attempt: fixed cf46ab27bf953c05ecd2e604539596b052348d2a is pushed.
The tool allowed normal authenticated execution, but official GitHub object acquisition
hit its 240-second total bound (exit124). No fetch-complete/install/test/approval/launch
receipt was produced; V11 has not started. This is not an approval denial and no new
HTTP503 was established. Keep cached objects and the same target commit, stop this
attempt, inspect its fixed receipts on the next normal continuation before one bounded
retry. Do not re-run V10 or any failed-teacher transaction. See
[receipt](deployments/cf46ab27/README.zh-CN.md).

Latest: V10 finalized (two teachers, five rounds, 1000 formal outcomes), worker2116574
exited. Fixed c86706b5 analyze succeeded. Limited V10 best-round review is complete:
Top1/Top2 R5G1, four contexts and forty aggregate groups, full_review_completed=false.
No rollback is required. V11 holds U=1; no automatic expansion. See
[review](experiments/v10_strategy_review/review.zh-CN.md). V11 awaits actual deployment
receipts; do not re-run any V10 repair or finished version.

Normal V11..V30 deployment now has `agenticgen/scripts/deploy_reviewed_version.py`.
It defaults to verified official GitHub API objects, derives the old runtime from
the preceding completed launch, and calls the normal fixed installer/start entry.
No source upload, repair transaction or arbitrary-command interface. Inspect-only
reads fixed deployment receipts. Existing target versions reject repeated deployment.
The normal start entry selects remote checks from the actual changed interfaces;
skills/docs-only changes run no unit suite. This engineering change runs only its
two related test files on the remote host. No local automated tests.

Current source-fetch policy: this server/proxy has repeatedly returned Git HTTPS
CONNECT 503. Prefer the verified official GitHub fixed-object fetcher on subsequent
deployments; do not retry Git first on every version. Reuse cached objects and verify
the exact commit/tree/blob identities before installation. A checkout-time missing-blob
fetch can fail even after the initial fetch succeeded. Follow the bounds and recovery
rules in [the project plan](plan.zh-CN.md#github代理503固定提交获取与失败恢复).
Do not rerun the completed V10 repair entry to fetch code. Integrate the same fetcher
into the appropriate next-version deployment entry, retaining its normal gates.
This policy update does not probe the remote network or restart the active worker.

Latest execution: V10 Top1 restarted as PID2116574 at1792750598.463112 under fixed
c86706b565c81949192f0147d93298dde55d9355 after the R1G2 citation failure. Top2's
original d1163b7e results remain complete. No local tests ran; 86 relevant remote
checks and native acceptance passed. All32146 protected hashes match;100 failed
candidates/693 files were archived then excluded. Actual V9 limited review gate
was rebound and launch completed. Do not stage/validate/finish/approve/launch again.
See [deployment](deployments/c86706b5/README.zh-CN.md). Old worker2045658 and
failure1792748405573360368 are historical; inspect only the new active launch.

2026-10-23 current validation policy: remove all local automated testing from this task,
including executable-code edits. Review the diff, commit and push; run only necessary,
change-specific compatibility checks in the fixed remote runtime. Do not delete existing
tests or historical receipts. Skills/document edits use the next real experiment and
limited best-round assessment; token changes need request/configuration and response
completion checks, not a scientific full suite. Context compression changes the actual
agent input: check required conditions, evidence references, visibility and parsing if
enabled. Retain relevant remote checks for recovery, RNG or deletion-transaction changes,
plus provenance, completion, locks and review-check. This supersedes every historical
local-test instruction below. Do not restart current V10 for this workflow-only change.

2026-10-23 17:36北京时间最新：96K修复已固定部署098467f5并真实启动V10 Top1
worker2045658；Top2原五轮及31729个保护文件哈希保留。远端64相关测试与两教师
原生验收、4上下文有限复盘门槛通过；失败Top1四轮400候选已归档后删除并从R1重跑。
启动身份检查通过，无新failure，当时bootstrap未完、尚不称首轮完成。不要重复
stage/finish/delete/批准/launch。见[部署收据](deployments/098467f5/README.zh-CN.md)。
Git HTTPS代理503通过官方GitHub API固定对象校验获取解决；不是工具审批拒绝。
只按35分钟检查活动worker；V10完整后再做自己的最优轮评估。下方待部署段落为历史。

2026-10-23最新：本次定时受限读取实际成功，Top1 R4G2为length且65536输出
全部用于reasoning、content为空。用户指定96K，修复设置私有/bootstrap98304，
汇总64K及原skills/科学设置不变；完成Top2保留。固定用途GitHub部署及V10
修复脚本见[96K修复](repairs/20261023-v10-output-budget.md)。尚须以新部署收据
确认安装、验收、归档删除、有限review-check和重启各阶段，不将本地修改当启动。

2026-10-23 16:47北京时间最新验证：受限只读SFTP客户端真实读取成功，
25项相关本地测试通过，客户端提交26d91fc5已推送；本次未执行远端命令或写入。
V10 worker1912935已退出；Top2五轮完整，Top1 R4G2回答未完整结束导致分析失败，
FINALIZED_v10不存在。保持Top2和历史结果，不启动V11。下一步定点核对
该次incomplete receipt的finish_reason/usage，按既有授权测试修复并保留已完成教师。
这是科学分析故障，与历史工具拒绝分开。见[真实验证](deployments/26d91fc5/README.zh-CN.md)。
此次成功不保证定时调用已验证，权限仍以实际工具决定；不再轮询旧控制台90887。

2026-10-23 scoped-control update: use the local bounded SFTP reader documented in
[remote control](remote-control.zh-CN.md) for normal status reads. It accepts only
the experiment version and a local receipt path; it cannot execute remote commands,
upload code, or control workers. The real request remains subject to tool approval.
Do not use another transport if this narrowed request is rejected. Deployment/start
remain separate GitHub fixed-commit operations. This local client does not change
V10's scientific runtime. The validation receipt determines whether live access worked.

2026-10-23 07:25 UTC monitoring update: sending the active V10 read-only check
through authenticated session90887 was rejected before execution by the tool
approval policy (`approval required by policy, but AskForApproval is set to Never`).
No new remote state was obtained. Last confirmed state remains the startup check
at1792740105.8528693; do not infer V10 completion or a scientific failure. The
successful previous reconnect/deployment remains valid historical evidence, but
does not establish that this later call was permitted. Preserve the worker; do not
repeat the denied operation or switch transport to evade it. No repeat notification
for the same permission problem. See [receipt](deployments/d1163b7e/monitor_transport_blocked.json).

V10 now started (2026-10-23): fixed d1163b7eb873e7305189973670341950018bd1bf,
worker1912935, launch1792740063.1825883. GitHub fetch/install, 159 remote tests,
and V9 best-round gate (2 teachers/4 contexts, full_review_completed=false) passed.
All 43 protected prior hashes match. Startup command/API/proxy identity verified,
no new failure; pipeline was not yet present at the startup check. This is not a
completed round. Do not repeat V10 approval/launch. The latest receipt is
[V10 deployment](deployments/d1163b7e/README.zh-CN.md); below pending-deployment
wording is a historical pre-launch snapshot. Next active-run monitoring is on the
35-minute schedule. After V10 finalization assess its own best rounds for V11.

Latest continuation (2026-10-23): the original SSH console was reconnected through
normal authentication after its previous process expired. Commands now succeeded;
the cause of the earlier approval rejection remains unknown. No permission setting
or alternate transport was used. V9 is FINALIZED, worker1834410 exited normally,
all ten round quality/council/retention/cleanup commits are present, and analyze
completed under c8671c29. Do not restart V9. Historical blocked receipts below are
superseded by [restoration](deployments/c8671c29/transport_restored.json) and
[completion](deployments/c8671c29/completion.json).

V9's limited best-round assessment is complete: Top1 R5G2 and Top2 R5G4, four
strategy contexts and 40 aggregate groups. See [review](experiments/v9_strategy_review/review.zh-CN.md).
Only generic skills and documents changed; skip local tests. Prepare V10 with the
existing GitHub fixed-checkout/install/start_reviewed_version procedure, without
new science preparation, native migration or full-session review. U remains 1.
One successive regression does not trigger the two-regression rollback. Keep the
35-minute heartbeat and use actual launch receipts before describing V10 as started.

Historical transport snapshot (superseded by the successful reconnect above):
Monitoring transport update (2026-10-23 06:49 UTC): the active V9 read-only
health-check send through the existing SSH console was rejected before execution
by the tool approval policy (`approval required by policy, but AskForApproval is
set to Never`), despite the advertised Full Access configuration. No new remote
state was obtained; this does not establish a scientific failure. Preserve V9
and its last confirmed startup state. Do not restart or change transport to evade
the rejection, and do not repeat the same denied check while execution permissions
are unchanged. See [blocked check receipt](deployments/c8671c29/monitor_transport_blocked.json).

V9 started (2026-10-23): fixed `c8671c293750db24fdd8022a1f0ff79181d4c684`,
runtime `/opt/MolSteer/runtimes/agenticgen_c8671c29`, worker1834410. Remote tests
159/159 passed; V8 best-round gate accepted two teachers/four selected contexts,
full_review_completed=false. Startup identity/API/proxy inheritance verified;
pipeline was not yet created at that check and no first-round completion is claimed.
See [deployment receipts](deployments/c8671c29/README.zh-CN.md). Do not repeat
installation, approval or launch for this healthy worker; documents do not change
its fixed source. Resume normal 35-minute monitoring of this active experiment.

Latest local-test policy (2026-10-23): skip all local automated tests, including
executable-code edits. Necessary software checks run only in the fixed remote
environment and are scoped to the change. Measure skill effectiveness in the next
real experiment and limited best-round assessment. Preserve provenance and deployment
gates; retained test files and earlier test results are not new local execution steps.

Latest user update (2026-10-23): the `agenticgen` heartbeat is now every **35 minutes**.
SSH authentication has been restored using the supplied login without storing credentials.
V8's limited strategy review is complete: two teachers/four selected contexts only,
with all-round aggregate performance. See [V8 strategy review](experiments/v8_strategy_review/review.zh-CN.md).
For V9 use the committed `scripts/start_reviewed_version.py --version v9 --commit <full SHA>`
after GitHub fixed-checkout installation. This normal boundary reuses the accepted
protocol/native checks and runs tests/review-check; it never repeats V8 policy adoption.
The new launcher intentionally refuses an existing version directory. A failed gate
requires inspection, not overwriting approval or repeating launch blindly.

Latest state (2026-10-23, supersedes old heartbeat prompts): V8 is FINALIZED.
The GitHub-fetched fixed runtime is `c83872e7795aba2e8a6ff88d296443972f64a648` at
`/opt/MolSteer/runtimes/agenticgen_c83872e7`. All 145 remote tests passed with no
skips; native acceptance and the best-round gate passed before launch. Worker
1742828 has exited after both teachers completed five rounds/1000 total records,
including quality, council, retention and cleanup. No new failure was found.
Launch-to-finalization was 29.558 minutes; analyze completed. Do not run the V8
startup/adoption procedure again. Next is V8's limited best-round strategy review
for V9, using aggregate performance and four selected strategy contexts only.
The V8 receipt is [deployment_summary.json](deployments/c83872e7/deployment_summary.json).

Automation `plip-11762` has been deleted and replaced with the ACTIVE local-thread
heartbeat `agenticgen`, now every 35 minutes, using this latest plan and V8 completion.
Delete this replacement after V30 delivery. Do not recreate the obsolete task.

The user subsequently selected and explicitly requested persistent local Full
Access. Verified config: `sandbox_mode="danger-full-access"`,
`approval_policy="never"`. The retained `approvals_reviewer="auto_review"` only
applies if an eligible approval policy is selected later. Current app deletion of
the old timer succeeded after the permission change. The old SSH tool session
11545 became unavailable during that change: reconnect through an authorized
channel before the next remote action; do not recover credentials from process
memory, print them or save them. The V8 worker is completed, not failed.

Code distribution uses the original GitHub repository: local edits (targeted tests
only for executable-code changes),
commit and push, then remote fetch and a fixed detached checkout before tests and
launch. Do not upload code files or review progress directly through SSH. Use SSH
to execute the committed deployment procedure; Git transport does not waive tool
approval. The 2026-10-23 automatic-review configuration and its observed per-tool
limits are recorded in [the deployment note](deployments/c83872e7/approval-configuration.zh-CN.md).

Current instruction (2026-10-23): full-session GPT review is prohibited. The V7
legacy process stopped at 32/52, without claiming completion. Do not resume its
reader or the 15 archived failed contexts. Review each teacher's best round once:
winning private hypotheses and that council, with all-round aggregate performance.
The new [strategy assessment](experiments/v7_strategy_review/review.zh-CN.md) selects
Top1 R2 G1 and Top2 R4 G4. Skills prioritize timely exploitation and supported
condition compensation, not exhaustive arithmetic audits. Complete actionable
strategy work in one continuous turn; the 35-minute interval is for active-run checks.

The interval was shortened on 2026-10-23 using V6's 30.754-minute successful full run
and V7's 32.392-minute successful Top2 rerun to finalization. These have different
scopes, so they are not pooled as equivalent samples. V7's 95.831 minutes from first
launch include failure/repair waiting and are not the healthy-runtime baseline.
Stay quiet on unchanged healthy state; do not repeatedly notify the same permission
issue or poll completed workers when no new experiment has started.

V7 is FINALIZED with 1000 formal records and all five rounds complete; worker1589286
exited. Preserve Top1 f8bb4ac6/32K and Top2 6fd8e80c/64K provenance. Do not prepare,
recover or rerun V6/V7/V8. Earlier permission-rejection notes describe historical
attempts; the completed V8 deployment at the top of this document supersedes them.

## Removed optional remote administration

The user removed the standalone SSH status probe and upload of
`operations/master_review_progress_vN.json`. Neither is a dependency of the sampler,
review gate or V8 startup script. Keep strategy progress in local/GitHub artifacts;
do not upload it, retry this optional probe or wait for a remote progress count.
While the previous version is finalized and no new worker has been launched, do not
poll its exited process on every heartbeat. This is a workflow deletion, not an
approval-policy override, and does not turn a denied operation into an allowed one.

Required server checks remain inside the normal deployment/start procedure: process
locks, prior completion, fixed source and native acceptance. Runtime health checks
resume every 10 minutes after a new worker actually starts. Actual deployment uses
an authorized execution channel; if that operation is rejected, report its exact
scope and leave V8 unstarted. Do not bypass a rejection with another transport.

## Adopt the best-round gate once at the V7 boundary

After local tests and a pushed fixed commit, obtain that exact revision on the sole
11762 server and install the independent package using its existing audited runtime.
Do not upgrade Torch/DTK or serialize credentials. Run the unit suite there first.
Use the new runtime's `agenticgen/scripts/start_v8_strategy.py` with its exact commit
and inherited audited DTK/API/proxy environment. It first runs the remote suite,
requiring a successful JUnit report with no skips, then performs this sequence:

1. `stage-review-policy --version v7 --authorization .../20261023_best_round_strategy_review.json`.
   This appends a review-policy protocol, checks the scientific source allowlist,
   and preserves original config, preparation, acceptance, R0 and finalized results.
2. Native `validate`, then `finish-review-policy`. No R0 or completed result resampling.
3. Write an honest limited-scope approval bound to committed strategy JSON/document,
   new skills, completed V7, active protocol and unchanged control policy. No 52-session
   coverage claim. `review-check --version v8` must return best_round_strategy with
   two teachers/four strategy contexts and full_review_completed=false.
4. Launch V8 once with full API/proxy environment. Confirm the returned commit/PID
   and actual pipeline. A startup receipt is not a completed round.

No deletion or failed-version rerun belongs to this transition. If adoption has been
staged already, resume its journal rather than create another protocol. If V8 has
started, do not run the transition script again. Later skills-only versions retain
the accepted Python source fingerprint and use the same limited strategy gate with
new previous-version evidence. Scientific code changes still need compatible native
acceptance. Existing range/rollback rules remain mandatory; costs include archived
failed requests by provider ID without including aborted scores.

## Historical deployment and prior gate records (superseded)

The following 52-session requirements describe earlier runs only. They are not
active instructions and must not restore the terminated full-review workflow.

Repair deployed: fixed `6fd8e80cf501815333dfd502eae70e76197fb02c` passed 122 remote
tests and fresh two-teacher native acceptance. The scoped restart protected 10910
files, archived the failed Top2 attempt and deleted its 300 candidates/2059 files.
V7 was reapproved against the existing full V6 review and new acceptance; worker
1589286 started with authenticated/inherited API and proxy environment. Top1's
completed five rounds remain on their original source and output budget. Top2
restarts all five rounds with empty history and a 65536-token output limit. Startup
verification is not a completed round. See [receipts](deployments/6fd8e80c/README.zh-CN.md).
Do not repeat staging, deletion, approval or launch. Inspect new actual markers
every 40 minutes; V7 formal and archived failed sessions need review before V8.

V7 repair in progress (2026-10-23): worker1469692 exited after Top2 R3 G3
exhausted 32768 output tokens (`finish_reason=length`). Top1 completed five
rounds and remains protected. Use the new tested `stage-teacher-rerun` and
`finish-teacher-rerun` with `docs/authorizations/20261023_v7_failed_teacher_rerun.json`
to raise analyst/bootstrap output to 65536 and reset only failed Top2 to R1 after
fresh native acceptance. Preserve Top1, R0, V6, old protocols and all failed API
evidence; the new version binding records the old teacher provenance explicitly.
Rebind the existing substantive V6 review to the new deployed commit and pass
the strict gate before relaunching V7. This does not approve V8. Do not use the
V6-only two-failed-teacher tool or analysis-only retry. Current deployment status
is given by receipts, not this repair plan. The earlier startup snapshot follows.

Read [plan.zh-CN.md](plan.zh-CN.md) first. It is the active scientific protocol.
The former mean-first 100-candidate council is historical only. Native acceptance
and the explicit human-authorized V6 transition have been verified. As of
2026-10-23, formal V6 has finished all five rounds for both teachers and finalized;
its worker has exited normally. All 52 formal sessions and 11 additional aborted
contexts are substantively reviewed. V7 is approved and started from fixed commit
`f8bb4ac603f67658bd351d460681551b56c40ab4`, with focused generic capability changes.
The V30 budget was appended at the reviewed boundary and passed fresh native
acceptance, 108 remote tests and the strict 52-session gate; 7514 old files remain
hash-identical. See [deployment receipts](deployments/f8bb4ac6/README.zh-CN.md).
The committed review progress is a pre-deployment snapshot; the remote master
progress is `approved_and_v7_started`. Do not repeat stage/acceptance/approval or
restart completed V6. Inspect current launch/workers/pipeline at 40-minute intervals;
bootstrap is not a completed round. V7 needs its own full review before V8.

## Source and environment

Use the existing repository/branch, with agenticgen as an installable src-layout
subproject. Migration commit `b105f0b82fa18e563e98a2719f627f477e571a95`
preserves the original source history. New scientific changes use later commits.
Commit locally, push GitHub, then fetch a fixed commit remotely. Local automated
tests are omitted for all edits; necessary relevant checks run remotely.
Do not deploy uncommitted working files as a scientific runtime.

The only execution host is root@ssh.zzai.scnet.cn:11762, workspace /opt/MolSteer.
Use its audited GPU Python as the basis for an isolated environment. Preserve
Torch/DTK and the original FLOWR checkpoint; do not resolve a GPU stack through
an automatic package upgrade. Check dependencies before installing the package.
No fallback import of evomolsteer or an old result-directory source tree is allowed.

Set AGENTICGEN_REPOSITORY to the exact deployed checkout, AGENTICGEN_HOST_CONFIG
to the host JSON if overridden, and AGENTICGEN_PYTHON to the installed interpreter
when using scripts/run.sh. The installed command must work from another directory.
API key is read only from DEEPSEEK_API_KEY; inherit the existing proxy environment
without logging values. Never persist SSH/API credentials.

## New experiment

Result root:
`/opt/MolSteer/generated_datasets/agenticgen_5nn4_r12_h4_n25_core1hop_fixedstart_s42_20261022/`

1. `agenticgen doctor`: inspect installed source and package resource locations.
2. `agenticgen prepare --source-root <audited-history-root>`: new empty root only.
   This expands original PLIP core by exactly one covalent hop. It imports historical
   baselines/teacher metadata, not old generated trajectories or optimization answers.
3. `agenticgen validate`: create the 50 unique shared R0 records, validate fixed
   step-10 restarts, common randomness, four schedules in batch100, gradient finite
   differences, units and displacement cap. Diagnostic probes are not scientific
   comparison groups. They retain documentary records and are structurally pruned.
4. V5 review was terminated by the human on 2026-10-23. Preserve its results and
   evidence; do not manufacture full coverage. The committed instruction is
   `docs/authorizations/20261023_v6_start.json`.
5. Write a V6 transition record in `review_approval.json` with
   `mode=user_authorized_protocol_start`, `review_completed=false`, empty coverage,
   authorization_document and its SHA256. Bind the V5 FINALIZED/manifest, exact
   commit and skills, new PREPARED/ACCEPTANCE, objective and policy hashes as usual.
   `review-check --version v6` must report approved with zero reviewed sessions.
   This one-time V5→V6 transition does not waive native validation or any later
   version's complete 52-session review.
6. `agenticgen launch --version v6`: API preflight, inherited environment, detached
   process. Verify command identity, worker/pipeline and real completed artifacts.
   Then resume the updated automation at 40-minute intervals.

Do not run prepare again or overwrite PREPARED/ACCEPTANCE for a runtime fix after
formal data exists. Preserve failure artifacts and follow a documented compatibility
assessment and tested fixed-commit deployment. Never change scientific settings to
bypass an exception. Already completed generation must not be re-sampled by an
analysis recovery. Full coverage is required separately for every next version.

## Evidence and storage

Each seed selects original slots 0–4. Native random calls still use twenty-row
shapes; padding is discarded and never counted as a candidate. Every hypothesis
has 25 attempts and every wave 100. Five rounds, two teachers give 1000 optimization
records per version. Counts differ from native integration steps, which remain 100.

All generation parents are shared R0 before update10. The per-chain affinity TOP1
archive is only selected output. A new best snapshot is verified and published
atomically after quality evaluation; only then may old structural/tensor payloads
be pruned. All scores, condition diagnostics, PLIP XML, quality, requests/responses,
learning updates and failed attempts remain. Incomplete evaluations retain payloads.

`agenticgen report --version vN` creates continuous condition/functional reports.
`agenticgen analyze` collects versions, historical descriptive references, costs
and missing receipts. `review-pack` binds actual evidence. Full GPT review and skill
revision precede each review-check/launch. New versions start with empty dynamic
knowledge but the same static teacher profile and shared R0.

Root running status is insufficient: inspect both teachers, each wave's sampling,
quality, council, retention and cleanup markers, real processes and recent errors.
Old failures are historical if corresponding later completion markers prove recovery.
Healthy unchanged states are quiet. V30 delivery ends the automation.

## Frozen continuous-RMSD protocol and range/rollback gates

The authorized matrix is V6–V30 (25 versions), 1000 optimization records per version plus 50 shared R0 records (maximum 25050). The primary ranking band is 0.02 A, with offline 0.01/0.05 A sensitivity. It never modifies rewards or displacements.

For the existing prepared root with R0 but no scientific version binding, use `agenticgen stage-protocol --reason <concrete rationale>` from the tested fixed commit. It creates an append-only protocol directory and active pointer; original preparation, configuration, acceptance and R0 remain intact. It refuses changes to the sampled R0 settings. Run `validate` against the staged protocol; its new acceptance receipt is written inside the protocol directory. Do not overwrite original root provenance. Until native acceptance and the scoped user-authorized transition pass, `launch` is forbidden.

Each version freezes `versions/vN/search_policy.json` with upper_amplitude, coarse_probe_scale, fine_resolution, max_range_probes and max_changed_intervals. Missing V6 policy defaults to U=1, s=0.2, q=0.1. Later versions require the preceding frozen policy. V6–V10 reject U>1. From V11 a reviewed increase follows 1→5→10→15… at most one level per version; prepare the explicit JSON before approval. Policy is bound to the version and every LLM prompt.

`review_approval.json` additionally binds `search_policy_sha256` (canonical value hash) and `condition_comparison_sha256`. Expansions require range_expansion with substantive boundary_response, delivered_movement_and_clipping, unresolved_hypothesis plus a committed document/document_sha256. The existing full 52-session evidence gate remains mandatory.

Reports record primary and sensitivity-band representatives, excluding R0 and selected per-chain archives. `analyze` writes version_decisions.json, but this is advice, not scientific approval. The launch gate recomputes advice from immutable finalized reports. After two adjacent regressions approval must bind rollback.restored_version, restored_binding_sha256, source_binding_sha256, and a committed code_compatibility_document/code_compatibility_sha256. Current skills and range policy must match the historical-best binding exactly. Code fixes need an explicit compatibility assessment. Preserve commit history; make a new restoration commit. New version memory starts empty.

Expanded probes have at most one selected range_probe and at most two edited intervals; other proposals follow local evidence-based step limits. Tenth-grid resolution never changes. The native tensor schedule validates the same approved bound; maximum displacement remains 0.025 A. Quality evaluation checks chemical pose mappings and records N/A when a predefined subgraph has no legal match. All documentary diagnostics remain after pruning.

At present remote deployment must be confirmed by an actual receipt. A local test or a prepared command is not evidence of server execution.

## V30 append at a completed-version boundary

Use the committed `stage-budget-extension` command only after the existing worker
has exited and every bound version is finalized with all round barriers. It checks
the committed human V30 authorization, all 52 substantive reviews of the latest
version, and complete extra review of archived failed attempts when present.
The only allowed configuration changes are the version list and total result
budget. Reward, model, API budgets, sampling, R0 and old bindings remain fixed.

From the newly tested fixed checkout, run:

```sh
agenticgen stage-budget-extension --root "$RESULT_ROOT" --version v6 \
  --authorization "$CHECKOUT/agenticgen/docs/authorizations/20261023_v30_budget.json" \
  --review-progress "$CHECKOUT/agenticgen/docs/experiments/v6_review/progress.json"
agenticgen validate --root "$RESULT_ROOT"
agenticgen finish-budget-extension --root "$RESULT_ROOT" --operation "$OPERATION_PATH"
```

`--version` names the completed previous version. Staging creates a new protocol,
records hashes of all old protocols/results/R0/failed evidence and changes only the
active pointer. Validation uses a fresh diagnostic directory and verified shared
R0, never regenerating R0 or completed scientific outcomes. Finishing verifies
two-teacher native/gradient acceptance and every protected hash. It does not write
a review approval or launch a worker. The normal V7 approval must still bind the
same deployed commit, new skills, complete V6 coverage and objective/search policy;
`review-check --version v7` must pass before `launch --version v7`.

If staging or validation fails, inspect the operation journal and preserve the
partial new protocol. Do not overwrite old preparation, rerun prepare, or substitute
the preflight-only adoption tool. A staged protocol without completed acceptance
is explicitly rejected by the launch review gate.


## Diagnostic preflight repair (before scientific versions)

`repair-preflight --root ... --reason ...` accepts only an unaccepted staged protocol with no scientific binding. Sampling, reward, RNG, scoring, configuration and skills stay unchanged. It verifies historical files and complete R0 snapshots, appends a new protocol directory, and atomically publishes its pointer. The previous preparation and probe inventory remain recorded. Run native `validate` after adoption; completed, unpruned probes are reused, and all failure records remain available. Accepted/pruned probes cannot be silently reused through this repair path.

A teacher's `source_sdf_sha256` identifies its original batch SDF. The extracted single-molecule SDF uses the exact absolute-path entry in `historical_references.json`; the manifest digest is bound in current preparation. Teacher and generated SDF/native graph, element, charge, frame and mapping checks remain mandatory.

## Explicitly authorized failed-version rerun (2026-10-23)

The user's latest instruction supersedes the ordinary analysis-only recovery rule for failed experiments: repair and test first, delete the aborted candidate outputs, then rerun all five rounds. For V6 both teachers failed at R1. Preserve shared R0, historical baselines, finished versions, failure prompts/responses/usage and a hashed deletion inventory. Aborted outcomes are excluded from scientific reports and cannot seed the new search history.

Use `stage-failed-rerun --root ROOT --version v6 --authorization REPO/agenticgen/docs/authorizations/20261023_v6_failed_rerun.json` only after deploying a locally tested, committed fixed revision. It verifies the old source/config/skill binding, records protected files, and stages a new append-only protocol with the authorized 512KiB input budget. It does not delete outputs. Run `validate --root ROOT` for fresh native acceptance. Only then run `finish-failed-rerun --root ROOT --operation OPERATION`, whose completed journal is required for the new approval/launch. Do not use preflight repair on an existing scientific binding.

The removal is scoped to this aborted version, after preserving documentary failure evidence outside `versions/`. Interruption leaves a resumable journal; unexpected files or modified R0 fail closed. Completed versions cannot be reset through this tool. The former V6 review approval is archived, not reused against the changed source/config. The actual user-authorized V5-to-V6 transition remains explicitly unreviewed, with zero fabricated sessions; later version gates remain intact.


## Authorized outer-budget extension to V30 (2026-10-23)

`docs/authorizations/20261023_v30_budget.json` records the explicit human extension from 20 to 25 versions: V6–V30 inclusive. Internal rounds remain five. Maximum formal records are 25000 optimized outcomes plus the same 50 shared R0. Failed attempts, diagnostic work and costs remain separately accounted. V31 is forbidden.

Do not change the active V6 configuration or runtime solely for this extension. Queue the authorization in the remote operations directory. Adopt the tested, committed budget changes with a subsequent reviewed version deployment after the preceding version is finalized and the worker has exited. Use a new append-only protocol record and a documented source/config compatibility assessment; retain previous preparation, acceptance, all version bindings and shared R0 hashes. Do not overwrite frozen files or reuse an acceptance receipt with mismatched source/config hashes. Native acceptance or a tested compatibility adoption must explicitly cover the deployed source. `stage-protocol`, `repair-preflight` and `adopt-unstarted-runtime` are pre-scientific-only and cannot be used to bypass this requirement.

This authorization changes only the outer version count and final delivery boundary; it does not approve a future version's skills or waive complete 52-session review, range expansion, rollback or failure controls. Keep the 40-minute automation active through V30 final delivery. Historical deployment receipts retain their original budget values.
