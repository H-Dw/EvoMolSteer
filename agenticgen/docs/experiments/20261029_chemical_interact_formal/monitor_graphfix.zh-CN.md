巡检并收尾本对话明确授权的“化学inpainting＋双教师同一互作位点”计算图修复后正式实验。先读取D:/llm_controlled_generation/EvoMolSteer-uniguide/agenticgen/docs/experiments/20261029_chemical_interact_formal/tensor_contract_audit.zh-CN.md、restart_graphfix_audit.json和本对话后续指令。只操作codex/uniguide-flowr-agent。原uniguide-flowr保持PAUSED、35分钟配置不变，不操作其他任务/自动化，不向其他聊天发送消息。

远端root@ssh.zzai.scnet.cn:11741，使用本对话授权凭据和D:/llm_controlled_generation/tmp/uniguide_ops.py（JSON stdin），凭据不得打印或写入仓库/自动化。固定执行目录/workspace/uniguide_runs/chemical-interact-graphfix-4ffd6e4c，已推送提交4ffd6e4c63bba17f9ae68eaf185a48f0d80a4985。新结果根/workspace/agenticgen_chemical_interact_20261029_graphfix；配置/workspace/uniguide_runs/chemical-interact-graphfix-4ffd6e4c-config.json，SHA256 987838e948e4377f0db4d62dcc91459f277f84b831b75bdc1c2c01c4333e1fd3。旧失败根/workspace/agenticgen_chemical_interact_20261029及旧执行目录chemical-interact-5f97acb3完整保留，不删除失败/claim，不复用旧验收或旧λ。

2026-10-29 14:54:15（Asia/Shanghai）已启动一次性后台链acceptance→calibrate→真实LLM run-version GD01→10/50诊断。定位父PID8557、验收PID8630；14:55:04真实进程与三把锁持有，处于acceptance，尚无标定/正式样本或真实LLM回执。仅作定位，后续必须使用新目录inspect_run.py读取真实命令、状态、全局/workspace/agenticgen_uniguide_flowr_20261026/.executor.lock、本专项.executor.lock/.campaign.lock、STATUS/HEARTBEAT、阶段log、逐组steps/progress、attempts和完成/失败标记，不能只凭PID或heartbeat判断健康。每次在本专项monitoring保存Asia/Shanghai时间快照。

修复：健康张量检查直接返回原对象，统一独立/合批距离计算图；数值检查、证据和失败隔离分离，增量溢出先交候选隔离。10行随机流分片仍为失败账本单位；3Å限幅为逐分子，不得混淆。远端111项测试通过。真实模型验收还必须包含双教师有限差分、无亲和力梯度/额外RNG、100区间独立合批逐位一致、恢复、90候选容量/真实OOM降档、3Å压力和第二分片第10步非有限梯度注入后第一分片逐位一致。验收内nonfinite_isolation故障注入是预期测试，不能仅凭其nonfinite证据文件宣布正式任务失败；以验收COMPLETE及CAMPAIGN_FAILED判断。任何测试失败禁止跳过或放宽容差。

保持教师6pyy_H7S_A_402/single_b004_final_015、3w54_RNB_A_504/single_b000_final_003。各九个原PLIP直接互作配体原子同时作condition和interact_atom，6pyy44对、3w5410对；仅原记录两端集合展开去重，不扩大位点，不当成独立互作事件。新旧SUBJECTS完全一致。化学状态inpainting开启，坐标不覆盖；raw999100区间0–0.999，10–99步坐标guidance，最后循环未覆盖预测；3Å候选局部上限配置不变，初始内部/外围/互作权重1/0/1。skills2.2.0不变，Agent可按现有工具局部调整。

重新逐教师探索λ：13格对数粗筛各10候选、局部5格逐步20/50复用已测槽位，每教师最多350次、共700。边界最优或证据不足停止，不扩界或自动弱化。正式5轮、三方向、每组50，两教师30组请求、最多1500新尝试；精确重复配置引用旧测量，不能重复生成或当独立验证。DeepSeek必须核对真实请求和回执：deepseek-flash、thinking/high、单响应16384及既定网络重试/一次格式修正；每教师每轮四角色共享30次调用、491520输出token。配置存在不代表真实调用，不启动GD02，不恢复UG05。

健康时不覆盖执行目录、源码、manifest、skills，不重复启动。若真实进程退出或出现CAMPAIGN_FAILED、INSUFFICIENT_EVIDENCE等实质停止，先保存阶段日志、源哈希、账本、状态和失败请求，核对原因并报告；不删claim/失败标记，不自动重跑或补样。launcher.log可能含交互式shell的环境输出，仅留远端，不直接打印或提交原文。GitHub异常按直连、ghfast.top、gh-proxy.com、ghproxy.net、gh.llkk.cc顺序尝试，均不可用才使用已验证且仅含已推送提交的SSH Git bundle。

本专项CPU quality_queue须检查worker.json/status.json/requests/POLICY、真实进程和.worker.lock；module_directory必须来自本次固定执行目录，commit及模块哈希按该目录验证。每组attempts=50、checked_attempts=10、not_selected=40、coverage=0.2；五种子各logical_slot=0、1，构建失败计入分母、不补样，其余40不追加慢检查。区分测量不通过、构建失败、无法评价和工程错误。必要的CPU恢复仅限本专项，先审计，再加载DTK和本目录agenticgen/src、src，屏蔽GPU，各CPU数值库1线程，不启动旧UG或旧实验worker。

SAMPLING_FINISHED后仅补后处理。CAMPAIGN_FINISHED后实际读取两教师标定、30组请求/复用来源、完整指标、四角色上下文和真实回执；按教师分析距离指标、互作维持、限制器、构建/连通/唯一性及10/50质量诊断。不得与旧绝对坐标协议混排名，不将未抽查当通过。SOURCE_PRESERVATION包含342文件（原134源文件、旧失败125产物、旧83源码），全部核对不变。下载结果/审计至本地实验目录，完成真实GPT复盘、提交推送；纯文档不部署至运行目录。仅在核对路径归属及队列空闲后停止本专项CPU worker，写FINALIZED与复盘绑定，通知本对话。完成或终态失败归档后删除当前触发的本专项heartbeat，其他自动化不动。

巡检间隔15分钟。健康/排队/状态不变时保持安静，仅报告完成、实质异常、修复或需用户处理的问题。中文文件UTF-8，读取或更新自动化后用JSON ensure_ascii=true读回核验，禁止Unicode替换字符。
