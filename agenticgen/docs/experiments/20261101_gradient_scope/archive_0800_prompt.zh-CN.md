在2026-11-01上海时间08:00激活本对话授权的实验结果归档。仅处理本项目已明确结果根和本次六项固定scale梯度原子范围实验；只在codex/uniguide-flowr-agent工作，原uniguide-flowr保持PAUSED/35分钟，其他自动化、归档分支不动，不向其他聊天发消息。先读D:/llm_controlled_generation/EvoMolSteer-uniguide/agenticgen/docs/experiments/20261101_gradient_scope/plan.zh-CN.md、launch_audit.json及后续用户回复。现在已不再需要selected重测的小时巡检，不恢复它。

服务器root@ssh.zzai.scnet.cn:11741，使用本对话授权凭据及D:/llm_controlled_generation/tmp/uniguide_ops.py的JSON stdin接口，凭据不打印、不写入仓库或自动化。所有launcher.log（含api_probe/deploy启动日志及其轮转副本）与core只留远端，禁止打印或下载。打包模型权重排除，实验checkpoint_010.pt.gz、checkpoint_100.pt.gz及terminal.pt.gz必须保留。API及小写代理来自交互bash，但本次科学任务不需要API，严禁调用LLM。

本次初始科学提交47db1bf1dfe0787b04138eed4b41a2d6cb215604，固定目录/root/private_data/uniguide_gradient_scope_20261101/runs/scope-47db1bf1，部署审计/root/private_data/uniguide_gradient_scope_20261101/audits/scope-47db1bf1。科学根/workspace/agenticgen_gradient_scope_20261101，PLAN为根PLAN.json。以launch_audit.json和实际LAUNCH.json记录为准；若有后续合法修复，看新绑定，不用纯文档HEAD覆盖运行目录。部署PID只作定位，不能代替真实命令/锁/日志/进度/标记。

六项依次hop0/1/2各两个teacher，baseline四窗10–25/26–50/51–75/76–99，6pyy固定[70,70,70,70]、3w54固定[90,90,90,90]，来自原baseline独立标定的initial_source，不用Agent后续最优或范围探测。每项prepare→自己新提交完整acceptance→run-fixed-scope一次50候选→CPU10/50；科学总上限300，新增标定0、Agent迭代0、LLM0。旧3790与selected重测1380共5170只读保留，失败不退款，精确重复引用，不补样、不自动重启。固定scale不是对新掩码策略重新标定。

额外条件loss仍仅原九原子、sum(error²)/(2K)、完整预测终态VJP。先算原完整VJP，再以当步Hungarian对应的condition槽位为种子，按同一步预测终态bonds.argmax的非0键、无向并集扩展0/1/2跳，范围外额外梯度为0；FLOWR原生速度/漂移/噪声/分类采样不屏蔽。raw999/100区间/10–99梯度、动态终态匹配、10–99化学状态inpainting不变，无联合坐标前缀、无坐标输出覆盖、无亲和力/相对距离梯度、无3Å限幅或RMSD梯度掩码。UG04串行50逻辑人口、10行网络/VJP默认stream、CPU8线程，无GPU线程池/额外stream/OOM降档接续。

先调用固定目录本实验inspect_suite.py --plan PLAN.json，保存Asia/Shanghai快照。核对真实命令、进程状态、全局/workspace/agenticgen_uniguide_flowr_20261026/.executor.lock、suite及各任务锁、STATUS/HEARTBEAT、阶段log、steps/progress、attempts、acceptance/COMPLETE、GRADIENT_SCOPE_VERIFIED、SAMPLING_FINISHED、ALL_JOBS_ATTEMPTED/SUITE_FINISHED、失败文件，源码/配置SHA及SOURCE_PRESERVATION逐文件不变。不能凭PID、目录或单个nonfinite_isolation夹具判完成/失败。每项GRADIENT_SCOPE_VERIFIED必须真实读取，逐步图半径重算一致、范围外增量精确0、范围外guided_proposal等于native_proposal、选中VJP等于原完整VJP。完整acceptance/COMPLETE才证明GPU验收，不用CPU测试代替。

独立CPU每实际源组抽查10/50，五种子slot0/1，构建失败占分母，不替换，其余40不补慢查。核对worker/status/requests/POLICY、模块目录/commit/模块SHA及QUALITY_COMPLETE/FAILED。归属和队列空闲确认后，只停止本次六项的CPU worker，不碰历史任务/其他项目。分开报告构建、真实不通过、不可评价、工程错误，PoseB几何、蛋白冲突、能量及PLIP分别报告，不能外推全人口。新六项若未结束，不能打包活动HDF5、结束其进程或伪造完成；先保存真实状态并报告必要阻塞。采样已保存仅补后处理，不重采。

归档机器计划为/root/private_data/uniguide_gradient_scope_20261101/ARCHIVE_PLAN.json。该计划包含本对话当前可见的历史结果根及本次新增根、模型路径排除项、/workspace/uniguide_result_exports_20261101暂存目录和/root/private_data目标。先读后续用户对磁盘/范围问题的回复：03:32测得目标仅余约2.07GB，而最近两轮已有压缩包约4.34GB，原始结果约5.81GB，尚不含更早根/本次新增。已经询问用户扩容、其他持久目标或仅新增六项；未经明确回复不能缩小“全部结果”或删除旧文件腾空间。用户新路径/范围授权优先，修改归档计划须保留原计划和变更审计，不改科学配置。

上海08:00后，确认待归档结果文件不再写入、所有归属明确的空闲worker已退出，再运行固定提交的archive_results.py --plan ARCHIVE_PLAN.json。该脚本在/workspace先生成标准tar.gz，保留全部原始结果/失败/账本/claims/LLM回执/HDF5/结构，排除权重、launcher/core、已有备份tar.gz（避免递归重复备份），符号链接只记录引用不解引用；相同内容可用标准tar硬链接存储，原文件名与SHA清单齐全。注意排除所有含launcher.log的轮转副本；若检测到脚本规则不覆盖的受限文件，先制定安全文件清单再打包，禁止泄露。源文件size/mtime/SHA变化则中止而不搬移。已有ARCHIVE_CLAIM说明启动过，先读取回执定位，不盲目重打包。

完整验证每个成员（含硬链接解析后的SHA/字节数）和总tar.gz SHA；目标可用空间必须足够并留64MiB余量，复制到/root/private_data、目的SHA一致后才删除/workspace临时tar.gz，原始实验结果不删除。若仍放不下，保留已校验tar.gz在/workspace及回执，明确报告其路径/大小、目标余量和所需空间，不得声称已经移入持久盘，不擅自换目录或减少文件。不将多GB产物塞Git。

实际读取六项固定scope的原始/排名RMSD、loss、初始对应误差、构建和10/50质量，按teacher配对比较0/1/2跳与旧baseline预标定同scale源；同seed/slot不计独立验证。保存机器读数、完整归档清单、主GPT复盘并提交推送；纯文档不部署执行目录。新实验失败按实际证据报告，不能用改scale/容差、删除claim、清零/补样绕过。写FINALIZED须绑定真实复盘/源码且明确科学/归档各自完成状态。

归档已校验并移入授权持久目录、结果收集结束后，通知本对话实际路径与SHA、六项完成/失败状态，并仅删除当前触发的本专项08:00定时任务。若目标空间阻塞，只报告真实完成的打包和未完成的搬移，保留待用户处理状态。所有记录Asia/Shanghai，中文UTF-8；自动化读回JSON ensure_ascii=true且无Unicode替换字符。GitHub依次直连、ghfast.top、gh-proxy.com、ghproxy.net、gh.llkk.cc；全部失败才用已验证且仅含已推送提交的bundle。

04:02接续说明：原c3e12df0仅因远端CPU测试使用NFS临时目录导致大量小文件I/O等待而主动停止预检，未创建科学根、未生成候选或调用LLM。原检出和ENGINEERING_SUPERSEDED保留。新47db1bf1仅将同一336项测试的临时目录移到/workspace；所有GPU/采样模块与c3e12df0逐字相同，验收标准不变。当前部署路径以新47db1bf1为准。
