# 一次Git拉取部署

2026-10-24最新用户授权：依次尝试四个镜像https://ghfast.top/ → https://gh-proxy.com/ → https://ghproxy.net/ → https://gh.llkk.cc/。本轮每个一次有界120秒clone，独立缓存/日志；前一镜像clone连接或传输失败后才试下一个。成功clone立即停止切换，再本地导入和固定提交/hash/原正常门槛验证；源验证/测试/安装失败不能当网络失败切镜像。全部失败则停止，不循环、不REST API回退、不改代理/TLS/凭据。此前只允许ghfast的限制由此次明确列表替代，工具拒绝仍必须停止。V28科学目标d7d35b13不变，本地无自动化测试，实际验证/启动待收据。

2026-10-24最新用户授权：GitHub连接异常时，允许使用`git clone https://ghfast.top/https://github.com/用户名/仓库名`作为后备来源。当前GitHub已有CONNECT503及TLS超时收据，因此本轮直接选择`--source-transport ghfast`，不重复测试直连。此前“任何换通道均不允许”的网络策略在此明确镜像范围内被替代；工具权限拒绝仍必须停止，不可绕过。

实现：一次120秒有界`git clone --bare --single-branch --branch codex/plip-adaptive-schedule --no-tags https://ghfast.top/https://github.com/H-Dw/EvoMolSteer.git /opt/MolSteer/source_cache/ghfast_<transport8>.git`。禁止覆盖已存在缓存，保留失败目录供inspect。克隆阶段禁用Git credential helper/extraHeader及system/global Git配置读取，保留原代理/TLS环境，不向镜像转交GitHub认证。仅此固定公开仓库和用户指定镜像获授权。

克隆成功后一次本地Git fetch导入旧仓库对象库（无网络），随后按原规则校验固定科学/传输commit、FETCH_HEAD祖先、来源SHA256和部署blob；不改旧runtime文件、HEAD、origin。验证不通过不安装/启动。此后仍走既有正常review-check。失败不回退REST API、不循环、不尝试其他镜像。已有GitHub网络失败时显式选择ghfast；没有该证据时正常github单次fetch。传输单独固定提交，V28科学目标d7d35b132a04c5b58964aad3cfcc552a792a9909不变。

本次工程改动本地只diff/commit/push；远端仅test_reviewed_deploy.py相关检查，增加克隆失败后不得导入/安装的行为验证。未实际执行或通过不得宣称通过。旧成功验证hash不同不可复用。下文为此前实施历史，发生冲突以本节为准。

2026-10-24用户明确要求替换逐对象GitHub REST API部署。正常部署只发起一次有界Git fetch（pull的拉取阶段），通过Git pack取得分支更新，再核验固定科学commit和传输工具commit，创建独立detached runtime，不在旧runtime执行merge。V28科学提交保持d7d35b132a04c5b58964aad3cfcc552a792a9909。

命令使用固定仓库HTTPS URL及codex/plip-adaptive-schedule分支，120秒上限、无tags/submodules、取消对象过滤，沿用原代理/TLS；禁止交互式Git凭据提示和隐式lazy fetch。拉取失败写git_fetch_receipt.json并退出，不尝试API、逐对象下载、换出口或换凭据。成功后检查固定提交属于FETCH_HEAD祖先、所需部署blob本地齐全；缺对象即停止。旧fetch_fixed_github_api.py及其收据仅保留历史，不作为当前部署入口或回退。

本地只diff、commit/push，不跑自动化测试。传输工程修改的定点验证在远端既有固定环境运行test_reviewed_deploy.py，源码由git show取得并按本地固定提交SHA256核验。验证覆盖单次fetch、固定目标/校验来源、失败后无安装启动，以及原完成版/活动worker保护。相同三个验证来源hash及原Junit hash完整时复用成功验证，不在后续仅skills/文档变更中例行重复测试。未通过时不安装科学runtime。

--transport-commit绑定已提交的传输脚本和检查来源，与--commit科学源分开，前者不改变实验skills/配置/原生采样路径。V28仍走原正常review-check、安装/原生兼容与启动门槛。新的git_fetch_receipt和transport_validation分别保存，与历史API收据及scientific tests_receipt区分。

原API reset等待和API重试方案已被本次用户要求替代。若Git仍报CONNECT503，只保留此次结果并等待网络恢复，不用API额度诊断决定Git重试。35分钟巡检不改变，不轮询已完成版本；V30完整收尾和报告、commit/push保存后删除定时任务。

首次实际尝试：传输088456aa、科学d7d35b13；一次fetch被原代理CONNECT 503拒绝，exit128、REST API请求0，无回退/重试，未抵达测试、安装或启动。等待实际网络恢复证据，35分钟心跳不自动盲重试。收据见../deployments/d7d35b13/。
