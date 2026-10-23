# V7输出截断修复：已部署并启动Top2五轮重跑

科学固定提交`6fd8e80cf501815333dfd502eae70e76197fb02c`，运行目录`/opt/MolSteer/runtimes/agenticgen_6fd8e80c`。原worker1469692已经退出，Top1完整五轮500条结果保留其原f8bb4ac6来源；Top2在R3 G3输出达到32768 tokens后被截断。新代码只将私有/Bootstrap输出提高到65536，汇总仍65536，输入仍512KiB。冻结skills、科学采样/奖励/幅度/RNG/模型、共享R0保持。

本地完整回归108通过、12依赖跳过，后补输出预算测试2通过；远端固定检出后的122项全部通过。两教师新原生验收通过，同调度恢复坐标/评分最大差均0，类别与RNG相同；batch100不同调度、共同随机流、质量、梯度和单位通过。诊断最大指导位移0.010812995/0.003432019 Å，未触及0.025 Å上限。探针不计正式结果。

活动协议`protocols/analysis_output_repair_v7_6fd8e80c`；操作`operations/failed_teacher_rerun_v7_6fd8e80c/journal.json`为`deleted_ready_for_new_approval`。10910个旧协议、R0、V6、Top1及旧故障来源文件哈希保持；仅删除Top2失败尝试对应的300条候选及需重新绑定的全局文件，共2059个文件。删除清单、失败prompt/响应/usage、旧pipeline/binding/deployment与批准已先归档。完成的Top1不重复bootstrap、采样、分析或清理，其原GPU事件进入新pipeline；新binding显式记录Top1旧来源。

重新绑定真实V6的52正式会话及11作废上下文复盘，新提交/原skills、新协议和验收后，严格review-check仍为52通过。不是V7复盘完成，也未批准V8。API认证HTTP200，新worker **1589286** 已启动；实际命令身份和完整API/代理环境继承核验通过。启动快照1792716156.173292确认Top2 bootstrap实际输入为空历史/空公共反思、25候选、请求上限65536，同一R0、原skills、10910保护文件再次验证。读取时bootstrap尚未结束、pipeline尚未创建，无新failure；这是启动核验，不是首轮完成。后续以launch/workers/pipeline和实际质量/保留/清理为准。

本次Top2作废尝试共19份provider ID去重API收据，681131输入、238339输出tokens，无已知缺收据；三次GPU波次166.052053秒，不含R0及原生验收。记录在`aborted_resource_usage.json`。正式报告要追加此成本并与versions内成功收据去重；保留Top1成本正常计入正式分支，不重复记。Top1使用32K、重跑Top2使用64K，不能声称本版两教师LLM预算完全一致。

22份部署来源文件已核验下载和逐文件哈希，见`source_manifest.json`。远端证据包883204字节，SHA256 `dd51b455e763d7a533c0aa6b933674b2a9f9e26dfd6feb22fe83b67a7a532f0a`。当前磁盘空闲584779931648字节。后续只作40分钟巡检；完成V7后必须重新审阅52正式会话及本次归档失败尝试，验证原三项能力假设，才能批准V8。V6作废审阅无需重做。后续文档提交不改变6fd8e80c科学部署。
