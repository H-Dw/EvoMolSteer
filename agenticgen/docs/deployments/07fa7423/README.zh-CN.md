# V11 失败教师接口修复与实际启动

固定 `07fa7423a64c497c8550fda99a040f3437ae3eec`，runtime `/opt/MolSteer/runtimes/agenticgen_07fa7423`。
官方GitHub API取得并验证固定对象，复用缓存；本次未遇403/503，未SSH上传源码。
四份直接相关远端测试文件全部通过；本地未运行自动化测试。两教师原生验收通过，原始结果见results.json。

Top1 R5G1首答重复已执行调度，中性修正产生不闭合JSON并漏字段；两答均stop，非96K耗尽。
修复把私有/bootstrap中性修正上限1→2，增加完整字段/嵌套对象提示；汇总上限仍1。
原JSON/schema/科学动作校验保留，无手工补JSON、代写方案或科学边界放宽。
96K/汇总64K、短引用、skills、奖励/采样/RNG/起点/U1/位移上限保持。

39218项旧协议/R0/V6–V10/Top2及历史失败来源hash保持。
Top1原500失败候选对应3451文件已先归档后删除，其调用/GPU成本见aborted_cost_summary.json；
Top2原cea3611c完成五轮500条保留，无重跑。原始失败回答/usage、删除清单与保护journal保留远端及本地.cache，来源清单逐字节hash验证。

真实V10→V11有限review-check通过，2教师4上下文，full_review_completed=false，不是V11完成评估。
worker2281628，启动1792765739.6019506；核验1792765770.8190935存活、命令身份匹配且无新failure。
Top1从空历史R1重跑五轮；初次核验不代表首轮完成。Top2保留原一次修正接口来源，重跑Top1使用两次修正，报告披露接口预算差异。

活动协议`protocols/analysis_output_repair_v11_07fa7423`。不得再次stage/native/finish/delete/approve/launch；
journal阶段deleted_ready_for_new_approval是事务阶段记录，后续已有真实launch。下一步35分钟活动实验巡检，
完整FINALIZED_v11后才进行其自身两教师最优轮评估。没有V12批准，不重启健康worker同步文档。
