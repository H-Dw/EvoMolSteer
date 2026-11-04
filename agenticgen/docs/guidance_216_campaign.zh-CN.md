# Guidance 2.16.0 三组串行实验

实验1/2沿用2.15阶段探索策略，每位教师重新标定最多500候选，正式10轮、每轮最多5组完整50候选，正式上限2500。实验1开启互作区域化学inpainting，实验2关闭；初态和种子相同但全部新执行，不导入对方测量。条件proper Kabsch后完整50均值定义不变。

实验3每位教师5轮，每轮新生成50候选，复用实验1最佳完整配置的scale；无Agent、无新标定。几何条件为本轮teacher的互作位点，化学inpainting为其完整分子图，覆盖1–98步；保留raw999输出，不做终态覆盖。固定蛋白、模型与口袋，使用独立的逐轮种子组，即使teacher保持也不重复初态。

终态使用本轮输入condition的原Hungarian对应做proper Kabsch，严格RMSD<0.5 Å；统一用t=.9999对实际返回类别和坐标重评分。可构建、连通、有限且pIC50严格超过teacher者排序top10，执行PB dock。PB通过且可重建PLIP条件者中最高分成为下一轮teacher；否则保留原teacher。PLIP为空、不可评价或非法图均不能晋升。谱系、原始定位和失败原因保留。

科学候选上限12500；工程验收独立登记。第三组在任何科学尝试前执行50条完整工程轨迹，验证整图注入范围、对应、终态评分和端点处理。失败仍消耗科学预留，不补样。中断轮次恢复时保留为失败，后续轮次使用新种子。

新增入口：scripts/run_stage_affinity_campaign.py；部署入口scripts/deploy_stage_affinity.py仅接受已推送至GitHub的固定提交。SSH密码仅读GUIDANCE_REMOTE_PASSWORD环境，不持久化；交互shell启动输出被屏蔽。固定checkout、独立六任务目录，全局GPU锁，前两组各两教师完成后才进入下一组。独立任务失败不取消其余任务；第三组依赖其对应实验1完整结果。

持久镜像仅含压缩回执、JSON记录、SDF和晋升teacher原始终态，HDF5和中间checkpoint留在/workspace。镜像空间不足独立记录，不触发重新采样。最终标记区分采样与CPU诊断结束。本次科学收益以线上新实验为准。
