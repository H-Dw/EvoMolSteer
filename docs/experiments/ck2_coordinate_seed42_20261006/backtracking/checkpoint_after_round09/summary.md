# seed42 坐标奖励实验汇总

已完成 9/30 轮；窗口 [0.0, 0.5]。

|组别|actual x_0.5形状距离 Å ↓|较 native 改善 %|有效/PB|唯一分子|all/valid/unique head|MMFF松弛/重原子 ↓|周围松弛 RMS Å ↓|
|---|---:|---:|---:|---:|---|---:|---:|
|Original Steer|—|—|98/98|24|7.5104/7.5011/7.2682|0.398679|0.202921|
|Native|0.237427|0.000|98/98|69|7.4343/7.4370/7.2719|0.548773|0.327214|
|coordinate_r01_spread|0.233446|1.677|98/98|72|7.4056/7.4078/7.2368|0.549577|0.343861|
|coordinate_r02_global|0.231873|2.339|98/98|71|7.4273/7.4299/7.2559|0.545973|0.340162|
|coordinate_r03_anchor|0.231284|2.587|100/99|69|7.4558/7.4558/7.2965|0.570958|0.366832|
|coordinate_r04_flowdose|0.237242|0.078|97/97|71|7.4395/7.4507/7.3079|0.548593|0.317770|
|coordinate_r05_val116|0.238067|-0.269|97/97|71|7.4282/7.4391/7.3091|0.548919|0.312728|
|coordinate_r06_replay|0.231284|2.587|100/99|69|7.4558/7.4558/7.2965|0.570958|0.366832|
|coordinate_r07_firstcap|0.237648|-0.093|99/99|74|7.4338/7.4413/7.3100|0.548863|0.320539|
|coordinate_r08_half|0.231343|2.562|99/99|71|7.4358/7.4462/7.3008|0.548077|0.359083|
|coordinate_r09_pairguard|0.237046|0.161|98/98|73|7.4224/7.4227/7.2796|0.552816|0.350713|

Exploration remains active. R6 reproduced R3; R7/R8/R9 identify dose and physical tradeoffs. Partial regression triggers factor rollback. R10 tests rigid-pose projection, followed by a separately audited selection/background contrast architecture.

本表不把100个同批候选当作100个独立重复，不给出由粒子数夸大的显著性。各项完整分母、失败对照及源文件SHA保存在CSV/JSON和逐轮报告中。
