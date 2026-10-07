# 亲和力优先：新30轮实验报告

已保留30/30轮。主种子42；动态学习窗口后继续原生推理至1；无Steer重采样。

|轮次|集合|奖励|剂量|相对配对无引导均值Δ|有效率|有效最高预测值|
|---:|---|---|---:|---:|---:|---:|
|1|discovery|motif_mixture|0.3|0.016488|98.0%|8.110496|
|2|discovery|motif_mixture|0.51|0.053348|98.0%|8.237673|
|3|discovery|motif_mixture|0.867|0.061806|97.0%|8.121699|
|4|discovery|affinity_landmark|0.867|0.030307|96.0%|8.119930|
|5|discovery|endpoint_direction|0.3|0.059752|98.0%|8.183568|
|6|discovery|endpoint_direction|0.3|-0.006339|98.0%|8.352077|
|7|discovery|endpoint_pointcloud|0.3|0.229293|99.0%|8.237215|
|8|discovery|endpoint_direction|0.3|-0.009526|95.0%|8.147816|
|9|discovery|endpoint_supported_attractor|0.3|-0.041429|97.0%|8.177889|
|10|discovery|endpoint_pointcloud|0.51|0.206789|96.0%|8.236473|
|11|discovery|endpoint_supported_attractor|0.3|-0.039052|96.0%|8.353724|
|12|discovery|endpoint_pointcloud|0.867|0.139001|100.0%|8.297089|
|13|discovery|endpoint_regional_pointcloud|0.3|0.212839|99.0%|8.298361|
|14|discovery|endpoint_pointcloud|0.3|0.194623|98.0%|8.238256|
|15|discovery|endpoint_regional_pointcloud|0.3|0.203369|99.0%|8.341009|
|16|discovery|endpoint_pointcloud|0.3|0.103399|99.0%|8.158355|
|17|discovery|endpoint_pointcloud|0.3|0.160317|96.0%|8.291833|
|18|discovery|endpoint_pointcloud|0.3|0.166963|97.0%|8.173245|
|19|discovery|endpoint_pointcloud|0.3|0.229364|96.0%|8.558043|
|20|discovery|endpoint_pointcloud|0.195|0.110902|94.0%|8.332711|
|21|discovery|endpoint_pointcloud|0.3|0.214475|100.0%|8.237318|
|22|discovery|endpoint_regional_pointcloud|0.3|0.213974|99.0%|8.237213|
|23|discovery|endpoint_pointcloud|0.375|0.181366|98.0%|8.193687|
|24|discovery|endpoint_pointcloud|0.18|0.081101|98.0%|8.141022|
|25|discovery|endpoint_pointcloud|0.24|0.178389|97.0%|8.251294|
|26|discovery|endpoint_pointcloud|0.33|0.230299|99.0%|8.237148|
|27|validation|endpoint_pointcloud|0.33|0.116303|99.0%|8.136656|
|28|heldout|endpoint_pointcloud|0.33|0.148553|96.0%|8.310325|
|29|heldout|endpoint_pointcloud|0.33|0.143465|95.0%|8.138247|
|30|heldout|endpoint_pointcloud|0.33|0.286960|97.0%|8.295715|

validation：100个引导样本，均值7.597575，匹配无引导7.481272，Δ+0.116303；批次bootstrap95%区间[0.037531,0.195074]。
应变中位数0.519157 vs 0.563659 kcal/mol/重原子；p90 1.026039 vs 1.197656。

heldout：300个引导样本，均值7.601172，匹配无引导7.408179，Δ+0.192993；批次bootstrap95%区间[0.131026,0.265349]。
应变中位数0.509709 vs 0.540330 kcal/mol/重原子；p90 0.992158 vs 1.291178。

all_frozen：400个引导样本，均值7.600273，匹配无引导7.426452，Δ+0.173820；批次bootstrap95%区间[0.113314,0.239207]。
应变中位数0.510402 vs 0.546064 kcal/mol/重原子；p90 1.017042 vs 1.264307。

历史Steer100的均值7.510351、有效最高8.347940分开比较；它不是同预算配对留出组。去重值保留首个出现构象，与历史统计定义一致。
平均Δ接近零也可能来自正负抵消，应结合实际位移、个体绝对Δ与奖励响应判断。控制器归一化梯度方向，因此整体放大奖励数值不等于提高注入剂量；需调整剂量比及实际限幅。
坐标关联不是因果结合区域，真实FLOWR坐标VJP也不是affinity head梯度。应变是孤立配体MMFF松弛，不是结合自由能。
