# New affinity-primary geometry campaign

16/30 rounds retained. Seed42, dynamic learned window, complete100 native steps; no SMC.

|Round|Split|Reward|Dose|Mean head delta|Best valid head|Unique Top5|Response|
|---:|---|---|---:|---:|---:|---:|---|
|1|discovery|motif_mixture|0.3|0.016488|8.110496|7.954408|flat_response|
|2|discovery|motif_mixture|0.51|0.053348|8.237673|8.044123|meaningful_gain|
|3|discovery|motif_mixture|0.867|0.061806|8.121699|8.092945|meaningful_gain|
|4|discovery|affinity_landmark|0.867|0.030307|8.119930|7.938727|promising_gain|
|5|discovery|endpoint_direction|0.3|0.059752|8.183568|8.039432|meaningful_gain|
|6|discovery|endpoint_direction|0.3|-0.006339|8.352077|8.004952|flat_response|
|7|discovery|endpoint_pointcloud|0.3|0.229293|8.237215|8.081687|meaningful_gain|
|8|discovery|endpoint_direction|0.3|-0.009526|8.147816|8.052607|flat_response|
|9|discovery|endpoint_supported_attractor|0.3|-0.041429|8.177889|8.055673|negative_affinity|
|10|discovery|endpoint_pointcloud|0.51|0.206789|8.236473|8.114897|meaningful_gain|
|11|discovery|endpoint_supported_attractor|0.3|-0.039052|8.353724|8.080254|negative_affinity|
|12|discovery|endpoint_pointcloud|0.867|0.139001|8.297089|8.097913|meaningful_gain|
|13|discovery|endpoint_regional_pointcloud|0.3|0.212839|8.298361|8.132443|meaningful_gain|
|14|discovery|endpoint_pointcloud|0.3|0.194623|8.238256|8.091015|meaningful_gain|
|15|discovery|endpoint_regional_pointcloud|0.3|0.203369|8.341009|8.125544|meaningful_gain|
|16|discovery|endpoint_pointcloud|0.3|0.103399|8.158355|8.026102|meaningful_gain|

Historical Steer100 mean7.510351, best-valid8.347940; not an equal-budget paired heldout arm.
Mean/maximum/top5 and physical tails are distinct outcomes. Recorded head predictions are not measured affinity.
26 adaptive discovery rounds; four subsequent frozen checks. Registered search implements the independently tested Skill policy.
