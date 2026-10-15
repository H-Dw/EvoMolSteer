# 局部motif auto-research v2：运行报告

|轮次|阶段/改动|状态|均值|最高合格|尾部产出U|合格/尝试|回滚|
|---|---|---|---:|---:|---:|---|---|
|R01|suffix/band_multiplier|completed|8.112374862035116|8.218710899353027|0.026946385701497395|6/6|False|
|R02|suffix/local_intervention|completed|8.01798423131307|8.265259742736816|0.042996724446614586|4/6|False|
|R03|from_zero/region|completed|7.662644545237224|8.127753257751465|0.0|24/24|False|
|R04|from_zero/region|completed|7.658508996168773|8.27283000946045|0.0057408809661865234|23/24|True|
|R05|from_zero/operator|completed|7.681791722774506|8.049042701721191|0.0|24/24|False|
|R07|from_zero/operator|completed|7.548984189828237|8.028063774108887|0.0|22/24|False|
|R07_v2|from_zero/dose.pointcloud_delta_A|completed|7.8189964691797895|8.168766021728516|0.001404881477355957|21/24|False|
|R08_v2|from_zero/target_schedule|completed|7.6721987922986346|8.132326126098633|0.0|24/24|True|
|R09_v2|from_zero/region|completed|7.522814313570659|8.027835845947266|0.0|23/24|False|
|R10_v2|from_zero/dose.native_rms_ratio|completed|7.551495869954427|8.027851104736328|0.0|24/24|True|
|R11_v2|from_zero/region|completed|7.6281831463178|8.12820816040039|0.0|24/24|False|
|R12_v2|from_zero/operator|completed|7.430088977018992|8.027891159057617|0.0|24/24|True|
|R13_v2|from_zero/region|completed|7.536413868268331|8.126762390136719|0.0|24/24|False|
|R14_v2|from_zero/operator|completed|7.490150272846222|8.0275239944458|0.0|24/24|True|
|R15_v2|from_zero/assignment_element_cost|completed|7.842058102289836|8.588829040527344|0.01890750726064046|23/24|False|
|R16_v2|from_zero/region|completed|7.629772384961446|8.127753257751465|0.0|24/24|False|

当前活动最佳：R15_v2；已完成16/20个研究决定。目标8.704806。所有分数为模型预测。

S0接口/API测试与冻结控制重放另计成本；后缀诊断不能代替从头生成。确认使用冻结程序，新池结果不参与调参。
