# 网格、输出来源与引导时间范围：配对实验

更新时间：2026-10-28T21:14:32.369906+08:00

| 教师 | 组 | 状态 | 条件原子RMSD/Å | 平均pIC50 | 构建/50 | PB抽样通过/10 |
|---|---|---|---:|---:|---:|---:|
| 6pyy_H7S_A_402 | gradient__native1__window | complete | 0.5832577311992645 | 4.587738146781922 | 50 | 10 |
| 6pyy_H7S_A_402 | gradient__native1__full_interior | failed | None | None | None | None |
| 6pyy_H7S_A_402 | gradient__raw1__window | complete | 0.5823873184621334 | 4.588088021278382 | 50 | 10 |
| 6pyy_H7S_A_402 | gradient__raw1__full_interior | failed | None | None | None | None |
| 6pyy_H7S_A_402 | gradient__raw999__window | complete | 0.58304138854146 | 4.588990077972412 | 50 | 9 |
| 6pyy_H7S_A_402 | gradient__raw999__full_interior | failed | None | None | None | None |
| 6pyy_H7S_A_402 | chemical__native1__window | complete | 0.49137067452073097 | 4.769111351966858 | 44 | 9 |
| 6pyy_H7S_A_402 | chemical__native1__full_interior | failed | None | None | None | None |
| 6pyy_H7S_A_402 | chemical__raw1__window | complete | 0.49219508081674573 | 4.813600935935974 | 49 | 10 |
| 6pyy_H7S_A_402 | chemical__raw1__full_interior | failed | None | None | None | None |
| 6pyy_H7S_A_402 | chemical__raw999__window | complete | 2.7198924204707144 | 4.662286462783814 | 50 | 10 |
| 6pyy_H7S_A_402 | chemical__raw999__full_interior | failed | None | None | None | None |
| 3w54_RNB_A_504 | gradient__native1__window | complete | 0.35804723247885706 | 8.082273769378663 | 50 | 10 |
| 3w54_RNB_A_504 | gradient__native1__full_interior | complete | 2.0421814849972724 | 6.296019144058228 | 49 | 2 |
| 3w54_RNB_A_504 | gradient__raw1__window | complete | 0.35853400722146034 | 8.094518566131592 | 50 | 10 |
| 3w54_RNB_A_504 | gradient__raw1__full_interior | complete | 2.0574425143003463 | 6.306609992980957 | 49 | 2 |
| 3w54_RNB_A_504 | gradient__raw999__window | complete | 0.37985245063900946 | 7.808360404968262 | 50 | 10 |
| 3w54_RNB_A_504 | gradient__raw999__full_interior | complete | 3.3789747440069915 | 6.463966751098633 | 49 | 4 |
| 3w54_RNB_A_504 | chemical__native1__window | complete | 0.25068212307989596 | 8.090530233383179 | 46 | 9 |
| 3w54_RNB_A_504 | chemical__native1__full_interior | complete | 2.2132920368015765 | 6.471743717193603 | 44 | 1 |
| 3w54_RNB_A_504 | chemical__raw1__window | complete | 0.2520033475011587 | 8.115802354812622 | 49 | 10 |
| 3w54_RNB_A_504 | chemical__raw1__full_interior | complete | 2.2288903428614137 | 6.431265382766724 | 48 | 3 |
| 3w54_RNB_A_504 | chemical__raw999__window | complete | 0.20782078206539153 | 8.31426417350769 | 50 | 10 |
| 3w54_RNB_A_504 | chemical__raw999__full_interior | complete | 3.273224191069603 | 6.393330106735229 | 49 | 4 |

## 配对条件原子RMSD差异（处理−对照）

| 教师 | 对比 | 平均差/Å | 五种子探索性95%区间 |
|---|---|---:|---|
| 6pyy_H7S_A_402 | gradient__raw999__window − gradient__native1__window | -0.0002163426578044858 | [-0.07441936220851288, 0.0739866768929039] |
| 6pyy_H7S_A_402 | gradient__raw999__window − gradient__raw1__window | 0.0006540700793266324 | [-0.0739296324768329, 0.07523777263548617] |
| 6pyy_H7S_A_402 | gradient__raw1__window − gradient__native1__window | -0.0008704127371311187 | [-0.0017796262208744228, 3.8800746612185346e-05] |
| 6pyy_H7S_A_402 | gradient__raw999__full_interior − gradient__native1__full_interior | None | unavailable_group_failure_or_pending |
| 6pyy_H7S_A_402 | gradient__raw999__full_interior − gradient__raw1__full_interior | None | unavailable_group_failure_or_pending |
| 6pyy_H7S_A_402 | gradient__raw1__full_interior − gradient__native1__full_interior | None | unavailable_group_failure_or_pending |
| 6pyy_H7S_A_402 | gradient__native1__full_interior − gradient__native1__window | None | unavailable_group_failure_or_pending |
| 6pyy_H7S_A_402 | gradient__raw1__full_interior − gradient__raw1__window | None | unavailable_group_failure_or_pending |
| 6pyy_H7S_A_402 | gradient__raw999__full_interior − gradient__raw999__window | None | unavailable_group_failure_or_pending |
| 6pyy_H7S_A_402 | chemical__raw999__window − chemical__native1__window | 2.2285217459499838 | [-3.920537424631003, 8.37758091653097] |
| 6pyy_H7S_A_402 | chemical__raw999__window − chemical__raw1__window | 2.227697339653969 | [-3.9215233408072088, 8.376918020115147] |
| 6pyy_H7S_A_402 | chemical__raw1__window − chemical__native1__window | 0.0008244062960147857 | [0.00042810626071510953, 0.0012207063313144618] |
| 6pyy_H7S_A_402 | chemical__raw999__full_interior − chemical__native1__full_interior | None | unavailable_group_failure_or_pending |
| 6pyy_H7S_A_402 | chemical__raw999__full_interior − chemical__raw1__full_interior | None | unavailable_group_failure_or_pending |
| 6pyy_H7S_A_402 | chemical__raw1__full_interior − chemical__native1__full_interior | None | unavailable_group_failure_or_pending |
| 6pyy_H7S_A_402 | chemical__native1__full_interior − chemical__native1__window | None | unavailable_group_failure_or_pending |
| 6pyy_H7S_A_402 | chemical__raw1__full_interior − chemical__raw1__window | None | unavailable_group_failure_or_pending |
| 6pyy_H7S_A_402 | chemical__raw999__full_interior − chemical__raw999__window | None | unavailable_group_failure_or_pending |
| 3w54_RNB_A_504 | gradient__raw999__window − gradient__native1__window | 0.021805218160152438 | [-0.10042401811480287, 0.14403445443510773] |
| 3w54_RNB_A_504 | gradient__raw999__window − gradient__raw1__window | 0.021318443417549134 | [-0.10032425892486019, 0.14296114575995847] |
| 3w54_RNB_A_504 | gradient__raw1__window − gradient__native1__window | 0.000486774742603302 | [-0.0008485418278714731, 0.001822091313078077] |
| 3w54_RNB_A_504 | gradient__raw999__full_interior − gradient__native1__full_interior | 1.3367932590097187 | [0.04604059775443359, 2.6275459202650038] |
| 3w54_RNB_A_504 | gradient__raw999__full_interior − gradient__raw1__full_interior | 1.321532229706645 | [0.01845469669086275, 2.624609762722427] |
| 3w54_RNB_A_504 | gradient__raw1__full_interior − gradient__native1__full_interior | 0.015261029303073884 | [-0.011658482942392023, 0.04218054154853979] |
| 3w54_RNB_A_504 | gradient__native1__full_interior − gradient__native1__window | 1.6841342525184153 | [0.8451799592083602, 2.5230885458284704] |
| 3w54_RNB_A_504 | gradient__raw1__full_interior − gradient__raw1__window | 1.698908507078886 | [0.8487147603804074, 2.549102253777365] |
| 3w54_RNB_A_504 | gradient__raw999__full_interior − gradient__raw999__window | 2.999122293367982 | [2.5774854762618773, 3.420759110474087] |
| 3w54_RNB_A_504 | chemical__raw999__window − chemical__native1__window | -0.04286134101450444 | [-0.08795097410408509, 0.002228292075076223] |
| 3w54_RNB_A_504 | chemical__raw999__window − chemical__raw1__window | -0.04418256543576717 | [-0.08705539799461304, -0.0013097328769213143] |
| 3w54_RNB_A_504 | chemical__raw1__window − chemical__native1__window | 0.001321224421262741 | [-0.0012331086019398666, 0.0038755574444653486] |
| 3w54_RNB_A_504 | chemical__raw999__full_interior − chemical__native1__full_interior | 1.0599321542680262 | [-0.2810067189892578, 2.4008710275253105] |
| 3w54_RNB_A_504 | chemical__raw999__full_interior − chemical__raw1__full_interior | 1.044333848208189 | [-0.26341595712121624, 2.352083653537594] |
| 3w54_RNB_A_504 | chemical__raw1__full_interior − chemical__native1__full_interior | 0.015598306059837341 | [-0.05539847217515007, 0.08659508429482475] |
| 3w54_RNB_A_504 | chemical__native1__full_interior − chemical__native1__window | 1.9626099137216806 | [1.3808918605381293, 2.544327966905232] |
| 3w54_RNB_A_504 | chemical__raw1__full_interior − chemical__raw1__window | 1.976886995360255 | [1.3724115985126033, 2.5813623922079065] |
| 3w54_RNB_A_504 | chemical__raw999__full_interior − chemical__raw999__window | 3.0654034090042113 | [1.7641235191082831, 4.366683298900139] |

完整原始差异、实用差异阈值、失败、位移诊断和质量覆盖见RESULTS.json。五个种子同时用于历史配置选择，所有区间仅为探索性配对描述，不是独立留出验证或多重比较校正后的显著性结论。失败不补样，不用中位数替代含极端值的全量均值。
raw999：100区间0–0.999，最后循环网络约0.98901；raw1：100区间0–1，最后循环网络约0.99；native1额外终态前向约0.9999。只有化学/native1在最终输出投影化学特征；全部组均不覆盖坐标。full_interior在1–99步引导，跳过奇异t=0。
