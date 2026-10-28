# 网格、输出来源与引导时间范围：配对实验

更新时间：2026-10-28T23:04:26.303119+08:00

| 教师 | 组 | 状态 | 条件原子RMSD/Å | 平均pIC50 | 构建/50 | PB抽样通过/10 |
|---|---|---|---:|---:|---:|---:|
| 6pyy_H7S_A_402 | gradient__raw1__window | complete | 0.5823873184621334 | 4.588088021278382 | 50 | 10 |
| 6pyy_H7S_A_402 | gradient__raw1__late_only | complete | 0.5819473637640477 | 4.588201704025269 | 50 | 10 |
| 6pyy_H7S_A_402 | gradient__raw1__full_interior | failed | None | None | None | None |
| 6pyy_H7S_A_402 | gradient__raw999__window | complete | 0.58304138854146 | 4.588990077972412 | 50 | 9 |
| 6pyy_H7S_A_402 | gradient__raw999__late_only | complete | 0.5825582896173 | 4.5890700340271 | 50 | 9 |
| 6pyy_H7S_A_402 | gradient__raw999__full_interior | failed | None | None | None | None |
| 6pyy_H7S_A_402 | chemical__raw1__window | complete | 0.49219508081674573 | 4.813600935935974 | 49 | 10 |
| 6pyy_H7S_A_402 | chemical__raw1__late_only | complete | 0.49172930754721167 | 4.8137670469284055 | 49 | 10 |
| 6pyy_H7S_A_402 | chemical__raw1__full_interior | failed | None | None | None | None |
| 6pyy_H7S_A_402 | chemical__raw999__window | complete | 2.7198924204707144 | 4.662286462783814 | 50 | 10 |
| 6pyy_H7S_A_402 | chemical__raw999__late_only | complete | 2.6379561342298983 | 4.6624193239212035 | 50 | 10 |
| 6pyy_H7S_A_402 | chemical__raw999__full_interior | failed | None | None | None | None |
| 3w54_RNB_A_504 | gradient__raw1__window | complete | 0.35853400722146034 | 8.094518566131592 | 50 | 10 |
| 3w54_RNB_A_504 | gradient__raw1__late_only | complete | 0.3579953959584236 | 8.094468631744384 | 50 | 10 |
| 3w54_RNB_A_504 | gradient__raw1__full_interior | complete | 2.0574425143003463 | 6.306609992980957 | 49 | 2 |
| 3w54_RNB_A_504 | gradient__raw999__window | complete | 0.37985245063900946 | 7.808360404968262 | 50 | 10 |
| 3w54_RNB_A_504 | gradient__raw999__late_only | complete | 0.3792613856494427 | 7.808479433059692 | 50 | 10 |
| 3w54_RNB_A_504 | gradient__raw999__full_interior | complete | 3.3789747440069915 | 6.463966751098633 | 49 | 4 |
| 3w54_RNB_A_504 | chemical__raw1__window | complete | 0.2520033475011587 | 8.115802354812622 | 49 | 10 |
| 3w54_RNB_A_504 | chemical__raw1__late_only | complete | 0.2514522390067577 | 8.115641384124755 | 49 | 10 |
| 3w54_RNB_A_504 | chemical__raw1__full_interior | complete | 2.2288903428614137 | 6.431265382766724 | 48 | 3 |
| 3w54_RNB_A_504 | chemical__raw999__window | complete | 0.20782078206539153 | 8.31426417350769 | 50 | 10 |
| 3w54_RNB_A_504 | chemical__raw999__late_only | complete | 0.20732150509953498 | 8.314137916564942 | 50 | 10 |
| 3w54_RNB_A_504 | chemical__raw999__full_interior | complete | 3.273224191069603 | 6.393330106735229 | 49 | 4 |

## 配对条件原子RMSD差异（处理−对照）

| 教师 | 对比 | 平均差/Å | 五种子探索性95%区间 |
|---|---|---:|---|
| 6pyy_H7S_A_402 | gradient__raw1__late_only − gradient__raw1__window | -0.000439954698085785 | [-0.0005597081940133912, -0.0003202012021581788] |
| 6pyy_H7S_A_402 | gradient__raw1__full_interior − gradient__raw1__late_only | None | unavailable_group_failure_or_pending |
| 6pyy_H7S_A_402 | gradient__raw999__late_only − gradient__raw999__window | -0.00048309892416000357 | [-0.0005684892107353952, -0.00039770863758461193] |
| 6pyy_H7S_A_402 | gradient__raw999__full_interior − gradient__raw999__late_only | None | unavailable_group_failure_or_pending |
| 6pyy_H7S_A_402 | gradient__raw999__window − gradient__raw1__window | 0.0006540700793266324 | [-0.0739296324768329, 0.07523777263548617] |
| 6pyy_H7S_A_402 | gradient__raw999__late_only − gradient__raw1__late_only | 0.0006109258532524114 | [-0.07393017687464182, 0.07515202858114664] |
| 6pyy_H7S_A_402 | gradient__raw999__full_interior − gradient__raw1__full_interior | None | unavailable_group_failure_or_pending |
| 6pyy_H7S_A_402 | chemical__raw1__late_only − chemical__raw1__window | -0.00046577326953411107 | [-0.0005766092958442925, -0.00035493724322392965] |
| 6pyy_H7S_A_402 | chemical__raw1__full_interior − chemical__raw1__late_only | None | unavailable_group_failure_or_pending |
| 6pyy_H7S_A_402 | chemical__raw999__late_only − chemical__raw999__window | -0.08193628624081613 | [-0.308013292996059, 0.14414072051442678] |
| 6pyy_H7S_A_402 | chemical__raw999__full_interior − chemical__raw999__late_only | None | unavailable_group_failure_or_pending |
| 6pyy_H7S_A_402 | chemical__raw999__window − chemical__raw1__window | 2.227697339653969 | [-3.9215233408072088, 8.376918020115147] |
| 6pyy_H7S_A_402 | chemical__raw999__late_only − chemical__raw1__late_only | 2.1462268266826867 | [-3.7769021848479443, 8.069355838213317] |
| 6pyy_H7S_A_402 | chemical__raw999__full_interior − chemical__raw1__full_interior | None | unavailable_group_failure_or_pending |
| 3w54_RNB_A_504 | gradient__raw1__late_only − gradient__raw1__window | -0.000538611263036728 | [-0.0005986886221997328, -0.0004785339038737231] |
| 3w54_RNB_A_504 | gradient__raw1__full_interior − gradient__raw1__late_only | 1.6994471183419229 | [0.8493008649156092, 2.5495933717682364] |
| 3w54_RNB_A_504 | gradient__raw999__late_only − gradient__raw999__window | -0.0005910649895668029 | [-0.0007781099740095192, -0.0004040200051240866] |
| 3w54_RNB_A_504 | gradient__raw999__full_interior − gradient__raw999__late_only | 2.999713358357549 | [2.578024852305947, 3.421401864409151] |
| 3w54_RNB_A_504 | gradient__raw999__window − gradient__raw1__window | 0.021318443417549134 | [-0.10032425892486019, 0.14296114575995847] |
| 3w54_RNB_A_504 | gradient__raw999__late_only − gradient__raw1__late_only | 0.021265989691019054 | [-0.10025871364502639, 0.1427906930270645] |
| 3w54_RNB_A_504 | gradient__raw999__full_interior − gradient__raw1__full_interior | 1.321532229706645 | [0.01845469669086275, 2.624609762722427] |
| 3w54_RNB_A_504 | chemical__raw1__late_only − chemical__raw1__window | -0.0005511084944009781 | [-0.0008110811587618717, -0.00029113583004008455] |
| 3w54_RNB_A_504 | chemical__raw1__full_interior − chemical__raw1__late_only | 1.9774381038546565 | [1.3727362824472245, 2.5821399252620885] |
| 3w54_RNB_A_504 | chemical__raw999__late_only − chemical__raw999__window | -0.0004992769658565522 | [-0.0006316023244609776, -0.00036695160725212675] |
| 3w54_RNB_A_504 | chemical__raw999__full_interior − chemical__raw999__late_only | 3.065902685970068 | [1.764552096700581, 4.367253275239555] |
| 3w54_RNB_A_504 | chemical__raw999__window − chemical__raw1__window | -0.04418256543576717 | [-0.08705539799461304, -0.0013097328769213143] |
| 3w54_RNB_A_504 | chemical__raw999__late_only − chemical__raw1__late_only | -0.044130733907222754 | [-0.08686241265924562, -0.0013990551551998909] |
| 3w54_RNB_A_504 | chemical__raw999__full_interior − chemical__raw1__full_interior | 1.044333848208189 | [-0.26341595712121624, 2.352083653537594] |

完整原始差异、实用差异阈值、失败、位移诊断和质量覆盖见RESULTS.json。五个种子同时用于历史配置选择，所有区间仅为探索性配对描述，不是独立留出验证或多重比较校正后的显著性结论。失败不补样，不用中位数替代含极端值的全量均值。
raw999：100区间0–0.999，最后循环网络约0.98901；raw1：100区间0–1，最后循环网络约0.99；native1额外终态前向约0.9999。只有化学/native1在最终输出投影化学特征；全部组均不覆盖坐标。full_interior在1–99步引导，跳过奇异t=0。

本专项仅新增两教师×两种化学条件×raw1/raw999的10–99步引导，共400次。10–90与1–99均复用，历史数值失败仅作为不可评价对照。raw返回第99步更新前预测，故新增的第91–98步更新影响返回值，第99步更新只影响未返回的最终状态。不得删离群点或补样。
