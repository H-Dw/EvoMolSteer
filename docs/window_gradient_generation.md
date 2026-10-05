# 学习窗口内的实际状态梯度控制

本入口用于让 FLOWR 实际中间态接近原始 Steer 群体。生成窗口从参考数据和奖励配置读取，两者必须一致；不沿用旧 multistage 程序的全程控制。所有 100 步原生积分继续执行，窗口以外不注入梯度。

## 状态和时间契约

原始记录的 `score_time=t` 对应当前状态 X_t；该次选择发生在积分后，复制的是 `state_time=s=t+dt` 的 proposal。严格的窗口末端 X_b 取 `current_*[score_time=b]`，并核对前一个事件的 `proposal_*[selected_indices]`。控制条件是 `t >= a` 且 `s <= b`。本案例学习窗口为 [0,.5]，所以最后修正 .49→.50，.50→.51 不修正。通用程序没有写死该边界。

`window_state.npz` 保存实际 X_b，坐标仍为模型坐标；世界坐标换算使用同一 campaign 的 `frame_batch_*.json` 和 `coord_scale`。评价器要求它与 HDF5 的实际 current 状态逐位一致。

## 奖励与梯度

对每个实际状态时间保存发现集原子点云。每批等权，批内确定性 medoid 按其代表的候选数加权；没有跨分子原子对应，也不对不同分子的坐标线性插值。受体坐标系固定，不对配体单独对齐。

令 G_sigma 为包含两侧自核项的点云 MMD²，A_sigma 为坐标高斯核乘以原子类型 Kronecker 核的 MMD²，B 为各硬键类型的距离一阶、二阶矩差。距离矩用 3 Å 归一化。不同时间的参考均从同刻实际数据计算，不能使用终态理想键长替代。

    L_r(q,s) = mean_sigma G_sigma(q,Y_r(s))
               + lambda_type * mean_sigma A_sigma(q,Y_r(s))
               + lambda_bond * B(q,Y_r(s))
    R(q,s) = T * log sum_r pi_r * exp(-L_r(q,s)/T)

先运行原生积分得到 q_s，再对该实际 proposal 的坐标直接计算 `grad_q R`。控制量按原生位移 RMS 乘以 eta 归一化，另受每原子位移、累计 RMS 和几何回溯限制。eta 与奖励内部权重 lambda 分开校准；不会再次乘 dt。零梯度或零原生位移不强制移动。

这里不需要经过 FLOWR endpoint 的坐标 Jacobian，也没有训练额外神经网络。奖励使用冻结的经验参考点云和解析核函数。原子、键的硬标签均 detached；当前 typed/bond 项只对坐标可微，不直接修改离散标签，也不声称实现了概率头 Jacobian。几何约束不是物理势能或结合自由能。

## 输入、输出与复用

先在本地生成固定参考；可显式传 `--window-start`、`--window-end`，省略时从数据中推导观测范围：

```text
python scripts/build_window_reference.py --dataset <original_dataset> --campaign <original_campaign> --batches 0,1,2,3,4,5,6,7,8,9,10,11,12,13 --output <reference.json.gz> --representatives 2
```

奖励 JSON 的 `window` 和 `reference_sha256` 必须匹配该参考。冻结配置在 `configs/experiments/ck2_window_iter8_v1/round_*.json`；更换学习范围时应重新提取参考和配置，而非只修改推理的结束时间。支持独立指定 FLOWR 源码路径和 checkpoint：

```text
python scripts/generate_window_flowr.py --flowr-root <FLOWR_ROOT> --input-dataset <original_dataset> --root <new_output_dataset> --checkpoint <checkpoint> --program <round_program.json> --reference <reference.json.gz> --campaign <new_campaign> --steps 100 --n 32 --batch 16 --seed 20261205 --arms unguided,gradient
```

也可安装项目后运行 `evomolsteer-window`。推理支持 `unguided`、`gradient`、`gradient_zero`，不接受 SMC 臂。上游选择函数的入口断言经显式适配允许 `apply_guidance=False`；真实重采样入口若被调用会抛错。原生 SDE 和原子/键的类别抽样仍保留，后者不属于粒子筛选重采样。

远端仅执行冻结推理、无损轨迹记录、校验和打包。沿用逐步 HDF5 提交，不累计原始 NPZ；保留原生 proposal、修正后 proposal、actual current、预测终态、分数、类别、键、时间、身份父子关系和控制遥测。`final_records.json` 中的构建状态为 `not_evaluated_remote_inference_only`，不能将其解释为化学构建失败。

本地核验下载并分析：

```text
python scripts/verify_generation_archive.py --archive <archive.tar.gz> --metadata <archive.tar.gz.json> --destination <new_local_dataset> --report <download_report.json>
python scripts/evaluate_window_flowr.py --dataset <new_local_dataset> --campaign <campaign> --original <original_dataset> --original-campaign <original_campaign> --output <analysis_directory>
```

主指标是固定受体帧下、最优原子指派 RMSD 的双向最近邻均值；两方向单独保存。它衡量形状及参考覆盖，不证明完整分布或复制频率等价。另保存硬类别、组成、有键集合差异、全部窗口节点的描述符及时间导数。缺少完成标记、批次、候选或窗口快照时拒绝评价；禁用窗口外控制和粒子重采样是硬检查。

## 八轮记录

`scripts/prepare_window_round.py` 冻结基于数据的声明式修订，记录来源哈希、精确参数变化、假设、备选方案和决策摘要，不保存私有思维链。`scripts/calibrate_window_component_weights.py` 在本地按梯度尺度给出辅助项的初始权重，不把工程比例解释为生物最优值。`scripts/compare_window_rounds.py` 按预声明准则记录保留与拒绝；最后新 seed 确认不参与调参。

本轮原始数据、完整结果及每轮设计记录分别位于 `data/window_iter8/`、`results/window_iter8/`、`docs/experiments/ck2_window_iter8_20261005/`。大数据不进 Git，源码、冻结参数和小型结果进 Git。每轮均为本地提交与 VPN push 后，远端 pull 再推理。
