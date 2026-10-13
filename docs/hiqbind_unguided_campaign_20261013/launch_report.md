# HiQBind unguided独立对照集合启动（2026-10-13）

服务器：`ksai.scnet.cn:10104`。后台controller PID：`32518`。启动时间：北京时间2026-10-13 12:28:44。固定执行源码commit：`59ae8aab7db076b24197cd94c855d9ea4909dd4a`。

## 对照与运行参数

- 同既有Steer集合的297个target；输入身份、顺序及文件SHA256已配对核验。
- 每target 100个候选槽位，5个batch，每batch 20粒子；共29,700槽位，构建失败记录仍计入，不补生成。
- 使用`flowr_root_v2.ckpt`，checkpoint SHA256与历史Steer manifest一致。
- 100积分步，seed=42，与Steer匹配的SDE坐标噪声0.2、终点解码及评分流程。
- 全程0–1无Steer重采样、无reward gradient guidance；[0,0.5]仅为保存中间结构的观察窗口。
- 每10个完成target打包tar.gz、逐成员校验后回收对应原始target目录；最后剩余7 target归档一包。
- 并行进度标记复用原子running/finished/error策略。

代码已在本地实现，经本机VPN接口推送GitHub；远端主仓库执行`pull --ff-only`后，使用独立固定commit worktree启动，避免后续其他实验更新源码影响此集合。FLOWR sampler文件的当前SHA256与历史Steer记录一致：`beaace3ab92a8377bc8eab95333c8f40ce370ca9224f7a78927c5a424bb38426`。

## 启动时观察到的实际状态

北京时间12:29:29：controller存活，campaign状态running；296 pending、1 running、0 complete、0 archived、0 failed，未发现OOM或任务失败。

首个target为`1bcj_NGA_1_1`。日志确认v2加载成功、seed=42，并出现真实`trajectory_step_committed`：arm=unguided、batch=0、step=0、resampled=false。原生设备为`K500SM_AI`，Torch 2.9.0、HIP 6.3.26093；runtime明确记录`apply_selective_guidance={"unguided": false}`、`particle_resampling=false`和`gradient_guidance=false`。

健康状态仍为pending：首个完整target尚未完成，不能将模型加载或第一步推进当作100候选全部成功。没有创建长期监控任务。

## 数据位置

实际新集合：

```text
/opt/MolSteer/generated_datasets/hiqbind_unguided_learning_s100_b20_20261013/
```

软链接入口：

```text
/root/private_data/MolSteer/flowr_root/experiments/hiqbind_unguided_learning/
```

后台控制目录与日志：

```text
/opt/MolSteer/run_controls/hiqbind_unguided_20261013/launch.json
/opt/MolSteer/run_controls/hiqbind_unguided_20261013/controller.log
```

其兼容入口：`/root/private_data/MolSteer/EvoMolSteer/test/hiqbind_unguided_control_20261013/`。

固定执行源码：`/opt/MolSteer/run_repositories/hiqbind_unguided_20261013_59ae8aab/`。既有Steer结果保持原位；本次没有删除旧Steer数据。新结果在节点本地磁盘，释放节点前需另行取回。

## 修改与验证

- `449d4989`：增加独立unguided入口、原生关闭SMC、模式证据/身份谱系核验、参考集合匹配及10-target归档复用。
- `631bcff9`：配对核验报告在campaign归属初始化后写入；unguided存储窗口明确标为observation_only。
- `59ae8aab`：补齐观察范围单元测试fixture。
- 相关Steer、归档、仪器化和健康测试原有31项通过；修复后的unguided模块5项通过，包括带参考集合的实际初始化/归档/恢复/跳过已完成target；新增观察范围fixture修复后对应单项再次通过。CPU fixture测试没有冒充GPU生成验证。

首次启动在领取任何target、加载模型之前被目录归属保护阻止，原因是reference_comparison.json写入顺序不当；本地修复并再次push/pull后才启动上述有效任务。首次日志、启动JSON及reference证据均保存在远端控制目录的`preclaim_failure.*`中；没有清空target进度标记，也没有丢弃任何已生成结构。

当前证据：`launch.json`、`startup_health.json`；首次启动记录：`preclaim_launch.json`、`preclaim_failure_health.json`。通用用法见`../unguided_generation.md`。
