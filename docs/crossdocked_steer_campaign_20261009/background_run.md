# CrossDocked 100-target 后台生成

2026-10-09使用新的SSH端口10104；认证信息仅使用用户提供的会话凭据，不进入代码、报告或自动化配置。

程序、模型和输入仍位于 `/root/private_data/MolSteer/`。新节点提供一张K500SM_AI GPU，Torch 2.9.0 / HIP 6.3.26093，预检可用显存67,844,964,352字节。模型为 `flowr_root/checkpoints/flowr_root_v2.ckpt`。

私有存储总配额5 GiB，可用约1.70 GiB，不能承载100个target的新轨迹。组存储可用约114 GiB。因此仅新增输出使用独立目录 `/root/group_data/daweihuang/MolSteer/flowr_root/experiments/crossdocked100_steer_learning`，在原实验目录创建同名链接；不搬移或删除模型、输入和历史Steer结果。新目录权限限制为所有者访问，控制文件位于同一用户目录下的 `controls/crossdocked100_steer_20261009`。

生成使用仓库配置 `configs/generation_crossdocked100_steer.json`：100组独立pocket/ligand输入，每target 1000候选槽位、10批×100、100积分步、seed=42。仅优化本target预测亲和力，0–0.5包含评分窗口内51个选择事件，随后推理至1.0。没有奖励梯度。

后台命令的主体如下。实际PID、启动时间、代码版本和输出路径记录于控制目录的 `launch.json`；标准输出与异常输出均记录于 `controller.log`。

```bash
cd /root/private_data/MolSteer/EvoMolSteer
nohup env CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=42 \
  bash scripts/scnet_steer_targets.sh \
  > "$control/controller.log" 2>&1 < /dev/null &
pid=$!
```

每10个**完成的target任务**组成一个tar.gz包。完整窗口候选、淘汰分支、全程谱系、终末结构、预测评分、失败结果、输入和来源信息全部包含于保留范围。流式压缩后重新顺序解压并逐文件验证SHA256，发布归档及恢复索引后立即回收该组工作目录。末尾不足10个成功target的余组在没有运行中任务时归档。保留少量进度文件、配置、归档索引及日志；原始输入与历史实验不会被回收。

只读监控接口：

```bash
PYTHONPATH=src /opt/miniforge3/envs/molsteer-flowr-dtk/bin/python \
  scripts/check_steer_campaign_health.py \
  --dataset /root/private_data/MolSteer/flowr_root/experiments/crossdocked100_steer_learning \
  --controller-log "$control/controller.log" --pid "$pid"
```

检查退出码：0表示至少一个完整target有原生GPU推理和核验记录且没有已检测异常；1表示尚待完成；2表示OOM、推理错误、完成证据缺失或控制器提前退出。已经归档并回收的target从tar.gz中流式读取少量来源JSON，不解压整个数据集，不重复生成或更改进度标记。

本任务每小时进行一次检查，不在对话中持续轮询。首次确认完整FLOWR target正常完成、无OOM后停止监控，100-target后台生成继续运行。启动和GPU预检本身不等于完整target已经验证完成。

本地 `tests/test_campaign_health.py` 6项测试通过，覆盖等待状态、完整原生证据、仅标记不足、OOM、失败/竞争标记及从已回收归档读取证据。这些测试使用记录夹具，不能冒充真实GPU推理结果。
