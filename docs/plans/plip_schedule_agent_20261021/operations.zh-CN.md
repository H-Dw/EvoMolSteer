# 四假设、20版本后台运行手册

本文件是当前操作协议。旧逐链/三教师/10轮命令只在Git历史90254155中留档，不得用于新任务。
唯一服务器ssh.zzai.scnet.cn:11762。环境/密钥完整继承交互环境，禁止输出凭据。Python为/opt/miniforge3/envs/molsteer-flowr-dtk/bin/python。

## 准备和运行

先本地测试提交推送，再远端检出固定commit。设置PYTHONPATH为该部署目录/src、OMP_NUM_THREADS=4。
命令入口为 `scripts/plip_group_pipeline.py`；默认root为plan中的新council结果目录。

1. `prepare --source-root /opt/MolSteer/generated_datasets/plip_schedule_agent_top1_5nn4_r123_i4_s42_20261021`
   仅从旧根引用历史配置、模板、基线及初态，不导入旧R0/优化结果。新根只能prepare一次。
2. `validate`：两教师真实batch400共同随机流及10/26/51/76混合父状态恢复。诊断不计入科学结果，证据保留，结构清理。
3. `launch --version v1`：验收通过后，认证网络预检成功才脱离SSH。完整继承API/代理，不记录值。
4. `status`：结合真实进程、pipeline.json、质量状态及日志判断，不能只凭running。
5. `report --version vN` / `review-pack --version vN`：重建阶段报告或审阅分片，不新增采样。

工作进程持有worker.lock；两个教师单GPU、CPU/LLM交错。实际部署以launch.json、workers.json及版本deployment.json为准。
失败写operations/failure_*.json并标记attention。成功采样/请求/质量缓存复用；不得覆盖失败证据。
修复必须本地测试、提交、推送再部署。冻结源码指纹变更时必须记录兼容性审计与重新验收，不能直接覆盖PREPARED绕过保护。

## GPT全量复盘及下一版

每版两教师各5轮，4000条优化记录。第5轮分析完成后生成FINALIZED_vN.json，状态awaiting_master_review。
读取skills/plip-schedule-review/SKILL.md，再读versions/vN/review/manifest.json。
逐片覆盖52个会话：2个bootstrap、40个私有组内会话和10个汇总会话。审阅全部独有prompt、工具证据、方案和结果，记录反例及未知问题。
将真实逐会话审阅JSON与复盘文档提交到本地仓库。每条审阅字段：id、evidence_sha256、assessment、decision_update、counterexample_or_limitation、evidence_files。
evidence_files必须引用该会话清单中的实际prompt文件；不要使用自动填充模板冒充实质审阅。
修改通用analyst/synthesizer skills，保留条件、证据、方向试验、反证结构，不携带teacher/slot最优幅度表。
本地提交推送、远端部署相同commit后，写versions/v(N+1)/review_approval.json，字段为：

```json
{
  "previous_version": "vN", "next_version": "vN+1",
  "finalized_sha256": "actual", "manifest_sha256": "actual",
  "review_commit": "deployed 40-character commit",
  "skill_sha256": {"analyst": "actual", "synthesizer": "actual"},
  "review_document": "repository-relative review.md", "review_document_sha256": "actual",
  "coverage": [{"path": "repository-relative reviewed-shard.json", "sha256": "actual"}]
}
```

`review-check --version vN+1`核验全部覆盖、证据、已提交文件与部署commit，再`launch --version vN+1`。
V2–V20均须前一版真实复盘，不允许空文件批准。每版共享R0、清空全部搜索历史；V20结束不再复跑。

## 清理与巡检

每链保留TOP1，四组评价及文档保存完成后校验快照并原子发布，之后仅删除非最佳pt.gz/h5/sdf/pdb等结构。
保留全部JSON/XML/prompt/响应、失败、清理哈希；共享R0和各版本最后100条/教师TOP1不删除。
程序故障时保留输入修复，不用质量失败标记跳过错误。不重新采样旧Unguided/Steer或零引导。
初始每20分钟巡检实际进程、完整轮次、错误、磁盘与质量/清理；稳定后按耗时调整，最高6小时。第5轮结束后及时进行GPT复盘。
只在实质变化/完成/异常通知。V20完整报告交付后删除plip-11762任务。
