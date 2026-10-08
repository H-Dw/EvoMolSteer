# 实际运行架构的补充核验

此核验在第 15 轮运行期间进行，只读取和传输上游源码，没有改变模型文件或生产推理。它是冻结 Agent 调用完成后的独立审计，不能声称已经被早先 Agent 消费。旧 `model_compatibility_audit.md` 作为已绑定输入保持字节不变。

远端 `sha256sum` 与本地下载文件一致：

|实际远端文件|SHA256|本地捕获|
|---|---|---|
|`flowr_root/flowr/models/pocket.py`|`ccc5199b91c46dba0c24b0a98f68085a7b7a9998970485a770d9823ae197a966`|`test/flowcompat30_driver/actual_pocket.py`|
|`flowr_root/flowr/models/pocket_util.py`|`8bdc8f3635ece1b8e16eb4e8846e464cb5e6f8dbdd2faa51e75f9adb86830cfd`|`test/flowcompat30_driver/actual_pocket_util.py`|

这补齐了旧审计中两份模块源码尚未逐文件与远端绑定的缺口。实际反传记录中的 `gen.ligand_dec.layers.*`、`coord_emb` 和 `coord_out_proj` 也对应此运行架构。

实际 `LigandDecoder` 将当前坐标与自条件坐标送入 `coord_emb`（约 754–793 行），经 `SemlaLayer` 的 self-attention、pocket conditional attention 与 feedforward 更新，再由归一化和 `coord_out_proj` 生成预测端点（约 900 行）。具体可选的 skip connection、额外 embedding 与时间设置是否启用仍应按 checkpoint hparams 判断，不能由源码中存在选项推断启用。

`pocket_util.py` 中 `SemlaLayer`（约 509 行）确实含不同通路：约 693 行执行 self-attention，约 707 行执行条件通路，随后更新 invariant/equivariant 特征。现有 hook 测量整个 layer 的多个输出，没有分别记录 attention 权重或做模块干预，因此不能把整个 layer 输出敏感度当作某个 attention 的亲和力贡献。

还有一个与化学图自由度直接相关的事实：`pocket.py` 921–924 行将 `out_coords` 送入 `bond_refine`，随后生成 `bond_logits`。坐标与化学图预测存在明确的架构耦合；坐标引导导致后续化学图变化是合理的模型响应。实验只检验最终有效率、构象和亲和力，没有以“化学图与基线不同”为失败 gate。

上述源码说明哪些通路可能传递坐标控制，不能识别哪些区域被模型因果重视。进一步因果归因需要独立区域输入或模块干预对照；本次未进行此类额外前向审计。
