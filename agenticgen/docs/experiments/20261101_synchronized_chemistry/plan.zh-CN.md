# 化学状态inpainting与梯度引导共用阶段

日期：2026-11-01（Asia/Shanghai）。接续用户要求，将新六窗口化学inpainting的开启/关闭步段与额外坐标梯度更新步段同步。只修改新代码、配置与离线核验，不部署远端、不重跑科学实验、不改变自动化或历史结果。

## 当前契约

[六窗口配置入口](../../../configs/uniguide_six_windows_zero_start.json)升级为不可变skills2.10.1，显式加入：

```json
"chemical_inpainting": {"enabled": true, "activation": "guidance_phase"}
```

| 步索引 | 额外坐标梯度/VJP | 化学状态inpainting |
| --- | --- | --- |
| 0 | 跳过 | 跳过 |
| 1–98 | 按各窗scale | 启用 |
| 99 | 跳过，返回未覆盖预测 | 跳过 |

六个控制窗口0–10/11–20/21–30/31–40/41–50/51–99不变；端点例外沿用上一提交。同步的是时间阶段，不按数值scale开关化学条件：某窗scale=0时仍在1–98注入化学状态，作为零额外坐标梯度的条件生成对照。enabled=false仍是关闭全部化学注入的独立开关。

化学注入仍在当步原生转移和坐标梯度更新之后，对同一步Hungarian对应的条件原子化学通道及条件内部键进行覆盖。坐标、self-conditioning和返回预测不覆盖，未选中的化学通道和外部键不改写；不增加模型前向、随机抽样或LLM调用。500次标定上限、实际0起点、标定根选优和Agent方法论保持。

## 接口与验收修复

- guidance_endpoints.update_bounds是共同阶段来源；额外梯度屏蔽与化学policy都读取它，避免再次分别写死步数。
- absolute/geometry配置规范化保留activation，避免它在validate_config后丢失。2.10.1拒绝缺少同步字段的配置；旧2.10.0继续使用原10–99化学阶段，不将新行为套到旧发布。
- CHEMICAL_CONDITION绑定activation、step_bounds和是否依赖scale，OUTPUT_POLICY和实际LLM上下文同时提供真实化学契约。改变阶段使兼容绑定改变，旧回执不可迁移；旧阶段的checkpoint拒绝冒充新阶段接续。
- 真实GPU验收的disabled-native比较显式关闭化学条件；原来只跑前3步时假定尚未注入，提前到步1后该假定已不成立。
- 化学前缀配对比较改为检查首次注入前及首次注入步的原生/梯度更新；新配置对应步0、1。完整100步验收按policy读取1–98的对应和化学数据，并逐步核对实际开关，不再硬切10:100。

## 离线核验与边界

新增CPU检查覆盖100步实际化学投影、0/99不投影、1和98边界、scale=0仍保留化学阶段、enabled=false、步1动态对应、坐标/self-conditioning/未选通道不变、无额外RNG、阶段绑定及旧checkpoint拒绝。同时回归旧10–99模式、六窗口端点和开放标定/Agent严格输出契约。

测试读数见validation.json与local_cpu.xml；新旧发布逐文件SHA核验。真实GPU acceptance仅修改了接口，未执行；没有加载真实模型、调用DeepSeek或生成新的科学候选。旧发布2.10.0及以前字节保持，原500次额度和生成账本不变。
