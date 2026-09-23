# V1.41: 接触后加强抓握及 CmLite 门控

- experiment_id: `EXP-20260923-V141-CONTACT-GRIP-REFLEX`
- branch: `agent/v141-contact-reflex`
- run_status: `PLANNED`
- conclusion: `INCONCLUSIVE`
- official actor checkpoint used: `no`

## 动机与预注册设计

V1.40 的固定整段路由在未见 seeds85–89 是254/320，
相对自训练 `back260` 的208/320有净增，但每 seed
75.0%–85.9%，未达到90%。部分失败有较长的手/物接触，
最大抬升在3cm严格线附近。测试在**同一冻结路由**上，
仅接触已连续5步且物体相对初始高度低于6cm时，是否
通过增大手指目标位置补足抓握；其他时刻不改 PPO 动作。

固定 V1.40 路由 SHA256
`7db4686f8eb273e0ac94bfbbd3625ddd35209a1f4922e8451bcbf519f6395a79`，
三个专家及训练输入均不变。每个符合条件的时刻，对
手指归一化动作索引6、8、10、12、14、15加 `+0.08`，
再截断到[-1,1]。固定三臂：

1. `off`：原路由，无动作修改；
2. `always`：条件满足就直接加0.08，不用 Cm；
3. `cmlite`：条件满足时让冻结 V1.37 CmLite 比较原动作
   与+0.08动作的预测一步物体目标进展，沿用正进展、
   预测接触≥0.5的安全门，其他情况维持原动作。

三臂在新 seed90、64个并行环境首个完整 episode、提前
终止关闭下配对；先记录 `off` 和 `always`，但无论它们
结果如何都执行 `cmlite`，避免选择性缺失。若 `cmlite`
比 `off` 至少净增5/64，且不比 `always` 低超过2/64，
再冻结在未见 seeds91–94 每 seed64环境比较三臂。
否则预注册停损，不扩大。只有 heldout 中 `cmlite`
比 `off` 和 `always` 都有一致净增，才称为 Cm 对此
干预有正向证据；稳定目标另要求每 seed≥90%。

该实验检验一步 Cm 的局部动作门控，不改变 Cm 架构；
它不同于 V1.38 对单个 e180 策略选五种含腕偏置的动作。
GPU5/6 至多两卡并发，每次实验均记录代码 commit、
路由/专家/Cm SHA 和失败 run，输出限额300GB内。
