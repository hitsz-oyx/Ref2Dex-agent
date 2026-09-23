# V1.35: 适配 CmLite 的 s3 在线成对微调

- experiment_id: `EXP-20260923-V135-ADAPTED-CM-ONLINE`
- branch: `agent/multitrajectory-v129`
- run_status: `PLANNED`
- conclusion: `INCONCLUSIVE`
- official actor checkpoint used: `no`

## 假设与冻结方案

V1.30 从同一源策略微调 s3，旧相对手腕 CmLite 奖励在 seeds71–74
均落后 Cm-off；V1.32 发现旧模型在高抬升策略真实转移上运动 EPE
甚至高于零位移。V1.34 在新 s3 seed76 达到 7.49/9.97 mm，
打乱动作对照通过，且 s1 DAgger 遗忘得到修复。现在检验
**适配后、低权重的 CmLite 在线目标进展奖励**，相对完全同源的
Cm-off 继续训练，是否能改善严格抓取。

两臂固定同一 V1.30 **Cm-off s3 e160** 自训练 checkpoint，SHA256
`ce62d6efb0c9a200583635eb942b45a07d972cfae098b8edb103e805a82cb8f1`。
这是新的、显式 SHA 锁定的微调试验，不是静默续跑旧 run。
输入仍为 s3 `corrected_manifest_r2.json`，seed70、64 env、
horizon32、minibatch256、学习率 `1e-5`，相同几何奖励和接触
课程。从 e160 各继续 20 epoch 到 e180，保存 e170/e180。
Cm-on 冻结 V1.34 相对手腕 CmLite checkpoint，SHA256
`1146025ca88b35f39a5fad7f5f899a88cca34b774fa9abe0c0d6f5d72519e1b8`，
目标进展奖励系数 `1.0`、预测接触门及 0.1m 几何可信区；
Cm-off 只去掉这一项。系数低于 V1.30 的 5，故本实验测试
“新模型 + 低系数”的联合配方，不能单独归因模型重训。

先每臂 e162 工程 smoke，确认从完全相同 e160 SHA 恢复、
学习率和模型 SHA 正确；smoke 不用于科研比较。正式训练后
在独立 seed77 评估两臂 e170/e180（完整轨迹、提前终止关闭、
每个 64 首 episode）。各臂按严格成功数选 checkpoint，
平局按平均最大接触抬升、再按较早 epoch。若 Cm-on 在两个
checkpoint 均不高于 Cm-off，且平均最大接触抬升也未超过
对应 Cm-off 的 1.25 倍，则预注册停损，不用更多 seed。
否则冻结各臂候选，在未见 seeds78–80 各 64 episode 复核。
支持 Cm 在线增益要求三个 seed **每个** Cm-on 严格成功数
高于 Cm-off，且合计至少多 10/192；多轨迹稳定目标另要求
s3 每个 seed ≥90%，已有 s1 门禁仍须保持。未见 seed 的结果
不应与 seed77 选择集混为一谈。

GPU 5/6 各一臂、同时最多两卡，预计 smoke + 正式训练与
门禁可在数十分钟内完成；输出总量 <300 GB。checkpoint、
输入 manifest、代码提交、精确命令及资源由 run manifest 锁定。
发现占卡、输入漂移、SHA 不符或恢复异常立即停止。
