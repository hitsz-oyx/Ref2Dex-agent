# V1.34: 来源平衡的跨轨迹 CmLite 一步模型

- experiment_id: `EXP-20260923-V134-BALANCED-CM`
- branch: `agent/multitrajectory-v129`
- run_status: `PLANNED`
- conclusion: `INCONCLUSIVE`
- official actor checkpoint used: `no`

## 假设与预注册门禁

V1.33 只增加 s3 高抬升转移，即在未见 s3 seed75 将运动 EPE
从 9.73 降到 6.82 mm，动作打乱对照也通过；但 s1 DAgger
seed5909 退化为 3.43 mm，高于零位移 3.31 mm，未过联合上线
门槛。原 s1 两源只有 862 条，新增 s3 高抬升源有 33,189 条。
本实验检验训练来源平衡能否保持 s3 改善并减轻 s1 遗忘。

模型、特征和优化器仍为 V1.33 的 49D、107,012 参数相对手腕
CmLite，训练源与 SHA 不变。唯一训练变量：对 s1 seeds5910/5929
各在**内部 holdout 划分之后**重复训练分区 16 次；三个 s3 训练源
重复 1 次。这样内部 holdout 无同样样本复制泄漏。训练 30 epoch、
batch2048、AdamW `3e-4`、seed42；不使用 s3 seed75/76 或 s1
seed5909/76 更新模型。一个 GPU 6，训练输出预计 <100 MB。

独立 s3 门禁：Cm-off e160 策略新 seed76、64 env 的首 episode
非终止转移；新模型运动 EPE 至少比零位移低 20%、打乱动作后
运动 EPE 至少增加 10%、接触 precision ≥0.5。s1 保留测试：
既有 DAgger seed5909 运动 EPE <零位移，以及自训练 s1 PPO e140
新 seed76 的运动 EPE <零位移。seed5909 已用于确定数据平衡方向，
因此不能作为独立确认；s1 PPO seed76 才是新分布复核。
所有门槛通过才可设计下一轮在线 Cm-on/off 匹配实验。若其中
任一失败，停止该 49D/来源加权配方，分析特征/动力学不足。

两个 seed76 评估各在 GPU 5/6 做一次，提前终止关闭并保存转移，
总共最多两 GPU。源 checkpoint 均为本地自训练、训练 run manifest
已完成；GPU 被占、源/输入 SHA 漂移或 schema 不符即停。
单步模型离线过门不等同于最终抓取提升。
