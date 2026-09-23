# V1.33: 用高抬升策略真实转移适配低延迟 CmLite

- experiment_id: `EXP-20260923-V133-CM-ADAPT-HIGHLIFT`
- branch: `agent/multitrajectory-v129`
- run_status: `PLANNED`
- conclusion: `INCONCLUSIVE`
- official actor checkpoint used: `no`

## 假设与冻结方案

V1.32 表明旧的相对手腕 CmLite 在 s3 高成功率策略 e160 上，
运动 EPE 反而高于零位移。先不继续在线奖励训练；检验是否仅靠
加入该策略的成功/接触转移、保持 49D/107,012 参数低延迟结构，
即可恢复未见 seed 上的动作条件一步预测。

训练输入固定为 V1.29 的四个原训练源，以及 V1.30 **Cm-off**
e160 在 seed 74 的首 episode、非终止真实转移。后者由 V1.32
冻结评估采集，先筛去重复 episode 和 reset 污染。训练 30 epochs，
batch 2048、AdamW `3e-4`、seed 42、相对手腕 feature mode、
宽 128/3 residual blocks；随机源内 10% holdout 仅供选 checkpoint，
不充当独立泛化证据。使用一张空闲 GPU 5。

独立门禁是**同一 Cm-off e160 策略**在未见 seed 75 的首 episode
非终止转移；同时检查原 s1 DAgger seed 5909，避免完全遗忘。
先用旧 CmLite 在 seed 75 建基线，再冻结新 checkpoint 检查：

- s3 seed 75 运动 EPE 比零位移至少低 20%，且比旧模型低；
- 固定状态打乱动作后，运动 EPE 至少比真实动作高 10%；
- s3 接触 precision ≥0.5，s1 seed 5909 运动 EPE 低于零位移。

所有门槛通过才考虑新的在线 Cm-on/off 匹配训练；否则停止此
仅追加数据的 49D 配方，转向补充动作动力学输入或重设目标。
单步预测通过也不等于在线抓取增益，仍须严格成对验证。

输入源、SHA、代码提交、训练参数、资源与结果记录在各自 manifest
或 summary；输出预计小于 100 MB，加上评估转移仍远低于 300 GB。
如 seed75 收集失败、输入 SHA 漂移、GPU 被占或评估 schema
不符，停止而不静默改用其他输入。
