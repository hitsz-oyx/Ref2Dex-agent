# V1.33: 用高抬升策略真实转移适配低延迟 CmLite

- experiment_id: `EXP-20260923-V133-CM-ADAPT-HIGHLIFT`
- branch: `agent/multitrajectory-v129`
- run_status: `COMPLETED`
- conclusion: `REFUTED`（四项联合上线门槛：s1 留出集未过）
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

## 结果与判定

seed74 Cm-off e160 首 episode 筛出 33,189 条非终止转移并完成
30 epoch 训练；checkpoint 为 `outputs/CmLite/V1.33/s1_s3_highlift_seed74_train/best.pt`，
SHA256 `2640f2e0b932be375f603bfecca4b3cb61310878b9a580e5512cecefa00272ee`，
参数 107,012，64 样本 GPU 前向 0.555 ms。训练 `summary.json`
为 `COMPLETED`。独立 s3 seed75 的 Cm-off e160 策略抓取 27/64，
其首 episode 33,094 条非终止转移中运动样本 10,627 条：

| CmLite | 运动 EPE | 零位移 EPE | 打乱动作 EPE | 接触 precision / recall |
| --- | ---: | ---: | ---: | ---: |
| 旧 V1.29 | 9.73 mm | 9.29 mm | 9.93 mm | 0.811 / 0.538 |
| 新 V1.33 | 6.82 mm | 9.29 mm | 8.71 mm | 0.913 / 0.955 |

新模型在 s3 seed75 运动 EPE 相对零位移降低 26.6%、相对旧模型
降低 30.0%；打乱动作使 EPE 上升 27.8%，通过前三项 s3 门槛。
但 s1 DAgger 独立 seed5909 的运动 EPE 从旧模型的 2.13 mm
恶化为 **3.43 mm**，高于零位移 3.31 mm，未通过防遗忘门槛。
新加入的 s3 高抬升样本约 33k，原 s1 两个源总共仅 862 条，
数据比例失衡是合理但尚未实证的解释。根据预注册联合门槛，
**不以 V1.33 checkpoint 启动在线奖励训练**；下一试验需显式
平衡来源并使用新的独立 seed 复核。离线预测增益仍非抓取增益。

评估与转移 SHA 在 Cm-off e160 run 的 `eval_s75_e160_full/run_manifest.json`；
旧/新模型完整审计分别为同目录的 `cmlite_audit_old.json` 和
`cmlite_audit_v133.json`，s1 指标见模型 `summary.json`。
