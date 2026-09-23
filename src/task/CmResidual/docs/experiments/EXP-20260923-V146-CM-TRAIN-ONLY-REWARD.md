# V1.46: 冻结 CmLite 只作 PPO 训练奖励

- experiment_id: `EXP-20260923-V146-CM-TRAIN-ONLY-REWARD`
- branch: `agent/v146-cm-train-reward`
- run_status: `PLANNED`
- conclusion: `INCONCLUSIVE`
- official actor checkpoint used: `no`

## 动机与冻结两臂

V1.45 的无 Cm 三专家路由相对单专家重复评测
有+10.6pp，但本身仅78.0%，距稳定抓取远。
V1.35 在较弱 e160 源上用旧 CmLite 系数1、
预测接触/几何可信门续训失败；V1.42 却显示
新版低延迟 CmLite 在当前 e240/e260 专家的
真实接触转移上比零位移低约35%–41%。因此
只检验一个明确改变的接法：Cm 预测进展奖励
系数 **0.1**、正进展截断、真实手物接触门，
并只在训练时使用；推理只保留同架构 PPO actor，
无 Cm 推理负担。

两臂均从自训练、Cm-off V1.39 `back260` e260
checkpoint SHA256
`16fd261b4b2de4cbdb257b09f1c7b363b384153103901ff831c825cf47d6a78f`
恢复，不使用官方权重。s3 corrected_r2 输入 manifest
SHA256 `2878bd20d1dd849f6844883c832a3777602d2d2bc73f7581b3bc23d30725f038`，
训练 seed70、64env、horizon32、minibatch256、
学习率1e-5、相同接触/抬升重置课程（接触前后
各3帧、0.5接触/0.25抬升、backtrack180–220）
和几何接近2/持物抬升10/抬升进展5奖励。
唯一区别：Cm-on 加冻结 V1.37 CmLite checkpoint
SHA256 `396f5e0c92ecf67b2a096568c1a538b7c86d40779a594729f786df6c64ed204a`
产生的0.1倍、真实接触门目标进展奖励；Cm-off
系数严格为0且拒绝 Cm 模型构造。两臂各自从
同源 e260 继续40 epoch到 e300，e280/e300保存；
**固定 e300**用于比较，不按训练/评测成绩挑 epoch。

工程门禁：先各跑 e262 两 epoch smoke，确认源 SHA、
恢复 epoch、学习率、Cm 模型 SHA、奖励日志有限、
checkpoint 正常；smoke 权重不用作正式训练。
任一工程门禁失败则先修接线，不启动正式实验。

正式 e300 后在全新 seeds104–108，各臂每 seed
两次64环境完整首 episode、关闭提前终止；两次
交换 GPU5/6，至多并发两卡。物理评测不逐 env_id
配对（V1.44 已发现重跑自身不重现）。Cm 有益
门槛与 V1.45 相同：总成功率相对匹配 Cm-off
高≥8pp，五 seed 各自两次均值均高于 off，
seed 聚类10000次 bootstrap 两侧95%区间下界>0。
稳定目标另要求 Cm-on 的十次运行**各**≥58/64。
不论中间分数如何，完成固定矩阵；若门槛失败，
不继续在这些 heldout seed 上调0.1系数或奖励门。

本试验比较“V1.37模型+0.1系数+真实接触门”
这个联合配方，不能单独归因模型结构或系数；
任何离线预测成绩都不能替代在线严格抓取。
预计两臂训练各数分钟、20次评测约十余分钟，
产物预算<20GB，总产物仍须<300GB。
