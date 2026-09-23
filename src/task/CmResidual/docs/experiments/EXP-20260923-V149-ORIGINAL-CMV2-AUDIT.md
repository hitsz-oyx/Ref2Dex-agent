# V1.49: 原版 Cmv2 在自训练 PPO 真实接触转移上的作用审计

- experiment_id: `EXP-20260923-V149-ORIGINAL-CMV2-AUDIT`
- branch: `agent/v149-cmv2-audit`
- run_status: `NOT_STARTED`
- conclusion: `INCONCLUSIVE`
- official actor checkpoint used: `no`

## 研究问题与冻结输入

用户强调论文必须证明 Cm 的作用，而不只是抓取。
V1.28 的 s1 稳定抓取含 CmLite 奖励专家但没有 matched
因果消融；V1.37/V1.42 只验证了低延迟 CmLite 的
一步预测；V1.46 的 CmLite 一步进展训练奖励失败。
在继续设计 PPO 接法之前，本实验直接检查原始
几何/接触 token Cmv2 在当前 s3 自训练策略状态上
是否提供动作条件的一步物体效应预测信号。

原版 Cmv2 V1.3 checkpoint 为外部只读
`/home2/wyy/oyx_ws/Ref2Dex/third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual/latest.pt`，
SHA256 `371fb3396d8fc4ecea61de25178e58954090e26b2f2856aad925f25cb3b01591`，
仅在本工作区建立软链接，不复制/修改外部项目。
轻量对照 V1.37 CmLite SHA256
`396f5e0c92ecf67b2a096568c1a538b7c86d40779a594729f786df6c64ed204a`。
真实轨迹为自训练 `back260` e260 策略的 s3
seed95/96 首个完整 episode 非终止转移，SHA256
分别 `ae4cd504912ba5abcf86d75519c2b8f372e84c0fa23a279827369f47165098bd`
与 `bc656eaed82b714cf1da1115e6d800261917e0f80a74dcc8823e9c1fa006e360`；
无官方 actor，也不重新运行物理仿真。

每 seed 用固定 RNG149 在已完成首 episode 的
转移中抽取 64 个**动作后真实手物接触**、64 个
无该接触的样本，总计256；保留原动作、前/后
物体状态和前手状态。随机打乱动作只在同 seed、
同接触分层内进行，是观察数据上的动作敏感性诊断，
不是同一物理状态的反事实结果。固定 DExplore
action→native target、Inspire URDF FK/surface geometry，
由前状态+动作构造 Cmv2 名义手点 swept flow；
不得读取 `next_q` 构造模型输入。GT 为前后物体
姿态的局部 SE(3) 差。Cmv2 与 CmLite 在完全同一
选中索引上计算局部平移 EPE；另报告零位移、
真实/打乱动作 EPE、预测动作变化、token 激活率、
旋转误差、模型+几何端到端延迟及参数量。

先8样本 smoke（每 seed 接触/无接触各2），验证
V1.3 strict 加载、坐标/shape/finite、名义 hand flow
和延迟可测；smoke 不作科学结论。正式审计全256
样本、Cmv2 microbatch ≤8、interaction object chunk32，
GPU5 空闲时单卡运行，最多10分钟；CmLite 同卡。
预注册“原版 Cmv2 在此分布有可用一步信号”的
证据门槛为**每个 seed 的接触子集**：
真实动作平移 EPE 比零位移低≥20%，且打乱动作
EPE 比真实动作高≥10%，并有至少10%样本激活
有效 contact token。全样本/无接触指标和实际
端到端延迟必须同时报告；即使通过也只证明
当前分布的离线预测信息，**不证明 PPO 抓取增益**。
若模型 strict 加载或输入合同失败，标记工程失败，
不把它解释为架构科研负结果。

新增产物<1GB，总产物<300GB；外部数据只读，
GPU5占用>1GiB时不启动，也不触碰他人进程。
