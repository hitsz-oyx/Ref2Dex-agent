# V1.48: 多指几何包围奖励能否稳定 s3 持物

- experiment_id: `EXP-20260923-V148-GRASP-LINKS`
- branch: `agent/v148-grasp-links`
- run_status: `NOT_STARTED`
- conclusion: `INCONCLUSIVE`
- official actor checkpoint used: `no`

## 假设与预注册对照

V1.45 的无 Cm 整段路由在 s3 重复评测为78.0%，
但十次没有一次达58/64。V1.47 从 e260 续至 e320
的两臂都在新 seed 上跌至约31%–35%；不同 seed
不能证明续训导致坍塌，因此本实验选择先前已有的
**相同 e300 续训对照**，只检验一个明确抓握目标。
V1.46 Cm-on 的单步预测进展奖励有害，故此实验
不使用 Cm，目标是改进待比较的稳定抓取基线。

固定 Cm-off 对照是 V1.46 `agent_v146_cmoff_s70_e300`
e300 SHA256
`36ff2ac7ffd4433b5f60b32dce7603cff5db24a0bab9866b9ab469d1ce645929`。
新多指臂和对照都从自训练 s3 back260 e260
SHA256 `16fd261b4b2de4cbdb257b09f1c7b363b384153103901ff831c825cf47d6a78f`
恢复，s3 corrected_r2 manifest SHA256
`2878bd20d1dd849f6844883c832a3777602d2d2bc73f7581b3bc23d30725f038`，
训练 seed70、64环境、horizon32、minibatch256、
学习率1e-5，接触前后3帧/0.5接触+0.25抬升/
backtrack180–220、几何接近2/接触持物抬升10/
接触抬升进度5，固定继续至 e300。

新臂**同时**增加两项预定的抓握配方：
`grasp_link_reward_coef=2`，奖励五个指端几何接近
物体表面的平均比例；`min_grasp_links=2`，让原有
持物抬升与进度奖励仅在真实手物接触**且**至少
两个指端几何接近时生效。对照两项都是0。
因此若有差异只能归因于这个联合配方，不能独立
归因奖励或门。几何指端是当前状态计算，不用
官方 ckpt，也不是 Cm 预测。

先从同源 e260 到 e262 做2 epoch 工程 smoke，
要求来源与超参一致、日志 finite、checkpoint 存在，
且 `qualified_grasp_contact_fraction` 在至少一个
horizon >0.05；若几何门完全失活，停机修接线，
不启动正式训练。smoke 权重不用于正式训练。
正式训练固定 e300，不按训练 reward 或中间表现选模。

在全新 s3 seeds114–118，每臂每 seed 重复两次64环境
首个完整 episode、关闭提前终止；r0/r1 交换 GPU5/6，
不作 env_id 级因果解释。新臂有益门槛：合计成功率
高对照≥8pp，五 seed 的两次均值均为正，seed聚类
10,000次 bootstrap 双侧95%区间下界>0。
稳定门槛另要求十次新臂各≥58/64。
若失败，不在这批 heldout seed 上调奖励系数或门槛。

最多同时两张 GPU5/6，使用前和评估每轮检查外部占用；
不触碰他人进程或外部数据。新增产物预算<10GB，
总产物<300GB，预计训练与20次评估<30分钟。
输入 SHA 漂移、非有限值、资源冲突或工程 smoke
不通过则停机。此实验不能证明 Cm 有用。
