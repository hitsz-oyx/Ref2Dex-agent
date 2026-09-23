# V1.45: 非确定性评测下的固定路由重复检验

- experiment_id: `EXP-20260923-V145-ROUTE-REPLICATION`
- branch: `agent/v145-route-replication`
- run_status: `PLANNED`
- conclusion: `INCONCLUSIVE`
- official actor checkpoint used: `no`

## 问题、冻结输入与预注册门槛

V1.40 路由在新 seeds85–89 以254/320 对单
专家208/320，但 V1.44 同 seed/GPU/策略重跑
自身可差7/64成功，逐环境状态不重现。本实验
不再解释同 env_id 的修复/破坏，只在**新的**
seeds99–103、每策略每 seed 两次完整首 episode
重复中检验分布级增益。

保持 DExplore s3 轨迹、64环境、提前终止关闭、
自训练 PPO checkpoint 不变；两臂都用同一
`eval_cmlite_bc_policy.py --mode router` 评测器、
同一自身训练 e180 checkpoint 构造播放器，
运行时均不使用 Cm。实验臂为 V1.40 冻结
开始帧四段路由，map SHA256
`7db4686f8eb273e0ac94bfbbd3625ddd35209a1f4922e8451bcbf519f6395a79`；
对照臂为仅有帧0→`back260` 的固定 map，
由评测器最近帧规则映射所有开始帧，SHA256
`1c396812c46d2e9d433e168925e73c2d3c55a5ac9bce6ff932a7c1845716e0bd`。
route 使用 `source/back240/back260` 自训练专家，
baseline 只使用自训练 `back260`；这些 SHA
在 V1.40 卡片及各 run manifest 中锁定。

每 seed 两次重复编号 r0/r1；r0 route GPU5、
baseline GPU6，r1 对换 GPU6/GPU5，均最多
并发两卡。各次运行是不同物理样本，不能
做 env_id 级配对因果推断。即使中途结果不佳，
仍完成固定5×2×2矩阵（技术失败则记录失败
后以新 ID 原参数重启）。

强稳定目标：route 十次运行**各**≥58/64。
路由优于单专家的较保守证据门槛：十次合计
成功率至少高8个百分点、五个 seed 各自两次
平均成功数均高于 baseline，且按 seed 为聚类
单位、固定 RNG145 做10,000次有放回 bootstrap
的两侧95%区间下界>0。任何一项失败则不
宣称重复验证通过；仍报告原始结果及区间。
记录整段运行时长供粗略延迟比较，但它含
环境初始化，不代表纯策略推理延迟。

不训练 actor、不引用官方 actor，输出预计<1GB，
总产物限额300GB内；外部占卡则等待，不打扰
他人的进程。
