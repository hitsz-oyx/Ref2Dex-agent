---
schema: ref2dex.probe.v2
probe_id: P-20261016-independent-motion-ppo
experiment_id: P-20261016-independent-motion-ppo
date: 2026-10-16
task: wam-rl-policy
branch: main
git_commit: recorded-in-run-manifest
claim_id: C1
hypothesis_family: HF-wam-motion-execution
probe_index_in_family: 1
seed_pool: probe
seeds: [245, 246, 247]
decision_changed_if_positive: extend independently initialized PPO with motion references only if learning and native execution qualify
decision_changed_if_negative: fix action or observation wiring and exploration before interpreting WAM utility
status: UNCLEAR
run_id: independent-motion-{none,geometry,wam}-20261016-r1
---

# 无BC的PPO与因果运动参考接口资格

Class: Blocker，随后Decision。服务小机器人数据下WAM辅助PPO，不改变MISSION。
检验：去掉BC/官方actor/BC正则后，独立动作学习是否正常执行？WAM的条件运动预测
能否成为真正被策略使用的输入？短期工程资格与任务收益分开，不以本次短训否定主假设。
最低成本是复用当前原生环境，8TRAIN物体×8环境，同587D小型actor/critic，三臂共享
native reward、初始化和探索流；不采教师、不新增BC、不用未来轨迹作在线条件。

## Decision Note / 固定协议

先前终点回归未过门槛，因此不部署该模型、不修改其gate。新的控制问题不要求预测旧
策略实际未来：目标按当前到task goal方向截断至5cm、姿态保持当前，代表期望0.8s位移，
不是已验证可达目标。物理可达性与收益由真实执行检验。

- `--bc-kind none --action-kind independent-gaussian`：不传BC checkpoint，运行副本
  跳过官方actor加载；直接24D高斯均值，原生clip[-1,1]，BC coefficient=0且计算项也为0。
  观测保留24个全零兼容槽位，无示范动作。公共current300含机器人过去已执行动作。
- 三臂motion-none/geometry/wam：公共current300、task goal3、motion258+age1、
  remaining1、zero24，共587D。none置零运动误差；geometry用同短目标刚体平移；
  wam用冻结V3 step80000的hand/scene预测，取16/32native步。
- 参考每16native步更新，世界坐标目标在区间内固定；当前hand11/CAD32到两时刻参考
  的误差转native yaw frame、除.05m、clip±10，附参考年龄/48。history只含已观测
  [-6,-4,-2,0]，开头重复最早帧。没有未来标签、动作反事实评分或tracking reward。
- panel：`outputs/wam-rl-policy/independent-motion-panel-20261016-r1/manifest.json`，
  原train-00按object code排序前8，64env。旧探索面板，不称预训练未见/独立TEST。
- 每臂先2episode工程smoke，debug seeds45/46/47，原生warm同步、保留reset index和
  单次target setter。通过后fresh25episode，reset245/policy246/exploration247，
  非smoke续训。每臂318400交互步；相同种子不代表物理初态已匹配，必须核对实际首帧。
- 小Tanh 256×256，initial log std=-1.5，rollout16，Adam actor1e-4/critic3e-4，
  value_scale100、5epochs×4minibatches，joint KL .02。沿用已测试GAE与实际next-state
  bootstrap，不宣称是调优完成的独立PPO配方。
- 固定25终点，不按中间结果挑checkpoint；仅TRAIN学习资格。报告actor更新数、logprob
  replay、原生执行误差、BC项=0、critic EV、return、lift/reach/hold，不能以训练reward
  或少数成功替代独立评估。

## Gate / 资源

工程gate：全程有限；没有BC/官方actor加载；实际canonical到world、force/torque/DOF
误差<=1e-4；参考goal位移<=5cm；WAM冻结；三个实际初态一致；actor有更新、logprob
replay<=1e-4。任一实现项失败先修复，只重跑受影响部分。

Decision gate：最后5episode至少4个的13段critic EV中位数>0.05，且学习过程无全部
13段EV近零的连续8episode；满足才允许扩大独立控制预算。WAM是否提高真实成功仍UNCLEAR，
需要固定评估/更多训练。无收益也不自动关闭先验，只定位学习和接触探索瓶颈。

GPU0/1/2空闲时每臂独占1卡，保留本地至少2卡；smoke每臂<=600s，fresh每臂<=1200s，
总墙钟<=30min、产物总<=1GiB、磁盘保留20GiB。非有限/输入泄漏/资源冲突即停，连续
critic崩溃保留checkpoint后停。完成后释放卡，不自动加大预算。

## Limitations / future evidence

训练仅8物体与短预算；重置含官方静态抓姿初始化先验、原生奖励和几何，不等于无机器人
先验。未使用官方actor或示范动作。未来需相同交互预算、独立对象/任务评估与多seed验证。
相同目标的WAM/geometry语义由重放审查，目标仍可能不可达，输入clip率必须报告。
