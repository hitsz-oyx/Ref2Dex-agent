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
status: PROMISING
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

## 运行进展

工程smoke（代码d4eb3fed）三臂均完成2episode、64env；实际9类首帧逐值一致，
无BC/官方actor加载，动作链路最大误差1.19e-7，行为logprob复算、BC项为0和WAM冻结
通过。分别有292/323/339次actor更新。首轮刷新参考未触发clip。
三臂总墙钟约30/30/50s，峰值采样显存约2.7/2.7/5.2GiB。
资格仅为工程PASS，没有抓取提升结论。按预先协议进入fresh25轮；补充记录区间内
参考保持与全episode输入clip率，不改变动作、奖励或参考计算。

## 固定终点结果（25episode）

运行代码 `01f9c80d`，三臂fresh各318400步、总955200步，全部完成；
输出为 `outputs/wam-rl-policy/independent-motion-{none,geometry,wam}-20261016-r1/`。
汇总及实际初态审查：`outputs/wam-rl-policy/independent-motion-panel-20261016-r1/training-audit.json`。

| 条件 | 前5轮平均native return | 后5轮平均native return | 后5轮chunk-EV中位数范围 | actor更新次数 | 25轮到达/保持 |
|---|---:|---:|---:|---:|---:|
| 无运动参考 | -91.42 | -4.65 | .814–.916 | 1860 | 0/0 |
| 几何运动参考 | -125.62 | -14.63 | .981–.991 | 2511 | 0/0 |
| WAM运动参考 | -94.72 | 8.79 | .890–.952 | 2131 | 0/0 |

三臂通过预先定义的**独立动作学习资格**，此范围PROMISING；WAM带来真实任务收益
仍UNCLEAR，不能把训练回报差当作成功率增益或预训练因果验证。EV在各训练chunk更新后
计算，只证明回归拟合没有原先的退化，不是留出价值泛化。最终学习率仍1e-4，非LR归零。

与随后严格复核的官方 initialized UniDex PPO 不同，本卡使用 8 物体、587D 小型网络和
独立运动参考目标；官方底座结果不能推翻这里的“独立动作学习资格”，但也不能把本卡的
PROMISING 外推成官方任务收益或 WAM 接入成功。

最后一轮平均指尖到物体中心距离从共同起点204.24mm降到113.55/126.41/94.58mm，
显示当前主要学到靠近；这是几何代理，不是实测接触。最终三臂最大抬升都为0。
WAM第20轮env58（CellPhone）曾高出起点0.2719m，但结束时距目标2.18m、无到达/保持，
不能称有效抓取；该轮没有完整轨迹，不能确认抛飞的接触因果机制。其余两臂全程无>10cm抬升。

### 关键链路复核

- 三臂九类实际首帧逐值一致；无BC/官方actor加载，原生动作=clip(latent)，base24全零，
  BC regularization全零。全程执行误差最大1.19e-7；logprob复算通过，WAM参数/梯度冻结通过。
- 同一世界参考在tick1/2/16逐值不变，年龄与16步刷新节奏一致，短goal位移<=5cm。
- 最终模型在第一轮保存状态上的参考置零干预：几何/WAM动作RMS变化.0741/.0618；
  WAM手参考单独置零.0221、物体参考.0570。这证明参考接入动作，但不能证明其有益。
- WAM初态参考指尖到预测物体表面的均距16/32步为153.52/140.47mm，几何为159.98/159.98mm。
  这是预测参考内部的协调差异，不是机器人已达到的接触或可实现性认证。
- 无参考/几何全程没有输入clip；WAM仅第20轮出现，最严重单拍0.563%标量被裁剪，
  最大参考误差.775m。与异常抬升同时出现，但不足以判定哪项导致哪项。
- 24项相关单测和owned文件verify通过。训练墙钟211.5/211.5/342.5s，显存最高采样
  2713/2713/5129MiB；六个smoke/训练run约291MiB，均已完成、GPU释放。

## 结果后的 Decision Note

- 已解决的blocker：独立于BC的策略、原生动作、因果固定参考和PPO学习可以共同运行。
  现在没有理由再把完善BC设为前置关卡，也不能把后续抓取失败归到已解决的critic崩溃。
- 保留WAM运动先验方向。单靠增加观测，本轮仍主要学会靠近，未形成受控抬升；当前证据
  不能区分更长探索、参考使用方式与接触控制难度，不做“数据不够”或“WAM无效”归因。
- 下一项方法工作：把WAM的手物**相对协调目标**接到有界辅助学习信号，检验是否更容易
  形成抓取接触；对象到task goal的推进仍按真实native任务计分。采用同目标几何对照，
  防止把任何额外shaping都误归为WAM。新方案先检查参考坐标、区间内进展定义和异常值，
  再单立短Probe；不直接奖励WAM预测的物体升高，不依赖虚构的动作反事实。
- 本轮不自动扩到200轮或更大面板，不以最终回报排名替代行为判断。下一轮只有在真实
  接触/受控抬升出现正向信号后才扩大。无新的外部授权需求，MISSION/最终claim不变。
