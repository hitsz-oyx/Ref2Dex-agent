---
schema: ref2dex.probe.v2
probe_id: P-20261016-wam-coordination-bonus
experiment_id: P-20261016-wam-coordination-bonus
date: 2026-10-16
task: wam-rl-policy
branch: main
git_commit: recorded-in-run-manifest
claim_id: C1
hypothesis_family: HF-wam-motion-execution
probe_index_in_family: 2
seed_pool: probe
seeds: [245, 246, 247, 249]
decision_changed_if_positive: validate WAM relative coordination for stable holding before scaling generalist training
decision_changed_if_negative: audit reference and auxiliary incentive before adding compute or robot data
status: UNCLEAR
run_id: coordination-{none,geometry,wam}-20261016-r1
---

# 保留抓取起点，检验WAM相对协调奖励

Class: Decision。服务WAM帮助真实机器人RL抓取的目标。前一轮冷启动学会靠近但未形成保持，
不证明冷启动无效；当前更有信息价值的问题是：在已有抓取能力上，WAM运动先验能否减少
抬起/到达后的丢失？不改变MISSION，不把完善BC设为前置关卡。

## Decision Note / 设计

固定本项目自训`shadow-current-bc-seed241-20261016-r1/step50000.pt`为三臂共同起点，
不再训练BC，不加载官方actor。24D reference-gaussian均值=BC+新网络，无残差幅度限制；
BC regularizer=0，仍保留BC这一初始化/参数化，不能称无BC或零机器人示范。

- 辅助目标每16 native步刷新并固定，短goal为朝真实task goal最多5cm/0.8s的平移。
  geometry取当前手相对物体构型；WAM取16native步手预测，并对32个预测物体点做
  proper rigid fit，把预测手转入该预测物体坐标系。实际奖励用真实next-state物体姿态
  将真实手点转入物体坐标系，再与这个冻结的相对目标比较。
- Eligibility在刷新时用真实状态判定：物体高于本episode起点>=3cm，5指尖到中心均距
  <=15cm，palm到中心<=15cm。目标和admission区间内保持，但score每步重新检查实际lift/proximity，跌落后立即停bonus。
  这是抓持几何代理，不是实测接触标签。
- Admission：预测物体非刚体残差<=1cm；相对手目标每点改变<=8cm；非有限立即停。
  未准入参考无bonus，并以当前相对手构型提供安全观测；记录准入率与启用率。
- bonus=0.2*eligible*exp(-mean(hand-point squared distance)/(2*.03²))，范围[0,.2]。
  只奖励实际相对构型，不奖励WAM预测物体运动。它改变训练目标，不宣称potential-based
  policy invariance。真实任务reward/holding是最终判据，必须检查停滞与退步。
- 三臂none/geometry/wam：none也提供几何相对参考以匹配观测，但bonus=0；geometry/WAM
  用相同函数、权重和gate。WAM与geometry差别包括目标及对应观测，不能分离归因纯reward。
  公共native300+goal3+relative_error33/age1/eligible1+remaining1+BC24=363D。

## 最小协议与Gate

1. 当前基线：TRAIN全train-00的32物体×4env=128；DEV固定core-dev-00前16物体×8env=128，
   分别reset245/249，BC确定性执行，WAM只做参考/bonus旁路诊断，不改变动作。
   至少有8条TRAIN和8条DEV保持轨迹；WAM参考在真实抬升阶段有非零启用且准入率>=50%，
   才启动训练，否则先诊断，不换面板追求好结果。
2. 三臂2轮工程smoke，debug seeds45/46/47；原生动作、bonus分解、冻结与实际初态匹配
   通过后fresh25轮，policy246/exploration247/reset245。各636800实际交互步，不按中期挑模型。
3. 小Tanh256×256、value_scale100、initial log_std=-4.3971199849、rollout16、5epoch×4minibatch，
   actor1e-4/critic3e-4、joint KL .02。使用同步warm/reset index保持/单次target setter。
4. 固定最终checkpoint，DEV reset249确定性完整199步。共享BC基线、九类真实首帧须一致；
   报告held60+lift10cm、reach、lift，以及BC-only / new-only、reach-without-hold。
   WAM比none与geometry的保持数均至少多5/128，且相对BC没有净退步，才PROMISING。
   否则按实现/信号覆盖和行为审查判UNCLEAR或当前接法UNPROMISING，不关闭WAM主假设。

## 资源与停止

本地GPU0/1/2各一臂，baseline/评估使用空闲卡，始终保留至少2卡。
每个baseline/smoke/eval<=600s，每臂fresh<=1200s，总墙钟<=35min，新增产物<=1GiB、
保留20GiB空盘；无checkpoint覆盖。非有限、参考/奖励错接、资源冲突立即停；连续8轮
13段critic EV近零保存后停止。模型计算GPU，极小数学单测/纯统计CPU。

## Limitations / future evidence

单seed、小DEV、既有探索面板；并非预训练未见/独立TEST。BC预训练使用的机器人数据必须
计入最终小数据claim；本轮只验证辅助方法，不证明小数据通用策略。正向结果后才多seed、
更广对象与任务验证。尤其检查保持奖励是否鼓励原地停留、忽视目标搬运，及参考拒绝是否
造成实际干预缺失。不为本轮探究新增BC数据或扩大BC训练。

## 启动前审查

独立只读审查核实Kabsch方向、真实next-state评分和初始actor均值=BC。
据审查把phase gate改成每步实际检查，避免跌落后继续bonus；eval必须保留初始化时的
skip-official-actor参数，防止加载记录与实际不符。none的几何观测差异已明确记录，
WAM/geometry比较不声称隔离了纯奖励贡献。四项协调数学测试覆盖刚体不变性、滑落、
proper rotation与跌落禁奖；小数学测试CPU，模型/仿真使用GPU。

## 基线与工程资格（代码360054d7）

BC固定动作下TRAIN保持45/128、到达68；DEV保持28/128、到达42，满足基线资格。
WAM参考在启用phase内100%准入，TRAIN/DEV启用比例25.1%/15.6%，相对手目标平均
改变4.25/4.27mm；说明这次主要检验毫米级稳定构型修正，不能宣传大幅技能生成。

三臂2episode smoke全部完成，真实九类首帧逐值一致；奖励按真实next-state独立重算
最大误差2.98e-8，none bonus严格为0，BC/WAM冻结和执行链路通过。28项相关单测通过。
`coordination-panel-20261016-r1/{baseline-audit,smoke-audit}.json`保存证据；smoke审查
中的历史字段bc_only实际以第一条none-smoke为anchor，仅检查配对，不当作BC退步指标。
后续审查字段改为reference_only，正式DEV以实际BC baseline为anchor。

按既定协议进入fresh25轮及固定DEV读出，不按smoke行为挑选参数。
