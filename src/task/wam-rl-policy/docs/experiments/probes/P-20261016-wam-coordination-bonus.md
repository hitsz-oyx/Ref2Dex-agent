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
status: UNPROMISING
run_id: coordination-{none,geometry,wam}-20261016-r1
---

## 2026-10-22 有效性边界（本注记优先于下方历史外推）

- **结论处置：加注。** 加注保留当前接法不追加预算的局部决策；不把该 gate 未过用于关闭 WAM、τ 或 v6 路线。
- **能支持：** 同初态 DEV 的 BC/none/geometry/WAM 保持代理 28/26/23/25，当前 bonus 配方未过 +5 且不退 BC 的 gate；弱 WAM 特异 bonus 诊断仍有效。
- **不能支持：** 不能支持 v6/τ 或 WAM 一般无效、BC-free 成功或纯 reward 因果结论。实际为 Shadow 旧 v3-step80000 WAM 副本加冻结 BC；保持是 goal-position/lift 代理，非实测接触/force 资格。单 seed，参考/观测/bonus 同变。

本卡不得再被引用为“v6/τ 无效”的证据。历史数值、状态、门槛和运行产物原样保留；
此处处置限定其当前可引用范围，不重写旧实验。501 轨迹评分、UNKNOWN 准入、voxel 与日志
修复的影响按实际执行链归属，不能无证据套用到本卡所有旧运行，也不能倒灌成修后结果。
共同事实、逐卡处置和复现入口见[历史卡片卫生报告](../../../../../../docs/audit/historical-card-hygiene-20261022.md)。


# 保留抓取起点，检验WAM相对协调奖励

Result: 固定DEV同初态保持BC/none/geometry/WAM=28/26/23/25，未过门槛；当前抬起后协调奖励接法UNPROMISING，不否定WAM主假设。
Decision: 停止增加该奖励的权重或训练预算；先在接触转折片段区分短目标条件错误与WAM先验不足，不扩大BC。

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

## 固定终点结果与执行审查

fresh三臂和最终DEV运行代码`4b82bd84`，基线/工程smoke为`360054d7`；两者模型与物理
执行代码相同，后一个提交只增加审查工具和基线记录。各25轮、636800实际交互步，
没有挑选中间checkpoint。DEV九类实际首帧逐值一致，配对审查PASS。

| 策略 | 抬升>=10cm | 到达 | 保持>=60帧且抬升>=10cm | 到达但未保持 | BC-only保持 | 新增保持 |
| --- | --- | --- | --- | --- | --- | --- |
| 共同BC | 52 | 42 | 28 | 14 | 0 | 0 |
| PPO none | 58 | 51 | 26 | 25 | 18 | 16 |
| PPO geometry | 48 | 46 | 23 | 23 | 19 | 14 |
| PPO WAM | 57 | 51 | 25 | 26 | 18 | 15 |

每行128条。WAM相对none为control-only17/WAM-only16，相对geometry为13/15。
既没有比两对照均多5条，也没有达到BC的28条，按预设门槛不进入Validation。
当前接法标记UNPROMISING只表示不值得继续加码本配方，不是统计上证明WAM无效。

三臂actor分别更新6128/6142/6182次，最后episode的chunk critic EV中位数
.912/.925/.920；未触发死critic停止。实际动作、behavior logprob、BC/WAM冻结与
奖励分解审查通过，真实next-state重算奖励最大误差2.98e-8。参考在第一episode真实phase
内均100%准入；没有发现导致本次负结果的明确接线错误。TRAIN最终保持36/39/42只作
运行诊断，不能替代DEV或与BC另一次reset直接作行为比较。

## 事后语义定位：模型信息与奖励信号的区别

对既有BC轨迹做CPU统计，不新增训练、模型推理或仿真，也不调整预设门槛。
固定16步WAM参考与“保持当前相对构型”比较，未来真实状态只用于诊断标签。
同一轨迹窗口相关，以下样本数不是独立实验数；lifted-nearby仍是几何代理而非接触标签。

- 随后仍满足抬起/靠近条件：TRAIN364窗，WAM/geometry误差4.02/2.47mm；
  DEV223窗为4.06/2.66mm。稳定阶段几何恒定已很强，WAM的毫米级改动不是可靠改进。
- 在完全相同的实际BC next-state和gate下，仅替换参考目标，DEV启用的3986个转移中
  WAM/geometry score均值.98285/.98855，实际bonus绝对差均值.001295、p95=.003834，
  对照bonus上限.2。当前标量奖励主要提供phase存活信号，WAM特异信息弱。
  观测中也有参考差异，因此不能把整个PPO结果归因到这个奖励差值。
- 扩展到抬起前且参考准入的窗口，DEV1284窗WAM/geometry为25.61/28.07mm；
  但其中16步后进入lifted-nearby的58窗为7.98/5.33mm。TRAIN对应1141窗29.18/31.87mm、
  79个转折窗8.88/6.37mm。一般预测优势没有自然集中在建立抓取的关键转折。
  这不是证明WAM目标有害或真实BC未来最优，也不支持简单把同一bonus前移即可成功。

工具：`tools/audit/audit_coordination.py`与`tools/audit/audit_coordination_targets.py`
（均相对本Task）。产物根`outputs/wam-rl-policy/coordination-panel-20261016-r1/`：
`train-audit.json`、`final-dev-audit.json`、`target-semantics-v3.json`、`decision-summary.json`。
target-semantics/v2为保留的早期审查，v3补充阶段划分；无原始运行覆盖。
train-audit中的配对字段仅相对第一条none训练记录，不能解释为BC保持退步。

## Decision Note / 收束与下一步

问题：是否扩大抬起后相对协调奖励训练？关键证据是同初态DEV未改善、原有成功病例大量
替换，以及WAM特异bonus很弱，稳定/抬升转折预测没有超越几何参考。
root选择停止本接法的权重和时长搜索，保留能抓取的共同起点，不重训BC、不关闭WAM。

下一项最小Decision应使用这些已有关键转折状态，固定状态与预测跨度，比较可部署短goal
和真实短期终点oracle下的WAM/geometry相对协调预测。oracle仅诊断：若关键阶段只在oracle
下改善，优先修复目标条件接口；若oracle仍无优势，优先审查该阶段预训练数据/目标覆盖，
暂停更多在线reward改法。不能再用总体预测均值改善跳过关键阶段资格。此项尚未启动，
需单立卡固定面板、预算和门槛后才执行，不把事后误差直接升级为任务收益。

本轮训练监督器墙钟none/geometry/WAM为271.9/271.9/523.8s，显存约2.8/2.8/6.8GiB，
WAM推理期间利用率可达100%。基线+smoke+训练+DEV总产物892.6MiB，低于1GiB。
所有进程正常结束，GPU已释放，无额外外部授权边界。

## 2026-10-22 负结果 CPU 审计：结论范围修订

28/26/23/25对应BC/none/geometry/WAM的goal-position保持代理：连续60帧位于目标5cm内且lift10cm，不是接触/force持握认证。正式4b82bd84是shadow旧v3-step80000 WAM副本+冻结BC，不是v6或BC-free。保留同初态DEV下本bonus接法未达+5且不退BC的gate、弱WAM特异信号与部分行为退步；单seed且参考/观测/bonus一起变化，不能证明WAM一般无效或隔离纯reward因果。

本追加不改变原始数字、gate或运行产物；完整身份、版本边界与排除项见[逐卡证据审计](../../../../../../src/task/ptv3-shared-wam-v6/docs/audit/negative-result-card-audit-20261022.md)。
