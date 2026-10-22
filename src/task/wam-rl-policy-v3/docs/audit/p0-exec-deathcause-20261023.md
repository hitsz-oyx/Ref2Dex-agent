# 结果解读表

| 实测结果 | 含义 | 决定 |
|---|---|---|
| 32回合：21关节中止、2物体/桌面几何门中止、8时域截断、1旧成功早停；关节占物理中止21/23=91.30% | 同一失败门很常见，但不等于同一可修代码原因 | 安全门和实发PD按原始7400状态重算；不调阈值 |
| 15/23物理中止=65.22%有“腕角在±2π附近跳变”签名（关节中止内15/21=71.43%，全部旧失败内15/31=48.39%）；另22次同类跳变未当步中止 | 这是跨臂共同候选；本机驱动跨支路误差与碰撞/求解器冲量的先后尚未取证 | **走C：缺少因果判定字段；B的单一可修主因门未满足，本轮不修代码、不启动n=5** |
| 2/2桌面门都是已在桌面下、中心从桌外重入footprint；6个contact未认证签名实际等到300/600时域末才结束 | 不能统称“手把物体压穿桌面”或“contact评分直接终止” | 保留原标签并细分实际触门；不删除/放松gate |
| 已有PPO随机采样同初态成功2例，9初态数组逐位差0；几何τ/GTτ仍经神经R | 至少这2初态不是严格物理不可达；位置式参考不是直接位置PD控制器 | 不把整批归为几何不可达，也不把未定位原因归为τ质量 |

# brief40 执行器死因归因（2026-10-23）

**范围与裁决：A完成；C完成；B未启动。没有得到足以确认“多数中止由同一条可修代码/参数导致”的因果证据，因此没有改默认实现/配方、没有换模型、没有新仿真/训练。** 本轮新增物理controls=0、模型推理=0、训练updates=0。只写本报告及`tmp/s40-deathcause/`；未改STATE/INDEX/v6/t38/s39，未删除outputs。

## 1. 原始分母、身份与协议边界

- S2e：`tmp/s2e-geometry-20261022/outputs/wam-rl-policy-v3/s2e-geometry-20261022-r1/`，WAM/geometry各5，共1941控制。
- S2d：`tmp/s2d-controls-20261022/outputs/wam-rl-policy-v3/s2d-controls-20261022-r4/`，a/c各5，共1592控制。
- BC/PPO：`outputs/wam-rl-policy-v3/bc-ppo-usefulness-20261023-r1/assessment-{bc,ppo}/`，各6，共2543+1292=3835控制。只取已指明的最终配对评估，不混训练采样分母。
- 合计32回合/7368控制/7400保存状态。原协议失败31、旧witness成功早停1；物理中止23与时域截断8不能混为同一中止。安全/PD逐tick原始重算32/32一致；阶段标签只沿用冻结评分输出，不把它当成成对接触真值。
- S2e/S2d为同一`s1_cylindersmall_pass_1`可见初态、native/model seed195/196的5次重复，**不是5个独立物体或5个独立seed**。BC/PPO为6序列，native294–299/model293。
- S2e几何τ也经过R：冻结`S2e/runner.py:135–139`依次actor→clip→grasp_intent→MeasuredGraspConsumer；S2d冻结`runner.py:175–176`是R直接动作。BC冻结`assessment-bc/runner.py:232,252,256`是R+残差+GeometryGraspConsumer。因此“位置式/网络”不是互斥分类，本表分别记录τ源与实际命令链。
- **历史终止合同不相同**：S2e/S2d冻结runner分别`:102,134`与`:148,169`是最多300控制、witness成功即停；BC/PPO冻结runner`:231,266`才是600控制末态+尺寸目标+物理安全。旧WAM成功1/5在tick103停止、目标误差100.793mm，不升级为600末态成功。若以后进入B，只能对共同前300控制做旧/新安全比较，另报新600末态；旧600结果“未测”，不能补造。
- 背景更正：S2d WAM block0在tick68达到5cm、held72–82、tick83首次高度回落（见原始报告/JSON），所以“两个臂都没达到5cm”不成立；GT各例未达到5cm保持原结论。

## 2. 逐回合表（action[t−1] → state[t]）

J=原URDF受限关节±0.10rad门；T=原物体CAD样本低于桌面20mm且中心在桌面footprint；H=预注册时域截断（不是contact评分即时abort）；S=旧witness成功早停。表中tick是state[t]，命令为此前实发action[t−1]；任务误差在state[t]测量。位置量均mm，腕量为三关节增量范数rad。B=本文定义的腕角分支跳变签名，不是已证实因果。clip列为该回合任一轴实质裁剪的控制次数，R内部与最终输出分层；S2d为同层直出不重复计数。

| 批次/臂/重复 | 初态 | 网络/参考链 | 终止tick/门 | 阶段 | 目标误差mm | 实发PD增量：位置mm/腕rad | raw动作∞范数 | clip：最终/R内部 | 触门DOF；B |
|---|---|---|---|---|---:|---|---:|---|---|
|S2e/block0-geometry|s1_cylindersmall_pass_1|几何τ→R/v7|106/J|approach|81.987|150.428 / 0.6796|0.4347|0/0|8,10,12,17；B|
|S2e/block0-wam|s1_cylindersmall_pass_1|WAM→R/v7|103/S|lift|100.793|5.039 / 0.0094|0.8783|0/0|—；—|
|S2e/block1-geometry|s1_cylindersmall_pass_1|几何τ→R/v7|156/J|approach|81.987|223.068 / 0.1939|0.4297|0/0|17；—|
|S2e/block1-wam|s1_cylindersmall_pass_1|WAM→R/v7|232/J|approach|81.987|143.520 / 0.8338|0.2797|0/0|15,16；B|
|S2e/block2-geometry|s1_cylindersmall_pass_1|几何τ→R/v7|300/H|approach|81.987|131.281 / 0.5940|0.8597|0/5|—；—|
|S2e/block2-wam|s1_cylindersmall_pass_1|WAM→R/v7|179/J|approach|81.987|117.727 / 1.7575|0.4746|0/0|10,12；B|
|S2e/block3-geometry|s1_cylindersmall_pass_1|几何τ→R/v7|141/J|approach|81.987|222.293 / 0.0341|0.2857|0/0|10,12；B|
|S2e/block3-wam|s1_cylindersmall_pass_1|WAM→R/v7|300/H|approach|81.987|75.327 / 0.0647|0.8762|0/0|—；—|
|S2e/block4-geometry|s1_cylindersmall_pass_1|几何τ→R/v7|124/J|approach|81.987|252.395 / 0.6842|0.3170|0/0|16；B|
|S2e/block4-wam|s1_cylindersmall_pass_1|WAM→R/v7|300/H|close|873.626|145.378 / 1.3282|0.3518|0/0|—；—|
|S2d/block0-a-original-wam|s1_cylindersmall_pass_1|WAM→R直出|93/J|直接动作/无v7相位|137.436|159.918 / 0.2558|0.7021|0/未单列/直出同层|10；—|
|S2d/block0-c-live-gt|s1_cylindersmall_pass_1|GTτ→R直出|112/J|直接动作/无v7相位|81.987|177.529 / 0.1904|0.3027|0/未单列/直出同层|6；B|
|S2d/block1-a-original-wam|s1_cylindersmall_pass_1|WAM→R直出|300/H|直接动作/无v7相位|81.987|84.046 / 0.0681|0.8449|0/未单列/直出同层|—；—|
|S2d/block1-c-live-gt|s1_cylindersmall_pass_1|GTτ→R直出|119/J|直接动作/无v7相位|81.987|121.708 / 0.6764|0.4151|0/未单列/直出同层|8；B|
|S2d/block2-a-original-wam|s1_cylindersmall_pass_1|WAM→R直出|91/J|直接动作/无v7相位|81.987|363.096 / 0.6810|0.5721|1/未单列/直出同层|16；—|
|S2d/block2-c-live-gt|s1_cylindersmall_pass_1|GTτ→R直出|144/J|直接动作/无v7相位|81.987|242.771 / 0.0637|0.3245|0/未单列/直出同层|10,16；B|
|S2d/block3-a-original-wam|s1_cylindersmall_pass_1|WAM→R直出|208/J|直接动作/无v7相位|81.987|70.985 / 1.1632|0.4832|0/未单列/直出同层|15,16；B|
|S2d/block3-c-live-gt|s1_cylindersmall_pass_1|GTτ→R直出|113/J|直接动作/无v7相位|81.987|275.594 / 0.7226|0.3190|0/未单列/直出同层|16；B|
|S2d/block4-a-original-wam|s1_cylindersmall_pass_1|WAM→R直出|300/H|直接动作/无v7相位|82.148|40.490 / 0.1204|0.6734|0/未单列/直出同层|—；—|
|S2d/block4-c-live-gt|s1_cylindersmall_pass_1|GTτ→R直出|112/J|直接动作/无v7相位|81.987|177.529 / 0.1904|0.3027|0/未单列/直出同层|6；B|
|BC/p0000-A1|s1_cylindersmall_pass_1|WAM→R+残差BC|600/H|approach|81.987|25.000 / 0.0908|0.6742|0/0|—；—|
|BC/p0001-A1|s10_alarmclock_pass_1|WAM→R+残差BC|600/H|approach|1391.437|70.449 / 0.1774|0.4102|0/21|—；—|
|BC/p0002-A1|s10_banana_pass_1|WAM→R+残差BC|292/T|close|928.774|90.739 / 0.3599|0.3517|0/0|—；—|
|BC/p0003-A1|s10_cubesmall_inspect_1|WAM→R+残差BC|286/J|approach|114.819|80.028 / 0.6507|0.8022|0/0|15,16；B|
|BC/p0004-A1|s10_cup_drink_1|WAM→R+残差BC|231/J|close|1806.283|188.048 / 1.3679|0.4176|0/3|10；B|
|BC/p0005-A1|s10_cylindermedium_inspect_1|WAM→R+残差BC|534/J|close|1579.043|333.366 / 1.5410|1.0000|0/7|16；B|
|PPO/p0000-A1|s1_cylindersmall_pass_1|WAM→R+残差PPO|600/H|approach|81.987|25.000 / 0.0000|0.7345|0/0|—；—|
|PPO/p0001-A1|s10_alarmclock_pass_1|WAM→R+残差PPO|159/T|close|995.303|25.000 / 0.0865|0.3433|0/0|—；—|
|PPO/p0002-A1|s10_banana_pass_1|WAM→R+残差PPO|62/J|approach|110.135|337.418 / 0.3710|0.3954|0/0|15；—|
|PPO/p0003-A1|s10_cubesmall_inspect_1|WAM→R+残差PPO|211/J|approach|99.833|191.151 / 1.7662|0.4219|0/0|16；B|
|PPO/p0004-A1|s10_cup_drink_1|WAM→R+残差PPO|211/J|close|1071.702|236.402 / 0.9390|0.6895|0/4|10；—|
|PPO/p0005-A1|s10_cylindermedium_inspect_1|WAM→R+残差PPO|49/J|close|79.528|828.567 / 0.5601|0.7918|0/0|10；—|

完整字段含每例末5状态的q/dq、实际PD、force_valid/五link力、table gap/footprint、该门关节上下界/超界量、命令/PD跟踪误差、clip数量和原文件路径：`tmp/s40-deathcause/evidence.json`；简表CSV为同名`.csv`。原有consumer日志保留位姿目标，未保存consumer的S2d为直接actor路径，不臆造相位。

## 3. 共同代码原因的比例：什么已经证明、什么尚未证明

### 3.1 可以直接复核的数字

- 21/23=91.30%物理中止触发同一关节门；21/31=67.74%旧协议失败最终落在该门。30个关节事件全部属于受限手指转动DOF，不是把prismatic米与revolute弧度混算。
- **0/21**中止在触门手指对应的末次PD目标直接越出该手指名义界限；**7/21**末次PD目标在其他DOF（均Z平移）越界；**11/21**历史上至少有一次受限目标越界，另外10/21整条历史无受限目标越界。不能把“触门手指目标合法”写成“全部18DOF目标均合法”。
- 21/21触门前一保存状态未abort、后一状态首次abort；最大PD重建误差`3.99e-07`。最终动作裁剪不是终止门；21个关节中止最后一拍的最终raw→clipped均未发生实质裁剪。S2e的R内部5次clip仍单独保留，不将最终0clip当全链0clip。
- 已保存相位的13个关节中止全部在approach/close（S2e6、BC3、PPO4）；S2d8个为直接actor，不能强加同一相位机。此数据不支持把共同原因定位为抓后servo。
- 触门前腕部PD位置增量范围**70.985–828.567mm**，腕关节增量范数**0.0341–1.7662rad**。这是平衡点偏移，不是实际每tick位移/速度；单凭量级不能宣布“scale错了”。

关节事件分布（旧S2e/S2d未保存名字，索引来自实际q/limit；名字按同一已验证Inspire URDF/新host名字表映射）：

| DOF | 关节 | 事件数 |
|---|---|---|
|8|middle_proximal_joint|2|
|10|pinky_proximal_joint|8|
|12|ring_proximal_joint|3|
|17|thumb_distal_joint|2|
|15|thumb_proximal_pitch_joint|4|
|16|thumb_intermediate_joint|9|
|6|index_proximal_joint|2|

### 3.2 跨臂最强共同候选：连续腕角分支

取证后固定的检索算法（非预注册因果测试）：某腕DOF满足`abs(q[t]-q[t−1]) > 2π`且`abs(q[t−1]) > 2π−0.25`。0.25rad只用于描述“接近分支”的检索窗口，不是新控制阈值；不用于改评分。实际15个终止事件跳变远大于该阈值。检索窗口0.05/0.10/0.25/0.50rad时，当步关节中止检出8/12/15/15例，全历史事件18/26/37/41次；这些是事后敏感性数字，不据此选控制参数。

15/23物理中止（65.22%；关节中止内15/21=71.43%；全部旧失败内15/31=48.39%）满足，跨S2e几何3/WAM2、S2d GT5/WAM1、BC3、PPO1。样本描述性Wilson95为[44.89%, 81.19%]，重复同初态且批次依赖，不当独立泛化CI，更不是因果贡献CI。

| 例 | 腕DOF | q[t−1] | 实发PD[t−1] | q[t] | 实测palm旋转变化rad |
|---|---|---:|---:|---:|---:|
|S2e/block0-geometry|4|-6.271044|-6.511477|4.152091|2.3399|
|S2e/block1-wam|3|6.193161|6.852151|-4.510048|1.7552|
|S2e/block2-wam|4|-6.187247|-7.678254|3.690735|2.6991|
|S2e/block3-geometry|4|-6.269238|-6.294003|4.672350|1.7279|
|S2e/block4-geometry|5|6.274720|6.624001|-3.380906|2.8622|
|S2d/block0-c-live-gt|4|-6.280584|-6.404014|4.015487|2.2747|
|S2d/block1-c-live-gt|4|-6.275624|-6.459987|3.616122|2.7415|
|S2d/block2-c-live-gt|4|-6.267972|-6.278357|4.684702|1.7107|
|S2d/block3-a-original-wam|4|-6.205589|-7.289879|3.413124|2.9493|
|S2d/block3-c-live-gt|4|-6.282788|-6.639295|3.888322|2.5270|
|S2d/block4-c-live-gt|4|-6.280584|-6.404014|4.015487|2.2747|
|BC/p0003-A1|4|-6.203987|-6.660948|4.837377|1.5262|
|BC/p0004-A1|4|-6.112158|-7.423974|4.423986|2.0304|
|BC/p0005-A1|3|6.086205|7.082608|-5.301828|1.2846|
|PPO/p0003-A1|3|6.147873|6.784483|-5.286105|0.9036|

可定位的相关代码：

1. 各冻结`physical_safety.py:17–20`按has_limits mask检查；腕3:6本来hasLimits=False，不进入这个门。门报的是随后手指状态越界。
2. 各冻结`tau_replay_contract.py:21–27`把根部target设为q+scale×action，native host也用同转换；BC冻结`grasp_geometry.py:57,70–71,79–84`不把无限腕视作有界DOF。它不是“已确认漏了机械限位”，而是尚无连续角驱动支路合同。
3. `src/task/wam-rl-policy-v3/src/wam_rl_policy_v3/continuous_wrist.py:1–4,16–17,29–35`明确只在reset归一化，运行期action与目标保持独立；native入口`inspire_physics_host.py:249–258`在task.step前生成目标、step后读状态，未保存中间驱动误差。S2e冻结host实际同一入口为`:247–256`。冻结host路径与行号见相应runtime-code（共享当前host会含之后新增的日志行，不混作旧运行身份）。

PhysX官方[Articulations文档](https://nvidia-omniverse.github.io/PhysX/physx/5.1.1/docs/Articulations.html)区分会在±2π折返的revolute与unwrapped类型；[NVIDIA维护者对PhysX旧版的说明](https://github.com/NVIDIAGameWorks/PhysX/issues/531)也指出关节位置分支与驱动误差有关。这为候选提供机制依据，**不是本机IsaacGym运行时joint枚举、子步drive误差或冲量的取证**；不据此宣称已确认库bug。

**反例与未证实部分**：全7400状态中检出37次此类跃变，15次在关节中止当步，另22次发生后当步仍未中止；4个最终非joint中止/时域截断回合也曾出现过该签名。0/21触门手指对应目标直接非法，且10/21历史受限目标始终合法，排除了“统一把手指目标裁进URDF即可解释全部”的说法；不能排除腕驱动、手/自身/桌面/地面碰撞与求解器的不同组合。

**问题的直接答案**：“由同一处可修代码/参数造成”的因果比例目前**未辨识，不能给点估计**。15/23是相同候选签名占比，不是已认证修复覆盖率；也不能把未确认写成0%真实因果贡献。因此B不满足，不做任意腕角上限/rate-limit或换控制器来换中止数字。

### 3.3 两个桌面门：触发机制不同于“压穿”

| 例 | t−1 gap/footprint | t gap/footprint | 可成立的描述 |
|---|---|---|---|
|BC/s10_banana_pass_1|-875.320mm / False|-886.118mm / True|物体已在桌面下，由桌外中心重入footprint|
|PPO/s10_alarmclock_pass_1|-862.863mm / False|-865.178mm / True|物体已在桌面下，由桌外中心重入footprint|

判据代码在各冻结`physical_safety.py:13–14`，中心footprint/min点高度计算在各冻结`teacher_quality.py::table_support`。2/2符合这种几何重入；没有成对接触/三角面碰撞证据，不能据此宣布发生了真实穿板，也不能把它计为成功或删除安全门。

### 3.4 并非全部目标严格不可达的既有反例

| 已有随机PPO成功（不新增臂） | 同初态失败评估 | 末态误差 | 原口径最长hold | 9个初态字段差 |
|---|---|---:|---:|---|
|ppo/p0005-A1|assessment-ppo/p0005-A1|3.4977mm|17.9667s|全部0|
|ppo/p0034-A1|assessment-ppo/p0004-A1|1.3859mm|16.6333s|全部0|

两例均原协议600controls末态成功，目标误差与9个初态字段已从raw独立复核；持续hold沿用原冻结评分输出（本次不把它当外部接触真值）。这只反驳这2个初态“物理上根本不可能完成”；另外4个BC/PPO初态、几何参考每一中间姿态的URDF/扫掠可达性仍未认证，不推算总可达比例。完整路径/初态差见`positive-witnesses.json`。

## 4. C：缺失字段与最小补记录方案（仅设计，未启动）

| 要区分的候选 | 已有字段 | 缺少的决定性证据 | 最小补记录 |
|---|---|---|---|
| 连续腕角折返→驱动误差脉冲 vs 碰撞先引发角跳变 | 30Hz q/dq、实发PD、palm pose、5个link净力 | 本机实际revolute/unwrapped类型；每次physics积分前后q、drive target/drive error、drive/constraint impulse | 在同一几何τ冻结入口，仅触发窗口保存各物理子步，附真实dt/substeps/control-frequency；不假定30Hz包含子步顺序 |
| 手/桌面/地面/自身碰撞 vs 物体接触 | 五link净接触力、object net force/valid、CAD样本 | 成对body/shape ID、接触点/法向/穿深、normal/friction impulse；旧S2e/d没有各link drive torque/DOF properties | 窗口内只保留手、物体、桌面、地面涉及的接触；GPU API不暴露时标MISSING，不能用净力冒充 |
| 接触未认证是没抓到、短暂掉落还是observer阈值问题 | hand11、force_valid、阶段汇总、部分consumer info | 每tick observer近邻/力/耦合/支持判据分量和触发原因，几何参考→R→最终PD间目标变化 | 同窗口记录原始各分量与完整目标，不改阈值；S2d无v7相位机必须标不适用 |

若机主裁决补录：**只一条几何τ臂，复用原5个block初态/原R与权重/原PD/原gate，不重跑配对套件**。预注册最多5×300=1500控制（原诊断时域），每个首次跨支路附近只写“前2控制间隔＋事件间隔＋后1间隔”的环形缓存，共≤20个控制间隔；每个间隔按实际native子步全部留存，非假定固定帧数。不保存视频/模型/全mesh。成功未到600的只报未测，不能当正式末态success；以后真进入B才使用600末态主指标。

补录预算≤16MiB，总物理子步样本≤2048、每窗口接触记录总数≤4096对（含所有子步）并记溢出；若溢出则该窗口接触归因无效，不能静默截掉接触。已有snapshot只含q/objects，不含PhysX隐藏求解器缓存，**不能从任意失败前tick设状态就宣称精确重放旧事件**；必须从原production reset走到触发窗口，记录能否复现。GPU接触/drive impulse接口若不暴露，先返回不可取证项，不擅自换CPU物理或新模拟器。本轮没有启动这项补录。

## 5. τ训练批只作为离线预测证据

`tmp/t38-r2/panel-trained-A/metrics.json`：step1223、512窗口/124母序列（TRAIN分布），三项EPE基线[16.1694,59.2219,11.2523]mm→[10.2884,31.7598,6.5043]mm；相对−36.37%/−46.37%/−42.20%。配对母序列bootstrap95差值区间分别[−6.5041,−5.2801]、[−31.9789,−23.3049]、[−5.7424,−3.8141]mm，均排除0。此处只独立读核既有JSON/更新数，不重跑其评估，也不把它混入32回合。`delivery-summary.json`明确execution_benefit=UNMEASURED；新的τ执行死因/收益**未取证**。

## 6. 一条可复现命令、预期输出与边界

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=2 \
 /home2/wyy/miniconda3/envs/graspenv/bin/python tmp/s40-deathcause/collect.py \
 --output tmp/s40-deathcause/evidence.json
# 预期 n=32 / states_replayed=7400 / controls_read=7368
# physical_abort=23 / joint_abort=21 / joint_events=30
# terminal_branch_joint_abort=15 / branch_transition_events=37 / nonterminal_branch_events=22
# 所有保存tick、原门、实发PD与重建一致；status是证据重放，不是控制器正确/修复成功。
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=2 \
 /home2/wyy/miniconda3/envs/graspenv/bin/python tmp/s40-deathcause/make_report.py
# 只再生本报告与同初态正见证元数据，不更新STATE/INDEX。
```

依赖原存档四个run、各自runtime-code、旧native audit、原canonical.npz与table_bounds；缺失就失败，不下载/替换样本。解释器graspenv Python3.8.20/NumPy1.24.4，CPU只做已授权只读取证（无网络模型推理）。代码调用的是每run冻结安全函数/PD公式，非当前生产版本。

反馈环数值敏感性检查：21例实际触门q恢复到名义范围后，原安全函数不再给关节门；若误用前一状态则21例全部漏判。它们只检查本次取证的时间/门判别敏感性，**不是修复回归或仿真反事实**。本次不声称“测试PASS证明因果正确”。

本次空间预算5MiB，实际取证产物约0.6MiB（最终以du为准），/home2检查可用约130GiB。未创建B修复worktree、未使用GPU、未停止/重启任何既有作业。其他会话已有训练/仿真不计本次新增，也不将其结果纳入本表。

**收口**：已确认共同失败门，定位连续腕角驱动支路这个最强共同候选；“同一处可修原因占比”仍未取证。按brief40走C交裁决，当前没有获证据支持的单点修复，不把候选签名比例冒充修复收益。
