# source控制器接近/闭合修复（v14）

## 结果解读表

| 结果 | 结论 | 下一步决定 |
|---|---|---|
| 修后source≥32/40，物理安全通过 | 本开发面板控制器资格达标 | 才与WAM同初态/预算/后段比较 |
| 提升但<32/40 | 修复不够，不能拿WAM解释数据量 | 按余下死因继续修source |
| 不提升或实际限位仍多 | 当前方案未解决物理瓶颈 | 保留全部失败和gate，继续修；不训练v6 |

当前状态：**R3–R6均已完成40且均0/40；未修好，不能称改善**。R7已运行且≥9失败，仍完成40。R8恢复source路径及完整腕PD目标，预注册同40/24000控制，尚未运行。旧source14/40；新四版合计93322控制、0更新。完整逐版数据在后文，各版原始失败均保留。
## 修复、可证伪预测与复现

R3历史定位（以提交99875f23核对；均以`src/task/wam-rl-policy-v3/`为前缀）：
`src/wam_rl_policy_v3/source_ceiling.py:16`物体SE3映射、`:33`余量准入、`:64`实测卸载/限速/交接；
`tools/run/run_source_ceiling_worker.py:111`不实例化WAM、`:214`source映射实际调用、`:233`游标暂停；
`tools/run/run_source_ceiling40.py:12`原40初态/源码冻结；
`tools/run/summarize_source_ceiling.py:63`初态/逐物体核对入口。
首版diff为`git show 9f578b7d -- src/task/wam-rl-policy-v3`，交接/日志修订见同文件后续提交；
用`git diff 9d005b4e -- <上述单个文件>`可查看本轮最终diff，不含并行v13的文件。

- `src/wam_rl_policy_v3/source_ceiling.py`：接近/闭合实测指尖力>12N卸载；真实关节位置+速度向限位滑行触发独立退让（指尖力不是全手接触检测）。卸载暂停参考游标，不让参考时间继续跑；无效力数据不触发力卸载。
- 同模块`MarginAdmission`：候选排序前按原生18关节含被动耦合检查，目标内余量40mrad/根位移10mm，范围小则限制为10%；没有扩大实际关节界限。手指目标1.2rad/s，接近0.30m/s/1.2rad/s，几何替换后仍限速。实际状态不被裁剪。
- `tools/run/run_source_ceiling_worker.py`：仅A0，WAM不实例化；旧R无τ后备保留，source表达失败照报；抓后使用原RelativePoseServo/force规则，不改成功、支持、碰撞、持续时间或终点目标。
- `tools/run/run_source_ceiling40.py`：冻结v11源码和同一40初态，独立空闲卡队列，24000控制上限；每回合保存q/dq/force/PD/游标/卸载/候选/阶段/限位。`summarize_source_ceiling.py`输出逐物体死因、哪个关节、阶段、q/上下界/超界量、PD目标、物体抬升、palm/指尖距离与目标误差；重新对齐40个旧初态。

预期数值：source成功14/40→≥32/40；接近/闭合限位12/40→≤2/40；初态对齐40/40；0训练更新。若未达到，仍不跑WAM。候选几何仍是稀疏表面/点位近似，尚无完整扫掠碰撞证明，不冒充安全保证。

```bash
export TMPDIR="$PWD/tmp"
PY=/home2/wyy/miniconda3/envs/graspenv/bin/python
$PY -m pytest -q src/task/wam-rl-policy-v3/tests/test_source_ceiling.py --basetemp=tmp/pytest-source-v14-review
$PY src/task/wam-rl-policy-v3/tools/audit/run_source_ceiling_mutants.py
$PY src/task/wam-rl-policy-v3/tools/run/summarize_source_ceiling.py --baseline outputs/wam-rl-policy-v3/grasp-v11-coverage40-20261022-r1/summary.json --output outputs/wam-rl-policy-v3/source-ceiling-20261023-r3/baseline-details.json
# 第一次执行；禁止对已有registration重复prepare，复跑请换fresh run路径。
$PY src/task/wam-rl-policy-v3/tools/run/run_source_ceiling40.py --run outputs/wam-rl-policy-v3/source-ceiling-20261023-r3 --prepare
$PY src/task/wam-rl-policy-v3/tools/run/run_source_ceiling40.py --run outputs/wam-rl-policy-v3/source-ceiling-20261023-r3 --detached
```

实测软件：专项10 passed/0 failed/0 skipped；完整v3 279 passed/0 failed/0 skipped（详见tests.log，3条万向节锁警告）；故意错误7/7 killed（忽略压力、允许目标跳变、遗漏余量、篡改抓后伺服、漏交接限速、日志目标失真、继续使用录制世界坐标）。旧真实step复现为2 failed，见原r1产物red.log（另有handover-red.log/telemetry-red.log）；没有把纯import失败冒充复现。
原生解释器`/home2/wyy/oyx_ws/.runtime_envs/dexplore_v120_train/bin/python`/Torch2.0.1+cu118；策略解释器graspenv Python3.8.20/Torch2.4.1+cu121；native先import IsaacGym。

## 逐物体旧source失败及几何证据

不是把WAM的13/14套给source：source实际12限位（approach1/close11），13个关节事件。
下表超界量单位rad，原始全量q/dq/PD/距离/支持面间隙见`outputs/wam-rl-policy-v3/source-ceiling-20261023-r3/baseline-details.json`。
`stage_contact_not_certified`指阶段认证未过，**不等于没有真实接触**；例如knife/phone末态伺服接近目标，仍可能因点位/阈值未获得阶段认证，不能把它们强行算成功。

| 物体 | 成功 | 死因计数 | 关节及超出名义界限rad |
|---|---|---|---|
|bowl|0/2|held_then_lost:1, physical_table_penetration:1|—|
|camera|1/2|success:1, physical_joint_limit:1|thumb_intermediate_joint:0.131|
|cylinderlarge|0/2|physical_joint_limit:1, contact_without_lift:1|ring_proximal_joint:0.143|
|eyeglasses|0/2|physical_joint_limit:1, lift_without_continuous_hold:1|thumb_proximal_pitch_joint:0.128; thumb_intermediate_joint:0.120|
|fryingpan|1/2|success:1, physical_joint_limit:1|thumb_intermediate_joint:0.118|
|hand|2/2|success:2|—|
|headphones|1/2|success:1, physical_table_penetration:1|—|
|knife|0/2|approach_without_contact:2|—|
|lightbulb|1/2|success:1, held_off_goal:1|—|
|mouse|1/2|success:1, physical_joint_limit:1|thumb_intermediate_joint:0.104|
|phone|0/2|approach_without_contact:1, physical_joint_limit:1|thumb_intermediate_joint:0.100|
|piggybank|0/2|physical_joint_limit:2|pinky_proximal_joint:0.145; thumb_intermediate_joint:0.106|
|pyramidlarge|2/2|success:2|—|
|pyramidmedium|1/2|success:1, contact_without_lift:1|—|
|pyramidsmall|0/2|physical_joint_limit:1, contact_without_lift:1|thumb_intermediate_joint:0.142|
|scissors|0/2|physical_joint_limit:2|thumb_intermediate_joint:0.353; thumb_proximal_pitch_joint:0.101|
|spherelarge|0/2|physical_joint_limit:1, contact_without_lift:1|thumb_intermediate_joint:0.185|
|spheremedium|2/2|success:2|—|
|spheresmall|0/2|approach_without_contact:1, lift_without_continuous_hold:1|—|
|stamp|2/2|success:2|—|

## 无法自证 / 待独立裁决

1. 12N与余量/速度常数是所有物体共用的首版控制参数，尚无新物理回合证明能改善成功率；测试只证明指定输入会退让与限速。
2. 候选使用当前手点几何，不是目标手形的全URDF碰撞/力闭合解；静态目标合法不等于实际接触动态安全。
3. 尺寸化landmark阶段是否对细薄物体过严仍不确定；本轮不偷偷改判据或重标旧成绩。真实接触与阶段认证分开记录，必要的合同更改须显式新版本并同时复核基线。
4. 同一开发面板不是新holdout；即便32/40达标，source与WAM的信息和参考执行链不同，差距不能单独证明只因数据量。
5. Hermes/GPT尚未复跑此实现；三方分歧逐项保留。本轮不动并行v13/原训练，不重训v6，不做RL或完整RELEASE。

预启动修订：原r1已冻结但未获GPU；查出close→lift手指目标跳变与最终目标日志失真，分别由红测试复现。r1 supervisor明确记CANCELLED_PRESTART、0子进程/0控制，原快照和日志保留；物理批次改为fresh r3，不混算。修后对12个真实首限位前状态单步重算：12/12目标准入、6/12触发卸载；这不是反事实仿真，不声称已防止任何旧失败。

第二次预启动修订：r2仍为QUEUED/无子进程/0控制时，生产source-PD→intent边界的红测试确认：物体平移50mm，原source抓姿仍移动0mm。已加入通用参考物体→live物体SE3映射（含姿态），手指目标不变，修后目标同移50mm；不是单初态常数，也不是GTτ/R上界。原r2 receipt明确CANCELLED_PRESTART和原因，fresh r3才是计划物理批次。新10项专项与7个mutant均须通过，40/40参考含实测palm_pose。

## R4：修源参考时钟与运动耦合等待环

R3至少13个失败/剩余最多27成功，已触发下版修复条件，但不提前终止原40。trace中p002/p005满600控制、0卸载、手仍远离物体时参考已结束，是参考时钟比限速后的物理执行跑得快，不可把0限位称改善。

- `source_ceiling.py::SourceReferenceClock`：当前/下一录制状态与实测腕在当前物体局部SE3内匹配；只有更接近下一状态才推进，最多1帧/tick，卸载暂停。39个参考的首次双指力接触+0.5秒是接近参考上限；p034无接触证据留在分母，不填假事件/不修改任务终态。
- 同`SourceCeilingConsumer.step`：复用生产FeedbackObserver.grasp_candidate及force_valid/contact_observed；未有运动耦合时最多10mm、0.06m/s起动，不累计，未凭探测就宣称grasp。失去有效接触清除起点。原成功门仍真实coupled/lift5cm/hold1s/尺寸误差，原抓后伺服不变。
- worker实际调用source_clock.advance再取source_target；记录匹配误差、推进/停顿、参考接触missing、探测起点/上限。queue在R3至少9个已完成失败后才允许fresh R4，R3其余回合不中止。

每版本仍n40、≤24000控制、0updates；R3/R4累计≤48000控制。R4不是把新控制器塞进旧run。
预期：远离下一参考时推进1→0；接触无运动耦合时0起动→≤10mm；新最终目标≥32/40及限位≤2/40，物理效果尚待测。
专项14 passed/0 failed/0 skipped；完整v3 283 passed/0 failed/0 skipped（24.85s，3警告）；10/10 mutants killed，新增抓住时钟越过实测状态、遗漏探测、探测累计3种错误。探测红测试在R3代码下1 failed，见R3/probe-red.log。
复现：运行本报告上述pytest/mutants命令（现预期14项/10mutants）；启动R4的两条命令见实验卡R4节，首次prepare需带 --predecessor .../source-ceiling-20261023-r3。

无法自证：接触候选+10mm探测能否建立真实运动耦合、实测游标会不会停滞、source接近的几何构型是否可行，均须原生40回合给数；不以这些CPU测试替代物理证据。

## R5：消费source关键抓姿，移除录制时钟约束

R4至少13个已完成失败，32/40已不可能达到；旧40不中止、不换样本。p005 trace显示PD平衡点误差current=3.936、next=4.160，cursor0卡满600controls。R5不继续调时钟常数，按v7保留完整抓取意图：source实际object-local腕抓姿+独立手指预形/闭合目标+同任务抓后目标。

`source_ceiling.py::SourceGraspReference`从原native轨迹提取实际抓姿；力接触必须同时满足生产InteractionConfig的18mm两指邻近，避免表面/自接触力冒充对象接触。39/40有source对象接触；p034最近双指几何候选仍在分母，source接触missing显式保留，不把UNKNOWN改HOLD。
`SourceCeilingConsumer.step`到达腕抓姿就尝试闭合，不要求开指先触物；闭合消费closure_q，失持/过载按实测卸载。只有生产反馈实测运动耦合才转lift；成功判据未改。
worker不消费R或source action，不再声称同source-PD原样重放；旧R forward仍保存作兼容诊断，source_R_consumed_controls=0，source_pd_inverse_applicable=false。诊断R的超界不能当作source实际动作裁剪。源指令的原生限位投影仍逐tick记录，不以0 canonical clipping冒称无饱和。

R5同40/≤24000controls/0updates，累计三版≤72000；每GPU等待所有前序source worker完成，继续等空闲4/5/6，不抢v13。预期最终≥32/40/预抓限位≤2/40。
新增回归：腕到位/开指时必须进入close且不假造contact；source独立预形/闭合/live SE3必须保留；离对象很远的手部力不能成为source对象接触。R4冻结运行时首条确实失败，见R4/key-close-red.log。
专项17项/0 failed/0 skipped、完整v3预计286项（以本run tests.log全量结果为准），13个故意错误（以mutants.log为准）。真实新成功率仍必须原生40验收，不能用这些数字替代。
启动fresh R5的prepare/--detached命令见实验卡R5节。全部源帧/接触依据见`outputs/wam-rl-policy-v3/source-ceiling-20261023-r5/source-keys.json`。

## R6当前复现与审阅索引

| 已完成批次 | 成功 / n / Wilson95 | 控制 / joint abort / table abort | 决定 |
|---|---|---|---|
| 原source | 14/40，[22.13%,50.49%] | 16745 / 12 / 2 | 仍是较强基线 |
| R3 | 0/40，[0,8.76%] | 23894 / 0 / 1 | 显著退步，Fisher3.07753e-5；不得称修好 |
| R4 | 0/40，[0,8.76%] | 24000 / 0 / 0 | 同上，不能只报限位减少 |
| R5 | 0/40，[0,8.76%] | 23648 / 0 / 6 | 同上，继续修消费/握力，不比WAM |

各版40个raw一致、40旧初态一致；这是本线程改进失败，不是τ或数据量的负证据。
R6预注册同40、≤24000新控制，累计四版≤96000；尚无R6物理结果。

最新文件定位（前缀src/task/wam-rl-policy-v3/）：
`src/wam_rl_policy_v3/source_ceiling.py:70` source完成抓姿/前一拍PD目标；`:131`18关节内余量准入；`:175`实测闭合、命令间限速、卸载/有界起动；
`tools/run/run_source_ceiling_worker.py:164`真实入口以native offset/scale/plane构造source key，`:211`消费该意图；
`tools/run/run_source_ceiling40.py`冻结v11评分/物理/后段并逐卡等前序完成。

最重要的两处修正及预期：
1. 用接触实测q当PD目标会卸掉正常握力。hand例实测thumb_pitch=.055578、前一拍PD目标=.459928；R6使用后者，前一拍是f-1，避免错一帧。新增实际step回归还抓住“相对实测q限速”会使q=.4时目标永远=.44，R6命令间限速允许12拍达到.8。危险时明确紧急卸载，非正常闭合跳变。见R5/force-target-red.log、R6/slew-red.log。
2. 原source抓姿误取初碰，R6优先source已观测运动耦合/抬升5cm/离桌2cm的完成状态。35/40具该source证据，5个不足的照留，不升级成live成功。闭合也读取匹配指节的实测接触；原live成功几何/时间/目标误差一个不改。
远端PD增量上限.01→.025m/tick，近端保留.01；实测接触预期≤7秒，而非R5约13秒。不是物理速度保证。

目标仍URDF内余量；卸载预警改为对既有physical abort±.10留一半余量（±.05），**没有更改physical_safety.py/评分边界**。保存并报告实际名义URDF偏差与中止不同口径。警告超限时interlock阻止发lift命令而不篡改feedback。
不再把跟踪误差当命令跳变：finger_target_step_rad、finger_tracking_error_rad、finger_emergency_slew_override分别记，力/手点槽显式映射。source参考actions只消费一个前驱闭合PD目标，不回放时钟；R forward仅诊断，不消费。

```bash
export TMPDIR="$PWD/tmp"
PY=/home2/wyy/miniconda3/envs/graspenv/bin/python
$PY -m pytest -q src/task/wam-rl-policy-v3/tests/test_source_ceiling.py --basetemp=tmp/pytest-source-ceiling-review-latest
$PY src/task/wam-rl-policy-v3/tools/audit/run_source_ceiling_mutants.py
# fresh独立复跑目录；已有registration拒绝重复prepare，禁止覆盖：
$PY src/task/wam-rl-policy-v3/tools/run/run_source_ceiling40.py --run outputs/wam-rl-policy-v3/source-ceiling-independent-review --prepare --predecessor outputs/wam-rl-policy-v3/source-ceiling-20261023-r5
$PY src/task/wam-rl-policy-v3/tools/run/run_source_ceiling40.py --run outputs/wam-rl-policy-v3/source-ceiling-independent-review --detached
```
当前软件全量302 passed/0 failed/0 skipped（19.35s，4warnings），专项26，21/21 mutants killed；actual-source支持条件/控制构型/最终成绩须物理40验收。原v11缓存、基线raw/reference和calibration-r1旧模型metadata、native IsaacGym/Torch环境都是必需输入；缺失不能独立复跑，命令不会下载/代换。
本轮git曾被并行暂存区挡住；用项目tmp/source-ceiling-sync的detached worktree完成pull/rebase及push，未stash对方文件；后续全仓检查断链已被相应会话修复。

## R7：闭合目标不追逐已被推动的物体

R6完整结果0/40（Wilson0–8.76%）、21780控制、1关节中止/2关节事件；不达32，继续修，不跑WAM。
复现红测试R6/chase-red.log：首次实测匹配接触后对象滑移50mm，旧控制在腕已达原目标时仍命令前进10mm。`SourceCeilingConsumer.step`捕获首次匹配接触时的世界抓姿；失触连续0.12秒/卸载后重接近，不把接触当成功。预期被推动对象移动50mm时抓姿漂移50→0mm；最终目标仍≥32/40，未知能否达到。
新增28专项/23mutants中包括故意恢复追逐与禁止失触恢复。完整v3测试需看R7/tests-ready-all.log的全量计数，不能据PASS宣布物理成功。
运行命令见实验卡R7节；同40/24000/0updates预注册，原评分、抓后伺服不动。运动学独立检查在R6 p005全部601状态中XYZ固定安装角漂移≤1.22e-6rad；小写xyz误差可达π，实际实现采用前者。实际外部与仓库URDF字节一致。

## R8：恢复实际source动作信息，不继续用静态key冒充原source

SourceGraspReference新增live_object/live_palm入口：当前物体局部腕在前方9帧中找最近已录制状态，单调游标，不按tick前进；接近消费真实source PD前馈，闭合完成腕姿也取前驱PD目标。R7之前只保留了手指PD，丢失腕PD前馈（n40位置差中位67.037mm、p90 153.097mm、max358.156mm，角差中位0.1142rad）。这些差异不等价于力，不声称恢复后必成功。
`tools/run/run_source_ceiling_worker.py`生产调用显式传live object/palm，R不消费；原评分/抓后伺服/native gate未改。复现沿本报告pytest/mutant命令，现预期30专项、26mutants；回退静态key、删腕PD、按tick推进分别必须被抓。两条语义红测试见R7/motion-red.log（2 failed/28 deselected）。命令、同40/24000/0updates预算及目标≥32/40见实验卡R8节。
无法自证：motion PD前馈与降速后的接触动态是否协调、冻结抓姿是否保留足够闭合预载，仍须本版40物理结果，不能拿CPU输出一致替代。
