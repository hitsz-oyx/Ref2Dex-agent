# source控制器接近/闭合修复（v14）

## 结果解读表

| 结果 | 结论 | 下一步决定 |
|---|---|---|
| 修后source≥32/40，物理安全通过 | 本开发面板控制器资格达标 | 才与WAM同初态/预算/后段比较 |
| 提升但<32/40 | 修复不够，不能拿WAM解释数据量 | 按余下死因继续修source |
| 不提升或实际限位仍多 | 当前方案未解决物理瓶颈 | 保留全部失败和gate，继续修；不训练v6 |

当前状态：**原source14/40；R3–R8各0；R11 4/40、R12 2/40、R13 12/40；仍未修好。** R13有19/40关节中止，R14对正常PD命令带宽/跟踪偏移做准入并同步暂停参考，按同40冻结。R10技术中断单列；不比WAM、不RL/不重训。
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

## R9：实测首次接触处停腕；候选探测不提前转持握

R7在21个回合首次锁定时，目标离实测腕中位63.232mm/max178.322mm，仍继续挤推。`SourceCeilingConsumer.step`改成锁实测palm而非未来目标；红测试源R8/contact-stop-red.log实际输出推进.01m，修后0。失触恢复合同仍在。
同模块probe分支删除“用持握微调覆盖尚未完成闭合”，保留source闭合目标及相邻.04rad上限、12N卸载；probe最多10mm/不是成功证书不变。红测试R8/probe-closure-red.log q=.4时旧命令.4，新.44；已抓住过早持握冻结。预期32专项/28mutants，现有命令可复核；物理n40门槛仍32，未达则不比WAM。
R9仅在R8≥9完整失败后排队启动；仍不终止R7/R8原40、不改评分或物理gate。完整v3结果以R9/tests-all.log全量pass/failed为准。新逐物体死因仍由summary.json保存全部q/关节名/阶段/几何量，不挑最好例。

## R10：目标准入与实测压力分离（R9未启动）

实际step红测试R9/margin-replan-red.log为1 failed/32 deselected：q=.01在名义0界内、未达实测危险−.05，父Geometry却因目标margin .04而换抓姿。`grasp_geometry.py`只增加可选admission.measured_pressure两行钩子；source MarginAdmission将真实危险按原名义边界±.05且向外速度>0.2预警。PD候选仍须目标inner bounds，physical_safety±.10不动，普通控制器无hook保持旧行为。预期不必要的重规划1→0；卸载和真实gate仍有效。
冻结prepare对v11 baseline仅注入同两行，不整份copy当前geometry，确保postservo/评分不被并行改动带入。R9 supervisor在QUEUED且pids/commands均空时被本线程取消，记录CANCELLED_PRESTART/0controls；从R8作为前序启动fresh R10，同40/24000/0updates，目标32。复现命令见实验卡R10；33专项/29mutants/全量309 passed/0 failed/0 skipped，20.85s、2 warnings。红测试与mutant都在真实step读reachability标记，非import失败。

## R11：腕PD命令限速、技术中断及补救

R10 supervisor.log明确JSONDecodeError发生于读前序manifest，随后两个本批client被终止。已完成3/40均失败；3个完整1624控制+2个partial870=2494，35未开始；保留所有记录，不挤掉失败或冒充40。R11 fresh同40，以R8为前序，不能等FAILED的R10。
`run_source_ceiling_worker.py::write`改原子替换；supervisor `read_manifest/ancestors_released`对瞬时空/截断文件返回pending，保留排队超时，不把pending当完成或致命错误。CPU三个queue回归抓住故意“空文件当完成”和“非原子直接覆盖”，不是物理成功证明。
`SourceCeilingConsumer.step`腕限速改相邻最终PD命令SE3之间，而非永久限制相对实测q的跟踪偏移；红测试六拍旧目标=.025m，新=.150m，每拍仍≤.025m。跟踪偏移与命令步长分字段记录；卸载/有界probe以实测锚点优先，抓后原伺服不改。mutant恢复相对实测限速会被实际step断言抓住。
复现命令/预算见实验卡R11节。全部版本已实耗141276（含R10中断），本版≤24000后≤165276，0updates，目标32/40仍未证实。全仓verify本次被并行v6卡P-20261023-tau-fixed-panel-metric缺claim_id/hypothesis_family/probe_index_in_family阻断，未修改对方文件，不报PASS。最初frozen preflight误读launch覆盖层缺inspire_contract，不能当测试通过；已在R10真实完整runtime复跑33/33，见frozen-preflight.json。

## R12：探测必须在闭合命令完成/原运动历史窗口后

R11初期失败p002/p005/p004在首次probe时closure raw余量.229/.269/.366rad；成功例约.04rad含投影。源码现先对closure做同inner admission再比命令差≤1e-6，完成后保持候选原force条件≥.25s（原observer历史窗），才允许≤10mm probe。真实耦合已成立仍按原反馈进入lift，不改评分。预期首次probe闭合指令余量→≤1e-6，等待→≥.25s；不能将此当实际q完成/力闭合证据。
红测试R11/premature-probe-red.log明确候选第一拍就probe，1 failed/34 deselected；修后35控制+3queue定向通过、32控制mutants killed，新增抓住“半张开就probe”和“省掉运动历史等待”。旧probe正控制fixture显式声明闭合已稳定，不用不完整手形冒充已闭合。独立复现仍按本报告pytest/mutant与实验卡R12启动命令。
同40/24k/0updates，source≥32仍门禁；前序R11全40保留。累计预算明确追加24k到≤189276，不隐瞒版本间成本。无法自证：命令稳定.25秒是否足够清除真实接触求解器抖动、是否因此增加握力超限，须本版完整40与q/dq/forces验证。

## R13：停止改写source，保住已工作baseline再加最小guard

R11全40仅4成功（Wilson3.958–23.052%，23080controls/1jointabort），不能拿局部阳性称修好，较原14/40退步；R12已≥36失败，继续原40。新SourcePDGuard保留原Geometry/MeasuredGraspConsumer消费原source PD意图，只加实测卸载与inner-target admission，不换key、不停腕/不probe/不做伪速度限制。健康安全输入、原后段命令均逐位对照；39控制+3queue专项通过，Guard独立4个注错新增，旧32个legacy mutants不冒称新Guard覆盖。
worker恢复source记录action/refq→target→live q strict inverse→intent路径；卸载暂停source帧、held后段不消费参考。原R无τ后备显式按strict inverse拒绝统计selected/consumed，拒绝原因与reference_PD_target逐拍保存；canonical最终clipping、inner源proposal投影、实际q限位分别统计。source key为metadata-only，不再宣称其腕/手形被执行。仍不是GTτ/R上界。
预期无危险/无需投影时命令改动=0，postservo改动=0；限位12/40→≤2，成功14/40→≥32，未知物理能否达到。复现按实验卡R13，本版40/24k/0updates；累计上限212356（明确追加24k），不比WAM。无法自证：原source暂停后是否仍能重入接触、inner投影是否削弱握力、后备R是否介入成功；必须原生40逐例报告，不隐藏。

## R14：PD指令与接触柔顺带宽，不用整数/代理门改成功

R13完整12/40（Wilson18.075–45.430%，Fisher对原14/40=.81165）/14279控制/19jointabort。末次失败PD目标跳变中位317mm位置、.199rad腕joint、.140rad指关节；最大762mm/2.787rad/.565rad，包含紧急回撤，不能全归源动作。Native dq预测仅对早期16个限位中的3个提前预警，不能依赖传感后再反应即可安全的假设。
SourcePDGuard现对pregrasp正常相邻命令设.75/.30m/s位置、2.4rad/s native腕joint范数、1.2rad/s指令带宽，限制PD tracking offset（远75mm/.15rad，近按既有wrist_gap及手跨度）。这不是实际速度/接触力证明；卸载例外显式记录，真实gate不改，后段不改。手指先限幅再进入原阶段机，避免close→lift把完整源closure一次跳进PD。
worker的advance_source_reference将reference推进与motor admission对齐：卸载/手指带宽/投影例外暂停；root受限且无生产多指接触也暂停；已多指接触可推进手形，避免接触预载不可达到造成死锁。held冻结参考。CPU新queue断言故意忽略带宽会被抓。
预期最大正常位置命令步长≤25/10mm、腕joint≤.08rad、指≤.04rad；实际异常投影/卸载须另报。目标joint19→≤2、成功12→≥32，仍待完整40。复现命令见实验卡R14；红测试为旧Guard第一步超带宽的实际step断言，非import错误。原“健康输入逐位相同”fixture改成同时满足新带宽的真实安全输入（.005m/.02rad指变化），不冒称全14旧成功轨迹逐位不变。

R14启动前补充：真实step红测试handover-clock-red.log抓住unused closure冻结executed-preshape cursor；仅当前执行手形决定finger_bandwidth_limited，未来持握初始化另记closure_handover_limited。实际close→lift finger目标仍限相邻.04rad，held之后原伺服不变。投影/柔顺优先造成的带宽例外和紧急卸载分别标记，不谎报绝对硬上界。生产Guard直接8项回归/11mutants，legacy模式35项/32mutants仅历史兼容；queue4项含3种注入错误。全量数以R14/tests-ready-all.log为准。

## R15：打断“需要先contact才能送到contact手形”的推进死锁

R14/p000在t180–599、reference36/source contact42、0卸载、root受限且手已接近，满20秒无持握。reference gate现接收生产同一InteractionConfig的实测finger gap；近物体可推进下一手形，不要求先多指力接触。手指指令带宽/卸载/投影仍阻塞，评分与物理接触布尔量原封不动。红测试实际返回36而非37；新增queue注入故意丢弃接近测量会被同一断言抓住，非import/TypeError冒充kill。
Guard动作算法沿R14，仅标签规范为schema force-margin-bandwidth.v1；clock协议schema measured-approach-paced.n40.v1，与run_id r15分开。预期root受限且实测40mm近物时36→37，远离仍停；真实force/coupling/lift/hold/目标误差不改。新同40/24k/0updates，前序R14，累计上限≤250430，目标32仍未知；不比WAM。

## R16：live对象参考 + 原始抓取阶段边界 + 首拍恢复保护

source_native_targets是生产worker实际入口：原source action/ref-q先变raw target，再用source palm/object同帧及live palm/object/native mount做SE3映射。记录recorded_PD_target与reference_PD_target两份，不能把raw/feedforward/mapped混报。source_pregrasp_length只限制ACQUIRE控制参考到已保存的前驱抓取command，不裁掉完整录制、不改TaskSpec/任务分母或标签；末端仍继续真实20秒物理，进入后段只靠原live反馈。
CPU3项生产helper回归分别抓住故意丢弃位移映射、丢弃旋转映射、使用完整postgrasp/release长度；红为3 failed（数值断言），见R15/reference-contract-red.log。预期50mm对象平移→50mm native目标平移；.4rad对象旋转→同腕旋转；hand案例188的reference不再越过51，而不是声明source frame52已在live成功。
SourcePDGuard同时把实测丢持后的第一拍按恢复处理；原Phase仍hold并不代表要继续绕过motor guard。数字红为400mm动作/应≤10mm，修后guard active且原正常held命令逐位同base。新增mutant重放旧首拍绕过行为会被数值断言抓住，不只检查metadata。
复现命令见R16实验卡及测试目录，完整计数见R16/tests-ready-all.log（须passed/failed全量）；9个production Guard step回归/12mutants与35个legacy/32mutants分开；queue/reference另有注错。物理目标32/40仍未证实；同40/24k/0updates，不比WAM。

## R17：参考推进看真实几何/离桌相位，不看PD终点/墙钟

R16 hand/camera末态力=0，腕相对source实测腕仍差93.7/109mm。select_source_reference用source object-local腕SE3与actual wrist、actual table gap匹配；phase未允许就不消费post-lift源姿态，2帧受phase限制的前瞻是控制参数，不改时间标签。允许回退并记录signed frame_delta/matched_state_frame/action_frame/phase_admission_missing；motor未就绪仍阻塞forward选择。无候选显式fallback、不删样本，不谎报完整source已可执行。
acquisition_feedback只在controller入口屏蔽supported/unknown时的grasp_established，防止桌面推物直接触发lift；原测量和评分反馈不动，实际unsupported仍用原coupling。新三项数值红在R16/measured-phase-red.log，3 failed；修后reference测试新增phase过滤、几何共变、supported接触三种注错，实际常态held算法不改。
预期supported状态的source参考不越过当前尺寸phase容差（缺候选例外明报）、controller不得仅凭supported推物就进入lift。物理≥32仍未知；同40/24k/0updates，预算上限298430，前序完整R16，不比WAM。
