---
schema: ref2dex.probe.v2
probe_id: P-20261019-inspire-task-contract
experiment_id: P-20261019-inspire-task-contract
date: 2026-10-19
task: wam-rl-policy-v3
branch: main
git_commit: recorded-in-run-manifest
claim_id: C1
hypothesis_family: HF-scheduled-wam-executor
probe_index_in_family: 12
seed_pool: probe
seeds: [246]
decision_changed_if_positive: train the full-data task-conditioned BC and official PPO under the repaired contract
decision_changed_if_negative: repair the first observation or native replay mismatch before further PPO scaling
status: COMPLETED
run_id: inspire-task7-fit-20261019-r1
---

## 2026-10-20 table support geometry repair

Blocker: the native tabletop mesh is thin along local Y, and local -Y faces
world +Z. The previous proxy assumed local +Z and local XY footprint.
Every old retained sequence had support=False throughout. The real-orientation
regression fails before repair and passes after selecting the thin-axis face
pointing against gravity, its tangent footprint, and its force normal. The
online BC support proxy uses the same repaired function.

Decision: re-audit the same immutable660 raw executions, preserving all other
failure thresholds,50-frame safety prebuffer, human review filters and split.
If prefixes/tasks change, old BC/tau results remain historical and cannot be
silently mixed with corrected data. Do not weaken filters. Cheapest test is
CPU geometry/observer replay; no neural computation is needed.

- run_id: inspire-teacher-rollout-audit-20261020-r3; task wam-rl-policy-v3,
  branch main, frozen git_commit in launch/manifest.
- source: outputs/wam-rl-policy-v2/inspire-teacher-rollout-20261019-r5/.
- output: outputs/wam-rl-policy-v3/inspire-teacher-rollout-audit-20261020-r3/.
- runtime: graspenv Python3.8/Torch2.4.1, CPU replay, bounded1800s,
  no GPU job/checkpoint, old audit/raw/cache read-only.
- admission schema v2 + explicit table-support contract; current corpus loader
  and auditor reject v1 inputs and sidecar reuse.
- full v3 suite135 tests passed. The old fixed two-motion panel had52/49 command
  prefixes but teacher first coupling84/55, so cannot alone screen transport.
  Old retained data include coupling in365/501 TRAIN and44/62 DEV prefixes,
  with23043/2541 coupled command rows. Not all old BC data lack grasping.

Re-audit completed at56339e41 in409.70s:660 engineering PASS,563 retained prefixes,
60529→62605 commands (TRAIN56418/DEV6187 before active gating),208 prefixes longer,
none shorter. All660 executions have some support frames (71522 total); the
259 matched human-review frames are preserved. No safety threshold was relaxed.
comparison-to-v1.json records the changes. The two old fixed-panel prefixes
remain52/49: their first actual coupling is later than the safe prefix, so they
are approach-only examples even after correct support. No new DExplore training.

## Original contract and historical probes

### UniDex data/interface separation probe (2026-10-20)

User requests UniDexGrasp before more Inspire tuning. Decision: if the known
Shadow565 PPO works but the task10 adapter fails under the same native initial
states, investigate TaskController/goal changes independently of DExplore data.
If both work, retain the new boundary for subsequent matched training; this
cannot prove that all DExplore failures are caused by data.

- run_id: unidex-task-goal-20261020-r1; same experiment/task/seed246/main.
- existing full UniDex BC parent plus trained PPO10000, no retraining/updates.
- GPU6 baseline565, GPU7 task575;16 fixed TRAIN object-scales,64 initial scenes,
  deterministic first complete native episode; common position_only goal,
  corrected reset, common_workspace termination and frozen WAM80000/local geometry.
- task575 uses robot300/task10/world tau264/valid1; exactly the shared
  TaskController used by Inspire. Native force5/velocity6 provide feedback.
- actor/critic task columns zero, named legacy columns/normalization preserved;
  no optimizer migration or learned-task-column claim. WAM goal changes to
  shared TaskController local goal; new goal columns themselves are untrained.
- bounded900s per worker, total1200s; outputs only
  outputs/wam-rl-policy-v3/unidex-task-goal-20261020-r1/. Leave GPU4/5 free;
  existing GPU0–3 tasks are untouched. Frozen source/commands/PIDs in run manifest.
- require initial robot300/pose/hand/goal/full root identity before matched
  interpretation. Report native/ever-held/final-stable/drop/workspace separately.
- only native state/action dimension and60Hz clock differ; same MLP capacity,
  native reward, upstream deterministic actor inference and simulator controls.

#### UniDex results and contact-source correction

#### BC-only correction requested by user

The user meant BC capability, not PPO. Prior UniDex task-goal runs loaded PPO
checkpoints and cannot answer whether BC itself succeeds. New run_id
unidex-bc-goal-20261020-r1 directly evaluates full-data BC best(step46000),
with its original WAM12000 and geometry. It loads no RL checkpoint, constructs
no PPO/optimizer/critic, and performs no updates. Compare BC565 and its zero-task
expanded575 onGPU6/7, same16 TRAIN objects/64 initial scenes/seed246,
corrected reset/position-only/common_workspace, full first native episodes.
Native-ever, final-position, ever-held60 and final-stable are separate metrics.
BC declaration explicitly records the checkpoint/step and absence of PPO.

Decision: if BC565 already fails, investigate BC or task setup independently of
DExplore data; if only task575 degrades, inspect added goals; if both execute,
compare DExplore supervision/state coverage next. Use existing BC weights as the
cheapest test, not a new train. Budget900s per worker/1200s total, no videos/cache
or new checkpoint. Outputs only outputs/wam-rl-policy-v3/unidex-bc-goal-20261020-r1/.
Zero new goal columns do not demonstrate learned task10 benefit. Initial-state
matching is checked; unmatched scenes cannot support paired method claims.

BC-only run completed at e9077e38: each arm64 first native episodes,0 optimizer
steps, explicit bc_step46000, ppo_checkpoint=null, optimizer_constructed=false.
All64 initial robot300/hand/pose/goal/full root arrays match bitwise; WAM12000
and its original geometry are held fixed. Capacity/weights come from full-data
self-trained BC; no RL weights are read.

| BC arm | Native-ever | Ever-held60 | Final-stable | Final-position | Workspace failure |
| --- | ---: | ---: | ---: | ---: | ---: |
| Scheduled565 | 32/64 | 25/64 | 25/64 | 28/64 | 0 |
| Task575 zero expansion | 35/64 | 25/64 | 24/64 | 28/64 | 0 |

Final-stable means object within5cm of goal and lifted10cm for the final60 native
steps (~1s). Native-ever is the official sticky success flag, not final stability.
Elapsed51.84s/75.75s onGPU6/7. Both BCs have real partial execution success on
this16-TRAIN-object/single-seed panel; this is not unseen-object generalization
or evidence that the untrained task columns improve anything. Do not compare
these counts causally to prior PPO runs with different WAM weights/reset draws.

Decision: UniDex BC is an executing positive control. Compare DExplore's data,
startup sampling and closed-loop state coverage against it, while retaining
the independently identified phase-gating issue. This alone cannot exclude
all DExplore data problems or identify their unique cause. No new long training.
Evidence: unidex-bc-goal-20261020-r1/comparison.json, both evaluation.json,
bc-evaluation.json and initial-state.npz; explicit --bc-evaluation rejects any
RL checkpoint before runtime initialization.138 tests pass, all GPU jobs ended.

#### Historical PPO-only results

r1 at90118e05 completed both64-scene first-episode evaluations: baseline565
native-ever46/final-stable32, task575 native-ever45/final-stable30. Only53/64
scenes have bitwise identical initial native300/pose/hand/goal/full root. The
common53 give38 native-ever and27 final-stable in both arms. The one task575
workspace failure is among the11 unmatched scenes. comparison.json preserves
both all-scene counts and the diagnostic intersection; no full-panel paired
benefit/degradation claim. PPO10000 is a retained endpoint, not the previously
best8500; the purpose is an executable interface control, not model selection.

The new Shadow adapter initially supplied the existing6D force sensor vector.
Installed Isaac Gym documentation (programming/forcesensors.html) specifies
external/contact AND internal solver forces; this is not pure contact. r2 at
63b0f58a only adds gating evidence and is historical diagnostic. Corrected9db09ec4
acquires/refreshes net_contact_force_tensor at the five distal bodies, matching
the contact-force meaning used by Inspire. Native actor300 sensor inputs remain
unchanged; only TaskController feedback changes. No threshold was changed.

r3 with actual contact forces completed in66.39s onGPU7:47/64 native-ever,
34/64 ever-held60,29/64 final-stable,0 workspace failures;0 optimizer updates.
Only11 environments ever enterTRACK.23 of29 final-stable environments never
enterTRACK, and no environment confirms a path milestone. WAM/task10 point goal
max discrepancy2.10e-9m. These are independent reset realizations, so r2/r3
success counts are not a matched policy comparison.

Of1595 near-finger+contact+moving observations,1209 exceed the relative-hand
speed limit and1262 exceed the relative/object-speed ratio test. Counts overlap;
they are not independent causes or grasp certificates. Among final-stable
environments, only1204/5800 observations have two fingertips within18mm of the
sampled object surface. This points to geometry/observability and conservative
motion gates, not proof of a single threshold fix. Also the scheduler8mm arrival
tolerance and two-endpoint0/1 path milestones differ from native5cm success.

Decision: do not attribute all Inspire failure to its dataset or scale training
yet. Align task-stage/readiness and success semantics on the known executing
Shadow control, then consider matched learned575 training. These zero task
columns do not prove that task10 or WAM independently improves performance.
Evidence: unidex-task-goal-20261020-r1/comparison.json and
unidex-task-goal-20261020-r3/feedback-analysis.json, task-diagnostics.json,
evaluation.json, task-migration.json. All bounded GPU jobs ended.137 tests pass.

### Historical implementation

用户ref1审计修订。Blocker：相同实测状态/终点而局部路径不同，在H4启动或τ被移除时，
旧577输入逐位相同；实际BatchedInspireInputs回归在旧代码失败。另复现静止接近物体而
无路径进度可每拍获得0.775奖励。优先补任务信息，保留全量数据和Shadow网络/PPO配方。

输入v2：physical312 + world tau264 + H4有效位1 + local task7 =584。
task7为当前物体坐标的局部goal平移3/rotvec3、confirmed_progress1，索引577:584；
tau仍312:576、valid576，已有物理与τ列身份不移动。task与WAM预测独立且从启动可见；
无WAM预测路径同样读取task7。current BC为physical312+task7，共319维；
scheduled初始化复制物理和task列、仅新增tau/valid列置零，保持当前BC函数。
task固定尺度平移.05m、旋转.35rad、progress1；物理归一化仍TRAIN-only/std下限.05。

完整660/50物体及589 TRAIN/71 DEV不变。只追加约4.64MiB task字段，
复用旧不可变物理/τ缓存与WAM80000身份，不重新推理神经缓存。
两阶段仍各50000更新、同1024/1024/512/512 ELU、同采样及DEV选择规则。
旧PPO已在972完整保存停止，不能把旧577/reward checkpoint直接当584普通resume。
新BC后PPO作为新合同运行，不能与旧972当成单变量连续学习曲线。

奖励v2：保留Shadow手掌/指尖接近项与高度项，将unfinished路径项改为局部目标点误差
-2*min(goal_point_m,.05)，去除持续的+0.9“贴路径”奖励；保留2*正confirmed进度增量。
完成bonus只在有序完成且当前实际SE3终态达标时支付，不依赖sticky complete单独判定。
为避免删除正偏置后主动越界提前结束，失败终止代价由非终态负项界限推导：
(.5*3+.5+2*.05)/(1-.96)=52.5，不改变PPO gamma=.96。
任务成功为有序完成后真实终态SE3持续1秒且不越界；高度10cm仅保留代理，
不作为放置终态要求，不宣称接触/释放证书。记录ever事件、当前终态、最终保持分别输出。

原canonical命令重放：当前纯物理host，不使用BC、WAM或PPO。先选cup/apple/cube三条TRAIN，
比较初态、第一拍PD、8拍手/物体偏差及全程首个分歧，不要求PhysX跨执行逐位相同。
初态/第一拍PD不对应则阻止长PPO并定位；短期对应但长时分歧需独立报告，
不能把全部闭环失败归因BC。首条cup初态q/hand/物体位置差0、第一拍PD差0，
8拍物体最大3.05µm、手平均最大3.72mm；全段物体差可达60mm，不称全程复现通过。

资源：模型graspenv/Torch2.4.1+cu121，物理dexplore_v120_train/Torch2.0.1+cu118。
准备与两阶段BC<=2小时、<=2GiB；CPU几何字段构造不是神经实验，不占GPU。
本地最多6卡、留出GPU6/7；重放只用GPU1–3，BC用GPU0。
PPO若工程检查通过，继续全局原48h截止与10000更新上限、滚动保存和20GiB磁盘保护。
不跑Shadow匹配KL续训，本次只补显式KL入口/身份；因果归因仍为未来Evidence。

产物：outputs/wam-rl-policy-v2/inspire-task7-bc-20261019-r1/cache/、
outputs/wam-rl-policy-v2/inspire-task7-fit-20261019-r1/、
outputs/wam-rl-policy-v2/inspire-task-audit-20261019-r1/。
实现提交45eb9837，BC冻结该提交；后续Shadow KL接线/重放审计提交53150deb。
85项Task测试通过。全量task字段准入检查通过：660序列、165677命令均有限，
50物体各前20拍的真实BatchedInspireInputs与离线task字段最大误差0；该检查不运行神经网络。

两阶段BC均已完成50000更新，单GPU0约309秒。current最佳DEV MAE0.03892193；
同一800行物体均衡DEV选择面板上，零扩展初始化与条件best均为0.03892196，
选择初始current扩展点，tau/valid权重为零，但新增task权重保留。
这没有优于旧312输入的离线MAE0.03850862，不能称局部任务输入已改善抓取或WAM已有收益。

原命令重放三条TRAIN的2环境结果：第一拍PD/初态q/hand/物体位置误差均0；
cup/apple/cube前8拍物体最大误差分别3.05/28/1.27µm，手点约3.72/2.72/4.51mm；
全程物体误差分别60.3mm/2.877m/83.4mm。因此只通过短程接口检查，长程动力学分歧仍记录。
进一步恢复cup原采集的16环境、批成员、seed120和fixed motion配置，重放所有16环境
共同可用的158拍：目标cup的q、手点、物体位置和PD目标全前缀误差均为0。
这是原canonical控制链可复现的证据，不证明不同采样/批配置的长期轨迹必然一致；
不能仅凭2环境漂移判定动作映射错误，也不能把16环境结果称为BC抓取成功。

原生工程检查inspire-task7-native-check-20261019-r1已完成：冻结53150deb，
64环境、50物体、4次官方PPO更新、80次optimizer step，2048交互；错开reset通过，
动作episode错配、PD目标误差、动作均值误差、logprob回放误差均0。
在线共享TF32对完整FP32最大tau差0.564mm，低于1mm门槛。
该检查只有3个终止（含2个强制reset），0任务成功，不能作为效果评估或长训起点。

新BC确定性首回合r2：50 TRAIN物体/64环境、300控制拍、0 optimizer step；
52个已结束回合中0任务成功、0最终高度保持、11个workspace失败，12个首回合尚未结束。
不能将其写为完整64回合成功率。r3将上限延长为1200控制拍，补齐完整回合；
使用GPU0/1、最多1200秒，不改变策略、输入或任务判定。这是是否继续扩大训练的Decision，
当前保持旧六卡停止，不凭工程通过重开万轮长训。
BC评估r1在启动时因未显式传--updates 0被拒绝，0物理步/0优化；正确重启为r2，
保留失败manifest，不将其计入评估。

r3完整评估已完成，484控制拍覆盖全部64首回合、50 TRAIN物体，0优化更新；
全部记录的初态字段与r2逐位一致。任务成功0/64、workspace失败17/64，
最终无越界高度保持1/64，不能把该高度代理写成完整路径任务成功。
结果与初态核对保存于`inspire-task7-bc-eval-20261019-r3/summary.json`。
决定：输入/在线/原生PPO工程检查通过，但新版BC仍非可靠控制底座，
保持旧六卡停止，暂不启动584合同的万轮训练；先定位BC闭环与任务条件的信息充分性。
task7解决明确的局部物体指令缺失，不保证相同物体任务下不同抓取/操作意图均可辨识。
未恢复手参考，未改变官方gamma/nsteps/noise数值，未进行Shadow KL匹配续训，
不将单seed TRAIN小面板、离线MAE或tau权重变化升级为WAM净收益结论。

后续bug排查（inspire-bug-audit-20261019-r1，修复3a542933）：

- 64首帧离线/在线输入最大差2.98e-7、同一BC动作差4.17e-7；
  对全部660序列存档body state重算physical248，裁剪后最大差1.073e-6。
  未发现该物理切片或首帧构造错误。
- 实际回归先失败：静止物体的300命令在第240拍scheduler超时后，缓存没有终止准入标记。
  修复后从超时状态开始禁止采样动作；current/scheduled训练和DEV选择统一使用task-active
  mask，scheduled额外相交既有workspace mask。旧缓存不静默默认全True，旧BC仍可历史评估。
  全量新小缓存为inspire-task-active-20261019-r1/cache，原始数据/神经tau缓存保持只读。
  TRAIN超时后12918/150044、DEV749/15633，总13667条；589/71全部源序列仍保留。
  87项Task测试通过，没有据此启动新BC/PPO或声称修复已经提高成功率。
- `check_inspire_demonstration_contract.py`逐序列确认canonical action标签与缓存一致，
  并区分原命令的任务相容性：TRAIN589中仅1条曾确认完整路径，2条末态在8mm/.12rad/8mm
  终点容差内；二者交集为0。DEV为0/1/0。TRAIN末端位置误差中位4.183cm。
  路径进度统计使用pre-action帧，末态误差使用terminal观测，不将其伪称完整1秒保持测试。
  同时50物体的完美reference自身重放全部完成，不能把真实示教偏离直接叫scheduler算术bug。
- 扩展重放指标到dq、physical248、物体速度。s1_cup_lift恢复原batch005/16环境/seed后
  仍有长期分歧：首次q>1e-5在71拍，全段物体差30.4mm、手点28.2mm；短程PD一致。
  此结果不推翻此前s10_cup_drink_1原batch000的158拍零误差，说明逐位一致不能外推所有批。

用户随后提供共享会话，要求以末尾关于统一TaskSpec、当前阶段及补力/形状观测的讨论为准。
当前只讨论新实现，未恢复721手参考、未改正在运行的WAM-v5、未启动新训练。
DOF force在原生仿真已启用但当前660存档未保存；连续body/object接触力已保存。
新任务协议应从完整原始660重新重放准入，不能将旧协议超时后的命令永久标成坏数据。


### 用户批准的统一 TaskSpec 与实测执行接口：第一批实现

本轮是工程 Blocker 处理，不作为训练效果实验。run_id 为
`inspire-measured-task-20261019-r1/r2/r3/r4`，产物位于同名 outputs/wam-rl-policy-v2 目录。
代码起点45ac5e9e，后续命名列/τ布局与事件合同对齐94b98f76，阶段确认修复b34cac0d。
外部WAM-v5的并行Git提交不代表本轮改变了其训练；实际运行版本另见budget/manifest。
未启动新的BC/PPO，旧六卡PPO仍停止，旧checkpoint与全部660原始源保持。

实现 TaskSpec、TaskScheduler、FeedbackObserver、task10、机器人adapter、命名列扩展、
PointNet shape64接口和增广阶段潜势奖励。Shadow current310/scheduled575；Inspire原312
加DOF力18、连续接触力15、shape64、两个力有效位为robot411，current421/scheduled686。
actor仍为1024/1024/512/512 ELU；新增列初始权重为0，原裁剪前缀不变。
Inspire τ从旧交错base/tip顺序迁移到共同palm/bases/tips时，归一化与首层权重一起置换。
事件字段与WAM-v5新增可选分支的next-contact-release-plan.frame30.v1数值合同对齐，
已知NONE与unknown mask区分；旧v4/v5权重未训练事件，不能据此宣称事件已接入当前模型。

原生力检查r1在GPU1、2环境13步加一次局部reset，13.17秒；r2用dense512交互表面复核，
13.99秒。两次均验证reset清零/invalid、step后valid、邻居状态/力/clock不变、
canonical episode错配0、PD最大误差0。686维只做输入格式检查，shape64用显式零测试fixture，
没有用随机/未训练形状特征控制机器人。模型graspenv/Torch2.4.1+cu121，
原生dexplore_v120_train/Torch2.0.1+cu118。

同轮新增力采集复用原官方teacher，仅作数据生成器：GPU0，cup和airplane两条，共426状态/
424命令，28.34秒；DOF18、contact validity与frame_tick对齐，首帧invalid且DOF力为0，
后续valid，canonical/applied逐位相同。DOF力绝对最大分别10.05、41.63（原生广义力单位），
说明原始[-5,5]裁剪不适用于新增力字段。全部新产物约数MiB，无大型缓存、视频或新模型保存。
全部660旧归档仍缺DOF力，不能将这两条新执行的力值拼接到旧动作；完整新BC仍需同轮全量数据。

r2完整660归档重放478.46秒：保留589 TRAIN/71 DEV、原始帧号及全部事件，77条存在多个
接触区间。全部参考尾部为计划释放，不能据此证明真实放置/支撑。该次暴露新时序bug：
几何tracker只在确认抓取后启动，丢失此前已观测的路径起点，导致全部确认进度0。
最小延迟确认回归先失败。r3修复时进一步明确用户要求的ACQUIRE cursor冻结，提前停止保留记录。
最终实现从首帧记录起点，抓取候选期间仅保留临时几何证据；实测抓取确认进入TRACK后才提交，
候选中断则撤销临时后缀，恢复失控时不推进。路径预算排除acquire等待，保留真实观测dt的穿越保护。
108项Task测试通过，包括外力移动不能积累成抓取前任务进度、候选撤销、已知NONE与实际WAM-v5
编码器兼容、旧Inspire τ列迁移、终止潜势望远镜关系。

另做50个真实物体参考的几何/阶段检查：用完美参考位姿和合成抓取/释放/支撑反馈，
50/50路径及合成任务完成，9.71秒。这是状态机工程验证，绝非机器人成功率。
真实teacher仍与严格路径容差不相容：例如s10_cup_drink_1首个milestone在原始47帧，
所有实际状态中最近位置状态也差15.7mm、旋转0.202rad、平均表面点16.8mm，
超过8mm/.12rad/8mm；即使几何readiness全True，该序列也不能确认第一个milestone。
因此时序修复不等于已解决任务标签与示教相容性，不能通过偷偷替换参考路径或放宽容差来制造结论。
完整r4重放结果另行追加；支撑反馈仍需原生接线，shape encoder未训练，奖励和新观测尚未切入
正式PPO入口，故不恢复长训。


真实旧BC权重/真实DEV输入的GPU0迁移检查（r4/model-migration.json）也通过：
Shadow565→575，32帧原始/迁移后动作最大差4.77e-7；Inspire584→686，32帧最大差1.79e-7。
不保存新checkpoint，不将函数保持检查解释为PPO迁移或新观测已学会使用；
Inspire新增的力/shape列仅为格式fixture，零扩展使这些列尚不改变旧策略。


最终r4完整重放已完成（b34cac0d，624.50秒）：660条/589 TRAIN/71 DEV，581条曾进入TRACK；
8条确认少量几何进度，最大2.26%，0条路径完成、0条任务完成。32条路径停滞超时、
2条acquire超时，626条在归档结束时尚未完成。均未进入RELEASE，因此不能把此轮用于
验证真实释放/支撑效果。此结果是已保存teacher执行与任务合同的重放，不是新BC或PPO成功率。
77条多次计划接触完整保留；全部660缺DOF力的边界仍明确。
起点/延迟确认bug已有回归锁定，但真实示教的任务相容性仍是独立blocker；下一步优先
核对已知参考与实际示教目标的构造/可实现性，再推进完整同轮力数据、shape与两阶段BC。
不在本轮私改原始参考、容差、冻结数据或恢复10000轮PPO。

验证：108项Task测试通过，22项governance测试通过，所改Python/结构化文件/Markdown链接检查通过。
全仓verify仍失败于trajectory-policy的6张历史v2实验卡缺probe_id等必填字段；本轮未修改
这些无关历史卡。索引检查另有并行WAM-v5新增实验卡尚待生成的差异，不属于本轮新卡。
本轮未新增experiment ID或消耗validation seed。所有资源任务均已结束，GPU归还，
本轮四个run目录合计约3.8MiB，无模型checkpoint新增；数据/指标不足不升级为方法结论。


### 用户纠正：任务必须来自teacher rollout，先启动六卡补全数据

此前将DExplore训练参考reference_obj/reference_current当作本任务指令是本轮实现错误。
新TaskSpec/goal应由checkpoint teacher在同轮物理执行所得object rollout构造，计划交互
标签也应从rollout实测证据离线提取；训练参考只允许留作teacher内部与诊断，不作为目标
或质量筛选依据。此前0/660不再用于阻止该任务路线，它反映的是错误目标定义。

run_id=inspire-teacher-rollout-20261019-r1。先用现有官方teacher采集器、GPU0–5并行，
每批16环境，完整660序列，继承原589 TRAIN/71 DEV序列划分，seed120+batch_index。
同轮保存q/dq、DOF力18及有效位、全部body状态/接触力、object状态/接触力、hand11、
canonical/applied动作与PD目标、原始tick、512点CAD几何；不拼接旧执行的力。
冻结采集器源码，2小时/4GB输出预算、20GiB磁盘保护，GPU6/7不启本任务。
原始采集先保留，完成后按既有物理/rollout审计筛选与标记，不按reference误差或旧scheduler
超时裁剪teacher动作，不启动PPO。先启动采集进程，再修正任务构造与新的字段准入。


### Teacher-rollout纠正实施完成：全量660回采、筛选与新state读取

最终raw run=`inspire-teacher-rollout-20261019-r5`；审计run=
`inspire-teacher-rollout-audit-20261019-r2`。先按用户要求启动六卡，才修改任务/输入代码。
采集复用官方checkpoint teacher，只生成数据，未训练BC/PPO或修改teacher权重。
TaskSpec/goal改由同轮实测obj轨迹与接触导出；训练reference字段不参与任务/筛选。
旧reference路径/容差不相容的0/660已被用户纠正为错误目标定义，撤销其作为当前路线blocker的判断。
旧神经τ缓存和物理缓存含旧目标条件，禁止无声复用；新的extended-force corpus也不能走
会丢force/shape/phase字段的旧319 BC入口。

采集时序：67822fc5六卡初启，首step部分PhysX非法访问；3d61ed66复用已验证的一次reset提交，
单独未消除；4f7034b8复用物理host8M/5缓冲，原故障批单独16条完成43.93秒，但并行仍会复现。
53dc0815加入每批最多3次保留日志的重试，让健康worker继续，r4最终保留596条。
四个剩余批次跨GPU仍失败。实测aggregate物体+桌面最多6个shape，未超过旧22容量，
因此不能将崩溃归因于容量不足或宣称reset/buffer是唯一原因。
原故障批的两个8环境半批均完整通过（32.97/36.16秒），单独bunny也通过21.56秒。
9239f673新增仅继承已完成同轮receipt、补采missing membership的入口；r5用GPU0–5、8环境
补齐缺失64条，122.94秒完成。596条通过只读链接复用，原始split仍589 TRAIN/71 DEV。
这是数据生成batch配置调整，不是严格配对物理对照；实际每条来源/seed/环境数由receipt追踪。
r4虽然整体FAILED，其596条是r5必要依赖，已加retained_by.json，不可清理删除。

原始660/50物体全部字段工程PASS，165677条canonical命令。每条同轮保存q/dq、DOF force18/
两个force有效位、完整body状态和连续contact、object位姿/速度/net force、hand11、action/applied/
PD/tick与CAD512/法线。reset force无效和零值、后续有效，DOF大于5的真实值不裁剪。
复用v2/audit_current_rollout的独立工程验证，通过include_reference=False去掉不相关参考条件；
新增force schema/timing、实测失控/unsupported loss、几何穿透、过大单步及可对齐旧人工标注。
失败50帧前缓冲与H4/future24窗口保护保持，BC使用连续安全前缀，任务也裁到同一实测前缀终点。
原始数据完整保留，训练从sidecar mask剔除。旧人工判定只在同帧q/obj/hand对应时复用，
纯tracking_error不作为新准入依据；接触/耦合/支撑仍明确是启发式。

筛选后563条前缀、60529命令；TRAIN501条/54544命令/50物体，DEV62条/5985命令/42物体。
TRAIN362条前缀含23043帧耦合、覆盖46物体；另4个物体当前保留前缀只有接近段，不宣称抓取覆盖齐全。
非互斥异常帧：失去耦合且无支撑41294，过大单步1997，桌面穿透476，可对应人工不适用/待定259。
这些是保守训练准入，不是成功率或失败真值，也没有按训练参考误差筛选。

TeacherRolloutCorpus及MeasuredInspirePolicy新增：raw measured347+CAD512，joint PointNet形状64
拼成robot411+task10=current421；加入规范τ264+valid为686。actor保持Shadow的
1024/1024/512/512 ELU，力不受原始312裁剪；shape64是BC可训练特征，不伪造为已采集的传感器字段。
114项Task测试通过；GPU0八个真实物体episode的421/686输入与forward均通过，2.75秒，
新增力到首层保真误差0，0优化。形状梯度测试通过，但未训练shape或新BC，没有闭环成功结论。
数据和审计均已完成，GPU已释放；后续将新数据/encoder/actor接入两阶段BC与官方PPO。


### Fresh UniDex BC and Inspire diagnosis protocol (2026-10-20)

Decision: distinguish a broken reusable BC learning/execution path from
Inspire sampling, demonstration coverage and closed-loop drift. User explicitly
requests new UniDex BC training, rather than another old-weight evaluation.

- run_id: unidex-fresh-bc-20261020-r1; task wam-rl-policy-v3, branch main.
- Freeze the committed v3 trainer before launch; each child records git_commit.
- GPU6, graspenv Python3.8/Torch2.4.1+cu121, seed241, no loaded policy weights.
- Current300: full existing current cache, 50k updates from random initialization.
  Scheduled565: full predicted-WAM12000 cache, 50k updates initialized exclusively
  from this run's newly trained Current300 best. No PPO updates.
- Both stages: Shadow 1024/1024/512/512 ELU, AdamW3e-4/wd1e-5, batch2048,
  standardized L1, half startup tick<8 and half full TRAIN sampling.
- Existing immutable caches: 1,536,897 TRAIN/354,761 DEV rows; no new tau cache.
  Two-hour deadline per stage, <2GiB new artifacts, 20GiB free-disk floor;
  GPU4/5 remain idle. Selection uses the existing fixed object-balanced DEV panel.
- Evaluate new Scheduled BC in the same 16-object/64-scene native TRAIN panel,
  seed246, WAM12000, corrected reset/common workspace/position_only. Report
  ever-native and final-held separately. This trains 565, not new task10 columns.

Inspire next probes use corrected support admission v2 and select TRAIN lift
motions by teacher coupled-prefix coverage before looking at student outcomes.
Compare canonical teacher command replay, teacher-state BC errors and autonomous
rollout. Then compare uniform versus physical-time-matched startup sampling,
keeping corrected corpus, random initialization, model, seed and budget equal.
Old support-v1 tau is excluded from new support-v2 training.
