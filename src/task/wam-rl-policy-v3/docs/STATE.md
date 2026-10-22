# 2026-10-23 v13已完成：联合输出依赖成立，泛化与GT执行上界未建立

| 结果解读 | 结论 | 下一步决定 | 来源 |
|---|---|---|---|
| 旧seen A1 35/40，WRIST/FINGER/TIME/T0各0/40 | 固定链依赖腕指联合输出；TIME无预测接触帧 | 保留联合接口，不宣称三项独立τ机制 | Hermes分解；本轮实测 |
| A3/A4两子集各0/40，全部未close | 源GT构造资格已过，R/执行器上界未过；WAM训练源兼容性未排除 | 先资格化接近/闭合，不宣称WAM独有信息 | Hermes对照；外审实现资格；信息解释未一致 |
| 17中止=9门槛跨越+7完全桌下重入+1边缘重入 | 12 approach/5 close；不是统一压穿原因 | 逐例支撑/掉落诊断，无“其他”合桶 | Hermes逐例；外审几何/形状 |
| 原unseen3/40且8序列均R TRAIN；另v11 A1/source=3/14 | 未证明训练未见或宽覆盖；部分任务有另一链见证 | 三臂RL HOLD、不训v6、不做完整RELEASE | 外审可推广性要求 |

[完整结果、17例与一条重复命令](grasp-v13-mechanism-20261022.md)；[卡](experiments/probes/P-20261022-grasp-intent-mechanism.md)。新增正式320回合147811controls；含工程实际153675–153677，0updates。raw/干预重放/R权重重算各320 PASS，初态40+80组、9源构造资格PASS；最终回归316 passed。最终产物 `outputs/wam-rl-policy-v3/grasp-v13-comparators-20261023-r3/summary-final.json` 与 `delivery-audit.json`。以下为并行路线与历史状态，不覆盖其证据边界。

# 2026-10-23 source 门控/引用离线分诊：先修接口，不开 R6 全批

审计报告与可复跑工具见 [`source-gate-triage-20261023.md`](audit/source-gate-triage-20261023.md) 和
[`audit_source_gate_triage.py`](../tools/audit/audit_source_gate_triage.py)。只读取 R3/R4/R5
已完成产物，没有启动仿真或 GPU，也没有修改 `source_ceiling.py`、`run_source_ceiling*.py`。

- R3/R4 40 个注册 reference 的物体坐标 frame0→首次有效双指力帧，位置差 p10/median/p90
  为 0.9801/1.1225/1.2495 m，旋转差为 1.5380/2.0256/2.8760 rad；按每例
  `wrist_gap_m`/`angle_tolerance_rad`，位置、旋转和双门命中均 0/40，且 40/40 均超过
  各自 5×门限。这支持引用 approach→grasp 几何不自洽的实现缺陷判断。
- 需要区分 runtime intent：A0 `plan=None` 时 `grasp_pose_object` 直接复制当拍
  `approach_pose_object`，R3/R4 的 intent 内部两 pose 差为零；历史 R3/R4 实际阶段门还要求
  `near_fingers>=2`，或腕位置≤`wrist_gap_m`且指尖 gap≤`tip_gap_m`，没有使用
  `angle_tolerance_rad`。
- R4 末控制时刻 31/40 仍在 approach、9/40 已进入 close/lift。31 个未推进例中，20 个腕
  位置+旋转静态门已满足但指尖门未满足，11 个腕静态门也未满足；R4 的 5 个
  `contact_without_lift` 属于已过阶段门但 live coupling/lift 认证未完成。
- R5 source 证据与 live success 明确分离：39/40 有 source object-contact 帧，raw/object
  帧总计 4724/4466（p006=71/69），但 manifest 的 `source_contact_is_live_success` 固定为
  false；运行时 25/40 曾 measured contact、13/40 曾 lift/hold、21/40 阶段 contact。
  按现有 summarizer，R5 死因为 `stage_contact_not_certified` 14、`contact_without_lift` 19、
  `physical_table_penetration` 6、`lift_without_continuous_hold` 1。

决策：不启动 R6 全 40。先在 3–5 个已有失败例上做一个最小接口修复（使引用几何/入门判据
自洽，同时保留 live contact/coupling 认证），再小样本回放；结果通过后才重新决定全批。

# 2026-10-23 v14：source前三版0/40，R6握力目标修复待物理验收

| 结果解读 | 结论 | 下一步决定 |
|---|---|---|
| R6≥32/40且安全门有效 | 本固定20物体面板工程资格达标 | 才做同预算/后段WAM比较，不单凭差距归因数据量 |
| 提升但<32/40或仍0 | 仍未修好，不以软件PASS代替物理 | 继续修source，三臂HOLD/不训v6 |

原source14/40、12joint abort；R3/R4/R5各完整0/40（Wilson95 0–8.76%，各vs原Fisher3.07753e-5），均0joint abort而非改善，table 1/0/6。
三个新版本共71542controls、120raw/120初态核对一致；报告[全部负结果、diff/复现/逐物体](audit/source-ceiling-fix-review-20261023.md)，[预注册](experiments/probes/P-20261023-source-controller-ceiling.md)。
R6闭合改取产生抓姿状态的前一拍PD目标，不能用受压实测q代替；命令间限速保留握力预载；source key优先已耦合/离桌完成抓姿（35/40有证据，其余照留）。
远离对象时加快命令、近处仍降速；目标URDF内余量、原actual gate/成功/后段不变，实际名义越界另报。26专项、全量302 passed/0 failed/0 skipped、21/21 mutants killed；不是物理通过。
同40/≤24000新控制/0updates，四版总上限96000；GPU仅空闲4/5/6逐卡等待前序，未见泛化不作门槛，无WAM对比/无完整RELEASE/无大训练。

# 2026-10-23 v11正式新物体覆盖：120/120完成，A1=3/40、A2=0/40、source=14/40

| 结果 | 结论 | 下一步决定 |
|---|---|---|
| 新20对象×2序列 A1 3/40 vs同R零τ0/40，Fisher .240506 | 未通过贡献/宽覆盖资格 | 不推广新几何控制为合格executor，不重训v6 |
| source接近→同后段14/40，12个同初态A1失败 | 部分任务有另一链物理见证，非GTτ→R上界 | 接近/闭合受压卸载优先，复用见证，不重复并行v13 |
| A1名义目标非法0但实际限位14，其中接近/闭合13 | 静态准入不是动态安全 | 保留真实gate；三臂RL HOLD，软件PASS不等于物理修好 |

[完整结果/复现](grasp-v11-coverage40-results-20261022.md)；[卡](experiments/probes/P-20261022-grasp-coverage40.md)。
52839controls、0updates、无中断/换样本；120raw一致、40组三链初态/40干净A2通过，当前v3回归269passed/0failed/0skipped。
新尺寸阶段+15mm次要指标仍为3/0/14；新A1没有“末态持握但未到位”例，主要失败在接近与闭合。
宽覆盖没有被三处修改解决；旧v10 seen35/40只限旧冻结控制/15mm协议，不混算或重标。
精确原生目标投影次数未保存，报告null而非0；完整GTτ上界与预训练样本效率仍未建立。
独立复跑待Hermes/GPT裁决；并行v13组件/GT验证已在推进，本线程不再起重复实验。

# 2026-10-22 v10：160回合与raw复核完成，seen35/40、原未见3/40

| 结果 | 结论 | 当前决定 | 来源 |
|---|---|---|---|
| seen A1 35/40 vs同R零τ A2 0/40 | 原15mm协议局部目标通过，Fisher2.27289e-17 | 保留R/WAM和消费修复 | Hermes冻结n40门 |
| 原未见A1 3/40 vs0/40，限位0但桌面穿透17 | 未见资格未通过，不能只以限位减少报通过 | 新几何合同/覆盖诊断，三臂HOLD | 外部审计跨初态要求 |
| 160raw一致、80初态/80干净A2均PASS | 有效局部证据，非独立物体泛化 | 不重训v6；旧15mm结果不重标 | 本线程复核；两方尚无独立复跑 |
| BC DEV上升、残差PPO前后0/2 | 未建立恢复泛化或物理收益，首步KL超target | 不扩训、DAgger后置 | Hermes的BC/残差方向，本线程有限pilot |

[交付报告与一条命令重复](grasp-handoff-results-20261022.md)；
[实验卡](experiments/probes/P-20261022-grasp-interface-repair.md)。实际82347controls含881中断，
有效160回合81466，0updates。新几何/SE3/原生目标准入已实现；36/36扩覆盖与raw复核完成，
A1/A2/A0总2/12、0/12、4/12，三条source-only严格见证；不冒充每子集n40资格。
BC漂移恢复400updates：train .422871→.0910735，DEV .386376→.460369，过拟合。
有界残差PPO826controls/2updates，确定性before/after均0/2，三阶段2669controls、raw6/6及
reward分项重放2/2一致；更新后KL .611887/.165042未达硬KL资格。不扩训，不恢复三臂，
不重训v6、不做完整RELEASE资格，DAgger后置。详见[残差卡](experiments/probes/P-20261022-drift-intent-residual.md)。

正式接口未见扩覆盖已另注册[每链n40卡](experiments/probes/P-20261022-grasp-coverage40.md)：
20个排除旧接口暴露的对象×2不同序列，A1 WAM/同R零τ A2/source-PD接近A0各40；
120回合≤72000controls、0updates，算法快照冻结于0e8c8214。前序小预算检查已完成，GPU4/5/6由此独立批次接续；
监督PID3631768，确切命令/预算/中断记录在`outputs/wam-rl-policy-v3/grasp-v11-coverage40-20261022-r1/`。
尺寸评分是新协议，旧15mm结果不回改；A0不作GTτ→R上界，n12工程检查不并入本分母。

# 2026-10-22 v9：320回合完成，seen局部收益成立、unseen及GT资格失败

[预注册、diff、统计与复现](experiments/probes/P-20261022-grasp-interface-stat40.md)：每臂每子集40，
seen/unseen分开。固定20s+S2c末态hold≥1s+物体目标≤15mm；seen A1=15/40，Wilson95
[.24223,.52968]，A2/A3/A4各0/40，[0,.08762]；A1对三臂Fisher p均1.21236e-5，
CI不重叠，达到seen局部门槛。A1末态hold31/40、最长连续持握中位17.233s，精确目标15/40；
不能把抓得住等同到位。unseen八个新接口物体各5seed，四臂各0/40，全部比较p=1；
A1新物体40/40关节限位中止，lift0。未见稳健性不成立，不把8物体说成40独立物体或预训练未见。
A4两子集均0/40、未建立上界，未触发重训条件；不能按WAM>GT排序解释τ质量。A2旁路R、
A3保留R但参考条件不同，现证据只限参考消费链局部收益，同R无τ的大样本因果隔离仍缺。
原型是HOLD，release不适用；不声称完整RELEASE成功率。P1 RL仍HOLD，不重训v6。
修正stage palm测量角色：实际URDF掌部mesh代替腕原点，S2c阈值不变；同真实S2b源四阶段
全拒绝→全通过、最长仍2s。仅stage测量适配，R/控制器/安全gate未改。正式代码冻结07d4db08。
三卡完成108+108+104=320，原始状态独立重算320/320，80/80配对九字段最大差0，实际PD
重构最大3.11642e-7；手点平移及发布标签完整CLI故障注入均被拒绝。237 passed/0 failed/
0 skipped。101570/192000controls、0updates，三worker评估wall之和9225.165s（2.563GPU·h），
最长53.49min（不含模型初次载入），无技术失败/partial/补抽。GPU4/5/6全部释放至2MiB。
全量`outputs/wam-rl-policy-v3/grasp-v9-stat40-20261022-r1/summary.json`含阶段、误差全向量/
分位数、12个Fisher、Wilson、配对补充与原始路径。下一步先修未见R/PD有限关节动态合同、
验收A4，再谈WAM；seen同时需补目标精度，不能由显著p值宣布稳定工程资格。

# 2026-10-22 v8：消融完成；5/5未稳定复现，资格退回HOLD

[完整diff/预注册/复现/全量结果](grasp-interface-v8-controls-20261022.md)。同一冻结v7消费器，
消融WAM2/5（hold[1,1,0,0,0]s）、零τ/无τ/固定几何模板各0/5；不把此局部差异当普遍τ
贡献或样本效率证明。旧5/5保留为真实旧run观察，撤回“稳定≥4/5”的工程资格判断。
新机器人初态airplane/apple/cubelarge/duck/mug为0/5、2例关节中止；五例是旧机器人BC val
且未用于v7调参，3例同源人类序列仍在WAM TRAIN，不能称严格未见WAM序列/物体泛化。
复用S2d整段live-object锚定、S2b205实测状态真实窄成功源（最长2s），同R/30Hz/阶段机/PD：
GT0/5，同批WAM1/5；未建立GT上界，不按排序推断τ质量。GT全approach，关节中止2/5，
两例在参考有效时退出。source-PD动作未当GT正控制回放。
35/35完整回合从原始物理状态重算连续持握一致；累计短区间的故意mutant被拒绝。
7579/10500controls（含KeyError中断partial3）、729.079执行秒、0updates；230 passed/
0 failed/0 skipped。代码入口已修元数据同名覆盖，r2携带原7例只补剩余13例，未筛失败。
GPU5/6已释放，4未用；三臂RL继续HOLD，不重训v6。下一步先修接近抓姿/预形与有限关节
可执行合同，验收同τ通道GT稳定正见证；当前不追加实验。最终全量summary在
`outputs/wam-rl-policy-v3/grasp-v8-summary-20261022/final.json`。

# 2026-10-22 v7：三项意图+测量相位，窄任务5/5

[最终接口、diff与复现](grasp-interface-v7-20261022.md)：保留腕抓姿、手指预形/闭合目标、
物体post-grasp目标；原R仍是冻结live WAM→机器人意图适配器。实测位置开始close，
保留live腕/手指解释直到实测force dwell+grasp_established，再交接实测物体目标servo。
最终同初态b持握[1,1,1,1,1]s、ever/final5/5；历史原R+WAM2/5，本轮未改baseline3/5。
新b关节安全中止0/5，实测交接5/5；9初态字段误差0。目标≥4/5达到，保留此接口继续开发。
工程旧版0/1、0/4及partial全部保留，累计2364/3000controls、330.888执行秒、0updates；
没有额外归因链/重训。220 passed/0 failed/0 skipped。GPU5/6均已释放。
这是已见单初态的1秒窄任务见证，非统计优势/泛化或完整RELEASE；物体精确目标≤15mm
仅1/5，成功即停截断了后续servo；不能传递成完整任务或GT executor资格。三臂RL继续HOLD。

# 2026-10-22 v6指令：位姿消费原型未达效果目标，保留原R

[本轮实现、固定预算和结果](pose-consumption-improvement-20261022.md)：实现object-local
precontact wrist pose消费、live物体/测量相位GT、显式approach mode/ETA/方向，以及S2c
只读阶段信号对接。本轮唯一3链×5回合，3849controls/402.318s/0updates；a原R+WAM
3/5，b位姿WAM0/5，c闭环GT1/5。b未达≥4/5，不替换默认原R；c也未稳定资格，不能
据此判定参考质量或排除τ问题。b全approach、新b/c测量contact切换0/10，接触后保持
分支尚无运行见证。211 passed/0 failed/0 skipped不代替效果门。未追加重复/训练/审计，
两项完整任务阻塞修法写在报告但未声称已修，三臂RL继续HOLD。GPU5/6已释放。

# 2026-10-22 v5：clipping全量复核，零裁剪仅限source见证

[v5定义/逐block/复现](audit/action-clipping-review-20261022.md)：此前0/0只属于六例生产
source-PD，不覆盖R链。冻结五链n=5的裁剪0/400、5/1500、50/1139、24/931、11/963；
关节安全中止0/0/4/2/5。block3校准GT25/300=8.333%，首次clip159晚于首次高度失败60，
held内部0/10；不把短时held当整链无饱和资格。六例source active3230独立核实实质clip0，
合法手指端点2。全25回合向量与ever/final均列报告，无新仿真/训练；完整RELEASE0/6保留，
三臂RL仍HOLD。206 passed/0 failed/0 skipped；13/13源码mutants被断言拒绝。

# 2026-10-22 v4：冻结条件重复完成；完整任务资格仍未通过

复核入口：[τ条件/闭环审计](audit/tau-executor-loop-audit-20261022.md)。旧81状态参考、
原/校准R和WAM11000冻结，隔离旧scheduler v2（生产v3未回滚）；单已见序列每链5次。
source-PD5/5、原R+GT0/5、校准R+GT0/5、原R+WAM2/5、校准R+WAM0/5；持握中位
1.000/0/0.400/0.9667/0s。4933envcontrols、0学习更新；25/25独立重算一致，可见初态
9字段最大差0、actual PD合同最大误差3.134e-7。排序只在该协议得到描述性重复，
不是WAM优于GT的统计证据，不是样本效率。GT时钟δ与live-WAM条件不同，R本身仍读
live state；旧监督RMSE.001849不认证闭环。旧64/66是最后满足帧，65/67首先高度<5cm，
并非已证实接触滑落；新5次首高度失败60..68时GT均有效。

完整RELEASE6例另审：2个实际安全中止、2个终点与当前固定桌面支撑几何矛盾，交集1。
cubesmall/cup末端物理条件实际连续7.167/6.733s，但缺更早严格路径milestone；full仍0/6。
不修改原始标签/旧成绩，不把无效任务排除后制造新成功率；任务意图与路径口径待独立
确认。三臂RL保持HOLD，不重训v6；S2b完整参考下一run还需核live锚定/相位合同。

## 以下为此前历史状态

# 当前状态

2026-10-21 用户授权让出GPU给本地WAM。r3通过owned STOP正常停在累计1575轮，
三rank状态STOPPED，六worker及原supervisor均退出。rank0最终checkpoint
`outputs/wam-rl-policy-v3/inspire-measured-official-20261020-r3/rank-0/official-iteration-1575.pt`
61,158,850bytes，CPU完整加载、model/optimizer有限且optimizer非空；可按原身份恢复。
GPU4/5/6已释放，随后按用户新指令三卡均用于扩大后的WAM训练。当前不再运行PPO，不把以下历史RUNNING状态当作现状。
切换回执在`outputs/ptv3-shared-wam-v6/local-subset-pretrain-20261021-r2/process.json`。

2026-10-21 10:24北京时间实查：r3三rank均到1570轮，supervisor及六worker存活，
本run结束53174回合、完整任务成功0、workspace失败27792(52.27%)；仍为随机训练计数。
最近100轮rank0均值36.31s，剩余预算约32.05h，磁盘约221.5GiB可用。
确认当前冻结runner的自定义ppo.log遗漏reward/return；三rank事件文件均仅88bytes。
历史reward曲线无法由这些日志还原；此问题尚未修复到运行中的版本，不能声称奖励上升。
当前BC起点是Scheduled686 r4 DEV-best(step0)，τ新增列起始为零；此前四TRAIN轨迹
300拍仅2/4曾满足lift5cm30且之后均掉落，窗口内完整任务0/4。早期teacher/辅助控制的
保持结果不能当作自主策略成绩，也不能直接与本次完整任务指标比较。
快照为outputs/wam-rl-policy-v3/inspire-measured-official-20261020-r3/status-20261021T022443Z.json；
详细证据及未验证假设见本Task的P-20261020-inspire-measured-official-ppo实验卡末节。
本次仅记录结果与局限，未新增评估、修改冻结代码/reward或停止训练。

2026-10-21 09:54北京时间实查：r3三rank均到累计1520轮，正常运行约13.1h，
六worker仍存活，未再出现PhysX崩溃。训练累计结束51322回合，完整TaskController任务成功0，
workspace失败26952(约52.5%)；这是在线随机训练计数，不是确定性评估，也不等于抓取/抬升率0。
当前runner未汇总抬升/阶段分项，不能从value loss约0.01宣称收敛或执行能力改善。
最近100轮约35.9s/轮，PD/回合错配/log-prob/同步误差0。30s GPU利用率均值
49.3/33.1/46.8%，显存峰值15,240/15,240/15,306MiB；磁盘可用222GiB。
剩余原预算约32.56h，仍运行；需要优先分项定位抓取、任务阶段推进及reward信号，
本次状态查询未修改、暂停或重启训练。快照status-20261021T015435Z.json在r3输出目录。

用户最新已授权恢复长训。inspire-measured-official-20261020-r3从GPU231模型和optimizer
继续，GPU4/5/6各512环境/全501 TRAIN/50物体，官方PPO超参和WAM80000保持。
冻结3a0779f6，strict同backend resume，不使用迁移例外；三rank已实际续训至至少235轮，
各新增80次优化，首轮实测同步/回合错配/log-prob误差0。GPU232断点完整CPU校验通过：
身份与231完全相同、optimizer推进20步、model/optimizer有限、CAD不变。当前约34s/轮；
60s启动采样GPU利用率37.8/37.9/43.9%，每卡峰值15,240MiB。六worker及supervisor运行中。
累计目标10000，原48h训练预算扣除CPU7801.46s和GPU复测683.64s后，剩164314s约45.64h。
到期或STOP/低磁盘边界保存退出；rank0 last3+500轮整点。原暂停run保持历史原位。

用户要求的暂停及GPU复测均已完成。CPU后备r2保存累计215轮后退出，原长训仍暂停。
GPU4/5/6各512环境、全501 TRAIN/50物体，从215模型/optimizer显式迁移cpu→gpu，
冻结fa10fcd5，完成16轮到231（各320次优化、128 native控制），六worker全部正常退出，
GPU4/5/6已释放。335次workspace终止/reset，未出现PhysX崩溃；PD、source错配、
log-prob重放及220轮实测同步误差0。这是短时运行稳定性结果，不是完整任务成功率，
128控制内完整任务成功0；不能称历史间歇故障已定位或彻底修复。
GPU231 checkpoint61,158,880bytes完整校验通过，model/optimizer有限、Adam各推进320步、
CAD encoder不变，CPU215另行保留；不自动恢复长训。稳态约36.75s/轮、峰值每卡15,240MiB，
利用率均值33–39%，WAM及CPU task构造仍为主要耗时。详见本Task实验卡和recheck acceptance.json。

用户要求优先启动PPO，指定GPU4/5/6。目标为官方UniDex PPO、最新r3准入和Scheduled686
DEV-best父模型，每rank512环境/合计1536，总rollout12288、minibatch3072，匹配原Shadow
六rank×256规模。两环境GPU物理异步reset已完成2迭代/40次优化，env0/1分别step2/4 reset，
其他时钟不动，动作跨回合错配0、logprob重放误差0；154项Task测试通过。
r1三rank在首个物理控制遇到PhysX convexPlaneNphase_Kernel非法访问，没有训练更新，已停止。
有界诊断中全50物体512环境单rank完成2次官方更新；256环境也会失败，不支持简单OOM解释。
提交83c746ce将扩展状态数组每snapshot转换一次并增加TF32数值检查；全24步误差0.377mm，
小于1mm门槛。同步reset/PD写入仍不能消除所有GPU接触核崩溃，不能声称GPU PhysX已修复。
r2已采用并通过CPU PhysX后备（8物理线程/rank），PPO与WAM仍在GPU4/5/6；物体覆盖、环境数、
动作/频率、网络、reward及PPO超参保持上述协议。CPU异步reset通过，三rank已完成同步参数更新。
实际状态以outputs/wam-rl-policy-v3/inspire-measured-official-20261020-r2/manifest.json为准；
启动不等于已有策略更新。物理后端与WAM推理设置纳入checkpoint续训身份，禁止静默切换。

启动时r2三rank各完成至少2迭代/40 optimizer steps，参数同步误差0、source错配0，
全501 TRAIN/50物体/1536环境已核对；真实自然workspace终止也经过indexed reset。
首个rank0 checkpoint official-iteration-1.pt(61,158,108 bytes)已完整CPU读取并校验有限，
包含actor/critic/optimizer/CAD和实际backend身份。第二迭代约32秒，10k估计89h；
本轮48h上限会先保存停止，不能称10k将在48h内完成。已有成功数不足以判断学习成效。

用户要求复核源码/运行环境后，已对照NVlabs官方源码并实测import：物理仍为与旧972轮
相同的Torch2.0.1+cu118/rl-games1.1.4，WAM/PPO独立Torch2.4.1；vendor含本地采样、
桌面reset、shape索引修正，不能称完全官方原样。新增native-only有界复现入口及GPU PhysX/
CPU tensor独立诊断选项。12组检查均未再现历史错误，包括完整512环境真实PPO2轮、
CUDA memcheck16环境0错误、4/5/6并发native，以及512环境128控制/32次子集reset。
这没有识别唯一根因或证明GPU长训稳定，不切换当前CPU后备。结果与限制在本卡及
outputs/wam-rl-policy-v3/inspire-physx-provenance-20261020-r1/result-summary.json。

并行WAM/MANO预览已完成，GPU7已释放：outputs/wam-rl-policy-v3/wam-mano-preview-20261020-r1/
有preview-middle.png、preview-contact-sheet.png和preview-slow4x.mp4。首个TRACK tick68，
WAM完整24步、相同物体回放和相机；真实Inspire FK最大误差0.00181mm，MANO为近似拟合，
残差均值22.83/max45.17mm，图中已明确标注。合成MANO回收误差0.0217mm，产物3.8MiB。
不将人手拟合表面、未来teacher差异或画面接触解释为真实物理执行或WAM因果收益。

最新DExplore goal+WAM BC已完成：inspire-scheduled686-bc-20261020-r4使用r3准入及其
startup_half Current421 best(step7000)，重建62605行同源τ并冻结WAM80000。
GPU6/7缓存440.6s、GPU6训练50k/600.8s；保留全部56036 TRAIN/6081 DEV active行。
独立同6081行DEV：父模型0.04012461、Scheduled DEV-best(step0)0.04012461、50k末态0.05317855。
best新增τ权重仍0，不能当WAM学习收益；末态τ权重最大0.19829，是真实训练结果。
相同四条TRAIN运动、300控制窗口、三臂初始q/dq/physical/object/hand/tick/table逐位相同：
抬升>=5cm连续30帧代理为Current2/4、step0扩展2/4、50k末态1/4；窗口内完整任务均0/4。
本轮未见WAM改善，但不是匹配训练预算归因或全回合泛化评估。输出约110MiB、GPU6/7已释放，
未启动PPO。结果见outputs/wam-rl-policy-v3/inspire-wam-goal-bc-20261020-r4/comparison.json
及P-20261020-inspire-scheduled686-bc实验卡。

用户纠正后的当前任务是实际训练UniDex新goal+BC，不能再以旧565结果代替。
run=unidex-task-bc-20261020-r1，三卡DDP GPU5/6/7（本次用户明确授权），
Current310随机初始化50k后，Scheduled575继承本轮已学goal的310再训练50k。
完整1536897 TRAIN/354761 DEV，task10与WAM来自同一次TaskController，旧565 τ不复用。
同轮旧记录没有net-contact字段，离线/在线统一显式motion-only反馈；当前速度由原生观测
yaw与角速度缩放解码，不使用未来，也不把总力传感器或applied wrist force冒充接触力。
全量task10/τ缓存已完成，TRAIN428972/DEV99248行非零goal；全1891658行task10逐位一致。
三卡梯度按683/683/682加权为同一global2048均值。310/575各完成50k更新约262/270s，
三卡参数差均0；310新goal第2步、575第1步有非零梯度，575末态τ权重最大改变0.213。
几何goal依赖审计保留phase/progress，仅置零SE3列即可改变输出；不是只学常数阶段偏置。
149项Task测试通过。
原生入口已支持直接载入训练310/575，zero_expansion=False，不再把零扩展评估叫训练复现。
自动流程已COMPLETED，产物outputs/wam-rl-policy-v3/unidex-task-bc-20261020-r1/，4.3GiB。
固定16个TRAIN物体/64场景：310(step50k)原生曾成功32、最终稳定21；575 DEV-best(step49k)
为33/24，预声明50k末态为38/29。三臂workspace失败均0，task_columns_learned=True、
zero_expansion=False，纯BC且无PPO更新；不事后用闭环结果替代DEV选择规则。
两个575 checkpoint的64初态逐位相同；与310的全状态只有61/64相同，不能称全64严格配对。
共同61场景原生曾成功31→32/35、最终稳定20→23/26。这里只复现新goal BC的实际执行，
不能证明WAM净收益或未见物体泛化。确认路径进度达1分别9/7/9，与原生成功指标仍须区分；
最终稳定但未进入TRACK仍有3/3/1个场景。完整comparison.json保留逐项初态检查和分母。
训练/评估已全部退出，不从这些结果自动启动Inspire长PPO。

新增用户要求的从头学习对照：UniDex Current300 随机初始化50k，再以本轮新父模型训练
Scheduled565 50k，不加载历史BC/PPO。新BC在同一16个TRAIN物体/64初态中原生曾成功35，
最终稳定26（40.625%），终态位置32，workspace失败0；64份观测/pose/hand/goal/full-root
与历史BC565面板逐位一致。这是565学习可复现性，不是新task10已训练或WAM净收益证据。
产物unidex-fresh-bc-20261020-r1/current与r2/{scheduled,evaluation}；r1 Scheduled在0更新
时因整块np.load异常慢停止，分块reader把TRAIN565读取降至4.92s，未改变模型/采样语义。

Inspire已按Shadow补齐startup_half默认采样，保持前8/60秒的物理窗口（30Hz的前4拍），
保留原始tick而不使用压缩后索引。用修正支撑v2数据分别从随机初始化完成uniform与
startup_half 50k：TRAIN56036/DEV6081 active行、501/62序列，两组种子/模型/配方一致。
DEV最优分别step2000/7000，MAE0.037471/0.040123；50k为0.054135/0.052154。
四条预先按teacher耦合覆盖选择的TRAIN lift运动：两组best均2/4曾抬高5cm连续至少1秒，
两组50k均1/4；完整路径任务全部0/4。旧Current在同一四初态中此物理代理为0/4。
所有初始q/dq/physical/object/hand/tick/table逐位一致；不能把物理抬升代理叫完整任务成功。

更明确的执行差异：DExplore腕部动作是相对当前q的米制PD平移目标，旧BC首动作目标误差
39–73mm，离线/在线初始预测却一致到1.79e-7；不能用UniDex的力/力矩动作MAE作为物理阈值。
startup_half50k在这四条TRAIN运动首动作误差降至1.7–3.0mm，仍仅1/4抬升代理；完整DEV
启动248行仍约103.5mm平均平移目标L2误差。采样修正没有解决泛化及闭环纠错覆盖。
诊断输出inspire-bc-sampling-20261020-r1，含full-dev-startup-audit.json、panel-comparison.json
和lift-panel.png。四序列拟合诊断inspire-bc-fit4-20261020-r1独立记录，不替代完整数据策略。

本轮另修复packed状态解码保留wire标记导致collector/observer重复解码失败的工程bug；
真实调用链回归先失败后通过，v3全套143项通过。历史JSON路径结果不受该bug影响。
teacher原始动作匹配16环境重放toothpaste：初态及前30拍手/q一致，后续也出现物理轨迹偏离；
stamp16环境在观测前PhysX非法访问，保留失败日志，不能归咎BC。四环境评估均已正常通过。
详细协议、运行版本和解释边界见P-20261019-inspire-task-contract实验卡。
四序列拟合已完成：2110 active TRAIN行、50k更新224.85s；教师状态前10拍腕部平移目标
误差0.69–1.20mm，仍只有1/4满足抬高5cm连续1秒，3/4曾TRACK、完整任务0/4，最大路径
进度约0.47%。支持优先处理接触阶段闭环纠错覆盖；不能靠继续重复BC epoch解决。
任何teacher接管/DAgger需区分真实behavior/applied与teacher标签，previous_action必须来自
真实执行动作，不能把混合执行伪装成现有同轮teacher监督。所有本轮任务已结束，GPU6/7已释放。


用户澄清要测BC而非PPO。已用纯BC入口完成unidex-bc-goal-20261020-r1：
加载全量自训Scheduled BC best(step46000)与其原WAM12000，不加载RL checkpoint，
不创建PPO/critic/optimizer，0更新；BC declaration与evaluation policy_kind均明确标注bc。
16个TRAIN物体×64场景，565与零扩展575的初始robot300/hand/pose/goal/full root全数逐位一致。
BC565：原生曾成功32/64(50%)、曾保持25、最终稳定25/64(39.06%)、终态位置28；
BC575：原生曾成功35/64(54.69%)、曾保持25、最终稳定24/64(37.5%)、终态位置28。
两组workspace failure0，各完整首回合；最终稳定要求近目标5cm且抬高10cm、末尾连续60步。
BC565/575评估分别51.84/75.75秒，两卡已释放；138项测试通过。
这证明UniDex的WAM+BC可在该TRAIN面板部分成功，可作DExplore的执行正对照；
不代表高成功率或未见物体泛化，也不证明DExplore失败唯一来自数据。
575新增task列未训练，不以35对32的差别宣称goal/WAM增益。

以下UniDex task-goal r1/r2/r3加载的是PPO，不能当作BC成功率。

最新决策：按用户要求先用UniDexGrasp排查新增goal/τ链路，暂不扩展DExplore训练。
Shadow575只做已训练PPO565的model-only零扩展评估，task10权重未训练，原v2代码不改。
净接触力版9db09ec4在64个TRAIN首回合中原生曾成功47、最终稳定保持29、workspace失败0；
task10/WAM点目标最大差2.10e-9m，但29个最终稳定场景中23个从未TRACK，所有路径进度0。
不能把DExplore零进度都归咎于数据；门控、可观测性和任务容差需独立校准。
Scheduler默认8mm容差及两端点路径的0/1确认进度不同于原生5cm成功，不能直接互换指标。
首轮对照两组仅53/64初态完全一致；共同53场景两版均原生曾成功38、最终稳定27。
不使用未匹配场景判断优劣，也不将新增task列尚未训练的结果当作方法增益。
初轮错误使用总力传感器作为接触反馈，已修为world净接触力；r1/r2仅作历史诊断，
当前反馈诊断以unidex-task-goal-20261020-r3为准。全部有界GPU评估已结束。

同时修复了桌面支撑的确定几何错误：DExplore桌板薄轴为局部Y，原代码误用Z；
改为从mesh bounds选薄轴、向上的面与切向范围。135项修复回归、加入Shadow后137项通过。
同一660条原始执行全量重审409.7秒完成，563条前缀保持，命令60529→62605；
208条前缀增长，无缩短，人工问题标注259帧保留，660条现均观测到部分桌面支撑。
新准入TRAIN56418/DEV6187为未做Task-active裁剪的命令数；不是可直接复用旧τ的行数。
新路径outputs/wam-rl-policy-v3/inspire-teacher-rollout-audit-20261020-r3/，schema v2；
旧v1准入、缓存与BC结果保持历史原位，当前loader拒绝v1，未开始新DExplore长训。

以下为迁移及修复前后历史运行记录。

v3 是新增显式任务 goal/task10 的执行路线，v2 继续保留 UniDex/Shadow300/565 主实验。
迁移提交为 da9f5b8b；共同工具在 v3 独立保留，无动态 v2 Python 包依赖。
历史原始数据和权重只读复用原 outputs/wam-rl-policy-v2 路径，新输出仅写 v3。

确认缺陷已修复：新增265列清零；task10/WAM同一TaskController；PPO行为动作采样；
单回合真实终止后收尾；实时支撑反馈；按环境绑定几何；shaping/GAE γ=.96一致。
v2 59项、v3 127项测试及22项治理测试通过。

冻结 da9f5b8b 的 v3 原生运行 inspire-measured-interface-fixes-20261020-r2 通过：
Current421 在450拍 acquisition timeout 后正常结束，Scheduled686 以 banana/apple 完成60拍。
PD误差0、跨episode动作错配0；Scheduled task10/WAM点目标最大差6.59e-9m。
Scheduled 使用的是未训练的精确 Current421 扩展，仅用于接口检查，不能视为新BC表现。
PPO两组actor候选都被KL门槛回退，没有策略改善结论。

统一goal τ缓存已在GPU6/7完成，约406秒，路径为
outputs/wam-rl-policy-v3/inspire-scheduled686-cache-20261020-r2；563条前缀/60529行，
有效TRAIN52660、DEV5693，task10/WAM点目标最大差7.61e-8m。
旧双调度源缓存及旧Scheduled权重保留，不能静默进入新训练。
真实Current421 best(step3000)在8物体256条新条件样本上的零扩展检查通过：
输出最大差2.38e-7，新增非零权重数0，首步tau/valid梯度最大0.01419/0.00448。
该检查只做1次工程optimizer更新，不保存checkpoint。shape标准化特征p99=2.48、max=3.24，
仅此小面板未见幅值爆炸，不能排除全量或后期漂移。完整报告为
outputs/wam-rl-policy-v3/inspire-current421-warmstart-audit-20261020-r1/report.json。
现有Current421 best仍保留为父策略，不需为迁移重训；正确条件的Scheduled BC重训结果见下。
新训练保留step0基线并记录shape特征漂移；不因ref1_2提出的学习风险修改容量或50k配方。
当前工具只有有界单回合collector，尚不作为多回合长PPO训练入口。

ref1.md 两项新增缺陷已补回归：Scheduled BC 保留所有 task-active 命令，
无效 τ 显式为零、valid=False，使用原时刻真实动作监督；配对 DEV 同样包含启动样本，
另报 valid/invalid 分组。意外掉落不再确认未到达位置的计划 RELEASE，恢复时输出
CONTACT 意图，重抓后保留待执行的 RELEASE；合法中间释放也有正向测试。
v3 共132项测试通过。四进程有序数据回放与串行结果逐数组一致。
r2 在 CPU 数据准备阶段、零次参数更新时停止，保留停止记录。
r3训练使用所有 active TRAIN54162/DEV5879 行，包含无效 τ TRAIN1502/DEV186 行。
事件修复不改变 task10/local goal 或当前不接收事件的 WAMv4 输入，既有统一目标缓存仍可复用。
这两项修复不等于已获得正向 BC 抓取结果；新训练和配对闭环结果见下。
额外覆盖“已经到终点但尚未进入 RELEASE 阶段就掉落”：最终释放必须等显式
RELEASE 阶段授权，不能只靠到达终点确认；相关24项定向回归通过。

r3 已在 GPU6 完成50000次更新，597.2秒，准备56.6秒；约84–93%训练利用率，
best/latest合计约39MiB。完整v3回归133项和本次Python/结构化/Markdown检查通过。
配对DEV5879行的Current421与选中Scheduled(step0) MAE均为0.03825807；
训练后step50000为0.05427545，valid组0.036321→0.052715，invalid组0.097547→0.102043。
step0仍为best，不能声称新τ训练获得收益。

GPU6/7原生配对也已完成：两条TRAIN运动各300拍，Current、选中step0、训练后step50000
初始q/dq/physical/object/hand/tick/table逐位一致；PD误差0、跨episode错配0，
Scheduled的task10/WAM点目标误差最大1.03e-8m，两组均有596次有效预测。
三组在300拍末均为ACQUIRE、路径进度0，不能作为全量成功率结果。
训练后权重在这两条TRAIN运动的首动作MAE从0.13543/0.07339降为0.00778/0.00705；
alarmclock在第60拍被observer短暂判为TRACK（前100拍40拍TRACK），末态又回到ACQUIRE。
这表明启动模仿改善仍不足以稳定维持观测耦合；不等于证明真实抓握成功或WAM净收益。
初始实测输入与teacher一致到2.98e-7、task10精确，下一步诊断耦合丢失与状态分布漂移，
暂不扩展长PPO。结果位于outputs/wam-rl-policy-v3/inspire-ref1-paired-20261020-r1/
及inspire-ref1-latest-20261020-r2/；两卡任务均已结束。
# 2026-10-23 PPO/BC 取证：首步 KL 实现缺陷已定位，规模训练继续 HOLD

真实部署 goal contract 仍未完全落地（`ptv3-shared-wam-v6/.../online_runtime.py:49` 的
`object_goal_valid=np.ones(32,bool)` 尚在修复中），本轮没有启动长训。冻结残差 pilot 的
旧实现把 sampled-action log-prob 差当作 KL，且 `if epoch` 跳过首步检查；离线精确审计为
p000/p001 Gaussian joint KL 0.611887/0.165042，旧日志 approx KL 约 0，clip fraction
0.8451/0.6800，entropy -12.972737→-12.972737/-12.973036，reward sum -1.181455/-0.845830。
同一 batch 的 lr 敏感性支持 1e-4 是强漂移因素，但不作唯一因果归因；12D joint target=.03
的尺度解释单独记录在 [P-20261023-ppo-bc-evidence](ppo-bc-evidence-20261023.md)。

`intent_residual.py` 现用 exact Normal KL、首步 rollback/retry（四次按 1/4 lr）、每轮记录
KL/clip fraction/entropy/reward，worker 预设 lr=1e-5、epochs=2；离线 p000 replay 已验证
最终 KL≤.03。下一步只有 10–20 分钟吞吐/数值 preflight；通过后才可排每 arm 1e5 的
BC-only vs BC+残差PPO 配对 probe。1e5 无改善但 KL 正常只扩预算，KL 未修好不下 PPO
无效结论。当前本地 `/home2` 约 140–141GiB 可用、历史 outputs 约 571GiB，不删历史。

## 2026-10-23 v14 source天花板修复（R7预注册，未过门）

原source14/40。R3/R4/R5/R6均完整0/40，实耗23894/24000/23648/21780控制；关节中止0/0/0/1。新控制退步，不是τ负证据，不声称已修好。R7只修闭合目标追逐物体的问题，同40/≤24000/0updates，接触后捕获世界抓姿、持续失触/卸载后重新接近；评分及抓后伺服不动。source未达32/40前不比WAM；RL HOLD、不重训v6。详见本task source-controller-ceiling实验卡/报告。

R7已≥9完整失败，仍完成40；R8预注册恢复source PD接近路径（实测最近前方状态游标）及前驱腕PD目标，避免静态key丢失动作信息。保留同评分/后段/gate，同40/24000/0updates。未过source32/40门，不做WAM/RL/重训。

R9条件预注册（R8≥9失败才启动）：首次接触锁实测腕、停止继续插入；探测保持source闭合目标，不提前转微弱持握调节。同40/24000/0updates，仍需32/40，不比WAM/不RL/不重训。

R9在0子进程/0控制时取消：新step红测试发现目标inner margin被父类当成实测碰撞压力。R10 fresh预注册（前序R8），同40/24000/0updates，保留闭合修复，加可选measured_pressure钩子，目标仍inner准入/实际gate不动。全v3 309 passed/0 failed；33专项、29mutants。R7完整0/40/22073控制/0jointabort，R8≥30完整失败仍跑满。未达32/40，不比WAM。
