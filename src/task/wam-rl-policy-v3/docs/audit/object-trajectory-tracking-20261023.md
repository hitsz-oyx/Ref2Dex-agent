| 观测 | 含义 | 决定 |
|---|---|---|
| 常见形状/参数下物体误差 ≤ 阈值、无碰撞、限位余量充足、漂移小 | **通用物体轨迹跟踪在现有执行器上可用**（走控制、不走学习） | 建立“物体轨迹 → 执行”通道为主线的标准能力，后续任务复用 |
| 只在高速/大曲率失败 | 存在**可行域边界** | 写明边界由哪一类约束决定；报告加前馈后边界移动多少 |
| 小参数也撞关节限位 | **可达性/限位是主线 P0**，与轨迹无关 | 停止扩展轨迹形状，回到执行器 P0（joint-10 限位） |
| 必须滑移/力反馈才不脱手 | 持握稳定是**必须新增件** | 列为下一项工程件（含最小可行实现方案） |

**最终实测解读：** A无FF 0/45、有FF 8/45满足数值阈值；正确保留抓取PD预压后的B无FF/有FF均0/45，但90/90完整走完且持续持握。剩余主约束为跟踪精度，不能再称主要死于滑移；未建立完整可用资格。A是运动学刚体代理，自碰撞未取证。

**brief48续跑核对：** 59/90是先前进度快照；本次读取r4的COMPLETED manifest，已存在90个唯一B格且全部完整执行，自有原生/客户端均已退出。因此本次不再启动仿真，不重跑任何格，不改写原始轨迹；只完成报告和已有收尾改动的提交。

**预注册归类：四行均未完整命中，当前为“跟踪精度资格未通过”。** 第一行不成立（真实B为0/90，且自碰撞未知）；第二行只得到A的局部边界证据，不能完整成立（最低速直线也超姿态阈值，真实B尚无可用格）；第三行不成立（限位中止0/180）；第四行没有新增反馈必要性的证据（原有力保持加正确预压已维持B全部90格）。不把“低速也失败”事后改写成“只在高速/大曲率失败”。下一步只做实测物体SE3误差外环的最小工程修正，不扩形状、不重训；碰撞资格仍须独立补齐。

# 通用物体 SE(3) 轨迹跟踪：预注册

实验 P-20261023-object-trajectory-tracking；Task wam-rl-policy-v3；branch main；run_id object-trajectory-tracking-20261023-r1。本轮新增仿真，历史 source-PD 5/6 仅作为初始化选择依据，不继承成功。无 τ、无神经网络调用、无训练、无 schema 变更。

决策：区分执行器跟踪边界与真实持握滑移；最便宜手段是已知 source-PD 获取初次实测持握，再在同一物体参考上切换刚体约束/真实接触，固定矩阵扫描。类别 Decision，低档限位为 Blocker。

## 冻结协议（计算/仿真前登记，不按结果改门）

- 形状 straight/circle/figure8/helix/random_spline(seed=4601)，闭合 circle/figure8；其它非闭合。三个长度尺度 20/50/100mm；平均弧长速度 30/100/250mm/s。直线的尺度是行程，曲率恒零；非圆形的尺度不是“处处曲率半径”，另报实际最小/分位曲率半径。禁止按物体身份分支。
- 5×3×3×2(FF off/on)×2(A/B)=180 格。先运行最低速度/最小尺寸直线 off/on，A 后 B；无小参数限位才扩展矩阵。若触发预注册 P0，则所有未测格明确记 NOT_RUN_P0，不填推测数字。
- 参考为物体 SE(3)+秒时间戳。路径弧长决定总时长；五次时间律 s(u)=10u³−15u⁴+6u⁵，端点静止。位置 CubicSpline、姿态 RotationSpline，逐30Hz tick 求值与一阶导；另报真实速度峰值/加速度/曲率。姿态变化幅度10度，所有形状同一规则。
- 每格重置后只回放现有 source-PD 获取持握；实测 ≥2指净接触力≥0.1N、离桌且抬高≥50mm，持续0.3秒才固定 T_hand_object。最多原 source 全长+120controls；获取失败独立记 ACQUISITION_FAILURE，不能算轨迹失败或删去。
- T_world_hand_des(t)=T_world_object_ref(t)·inv(T_hand_object)。速度含旋转杠杆臂 ω×r；姿态误差为 SO(3) 测地角。phase=TRACK。沿用原 PoseConsumer 位置/角度比例与阻尼 .45/.012、.40/.008，以及原 canonical→PD 映射；FF仅加导数×dt，不调 PD、关节门和目标。
- A：刚体运动学约束探针，按实测掌部位姿将物体每控制 tick 固定到已估 T_hand_object；不是有限刚度焊接的动力学验证，不认证负载/接触力。A 的零漂移是施加的假设，不能算控制器成果。B：不施加约束，真实物体与接触；同参考生成器、伺服与预算，保留既有实测力闭合保持规则。
- 跟踪通过：位置误差P95≤15mm、姿态P95≤5deg；闭合误差≤15mm/5deg；全程手物漂移≤10mm/10deg；无物理安全中止、无手/物体-桌或自碰撞、有限关节距 nominal 边界≥0.02rad（平移≥0.02m）。阈值在本任务不放宽。跟踪误差必须使用执行后物体状态，A刚体代理单独标记。
- 滑移：相对位移>10mm或旋转>10deg；脱手：>50mm或连续0.2秒失去双指实测接触。不同约束可多标签；主因按首次越界时间，不能只保留最后门。
- 真正安全门仍为原 physical_safety；另报 nominal joint margin（有限关节、单位分开），不把连续腕角套限位。碰撞尽量记录原生接触对；接口不可用则写未取证，不能称无碰撞。
- GPU仅4–7中空闲卡，首批4；运行前检查。client graspenv，native dexplore_v120_train Torch2.0.1+cu118。native GPU PhysX/CPU tensor exchange用于接触对诊断，若无法取接触对不偷偷切换后端。预估180格≤300000controls、2GiB新输出、总wall≤7200秒；每格轨迹≤90秒物理时间。可用盘<20GiB、非有限状态/原生失败即停止并保留证据。

## 实测结果

已完成；最终实测与完整矩阵见下方结算。工程测试不认证物理能力。

## 记录接口故障与修复（不是物理能力结果）

r1（33177f56）原生执行1control后记录转换失败：`ValueError('no field of name initial_overlap')`，client未收到回包所以client.controls=0；跟踪0。原始故障目录保留。初始“GPU接口不兼容”的猜测被明确异常否证：本机实际接触dtype缺文档字段；没有证据指向GPU PhysX不能查询接触。修复仅让缺失overlap为null（不伪造0），body0/body1/lambda照常记录；回归构造相同缺字段结构。8 passed/0 failed；物理后端仍gpu-cpu。新输出r2，控制、门槛、矩阵不改。

## r2阶段B协议偏差与定向纠正（追加登记，保留原数据）

r2入口实际将force_valid传False，两阶段均固定抓取时手指关节目标；这与上方冻结协议中“B保留既有实测力闭合保持”不一致。r2 A仍为约定的刚体探针；r2 B保留为固定关节持握的工程记录，不计作约定的B。不能把r2 B失败推成“现有力闭合保持无效”。修复仅B的调用条件：`case['stage']=='B' and contact_force_valid`，复用既有 .25N/+0.0015rad、4N/−0.001rad规则与FINGER_UPPER，不调PD、不改阈值；A不重跑。

r3只运行B：5形状×3尺寸×3速度×2FF=90格，初始物体参考从r2/reference-start.npy冻结复用；seed195，source-PD同源，空闲GPU5。同样小参数关节门停止扩形状，最大controls100000（实际入口仍共同总上限300000）、总wall7200秒与新增2GiB总预算不变；最终同时报告r2 B的偏差与r3实测，不抹去不利结果。该纠正恢复预注册，不增加新的学习/基线。

## r4交接预压修复（先登记，再只重测B）

r3（已恢复force-valid调用）第一TRACK指令仍错误地用实测q初始化手指目标，从source实际PD [1.364090,1.050968,0.807545,1.385341,1.111492,0.426087]切为[0.721299,0.885526,0.730134,1.390613,1.072549,0.266685]rad，最大卸载0.642792rad。这是新增交接入口自己丢失预压，不能用其0/45否定保持原抓取时的执行能力。最小修复：B的初始手指状态继承最后实际施加的PD targets，后续沿用原force规则/原upper；A不改不重跑，原始B全部保留，最终结算以r4替代B资格。回归必须抓住“把PD目标替换成实测q”的故意错误。

期望变化：首TRACK手指目标最大突降0.642792rad→不超过原force一步更新及既有upper钳位（本初态thumb1.111492→1.10的0.011492rad），食指目标约1.364rad，不再重置0.721；实际B保持时长/跟踪成功是否改善以r4为准，不预设必升。

r4只B、同90格/同参考文件/seed195/新output、空闲GPU5，参数与判据不改；允许joint P0停止，其余依照原预算300000controls/7200s/2GiB，不启动其它方法。

## 结算：持握交接已修，精度仍未通过

有效矩阵180格=5形状×3尺度×3速度×2FF×2阶段。A用r2的90格，B用r4的90格；相同object SE3/timestamps逐位核对90组，reset及交接q/object最大差均0。只选一只Inspire浮动腕手、一个已知抓取初态`s1_cylindersmall_pass_1`，不外推其它物体/机械臂；形状/控制均不按物体身份特化。

| 组别（45个参数格，每格1次） | 走完整段 | 数值阈值通过 | 各格位置P95的中位 mm | 各格姿态P95的中位 deg | canonical裁剪/ TRACK controls |
|---|---:|---:|---:|---:|---:|
| A无FF | 45/45 | 0/45 | 27.378 | 8.534 | 0/6215 |
| A有FF | 45/45 | 8/45 | 22.241 | 6.228 | 0/6215 |
| B无FF | 45/45 | 0/45 | 33.267 | 7.861 | 0/6215 |
| B有FF | 45/45 | 0/45 | 26.843 | 5.741 | 0/6215 |

门限冻结为位置P95≤15mm、姿态P95≤5deg、闭合≤15mm/5deg、全程相对漂移≤10mm/10deg、nominal有限关节余量≥0.02rad/0.02m、无安全中止；不放宽。每组45是不同条件，不是同分布45次重复，不伪造独立成功率置信区间。完整路径上的逐tick误差/分位数均按物体计算，未用累计进展当成功。

### 修复前后：不能把新入口的错误算作物理能力负证据

r1新增诊断曾假定SDK有initial_overlap字段，实际缺失；原生执行1control后ValueError，跟踪0，保留FAILED目录。修复把缺字段记null，不是0，后端不变；先前GPU查询不兼容的猜测不成立。

r2的B误把force_valid传False（固定手指）；r3恢复既有力保持，但又用实测q而非最后PD目标初始化手指，食指1.364090→0.721299rad，卸掉0.642792rad预压。两批都保留90格，各自0/90通过、84/90失持中止；只作旧入口结果，不用它们否定保留原持握设定点的执行器。

最小物理修复只有B交接的手指初值：继承最后实际施加的PD目标，后续仍用原.25N/+0.0015rad、4N/−0.001rad以及原FINGER_UPPER，不改PD增益或物理门。

| 指标（同初态/同参考） | r3修前 | r4修后 | 判读 |
|---|---:|---:|---|
| 首TRACK最大手指目标突降 | 0.642792rad | 0.011492rad（原thumb upper钳位） | 保留预压，食指不再归零到实测q |
| 失持门中止 | 84/90 | 0/90 | 立即失持已修 |
| 完整执行参考 | 6/90 | 90/90 | 未增加每格最大时域 |
| 数值阈值通过 | 0/90 | 0/90 | 持握修复不等于轨迹精度通过 |

r4的首TRACK手指PD独立复算最大差4.91e-08rad。旧80余例的失败不能继续称“必须新增滑移反馈才不脱手”；现有力保持加正确预压已持续持握全部参考。

### 可行域与前馈：只报告离散格，不内插

A有FF的8个候选均平均30mm/s：circle尺度20/50/100mm，figure8尺度20mm，helix尺度20/50/100mm，随机样条尺度20mm；实际峰速56.10–109.36mm/s，不是近乎静止。无FF0/45→有FF8/45（+8格），100/250mm/s档均未过；直线低速也会失败，不能说只有高速/高曲率失败。

真实B两个FF条件均0/45，**本阈值下未找到已测真实持握的可用格**。前馈仍改善误差：位置P95跨格中位33.267→26.843mm（−19.31%），姿态7.861→5.741deg（−26.97%）；A对应27.378→22.241mm（−18.77%）、8.534→6.228deg（−27.02%）。不是每格、每项都改善，完整45组在下表。

A的45组完整配对，有FF−无FF的位置P95差中位−6.423mm，范围−22.459至+0.023mm；姿态差中位−1.995deg。A候选是运动学假设下的上限，不能当作严格动态刚体持握的正控制：附件每control tick固定对象root，未验证子步焊接/负载惯性传递，A/B差异不能只归为滑移。

### 闭合、持握完整性、限位与碰撞

| 指标 | A无FF | A有FF | B无FF | B有FF |
|---|---:|---:|---:|---:|
| 最大相对位移 mm | 0.000097 | 0.000105 | 0.626261 | 0.631949 |
| 最大相对转角 deg | 0.000019 | 0.000019 | 1.372214 | 1.381110 |
| 最小有限关节余量 rad | 0.078890 | 0.078890 | 0.047011 | 0.047013 |
| 闭合形状格数 | 18 | 18 | 18 | 18 |
| 闭合位置中位 mm | 3.358 | 2.744 | 11.891 | 11.994 |
| 闭合姿态中位 deg | 1.030 | 0.573 | 3.371 | 3.255 |

A的零漂移是强加假设，不是抓力成果。B全程双指实测接触保持、手物相对漂移≤0.632mm/1.382deg；连续合格持握elapsed范围0.1–21.033s、中位2.533s，等于各参考时长。最短0.1s只对应短路径，不冒充长期持握。

A最小余量在native q[14] `thumb_proximal_yaw_joint`=0.07888985rad，连续腕角不套有限门；全部关节逐tick余量在CSV/原始tick记录。

B最小余量在native q[10] `pinky_proximal_joint`=0.04701114rad，连续腕角不套有限门；全部关节逐tick余量在CSV/原始tick记录。

canonical裁剪0不等于servo未饱和：

| 组别 | 平移±25mm cap命中 | 角±0.12rad cap命中 | TRACK controls |
|---|---:|---:|---:|
| A无FF | 87 | 0 | 6215 |
| A有FF | 110 | 12 | 6215 |
| B无FF | 94 | 0 | 6215 |
| B有FF | 125 | 12 | 6215 |

关节安全/physical_table中止均0/180；筛选到的手/物-桌接触对（native lambda>0.1）0。**不能认证无自碰撞**：既有`dexplore_inspire.py:15`设COLLISION_GROUP_SELF=1，:20的shape filter为2/3；本机Isaac文档assets.html:198说明actor filter>0关闭self collisions，未为本实验改filter。SDK缺overlap字段，当前已跑数据也未落完整负接触对；无接触记录不等于严格无碰撞。交付入口补全native_contacts落盘只适用于后续run，未伪造本轮缺失记录。

### 每个失败的观测归类

| 阶段 | 失败格/总格 | 限位 | 跟踪误差/滞后 | 滑移 | 记录的碰撞 | 其它 |
|---|---:|---:|---:|---:|---:|---:|
| A | 82/90 | 0 | 82（失败格100%） | 刚体假设 | 0，自碰撞未知 | 0 |
| B | 90/90 | 0 | 90（失败格100%） | 0 | 0，自碰撞未知 | 0 |

这里“滞后”是对象位置/姿态/闭合超阈值的观测类别，不是已识别的单一PD故障。B相对漂移远小于阈值，当前不支持“主要死于滑移”；也没有关节门证据支持回到joint-10。初始腕速0.855479m/s、角速约0.425rad/s，参考初速度0，仍有交接瞬态；但相同速度下保留预压已消除全部失持，不能再把速度断点当作失持的唯一原因。

**下一项最小工程件（未实施/未验证）**：保留修好的抓取PD记忆，在实测物体SE3上增加带抗饱和/限速的误差外环，结合已有位置积分与姿态修正，保留样条速度前馈；先复用现有GoalServo思路消除对象误差，不调大PD硬顶、不因本轮结果重训模型。初速度不匹配可做一阶连续过渡，但这不是本轮已证实的唯一原因。若要认证A/B因果差异，还需保持负载/惯性的真正rigid fixture以及独立自碰撞几何检查。本轮不额外启动这些实验。

### 通用输入与时间/导数契约

T_HO=inv(T_WH)·T_WO在获取时固定；T_WH_des(t)=T_WO_ref(t)·inv(T_HO)。v_hand=v_object+ω_world×(p_hand−p_object)。控制明确为TRACK，位置/角度增益与阻尼沿用PoseConsumer（.45/.012、.40/.008），手指保持沿用MeasuredGraspConsumer，最后走原canonical→原生PD；R/WAM不调用。这个新TRACK适配器不等于所有历史执行链的上限。

弧长/平均速度定总时长并向上对齐30Hz；s(u)=10u³−15u⁴+6u⁵生成121个knot。位置CubicSpline clamped，姿态RotationSpline；实际导数是position(t,1)、ω_world=R(t)·rotation(t,1)，不是把body rate误当world rate。FF是原位置目标中的v·dt与Euler-rate·dt一阶预补偿，不是另换原生速度/力矩驱动。实际峰速、峰加速度、峰角速度、曲率半径min/median均在CSV；直线半径为空=无穷/不适用，尺度是行程；非圆形的尺度也不冒充处处曲率半径。

位置端点速度严格0；RotationSpline角端点不是严格0，本矩阵端点角速率最大0.00752056rad/s，原“端点静止”对姿态只是近似，未隐瞒。`--reference-npz`接受world poses Nx4x4+times秒；任意非圆轨迹同一入口。交付版对不在30Hz网格的终点作末端hold、不外推；这一通用边界只做CPU接口检查，本次已测矩阵全部原生对齐，不另跑。

### 证据身份、环境与资源

| run | 实际运行Git | 格数/范围 | controls | 结论用途 |
|---|---|---|---:|---|
| r1 | `33177f56` | 0 | 1 | 诊断故障，非物理能力 |
| r2 | `632a307c` | 180 | 22048 | A90有效；B90固定手指偏差保留 |
| r3 | `d057487d` | 90 | 5125 | B90丢预压偏差保留 |
| r4 | `d65dee9e` | 90 | 16930 | B90保留实际PD预压，有效 |

总实际44104controls，有效协议A16930+B16930=33860；其余旧偏差B与故障全部计入消耗，未删除。有效TRACK controls A12430+B12430，获取各4500controls。所有run learning_updates/model_calls=0，入口和native pure-host代码可核；这不是学习效率结论。

client Python3.8.20/numpy1.24.4/scipy1.10.1，graspenv；native dexplore_v120_train/Torch2.0.1+cu118，GPU PhysX/CPU tensor exchange，seed195、assetScale1。只用GPU4/5，所有正常run原生退出0，GPU已释放；不影响0–3/6/7或任何训练。r2/r3/r4物理wall分别793.056/182.953/632.037s，部分重叠。输出合计约99.2MiB（写报告时，派生小文件后略增），低于2GiB；上轮盘余量约208GiB，本次brief48收尾核对约167GiB（共享盘快照，不能将差额归因于本任务），未删除历史outputs。

### 独立重算、回归与可核对代码

r2与r4独立重算均audit_failures=[]；r4位置误差最大差1.39e-17m、姿态2.98e-08rad、PD映射2.12e-07，判据与报告计数无不一致。跨run90组参考逐位一致，reset/交接q与object最大差均0。三个运行所用三份核心源码与所报commit共9/9逐字匹配，execution-proof.json留证；自有6个client/native PID均已退出。工程测试30 passed/0 failed（5文件，上轮1.91s，本次brief48复核2.70s；新跟踪14项），已有10/10故意错误被检出，含漏前馈、错导数/坐标/杠杆臂、连续腕套有限门、缺字段、caller关闭力保持、caller/class丢预压。测试不证明未取证的碰撞/动态rigid fixture。

- `object_tracking.py:21`通用SE3插值、`:45`五形状生成器、`:75`固定手物映射/杠杆臂、`:82`TRACK/原控制律；`object_tracking.py:88`的ObjectPoseTracker.__init__新增held_finger_targets继承实际PD，None只用于rigid A。
- `inspire_physics_host.py:214`opt-in运动学附件，`:229`接触诊断；默认不启用，不改task schema。
- `run_object_tracking.py:25`通用入口；`:132`的last_acquisition_pd=step(...)捕获真实PD，`:152`继承并传入B。精确diff：`git show d65dee9e -- src/task/wam-rl-policy-v3/src/wam_rl_policy_v3/object_tracking.py src/task/wam-rl-policy-v3/tools/run/run_object_tracking.py`。
- `tools/audit/audit_object_tracking.py:20`独立逐tick重算/首PD预压验收/全部格CSV，`:177`为A/B资格与参考一致性join。

原始每格：trajectory.npz（物体/palm/q/dq/力/tick）、reference.npz（SE3+秒+固定手物变换）、controls.npz（raw action/实际PD）、ticks.json（每tick误差/余量/门）、commands.json（目标/FF）、acquisition.json、handoff.json（r4）。完整原始目录均在`outputs/wam-rl-policy-v3/object-trajectory-tracking-20261023-r{1,2,3,4}/`，不入库。r4/qualified-matrix.csv含有效180格所有指标，qualified-matrix.md含45行四臂总表，stage-join-audit.json记录90组一致性；旧r2/r3各自matrix.csv保留偏差数据，旧r3的qualified标签已失效，当前join会拒绝缺预压记录的B。

复核命令（CPU，只重算派生文件，不跑物理/训练）：
```bash
PY=/home2/wyy/miniconda3/envs/graspenv/bin/python
$PY src/task/wam-rl-policy-v3/tools/audit/audit_object_tracking.py --run outputs/wam-rl-policy-v3/object-trajectory-tracking-20261023-r2
$PY src/task/wam-rl-policy-v3/tools/audit/audit_object_tracking.py --run outputs/wam-rl-policy-v3/object-trajectory-tracking-20261023-r4 --stage-a-run outputs/wam-rl-policy-v3/object-trajectory-tracking-20261023-r2
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' $PY outputs/wam-rl-policy-v3/object-trajectory-tracking-20261023-r4/check_mutations.py
PYTHONPATH=src/task/wam-rl-policy-v3/src PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' $PY -m pytest -q -p no:cacheprovider src/task/wam-rl-policy-v3/tests/test_object_tracking.py src/task/wam-rl-policy-v3/tests/test_source_pd.py src/task/wam-rl-policy-v3/tests/test_continuous_wrist.py src/task/wam-rl-policy-v3/tests/test_inspire_state_codec.py src/task/wam-rl-policy-v3/tests/test_inspire_reset_submission.py
```
期望：两份audit_failures=[]，90组reference_mismatches=[]，mutants10/10，pytest30 passed/0 failed。

实际物理命令（已完成；复跑必须新output且先查卡，不应覆盖）：
```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' /home2/wyy/miniconda3/envs/graspenv/bin/python -u src/task/wam-rl-policy-v3/tools/run/run_object_tracking.py --output outputs/wam-rl-policy-v3/object-trajectory-tracking-20261023-r2 --gpu 4 --matrix full --seconds 7200
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' /home2/wyy/miniconda3/envs/graspenv/bin/python -u src/task/wam-rl-policy-v3/tools/run/run_object_tracking.py --output outputs/wam-rl-policy-v3/object-trajectory-tracking-20261023-r3 --gpu 5 --matrix full --stages B --reference-start outputs/wam-rl-policy-v3/object-trajectory-tracking-20261023-r2/reference-start.npy --seconds 7200
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' /home2/wyy/miniconda3/envs/graspenv/bin/python -u src/task/wam-rl-policy-v3/tools/run/run_object_tracking.py --output outputs/wam-rl-policy-v3/object-trajectory-tracking-20261023-r4 --gpu 5 --matrix full --stages B --reference-start outputs/wam-rl-policy-v3/object-trajectory-tracking-20261023-r2/reference-start.npy --seconds 7200
```
r2/r3旧偏差需对应runtime-code才能复现，不能把当前修正入口说成同一代码。通用任意轨迹用`--reference-npz <times+poses.npz>`取代生成矩阵；该CLI边界仅CPU验证，不虚构另一次物理结果。

### 完整45组×A/B×FF（每格n=1）

每格为物体“位置P95 mm / 姿态P95 deg”。通过*仅指A运动学代理满足数值阈值，自碰撞和子步rigid负载仍未资格化；本表全都走完整段。P50/P90/max、每tick、闭合、相对漂移、关节名/余量、真实速度/曲率、servo cap都在qualified-matrix.csv及原始数据。

| 形状 | 尺度mm | 均速mm/s | A无FF P95 mm/deg | A有FF | B无FF | B有FF |
|---|---:|---:|---|---|---|---|
| straight | 20 | 30 | 9.60/10.40 未过阈值 | 9.63/7.95 未过阈值 | 10.28/9.71 未过阈值 | 10.31/7.33 未过阈值 |
| straight | 20 | 100 | 12.61/10.41 未过阈值 | 10.23/10.10 未过阈值 | 11.39/10.24 未过阈值 | 11.00/9.85 未过阈值 |
| straight | 20 | 250 | 14.21/10.71 未过阈值 | 12.47/11.70 未过阈值 | 13.85/10.56 未过阈值 | 12.24/11.56 未过阈值 |
| straight | 50 | 30 | 10.86/9.86 未过阈值 | 8.38/7.24 未过阈值 | 10.11/9.06 未过阈值 | 10.13/6.59 未过阈值 |
| straight | 50 | 100 | 26.27/10.87 未过阈值 | 18.44/8.22 未过阈值 | 22.48/10.24 未过阈值 | 15.18/7.56 未过阈值 |
| straight | 50 | 250 | 36.07/10.41 未过阈值 | 27.93/10.10 未过阈值 | 33.96/10.24 未过阈值 | 26.37/9.84 未过阈值 |
| straight | 100 | 30 | 11.96/7.52 未过阈值 | 8.83/5.48 未过阈值 | 10.54/6.93 未过阈值 | 11.21/5.10 未过阈值 |
| straight | 100 | 100 | 37.33/10.66 未过阈值 | 27.29/7.88 未过阈值 | 32.26/9.87 未过阈值 | 22.41/7.18 未过阈值 |
| straight | 100 | 250 | 62.97/10.62 未过阈值 | 51.42/8.90 未过阈值 | 58.64/10.10 未过阈值 | 47.59/8.26 未过阈值 |
| circle | 20 | 30 | 16.43/6.53 未过阈值 | 10.83/4.68 通过* | 26.02/6.04 未过阈值 | 22.43/4.64 未过阈值 |
| circle | 20 | 100 | 27.38/10.46 未过阈值 | 21.08/7.61 未过阈值 | 33.27/9.66 未过阈值 | 26.84/6.91 未过阈值 |
| circle | 20 | 250 | 36.44/10.05 未过阈值 | 29.48/8.17 未过阈值 | 40.34/9.39 未过阈值 | 32.66/7.48 未过阈值 |
| circle | 50 | 30 | 18.46/3.18 未过阈值 | 14.30/2.34 通过* | 28.13/3.80 未过阈值 | 24.11/3.55 未过阈值 |
| circle | 50 | 100 | 42.67/7.68 未过阈值 | 32.20/5.66 未过阈值 | 51.39/7.09 未过阈值 | 40.98/5.26 未过阈值 |
| circle | 50 | 250 | 65.71/10.51 未过阈值 | 50.22/7.61 未过阈值 | 70.53/9.67 未过阈值 | 55.79/6.91 未过阈值 |
| circle | 100 | 30 | 18.73/1.67 未过阈值 | 14.53/1.24 通过* | 28.46/3.47 未过阈值 | 24.39/3.36 未过阈值 |
| circle | 100 | 100 | 50.99/4.88 未过阈值 | 38.60/3.58 未过阈值 | 60.60/4.62 未过阈值 | 48.03/4.02 未过阈值 |
| circle | 100 | 250 | 102.15/8.52 未过阈值 | 82.79/6.23 未过阈值 | 112.66/7.86 未过阈值 | 92.67/5.74 未过阈值 |
| figure8 | 20 | 30 | 18.35/6.68 未过阈值 | 14.12/4.88 通过* | 28.37/6.16 未过阈值 | 24.13/4.70 未过阈值 |
| figure8 | 20 | 100 | 25.42/10.46 未过阈值 | 19.00/7.78 未过阈值 | 31.46/9.65 未过阈值 | 25.36/7.09 未过阈值 |
| figure8 | 20 | 250 | 26.92/10.90 未过阈值 | 22.24/8.26 未过阈值 | 30.72/10.26 未过阈值 | 25.65/7.58 未过阈值 |
| figure8 | 50 | 30 | 22.12/3.27 未过阈值 | 16.43/2.37 未过阈值 | 32.37/3.83 未过阈值 | 27.35/3.58 未过阈值 |
| figure8 | 50 | 100 | 46.00/7.82 未过阈值 | 34.70/5.73 未过阈值 | 55.12/7.20 未过阈值 | 43.67/5.31 未过阈值 |
| figure8 | 50 | 250 | 58.42/10.48 未过阈值 | 43.99/7.80 未过阈值 | 64.83/9.67 未过阈值 | 50.31/7.11 未过阈值 |
| figure8 | 100 | 30 | 23.06/1.72 未过阈值 | 17.71/1.27 未过阈值 | 33.44/3.48 未过阈值 | 28.12/3.37 未过阈值 |
| figure8 | 100 | 100 | 59.80/5.01 未过阈值 | 45.34/3.67 未过阈值 | 69.89/4.71 未过阈值 | 55.42/4.07 未过阈值 |
| figure8 | 100 | 250 | 95.96/8.64 未过阈值 | 73.50/6.32 未过阈值 | 105.21/7.95 未过阈值 | 82.85/5.82 未过阈值 |
| helix | 20 | 30 | 16.40/6.47 未过阈值 | 12.73/4.76 通过* | 26.14/5.99 未过阈值 | 22.53/4.63 未过阈值 |
| helix | 20 | 100 | 27.69/10.49 未过阈值 | 21.23/7.61 未过阈值 | 33.50/9.66 未过阈值 | 27.03/6.91 未过阈值 |
| helix | 20 | 250 | 36.95/10.05 未过阈值 | 29.79/8.17 未过阈值 | 40.79/9.39 未过阈值 | 33.04/7.48 未过阈值 |
| helix | 50 | 30 | 18.45/3.16 未过阈值 | 14.30/2.33 通过* | 28.29/3.80 未过阈值 | 24.24/3.55 未过阈值 |
| helix | 50 | 100 | 42.13/7.65 未过阈值 | 32.34/5.66 未过阈值 | 51.73/7.09 未过阈值 | 41.28/5.26 未过阈值 |
| helix | 50 | 250 | 66.19/10.51 未过阈值 | 50.03/7.61 未过阈值 | 71.10/9.67 未过阈值 | 55.48/6.91 未过阈值 |
| helix | 100 | 30 | 18.71/1.66 未过阈值 | 14.53/1.23 通过* | 28.57/3.47 未过阈值 | 24.49/3.36 未过阈值 |
| helix | 100 | 100 | 51.22/4.86 未过阈值 | 38.33/3.56 未过阈值 | 60.74/4.60 未过阈值 | 48.18/4.01 未过阈值 |
| helix | 100 | 250 | 99.79/8.53 未过阈值 | 79.61/6.22 未过阈值 | 109.82/7.86 未过阈值 | 89.17/5.74 未过阈值 |
| random_spline | 20 | 30 | 16.34/6.74 未过阈值 | 12.37/4.92 通过* | 19.96/6.21 未过阈值 | 17.75/4.72 未过阈值 |
| random_spline | 20 | 100 | 23.02/10.43 未过阈值 | 17.54/7.58 未过阈值 | 24.79/9.62 未过阈值 | 19.76/6.88 未过阈值 |
| random_spline | 20 | 250 | 25.21/10.87 未过阈值 | 22.49/8.22 未过阈值 | 27.49/10.24 未过阈值 | 25.20/7.55 未过阈值 |
| random_spline | 50 | 30 | 21.64/3.32 未过阈值 | 16.20/2.32 未过阈值 | 23.65/3.84 未过阈值 | 19.93/3.59 未过阈值 |
| random_spline | 50 | 100 | 44.88/7.88 未过阈值 | 35.50/5.76 未过阈值 | 45.52/7.26 未过阈值 | 34.45/5.34 未过阈值 |
| random_spline | 50 | 250 | 55.42/10.44 未过阈值 | 49.36/7.55 未过阈值 | 56.33/9.61 未过阈值 | 46.62/6.86 未过阈值 |
| random_spline | 100 | 30 | 24.41/1.63 未过阈值 | 17.34/1.29 未过阈值 | 26.65/3.49 未过阈值 | 21.78/3.37 未过阈值 |
| random_spline | 100 | 100 | 73.00/5.06 未过阈值 | 60.98/3.72 未过阈值 | 65.41/4.78 未过阈值 | 52.36/4.08 未过阈值 |
| random_spline | 100 | 250 | 115.62/8.68 未过阈值 | 110.24/6.36 未过阈值 | 110.73/7.98 未过阈值 | 104.24/5.84 未过阈值 |

### 实测/自报/待定与Git

实测：本轮物理pose/q/接触与实际PD返回、180格全路径指标、缺陷前后持握改善。独立核算：误差/判据/PD/首目标/同参考同初态。自报且入口可查：0模型/训练调用、资源预算与进程退出。待定：真实子步刚体固定的载荷、自碰撞、剩余跟踪误差的动力学单因、其它初态/物体；不把它们写成PASS，也不拿本轮否定τ或学习路线。

提交只含本轮显式路径；pull --rebase被他人脏文件拒绝，未stash/覆盖/加入他人修改，普通push。最终HEAD/origin/main一致证明在交付回答给出。
