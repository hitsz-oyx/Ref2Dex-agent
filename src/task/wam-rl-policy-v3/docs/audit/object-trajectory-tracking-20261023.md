| 观测 | 含义 | 决定 |
|---|---|---|
| 常见形状/参数下物体误差 ≤ 阈值、无碰撞、限位余量充足、漂移小 | **通用物体轨迹跟踪在现有执行器上可用**（走控制、不走学习） | 建立“物体轨迹 → 执行”通道为主线的标准能力，后续任务复用 |
| 只在高速/大曲率失败 | 存在**可行域边界** | 写明边界由哪一类约束决定；报告加前馈后边界移动多少 |
| 小参数也撞关节限位 | **可达性/限位是主线 P0**，与轨迹无关 | 停止扩展轨迹形状，回到执行器 P0（joint-10 限位） |
| 必须滑移/力反馈才不脱手 | 持握稳定是**必须新增件** | 列为下一项工程件（含最小可行实现方案） |

**实测解读：** A无FF 0/45、有FF 8/45满足数值阈值；B无FF/有FF均0/45。真实持握通用跟踪尚未资格化；8格仅为运动学刚体上限候选，自碰撞未取证，不能升级成完整可用。

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

## 结算：不是通用真实持握能力通过

有效协议180格全部有数字：A来自r2的90格，B来自修正r3的90格；无缺格、无删样本、无学习、无模型/τ调用。r2额外90格固定手指B与r1故障完整保留，不并入约定B。

| 组别（每组45格，非重复成功率） | 完成整条轨迹 | 数值阈值通过 | 各格位置P95中位 mm | 各格姿态P95中位 deg | canonical裁剪 | 物理关节/桌面中止 |
|---|---:|---:|---:|---:|---:|---:|
| A无FF | 45/45 | 0/45 | 27.378 | 8.534 | 0/6215 | 0/45；0/45 |
| A有FF | 45/45 | 8/45 | 22.241 | 6.228 | 0/6215 | 0/45；0/45 |
| B无FF | 3/45 | 0/45 | 10.556 | 39.060 | 0/309 | 0/45；0/45 |
| B有FF | 3/45 | 0/45 | 10.566 | 39.276 | 0/316 | 0/45；0/45 |

B的84/90路径因失持提前停止，位置/姿态分位数只覆盖已执行前缀，不与完整A作性能排名；6条短路径虽走完仍未达标，不能用短时未中止当持握成功。每组n=45是45个参数格，各格仅1次；不当作独立重复/泛化成功率，不补无意义的臂间显著性检验。

### 可行格与前馈边界

A有FF的8个候选全部在平均30mm/s档：circle尺度20/50/100mm，figure8尺度20mm，helix尺度20/50/100mm，固定seed随机样条尺度20mm。其实际速度峰值分别见CSV，整体范围 55.83–109.36mm/s；这不是“减速到几乎不动”。100/250mm/s档没有数值通过格；无FF任何档均0。离散域从0→8格（+8/45），不插值成连续可达边界。直线低速也未过姿态阈值，因此**不是只有高速/高曲率才失败**。

45个同参考完整A配对：有FF−无FF的位置P95差中位 -6.423mm（范围 -22.459至0.023），姿态P95差中位 -1.995deg；位置P95跨格中位下降18.77%，姿态下降27.02%。个别格会变差，不挑最好值。相同reference的逐tick对照保存在两格的ticks.json/trajectory.npz。

闭合circle/figure8共有18格/臂，A全执行完。闭合位置误差跨格中位：无FF3.358mm→有FF2.744mm；闭合姿态1.030deg→0.573deg。8个通过候选中的闭合4格，末端位置误差0.533–2.773mm、姿态0.129–0.607deg。B的闭合形状均未走完整环，**闭合误差未定义**；不拿中止点到起点的距离当闭合误差。

### 持握完整性、限位、碰撞与失败分类

- A相对漂移最大0.000105mm、0.000019deg是**施加的刚体假设和数值舍入**，不是实测抓力稳定。约束仅每控制tick作用在root状态，不能当作子步刚性焊接或负载验证；实际物体碰撞/惯性在tick间仍存在，不能称完全排除所有动态影响。
- B无FF最大相对漂移6.147mm/41.443deg；有FF12.764mm/44.589deg。90/90在TRACK第1步（0.03333s）失去双指力接触；力+相对姿态同时合格的最长连续持握elapsed全部0s。两臂各42/45由连续0.2秒失持门中止；其余3/45为短路径（0.1–0.2s）完成，但也不合格。
- 有效A最小nominal限位余量0.07888985rad，B最小0.06219673rad，均为native q[14] `thumb_proximal_yaw_joint`；超过预注册0.02rad余量。连续腕关节不套nominal有限边界。关节安全中止0/180，不能把此前其它初态的joint-10结论迁移到这一个已知持握初态。
- 0 canonical裁剪不等于内部伺服未饱和：A无FF平移cap 87/6215、角cap0/6215；A有FF平移110/6215、角12/6215；B无FF均0/309，B有FF平移4/316、角6/316。caps沿用原位置±25mm/角±0.12rad每tick，未修改。
- 原生筛选记录中的手/物-桌接触（lambda>0.1）及physical_table门中止均0；不能升级成严格“无碰撞”。本机绑定不提供initial_overlap，完整负接触对也未独立落盘。更关键：`third_party/DExplore/dexplore/env/tasks/dexplore_inspire.py:15` 的COLLISION_GROUP_SELF=1，`_apply_collision_filter:20` 又设shape filters 2/3；本机Isaac文档assets.html:198说明actor filter>0禁用self collisions。**自碰撞几何未取证**；没有为了通过去改filter，不认证无自碰撞。

| 阶段 | 失败分母 | 限位/余量不足 | 滞后/姿态/闭合阈值 | 相对滑移或失持 | 记录到的碰撞 | 其它 |
|---|---:|---:|---:|---:|---:|---:|
| A | 82 | 0 | 82（100%） | 不适用：刚体假设 | 0（自碰撞未取证） | 0 |
| B | 90 | 0 | 1（1.11%） | 89（98.89%） | 0（自碰撞未取证） | 0 |

分类先用实际中止门，再看完成路径的相对漂移、限位和误差阈值；89含84个失持中止与5个已完成但姿态相对漂移超10deg，余1个短轨迹为跟踪误差超限。这是观测约束归类，不是因果识别。另保存“首次任一阈值越界”时间：它常先是姿态误差，不能从先后顺序证明滞后导致滑移。

### 已查到的启动条件限制与下一件工程工作

共同初态、共同源回放的交接速度并非静止：腕线速度0.855479m/s（90格完全相同），腕角速度约0.425rad/s；物体参考位置端点速度严格0。最后0.3秒手物相对几何变化约5.419mm/7.062deg（按已测palm/FK从保存q重建），未超10mm/10deg，但不等于静止持握。原XYZ腕FK与实测palm旋转的最大偏差2.73e-5deg，未看到Euler顺序错误证据。

因此不能把B结果推成“任何物体轨迹都不能跟踪”，也不能宣布“力反馈是必要且充分新增件”：已有弱力保持已开启，仍未过；**交接瞬态/持握未资格化**与稳态轨迹难度混在一起。下一件最小工程件是通用GRASP→TRACK交接：用实测手物相对漂移、接触与速度门确认抓稳，保持真实接触下使初始设定点/一阶导连续（或用有界过渡段匹配live速度），然后再消费原轨迹。若仍滑移，再以物体相对手SE3误差与双指力作为持握反馈，限速并作抗饱和；不需模型、τ或训练。该方案本轮未实施/未验证，不用静止不接触来计通过。

### 时间律、导数和通用输入

T_HO=inv(T_WH)·T_WO在获取时固定；T_WH_des(t)=T_WO_ref(t)·inv(T_HO)。v_hand=v_object+ω_world×(p_hand−p_object)。s(u)=10u³−15u⁴+6u⁵，采样121个SE3 knot，弧长/平均速度决定时长，向上对齐1/30秒；实际执行位置CubicSpline(t)及其一阶导，角速度R(t)·RotationSpline(t,1)，以真实导数而非把24帧当时钟。`RotationSpline`返回body angular rate，世界旋转已转换，非单位混用。

位置clamped端点导数严格0；RotationSpline不是强制角端点导数为0，原预注册“端点静止”对角度只是近似，本矩阵端点角速率最大 0.00752056rad/s，已与0.425rad/s交接速度分开，未悄悄改样条。实际速度、加速度、角速度峰值及实际曲率半径min/median全部在CSV；直线曲率0，半径空值=无穷/不适用，尺度不能冒充其曲率半径。

通用入口支持`--reference-npz`（times秒、poses Nx4x4、world frame）；默认矩阵仅是该同一SE3消费通道的测试生成器。提交版另外处理任意终点不在30Hz网格上的输入：最后不足一tick的时间按原终点hold比较，不外推；这项仅CPU接口检查，未另加物理rollout。阶段信息为`phase=TRACK`，目标经原canonical→PD，无task schema变更。

### 独立复核、版本与产物

r2运行代码632a307c7b627ed753e5d06097d45b021ab8bb53，r3运行代码d057487daf8a4f02047b00746e3491075c4cd9f8；与交付后补的审计器/通用入口鲁棒性分开，不伪称后补代码参与了已跑物理。每轮runtime-code保存当时源码，native.log/physics/manifest.json保存环境和控制器属性；client graspenv，native Python3.8/Torch2.0.1+cu118，seed195，GPU4/5，GPU PhysX/CPU tensor exchange。

两轮全部270格独立重算0不一致：r2位置误差最大差1.39e-17m、姿态2.98e-8rad、PD1.23e-7；r3分别6.94e-18m、2.98e-8rad、1.87e-7。90组有效A/B参考SE3与时间戳逐位一致；reset q/object、交接 q/object最大差均0。只认证所重算字段，不认证缺失的自碰撞真值。

工程回归28 passed / 0 failed（5文件，1.79s），其中新跟踪12项；8/8故意错误被检出：导数符号、body/world角速度混淆、手物变换方向、漏杠杆臂、连续腕套有限门、缺失接触字段、漏前馈、真实runner关闭B力保持。测试不替代物理资格。

计量：r1故障物理1control（client未收到回包计0）；r2实际22048，r3实际5125，共27174controls。有效协议A16930（含获取4500、TRACK12430）、B5125（含获取4500、TRACK625），另有r2偏差B5118与故障1，全部明列。r2/r3原生退出均0；运行耗时793.056s/182.953s，部分重叠；无训练更新，不混作学习样本效率。新增产物约59.5MiB，低于登记2GiB；磁盘最后检查约223GiB，未删任何历史产物。GPU4/5进程均退出/显存4MiB；未使用0–3/6/7，其余卡占用不干预。

原始目录：`outputs/wam-rl-policy-v3/object-trajectory-tracking-20261023-r{1,2,3}/`。每格`trajectory.npz`为逐执行后物体/palm/q/dq/力，`ticks.json`为逐tick物体误差、相对漂移、限位、门；`reference.npz`为SE3+时间戳，`controls.npz`为raw action和实际PD target，`commands.json`为TRACK目标与前馈；`acquisition.json`含前置抓取。r3/qualified-matrix.csv为有效180格所有指标，r3/qualified-matrix.md为45组四格总表，r2/matrix.csv保留偏差B全部数字。outputs不入库。

### 可复核diff与命令

- `src/task/wam-rl-policy-v3/src/wam_rl_policy_v3/object_tracking.py:21`：通用SE3插值；:45 五种同机制测试生成器；:75固定手物映射及杠杆臂；:82复用位置式PD消费与FF；:10绑定dtype兼容。
- `src/task/wam-rl-policy-v3/tools/run/inspire_physics_host.py:214`：opt-in运动学附件；:229原生接触诊断。默认未加flag不改变生产RPC/物理。
- `src/task/wam-rl-policy-v3/tools/run/run_object_tracking.py:25`：通用入口、显式时间、阶段A/B、固定矩阵；r3恢复B既有实测力保持的调用已留diff。`git show d057487d -- src/task/wam-rl-policy-v3/tools/run/run_object_tracking.py`。
- `src/task/wam-rl-policy-v3/tools/audit/audit_object_tracking.py:20`：独立误差/PD/判据/servo cap重算与全矩阵；:167后续增加的跨阶段join函数以当前源码为准。

复核只读原始记录，重写派生CSV/JSON（不跑物理、不训练）：
```bash
PY=/home2/wyy/miniconda3/envs/graspenv/bin/python
$PY src/task/wam-rl-policy-v3/tools/audit/audit_object_tracking.py --run outputs/wam-rl-policy-v3/object-trajectory-tracking-20261023-r2
$PY src/task/wam-rl-policy-v3/tools/audit/audit_object_tracking.py --run outputs/wam-rl-policy-v3/object-trajectory-tracking-20261023-r3 --stage-a-run outputs/wam-rl-policy-v3/object-trajectory-tracking-20261023-r2
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' $PY outputs/wam-rl-policy-v3/object-trajectory-tracking-20261023-r3/check_mutations.py
PYTHONPATH=src/task/wam-rl-policy-v3/src PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' $PY -m pytest -q -p no:cacheprovider src/task/wam-rl-policy-v3/tests/test_object_tracking.py src/task/wam-rl-policy-v3/tests/test_source_pd.py src/task/wam-rl-policy-v3/tests/test_continuous_wrist.py src/task/wam-rl-policy-v3/tests/test_inspire_state_codec.py src/task/wam-rl-policy-v3/tests/test_inspire_reset_submission.py
```
期望：两个audit_failures=[]；stage join reference_mismatches=[]、90组；mutants8/8；pytest28 passed/0 failed。时间与输出bytes可能因派生产物增加略变。

实际物理命令（已有结果，不应因读报告再重跑；复现实验必须改新output，先查GPU）：
```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' /home2/wyy/miniconda3/envs/graspenv/bin/python -u src/task/wam-rl-policy-v3/tools/run/run_object_tracking.py --output outputs/wam-rl-policy-v3/object-trajectory-tracking-20261023-r2 --gpu 4 --matrix full --seconds 7200
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' /home2/wyy/miniconda3/envs/graspenv/bin/python -u src/task/wam-rl-policy-v3/tools/run/run_object_tracking.py --output outputs/wam-rl-policy-v3/object-trajectory-tracking-20261023-r3 --gpu 5 --matrix full --stages B --reference-start outputs/wam-rl-policy-v3/object-trajectory-tracking-20261023-r2/reference-start.npy --seconds 7200
```
r2须使用当时runtime-code的固定手指实现才能复现偏差B；当前入口已纠正，不能声称它会逐位复现旧偏差运行。r3命令对应纠正后的既有力保持。通用用户轨迹只需用`--reference-npz <输入.npz>`替换矩阵，创建新output；未另外运行。

### 全45组×A/B×FF物体误差（每格n=1）

下表每格“位置P95 mm / 姿态P95 deg”。“通过*”仅数值上满足，A的运动学假设与自碰撞缺口仍在；中止格是前缀分位数。所有P50/P90/max、逐tick、闭合、漂移、最小关节、实际速度/曲率、裁剪/伺服cap见qualified-matrix.csv，不删失败。空白闭合值仅不闭合/未走完整环。

| 形状 | 尺度mm | 均速mm/s | A无FF P95 mm/deg | A有FF | B无FF | B有FF |
|---|---:|---:|---|---|---|---|
| straight | 20 | 30 | 9.60/10.40 未过阈值 | 9.63/7.95 未过阈值 | 10.51/44.53 中止:slip | 10.54/46.00 中止:slip |
| straight | 20 | 100 | 12.61/10.41 未过阈值 | 10.23/10.10 未过阈值 | 15.16/11.11 未过阈值 | 11.78/10.60 未过阈值 |
| straight | 20 | 250 | 14.21/10.71 未过阈值 | 12.47/11.70 未过阈值 | 19.04/28.21 未过阈值 | 18.16/23.26 未过阈值 |
| straight | 50 | 30 | 10.86/9.86 未过阈值 | 8.38/7.24 未过阈值 | 10.56/40.03 中止:slip | 10.54/40.76 中止:slip |
| straight | 50 | 100 | 26.27/10.87 未过阈值 | 18.44/8.22 未过阈值 | 17.82/40.17 中止:slip | 14.13/17.07 中止:slip |
| straight | 50 | 250 | 36.07/10.41 未过阈值 | 27.93/10.10 未过阈值 | 40.73/31.81 未过阈值 | 34.33/25.65 未过阈值 |
| straight | 100 | 30 | 11.96/7.52 未过阈值 | 8.83/5.48 未过阈值 | 10.56/39.03 中止:slip | 10.58/41.23 中止:slip |
| straight | 100 | 100 | 37.33/10.66 未过阈值 | 27.29/7.88 未过阈值 | 10.50/38.92 中止:slip | 10.47/41.18 中止:slip |
| straight | 100 | 250 | 62.97/10.62 未过阈值 | 51.42/8.90 未过阈值 | 51.04/44.41 中止:slip | 40.24/10.34 中止:slip |
| circle | 20 | 30 | 16.43/6.53 未过阈值 | 10.83/4.68 通过* | 10.56/37.56 中止:slip | 10.58/40.21 中止:slip |
| circle | 20 | 100 | 27.38/10.46 未过阈值 | 21.08/7.61 未过阈值 | 10.52/41.09 中止:slip | 10.52/41.43 中止:slip |
| circle | 20 | 250 | 36.44/10.05 未过阈值 | 29.48/8.17 未过阈值 | 32.28/46.08 中止:slip | 38.83/45.83 中止:slip |
| circle | 50 | 30 | 18.46/3.18 未过阈值 | 14.30/2.34 通过* | 10.55/38.11 中止:slip | 10.57/40.95 中止:slip |
| circle | 50 | 100 | 42.67/7.68 未过阈值 | 32.20/5.66 未过阈值 | 10.55/39.68 中止:slip | 10.56/39.33 中止:slip |
| circle | 50 | 250 | 65.71/10.51 未过阈值 | 50.22/7.61 未过阈值 | 12.82/40.47 中止:slip | 11.43/41.20 中止:slip |
| circle | 100 | 30 | 18.73/1.67 未过阈值 | 14.53/1.24 通过* | 10.55/39.05 中止:slip | 10.56/37.62 中止:slip |
| circle | 100 | 100 | 50.99/4.88 未过阈值 | 38.60/3.58 未过阈值 | 10.54/39.02 中止:slip | 10.58/39.71 中止:slip |
| circle | 100 | 250 | 102.15/8.52 未过阈值 | 82.79/6.23 未过阈值 | 10.52/38.48 中止:slip | 10.54/38.68 中止:slip |
| figure8 | 20 | 30 | 18.35/6.68 未过阈值 | 14.12/4.88 通过* | 10.56/37.14 中止:slip | 10.56/38.30 中止:slip |
| figure8 | 20 | 100 | 25.42/10.46 未过阈值 | 19.00/7.78 未过阈值 | 10.57/38.98 中止:slip | 10.59/42.40 中止:slip |
| figure8 | 20 | 250 | 26.92/10.90 未过阈值 | 22.24/8.26 未过阈值 | 18.17/45.27 中止:slip | 19.90/44.68 中止:slip |
| figure8 | 50 | 30 | 22.12/3.27 未过阈值 | 16.43/2.37 未过阈值 | 10.55/38.11 中止:slip | 10.56/38.94 中止:slip |
| figure8 | 50 | 100 | 46.00/7.82 未过阈值 | 34.70/5.73 未过阈值 | 10.56/38.61 中止:slip | 10.57/39.28 中止:slip |
| figure8 | 50 | 250 | 58.42/10.48 未过阈值 | 43.99/7.80 未过阈值 | 18.29/40.92 中止:slip | 15.20/40.60 中止:slip |
| figure8 | 100 | 30 | 23.06/1.72 未过阈值 | 17.71/1.27 未过阈值 | 10.55/39.46 中止:slip | 10.56/38.26 中止:slip |
| figure8 | 100 | 100 | 59.80/5.01 未过阈值 | 45.34/3.67 未过阈值 | 10.55/40.49 中止:slip | 10.59/41.71 中止:slip |
| figure8 | 100 | 250 | 95.96/8.64 未过阈值 | 73.50/6.32 未过阈值 | 10.57/39.58 中止:slip | 10.60/41.08 中止:slip |
| helix | 20 | 30 | 16.40/6.47 未过阈值 | 12.73/4.76 通过* | 10.56/37.56 中止:slip | 10.57/39.42 中止:slip |
| helix | 20 | 100 | 27.69/10.49 未过阈值 | 21.23/7.61 未过阈值 | 10.52/41.18 中止:slip | 10.52/40.10 中止:slip |
| helix | 20 | 250 | 36.95/10.05 未过阈值 | 29.79/8.17 未过阈值 | 32.78/43.28 中止:slip | 30.99/19.26 中止:slip |
| helix | 50 | 30 | 18.45/3.16 未过阈值 | 14.30/2.33 通过* | 10.55/38.11 中止:slip | 10.57/40.95 中止:slip |
| helix | 50 | 100 | 42.13/7.65 未过阈值 | 32.34/5.66 未过阈值 | 10.55/39.89 中止:slip | 10.56/38.87 中止:slip |
| helix | 50 | 250 | 66.19/10.51 未过阈值 | 50.03/7.61 未过阈值 | 12.80/41.81 中止:slip | 11.58/41.59 中止:slip |
| helix | 100 | 30 | 18.71/1.66 未过阈值 | 14.53/1.23 通过* | 10.55/39.06 中止:slip | 10.56/37.88 中止:slip |
| helix | 100 | 100 | 51.22/4.86 未过阈值 | 38.33/3.56 未过阈值 | 10.54/38.83 中止:slip | 10.58/39.96 中止:slip |
| helix | 100 | 250 | 99.79/8.53 未过阈值 | 79.61/6.22 未过阈值 | 10.53/38.40 中止:slip | 10.54/39.45 中止:slip |
| random_spline | 20 | 30 | 16.34/6.74 未过阈值 | 12.37/4.92 通过* | 10.56/40.66 中止:slip | 10.57/39.19 中止:slip |
| random_spline | 20 | 100 | 23.02/10.43 未过阈值 | 17.54/7.58 未过阈值 | 10.56/40.64 中止:slip | 10.56/38.99 中止:slip |
| random_spline | 20 | 250 | 25.21/10.87 未过阈值 | 22.49/8.22 未过阈值 | 16.61/43.05 中止:slip | 19.57/10.39 中止:slip |
| random_spline | 50 | 30 | 21.64/3.32 未过阈值 | 16.20/2.32 未过阈值 | 10.55/37.52 中止:slip | 10.56/38.79 中止:slip |
| random_spline | 50 | 100 | 44.88/7.88 未过阈值 | 35.50/5.76 未过阈值 | 10.55/37.10 中止:slip | 10.56/38.54 中止:slip |
| random_spline | 50 | 250 | 55.42/10.44 未过阈值 | 49.36/7.55 未过阈值 | 10.56/40.69 中止:slip | 10.56/38.09 中止:slip |
| random_spline | 100 | 30 | 24.41/1.63 未过阈值 | 17.34/1.29 未过阈值 | 10.55/39.38 中止:slip | 10.56/38.77 中止:slip |
| random_spline | 100 | 100 | 73.00/5.06 未过阈值 | 60.98/3.72 未过阈值 | 10.56/37.20 中止:slip | 10.58/38.54 中止:slip |
| random_spline | 100 | 250 | 115.62/8.68 未过阈值 | 110.24/6.36 未过阈值 | 10.56/38.35 中止:slip | 10.56/37.25 中止:slip |

### Git与可信度边界

推送前git pull --rebase被他人的未提交改动拒绝，未stash、未覆盖或加入他们的文件；使用显式路径commit --only与普通push。原始结果已冻结并可由上述命令独立复核。运行计数/model_calls=0由入口与native pure-host路径核对；碰撞完整性、rigid负载等未取证项明确不升级。最终HEAD/远端一致性在交付回答给出。

## r4交接预压修复（先登记，再只重测B）

r3（已恢复force-valid调用）第一TRACK指令仍错误地用实测q初始化手指目标，从source实际PD [1.364090,1.050968,0.807545,1.385341,1.111492,0.426087]切为[0.721299,0.885526,0.730134,1.390613,1.072549,0.266685]rad，最大卸载0.642792rad。这是新增交接入口自己丢失预压，不能用其0/45否定保持原抓取时的执行能力。最小修复：B的初始手指状态继承最后实际施加的PD targets，后续沿用原force规则/原upper；A不改不重跑，原始B全部保留，最终结算以r4替代B资格。回归必须抓住“把PD目标替换成实测q”的故意错误。

期望变化：首TRACK手指目标最大突降0.642792rad→不超过原force一步更新及既有upper钳位（本初态thumb1.111492→1.10的0.011492rad），食指目标约1.364rad，不再重置0.721；实际B保持时长/跟踪成功是否改善以r4为准，不预设必升。

r4只B、同90格/同参考文件/seed195/新output、空闲GPU5，参数与判据不改；允许joint P0停止，其余依照原预算300000controls/7200s/2GiB，不启动其它方法。
