# Teacher/source-PD与真实动作/reset验收（2026-10-22）

本轮完成：生产动作/reset工程通过24/24方向、两次14字段indexed隔离差0，最小continuous腕初始化修复后cubemedium首tick实测旋转.000435rad。完整任务正控仍未资格：原teacher/source-PD双评分0/6（teacher5删失），branch0/1；这些物理run在生产reset修复之前或审计branch下，不能套作修复后完整链0/6。总9366envcontrols，无训练，所有自有进程退出。源raw RELEASE由接触结束推断，可能是滑落，尚非认证计划意图。

## 事前协议与实际结果

预注册卡：`src/task/ptv3-shared-wam-v6/docs/experiments/probes/P-20261022-teacher-action-reset-validation.md`。
原生入口：`src/task/wam-rl-policy-v3/tools/run/inspire_physics_host.py`；客户端：`src/task/ptv3-shared-wam-v6/tools/audit/validate_native_action_reset.py`。运行身份由manifest的完整命令、原生hello和runtime记录；按机主要求没有执行Git操作，没有把当前main修复推称为历史版本。

完成run：`outputs/ptv3-shared-wam-v6/teacher-action-reset-validation-20261022-r3/`。
`manifest.json`、`trace.npz`、`reset-snapshots.npz`为实测；`acceptance-summary.json`为CPU派生汇总。native独立审计在`outputs/wam-rl-policy-v3/teacher-action-reset-validation-20261022-r3/manifest.json`。

| 指标 | 样本量/基线或预期 | 实际 | 结论 |
|---|---|---|---|
| 实际PD对独立NumPy合同 | 388控制×2env×18DOF；误差≤1e-6 | max abs 0 | 通过 |
| 有符号实测q响应 | 12 active轴×正负=24；响应≥25%、终点误差≤75%幅值 | 24/24；响应98.1494%–99.2358%，最大终点误差1.8506% | 通过本局部小扰动验收 |
| q/dq有限，tick+1、episode不变、poststep contact有效 | 388×2转移 | 全部通过 | 通过 |
| 第2步reset env0、第4步reset env1的untouched影响 | 2次；13个实测/身份字段应逐数组差0 | 全部max abs 0 | 通过即时indexed隔离 |
| 目标env reset | 2次；episode+1、tick0、force invalid | 2/2；下一控制tick+1且有效 | 通过 |
| 完整任务teacher/source-PD≥95% | 固定前6条；6/6才过 | 未测，null | 待任务合同冻结，不宣称通过 |

物理预算实际r3=388batched=776envcontrols；r2失败前5batched=10envcontrols；r1在第1控制前停止=0。累计786envcontrols，低于10000。r3包含native初始化总wall27.06s。native Python为`/home2/wyy/oyx_ws/.runtime_envs/dexplore_v120_train/bin/python`、Torch2.0.1+cu118、Isaac GPU PhysX；client为`/home2/wyy/miniconda3/envs/graspenv/bin/python`，CPU NumPy。每次native启动前检查GPU4占用。

## 合同和代码定位

- 客户端L7–24固定active轴、coupling及独立PD/逆映射；L51–59 indexed reset逐字段差；L60–68实际PD、tick/episode、实测q/dq与forcevalid；L72–87四步reset及24方向响应。它没有调用被验证的Torch PD函数来构造期待值。
- 原生`inspire_reset_submission.py:L8–34`排队逻辑reset，在下一次prephysics合并actor-root/DOF indexed setters，并持有index tensor至simulate。此次没有修改native reset源码。
- `trajectory_policy/pd_retargeter.py:L38–62`是读到的参考合同：腕相对live q、指关节[-1,1]映射[0,1]、固定被动coupling。实际PD来自host截获native `_action_to_pd_targets`，不是仅验证客户端回环。
- 手指warmup使用合法canonical0（指关节目标中点），腕每步zero delta；20步后以实测q构造正负轴目标，再固定目标12步。腕平移3mm、旋转.03rad、指关节.08rad。轴门槛未因结果改变。

## 失败保留及未自证范围

r1：用reset实测q直接反解finger自保持，因某些finger reset q不在合法PD目标区间，反解超[-1,1]，0控制中止。r2：改成合法fixture，但warmup仍固定初始腕绝对角，5控制后中止。cubemedium reset wrist yaw=6.753245rad，首步实测=-3.017649rad，导致回初始数值所需增量约9.77rad超动作界。r3只修复工具预置动作，不改native，不降低验收阈值。

这说明初始q的数值自保持不是普遍可执行命令；Euler/DOF表示与首物理步约束的关系仍需独立定位。尚未证明这9.77rad数值跳变是等效姿态重参数化或真实大旋转，不能称其已修复。工程通过仅说明预置后局部响应和即时reset隔离；未做counterfactual双模拟证明reset后的物理演化与另一条无reset世界完全相同，未测其他14条panel、碰撞中大动作或完整任务可达性。

## 有判别力的mutants

在同一实测q/action上CPU重算错误PD，未额外跑物理。遗漏腕current_q的max PD差6.753245；错误指关节[-1,1]映射（其余腕/耦合保持正确）差.925412；错误重复**相加**distal coupling差.924244，均大于1e-6而被判别。

原r3 manifest的wrong-finger mutant同时漏了腕current_q，不能据其6.753245归因finger。原manifest保留，隔离修正值写`acceptance-summary.json`，工具L89也已修正。纯粹再次调用固定赋值coupling是幂等，差0，**未杀死**且不宣称杀死。mutants验证验收算式有判别能力，不证明所有native实现bug已排除。

## 可核命令

```sh
/home2/wyy/miniconda3/envs/graspenv/bin/python -u src/task/ptv3-shared-wam-v6/tools/audit/validate_native_action_reset.py --panel outputs/ptv3-shared-wam-v6/executor-qualification-20261022-r1/panel.json --output outputs/ptv3-shared-wam-v6/teacher-action-reset-validation-20261022-r4 --gpu 4
```

该命令需GPU4仍获分配且空闲；输出必须新目录，r3不可覆盖。完整native命令见r3 manifest。后续teacher/source-PD最多6序列×600controls×2臂=7200envcontrols，与已用786合计≤7986；任务语义版本和完整终点/settle可观测性先冻结，删失单列，失败保留，不追求把指标放宽到满分。

## Fresh official与source-PD六例上限：双评分结果

Fresh official实际用原collector行为与权重，wrapper只作两处可核源码instrumentation：600controls硬上限、Git查询替换为明确dirty身份。两处diff与源码位于`outputs/ptv3-shared-wam-v6/teacher-upper-bound-validation-20261022-r1/{collector.original.py,collector.bounded.py,collector.diff,source-index.json}`；原生checkpoint是`/home2/wyy/oyx_ws/_external/dexplore_official_v120/checkpoint/inspire.pth`，未复制、未训练。新工具`collect_bounded_teacher.py`，保留原official observation normalization、deterministic get_action、[-1,1]clamp和reset补丁。无teacher输入构造变更。样本量是固定panel前6条，不按fresh结果筛选：airplane/cubemedium/cubesmall/cup/cylindermedium/cylindersmall，各1序列。原生按motion名排序；manifest保存实际顺序。

Fresh official证据`outputs/wam-rl-policy-v3/teacher-upper-bound-validation-20261022-r1/`，600batched×6=3600envcontrols，64.91s；source原frames按实际顺序为167/233/207/653/194/166。airplane上限601帧，其他按原collector保存逻辑裁剪到原source长度，**5条缺额外末端settling实录**，不可编造后缀补成成功。对原几何ref的object位置RMSE分别4.26/3.70/4.21/17.70/39.96/4.74cm；这项RMSE不是任务成功。

Source-PD证据`outputs/ptv3-shared-wam-v6/source-pd-upper-bound-validation-20261022-r1/`，600×6=3600envcontrols；全部6env都保留完整601状态。沿每tick真实live q将fresh官方实际PD target反解为合法canonical action；较短source结束后保持最后PD target，`within_reference`单列。初态q/dq/hand逐数组exact；object quaternion→SE3数值重构误差9.94e-8，不能写作所有数组exact。未使用旧manifest训练准入筛选，不训练。

评分使用同一份**旧source full raw-task声明**`outputs/wam-rl-policy-v3/inspire-teacher-rollout-audit-20261020-r3/sequences/*/raw-task.json`，没有用fresh future重新标目标。六条均DEV，新的501 TRAIN sidecar不包含它们；读不到新DEV sidecar明确使用原full raw，未使用旧prefix task.json的HOLD重标。目标意图仍未经独立认证。

两个评分器独立子进程导入完整冻结package：

- 旧：`outputs/ptv3-shared-wam-v6/teacher-task-scoring-20261022-r1/old/src/task/wam-rl-policy-v3/src/wam_rl_policy_v3/`，root导出完整SHA`60efadce2444709bbbf9802107de7bc5e3f9eef6`。
- 新：同run的`new/src/task/wam-rl-policy-v3/src/wam_rl_policy_v3/`，当前dirty源码快照，scheduler schema`ref2dex.measured-task-scheduler.release-at-declared-pose.v2`；root通知28项相关回归通过（本子任务未独立重跑）。

`score_teacher_action_reset.py`在CPU重放实测pose/hand/velocity、native五body contact和force validity，独立重算coupling、lift/hold、support及admission failure masks。source-PD支持用实际动态table pose和object净contact force；teacher未保存table pose，用旧source几何初始化固定table，须保留这个限制。source-PD实测table600步max变化0，与旧sidecar初始值误差≤3.58e-7；未把旧supported mask复用为新实测值。

| 指标 | 样本量与门槛 | 旧评分 | 新评分 | 结论 |
|---|---|---|---|---|
| Fresh official完整task | 6；≥95%需6/6 | 0/6；5删失、1明确path failure | 相同 | 未资格，不能估计成已证实的0% teacher能力 |
| Source-PD完整task | 6；≥95%需6/6 | 0/6 | 0/6 | FAIL，未证实完整task正控 |
| Fresh official force/lift hold30 | 6；lift≥5cm、unsupported、palm≤.2m、至少2body>0.1N、连续30controls | 6/6 | 6/6 | 仅代理通过 |
| Source-PD force/lift hold30 | 同6与同门槛 | 6/6 | 6/6 | 仅代理通过 |
| Path完成 | 2臂各6 | 0/6、0/6 | 0/6、0/6 | 失败发生在release前 |
| source期PD clipping | 6条 | 仅cubemedium10controls | 同一物理run | 精确PD上限合同失败，其他5条source期0 |

逐case新旧JSON：`outputs/ptv3-shared-wam-v6/teacher-task-scoring-20261022-r1/{old,new}-summary.json`与`*-task-traces.json`。新source-PD：cylindersmall/cubemedium/cylindermedium为acquire_timeout，airplane为path_no_confirmed_milestone，cubesmall/cup为600controls内未完成（有限预算失败，非观测缺帧）。maxpathprogress按顺序为.213/0/.113/0/.004/.017，全部只有grasp_acquired、无release_authorized。Fresh teacher 5条因缺endpoint+1s记录删失；airplane在此前明确path failure。时序修复对这组较早阶段blocker无提升，**不否定其他轨迹已修好的release时序现象**。

Source-PD总clipping330envcontrols，其中320在保持末target的后缀。首错cubemedium t164，continuous pitch target=-7.282474rad，反解canonical=-4.159958超[-1,1]；native actualPD与rawtarget差9.927302rad，全程max10.461373rad。原始targets/intended/action/actualPD和within_reference全部保存；没有把clipping silently计作精准上限。

Fresh与source-PD加工程累计7986envcontrols。两个run均已退出，未触碰其他进程。完整task正控依然失败/未资格，无法由本组负结果定位WAM解码质量；下一步应解决初始化/角代表、获取控制与有序路径的明确blocker，再谈样本效率。

可核执行命令：

```sh
/home2/wyy/oyx_ws/.runtime_envs/dexplore_v120_train/bin/python src/task/ptv3-shared-wam-v6/tools/audit/collect_bounded_teacher.py --panel outputs/ptv3-shared-wam-v6/executor-qualification-20261022-r1/panel.json --output outputs/ptv3-shared-wam-v6/teacher-upper-bound-validation-20261022-r2 --native-output outputs/wam-rl-policy-v3/teacher-upper-bound-validation-20261022-r2 --gpu 4
/home2/wyy/miniconda3/envs/graspenv/bin/python src/task/ptv3-shared-wam-v6/tools/audit/replay_bounded_source_pd.py --teacher outputs/wam-rl-policy-v3/teacher-upper-bound-validation-20261022-r1 --output outputs/ptv3-shared-wam-v6/source-pd-upper-bound-validation-20261022-r2 --gpu 4
/home2/wyy/miniconda3/envs/graspenv/bin/python src/task/ptv3-shared-wam-v6/tools/audit/score_teacher_action_reset.py --package outputs/ptv3-shared-wam-v6/teacher-task-scoring-20261022-r1/new/src/task/wam-rl-policy-v3/src/wam_rl_policy_v3 --label new --teacher outputs/wam-rl-policy-v3/teacher-upper-bound-validation-20261022-r1 --source-pd outputs/ptv3-shared-wam-v6/source-pd-upper-bound-validation-20261022-r1 --declarations outputs/wam-rl-policy-v3/inspire-teacher-rollout-audit-20261020-r3 --output outputs/ptv3-shared-wam-v6/teacher-task-scoring-20261022-r1
```

前两条为复现说明，未额外运行；须新目录与授权GPU预算。旧评分命令只替换package `/new/`→`/old/`、label`old`。两评分器完整默认阈值来自各自冻结包；未放宽path/稳定/接触/支撑。

## 秒制持握与continuous腕代表定位

持握n个连续positive**状态**的真实elapsed为`(n-1)/30s`，所以真实≥1s要求≥31状态；旧hold30保留为既有代理，不称精确1s。CPU重算：fresh teacher 6/6达到真实≥1s，最长时长依native排序2.30/3.1667/2.6333/2.30/1.6333/3.20s；source-PD 6/6，2.30/3.1667/2.60/2.2667/1.60/3.20s。六个不同序列/物体、同seed195，每物体1条，不能外推一般分布成功率。JSON同时保存旧hold30与elapsed1s。

URDF实际路径`/home2/wyy/oyx_ws/dexplore/dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf`。腕joint4/5/6是continuous，原生import实际hasLimits均false，lower/upper0是无角限的占位，不能解释成限制角必须0。native `_set_env_state`只clamp6个active手指，保存腕q不normalize；原collector未wrap腕，PD直接live q加相对动作。故超过±π的q可合法作为同一角的代表被保存，但PhysX初始化/驱动可能对代表敏感。

审计独立NumPy URDF FK以原生返回的真实DOF名顺序映射，不能使用URDF XML枚举顺序（thumb/index排列不同）。`audit_continuous_wrist.py`、`continuous-wrist-fk-audit.json`及冻结URDF在评分run可核。reset腕角mod2π前后所有link变换差3.89e-16；branch实际initial vs原initial差4.26e-7。

新增branch run为`outputs/ptv3-shared-wam-v6/source-pd-continuous-branch-validation-20261022-r2/`，单例600envcontrols；仅审计host `inspire_branch_reset_host.py`把continuous腕reset正规化、source-PD取距live q最近的等价角，所有rawtarget保留。instrument diff在`outputs/ptv3-shared-wam-v6/branch-host-source-pd-continuous-branch-validation-20261022-r2/`，没有修改生产host。r1在0physics被host最小2env guard拒绝；r2仅instrument允许1env，未扩大物理预算。

| 已注册branch门槛 | 原source-PD | branch | 判定 |
|---|---|---|---|
| 首tick实测q→独立FK的palm旋转<.1rad | 2.483825rad | .472321rad | FAIL |
| 真source期clipping=0 | 10controls | 9controls | FAIL |
| raw/equivalent target所有link FK差<1e-6 | 不适用 | 4.45e-7 | 等价通过 |
| actualPD/equivalent target所有link FK差<1e-6 | 原numeric差10.461rad | .040148最大变换差 | FAIL |
| 完整task新v2 | 0/1 | 0/1 acquire_timeout | 未资格 |

branch最长forcehold真实elapsed1.6333s，达到≥1s代理，仍不代表fulltask。actualPD−原rawtarget数字差仍6.283186rad，未隐去。9次clipping全在平移Y：t179–184、190–192，意图canonical约−1.04015/−1.02014；腕角超限消除，但平移追踪不可执行，不能用wrap解决。

源行为/索引复核工具`audit_source_pd_trace.py`与`source-pd-provenance-geometry.json`：六个序列每条first target和全部held target均与对应fresh source archive逐数组exact；branch raw_target[0]同对应cubemedium pd_targets[0]exact。排除env排序/一帧移位作为这次强首步的解释。首个target−q腕为[+.59495,−.13621,+.14262m,+.25102,+1.79118,−.83671rad]，请求palm目标旋转1.815rad、位移.62678m；实际首dq腕为[1.523,−.621,−.150m/s,1.872,12.812,−5.594rad/s]。这是原teacher实录的强命令，不能把它的.472rad响应混为“无命令reset踢飞”。原始teacher独立FK vs实测hand11最大4.12e-6m；branch整段最大2.21cm，可能有约束误差，未经独立姿态测量不能把整段FK当真实body姿态。

初始几何：cubemedium CAD样本与table gap−2.469mm，未达原>2cm穿透proxy；palm距object中心1.269m，五tip距object CAD最近样本1.053–1.131m。step0 force invalid，不能用零force宣称无接触。pairwise signed hand/object穿透未采集，保留UNKNOWN；这些距离只说明开始时手远离物体，不证明所有碰撞正确。

最后仅4envcontrols的隔离run `outputs/ptv3-shared-wam-v6/continuous-neutral-reset-validation-20261022-r1/`：与r3相同canonical全零（腕无增量、手指合法中点），初palm位姿差4.05e-7；raw-neutral首tick真正基线旋转是2.734649rad，normalized-neutral为.002620rad<.1 PASS，平移11.87→5.55mm，独立PD误差0。该结论来自实测q→独立FK；raw是2env、normalized是1env，有batch混杂。`neutral-comparison.json`保留这项限制。它支持最小初始化角代表修复，**未修好强动作跟踪或完整task**。累计8590envcontrols。root准备生产最小reset修复，后续仅同r3的2env工程776envcontrols，预计总9366；未获得新的teacher/RL训练授权。

下一项最便宜可证伪实验（本报告不额外执行）：同批2env、同初态、同finger目标，比较raw/normalized在zero腕增量下首tick实测body quaternion；若raw大跳、normalized小且collision几何不变，支持角代表初始化问题。再用首个强target的分段幅度控制（独立标明改变行为），测实际目标误差/限幅发生轴；若静态初态无命令也大跳则优先约束/碰撞，若只有强target才跳则优先源动作瞬变，若目标合法但漂移后平移差>1m而逆动作越界则为受限跟踪。必须保留原首target做负控制；不能把平滑后成功套回原source-PD精确可执行主张。当前缺实际palm quaternion和初态signed碰撞对穿透，不能自证完整因果链。

## 最后生产入口回归：完成且停止

Root实现生产最小修复：`src/task/wam-rl-policy-v3/src/wam_rl_policy_v3/continuous_wrist.py:L11–35`，仅reset位置3:6 continuous腕角正规化；平移/手指/速度不wrap，原rawsource target不重写。`inspire_physics_host.py:L110–112`安装，L154核实际原生DOF名和hasLimits；L208 extended_state返回实际rigidbody palm_pose。此生产修改由root负责，本子任务未覆盖共享源码。

最终run `outputs/ptv3-shared-wam-v6/teacher-action-reset-validation-20261022-r4/`保存冻结source与mtime索引，base`60efadce2444709bbbf9802107de7bc5e3f9eef6`+dirty源码快照。入口为**生产host**，未使用branch instrumentation；同r3序列、2env、seed195、同canonical中点fixture与轴幅值。388batched=776envcontrols，26.61s，完成：

| 指标 | 预期/基线 | r4实测 | 结论 |
|---|---|---|---|
| 12轴×正负真实q响应 | 24/24，响应≥25%，误差≤75%幅度 | 24/24；响应98.1494%–99.2358%，最大误差1.8506% | PASS |
| 全部实际PD映射 | maxabs≤1e-6 | 0 | PASS |
| 第2/4步indexed reset | 旧13+新palm_pose，14字段untouched差0 | 两次全部精确0 | PASS |
| cubemedium首tick**实测rigidbody quaternion**旋转 | <.1rad；旧neutral q→FK为2.734649rad | .000435070rad | PASS最小reset工程验收 |
| airplane首tick实测palm旋转 | 记录 | .002433885rad | 记录，不外推task |
| 首tick实测palm位移 | 记录airplane/cubemedium | 5.425mm/.370mm | 记录 |

r4与r3早期角度的证据类型不同：r4直接来自实测rigidbody quaternion，早期r3/branch/neutral来自实测q的独立URDF FK，后者均明确标为FK导出，不能写成直接读取的body quaternion。r4 trace保存每步palm_pose_before/after。

三个错误PD mutants差分别.924706（指关节映射）、2.940888（腕不加liveq）、.924244（错误重复相加coupling），均>1e-6。纯重复固定赋值coupling仍幂等，差0，未杀死。CPU回归额外14case现在直接测生产v3helper，文件`src/task/ptv3-shared-wam-v6/tests/test_continuous_angles.py`：正负多圈、非continuous不wrap、故意all-axis错误判别、形状/dtype/有限性和不改caller输入。原root三测试保持；合并实跑17passed，记录`tmp/continuous-wrist-combined-regression.log`。重复v6 helper已删除，不引入第二生产接口。

最终累计9366envcontrols（工程r2 10+r3 776+teacher3600+sourcePD3600+branch600+neutral4+r4 776；两次0physics启动失败不加步），没有RL更新。`outputs/ptv3-shared-wam-v6/teacher-task-scoring-20261022-r1/final-stop-receipt.json`逐PID核实所有自有client/native均退出，核查时GPU4=6MiB/0%。本轮不再新物理。

**适用范围：**fulltask原source-PD 0/6、branch 0/1是各自旧初始化或审计branch物理run，经冻结新旧task scorer重放的真实结果；不能把它们写成生产reset修复后完整链0/6。生产修复后只通过这次2env动作/reset工程验收，未重新跑完整teacher/sourcePD，因此完整正控制仍未资格，而非已被修复。不能将较早负结果归因到“任务阈值太高”，也不能把工程通过升级为WAM解码瓶颈或样本效率证据。下一项完整正控需同一生产reset链、明确可执行且意图认证的source声明、完整末端settling实录；未满足前不得扩大训练。

CPU可核命令：

```sh
/home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest -q src/task/wam-rl-policy-v3/tests/test_continuous_wrist.py src/task/ptv3-shared-wam-v6/tests/test_continuous_angles.py
/home2/wyy/miniconda3/envs/graspenv/bin/python src/task/ptv3-shared-wam-v6/tools/audit/audit_continuous_wrist.py --teacher outputs/wam-rl-policy-v3/teacher-upper-bound-validation-20261022-r1 --branch outputs/ptv3-shared-wam-v6/source-pd-continuous-branch-validation-20261022-r2 --output outputs/ptv3-shared-wam-v6/teacher-task-scoring-20261022-r1
/home2/wyy/miniconda3/envs/graspenv/bin/python src/task/ptv3-shared-wam-v6/tools/audit/audit_source_pd_trace.py --teacher outputs/wam-rl-policy-v3/teacher-upper-bound-validation-20261022-r1 --replay outputs/ptv3-shared-wam-v6/source-pd-upper-bound-validation-20261022-r1 --branch outputs/ptv3-shared-wam-v6/source-pd-continuous-branch-validation-20261022-r2 --neutral outputs/ptv3-shared-wam-v6/continuous-neutral-reset-validation-20261022-r1 --declarations outputs/wam-rl-policy-v3/inspire-teacher-rollout-audit-20261020-r3 --output outputs/ptv3-shared-wam-v6/teacher-task-scoring-20261022-r1
```
