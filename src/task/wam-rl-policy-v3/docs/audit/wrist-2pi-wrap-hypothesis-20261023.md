# 结果解读表（brief44预注册原文，计算前写入）

| 观测结果 | 含义 | 决定 |
|---|---|---|
| 把腕角按正确口径 **unwrap/统一参考系**后重算关节门判据，21 次中止中 **≥50% 不再触发** | 定位到**一个可修的单点**（角度环绕/参考系不一致） | 才允许进入"**只修这一处** + 单臂 n=5 复测"；修法与最小 diff 先写进报告，**本轮不自行开跑仿真** |
| 只有 <20% 不再触发 | 跳变是**伴随现象而非原因**，假设**否证** | 停止在腕角上做文章；下一步是补 brief40 列出的三项缺失证据（子步驱动误差/冲量、成对接触、腕关节类型），**不得**改门阈、不得调 PD 增益、不得改执行器参数 |
| 20–50% 之间 | 部分相关、证据不足 | 判定 INVALID/PARTIAL，写清还差哪个字段与哪个实验版本，不修 |
| 必要记录缺字段（腕角单位、关节类型、每步 PD 目标/误差未落盘） | 数据不足 | 明确列出缺什么、在哪一层能补（只做**申请**，不自行跑仿真），报告标 INVALID_INSTRUMENTATION |

预注册写入于2026-10-23，本轮尚未计算。基线事实只引用brief40提交`8df1ae8a`及其只读产物，不重跑collect.py。只写本文件；本轮新增仿真/训练/模型推理/GPU调用均为0。后续追加计算结果，不改上表。

## 实测裁决：REFUTED

**21次原关节中止中，统一腕角后消失0/21=0%。2π与4π两种周期口径均为0/21；21/21原触门DOF列表、abort/reason完全不变。** 按预注册的<20%数字分支，单一裁决为REFUTED，不进入修复/n=5。

**裁决范围必须限定**：否证的是“腕角环绕/参考系差异直接造成已有安全门误判、仅换腕角代表即可撤销多数中止”这个可离线检验的单点假设。预注册第二行中“伴随现象而非原因”的因果措辞，超出了此静态重算的识别能力：本计算不能排除腕驱动跨支路先产生物理冲量、再使手指越界。后者仍未辨识，不能写成物理因果贡献0%。决策照预注册执行：停止基于该签名修改腕/PD参数，下一步只申请三项缺失记录。

### 为什么重算不依赖未保存的SDK关节枚举

原门在各run冻结`physical_safety.py:17–20 / physical_safety()`中使用`has_limits`；本轮所读audit的腕3:6均为False。对任何仅改腕的映射U，以及任意周期/腕姿参考代表，所有被该门检查的DOF都有`U(q)[j]=q[j]`，因此`gate(U(q))=gate(q)`。触门手指、边界、0.10裕量、物体姿态和桌面判据均保留原值。改变这些手指或阈值来让gate消失会违反本试验。

这里的必要输入均具备：保存q/实发PD、原mask/边界、同一冻结门函数；弧度口径由原调用和实测固定安装角一致性核对。本机SDK内部eREVOLUTE还是unwrapped枚举仍未落盘，因此**不冒认4π就是本机已认证周期**；同时检验2π与4π，且上面的输入不相关性证明不依赖这个未知枚举。它仍是未来动力学归因的缺失字段，不能因缺它而掩盖本静态门假设可被否证。

## A. 表示、跳变在哪一层

下文S2e/S2d根路径沿用[brief40](p0-exec-deathcause-20261023.md)第1节；BC/PPO根为`outputs/wam-rl-policy-v3/bc-ppo-usefulness-20261023-r1/assessment-{bc,ppo}/`。所有runtime路径均指各run的`runtime-code/`，不用当前生产实现替代旧入口。

| 环节 | 已核表示/变换 | 精确代码位置（以S2e冻结路径为例） |
|---|---|---|
| 原生q/安全门输入 | q来自native DOF标量；腕3:6为rad；不由palm四元数反算，packet未取模/unwrap；安全门直接读q | `runtime-code/src/task/wam-rl-policy-v3/tools/run/inspire_physics_host.py:198–209 / Host.packet`；同包`physical_safety.py:9–20 / physical_safety` |
| reset | 腕初态取`(q+π) mod 2π−π`；只作用reset，不覆盖运行轨迹 | `src/task/wam-rl-policy-v3/src/wam_rl_policy_v3/continuous_wrist.py:11–17 / normalize_wrist_positions`，`:27–35 / install_continuous_wrist_reset`；旧audit记录相同reset policy |
| R输入观测 | current_state串接原q，clip至[-5,5]；不是环绕归一化；存档observation[:,3:6]逐位等于clip(q[:,3:6]) | 冻结`inspire_shadow.py:97–105 / current_state`；S2e `runner.py:115–117,131` |
| τ预测/几何τ | 原始量是世界手点位移m，不是腕角；不能对τ直接做2π unwrap。物体/palm旋转来自SciPy默认xyzw四元数→矩阵 | 冻结v6`online_runtime.py:81–112 / OnlineV6Runtime.predict`；S2e `runner.py:120–128`。没有运行模型 |
| 点集→腕姿约简 | SVD刚体配准得到旋转矩阵，再到object-local SE3；没有标量腕角unwrap | 冻结`pose_consumer.py:54–81 / reduce_to_wrist_plan` |
| 神经R输出 | 12维canonical action；腕3:6是旋转增量/scale（默认scale=π rad），不是绝对欧拉角；MLP标准化/仿射，无atan2/取模 | 冻结`bc_mlp.py:29–34 / NormalizedCommandMLP.forward`；`tau_replay_contract.py:21–27 / pd_targets` |
| R→intent→PD | `target[:6]=q[:6]+offset+scale*action[:6]`。腕姿使用大写XYZ内禀欧拉，默认rad；`approach_joint_angles`保留R原分支 | 冻结`pose_consumer.py:172–191 / grasp_intent_from_retargeter`，`:244–264 / MeasuredGraspConsumer.step` |
| 由姿态反求角 | `as_euler("XYZ")`后枚举等价欧拉分支，再加2π整数使其接近measured q；是单时刻分支选择，不是全序列unwrap | 冻结`pose_consumer.py:84–88 / closest_wrist_angles`；有R提供approach_joint_angles时该路径不被选用 |
| 实发目标与测量时序 | native先转换target，再task.step，再回传q与actual_PD；未保存物理子步drive误差/接触冲量 | S2e冻结host`:247–256`；`runner.py:134–144`在记录state[t]后才执行action[t] |

URDF证据：`third_party/DExplore/dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf:64,78,92`中joint4/5/6标为continuous；这是URDF/限位语义，不是SDK内部joint类型取证。对全部既有q/palm序列检验`R_XYZ(q_wrist)^T R_palm`恒定，其最大漂移为**1.34e-06 rad**，与rad/XYZ固定安装口径一致；没有把四元数正负号切换当关节标量跳变。

### 已有事件索引上的新幅度/层级检查

37个事件tick只读引用brief40的`wrist_branch_transition_state_ticks`，不重跑其死因计数或检测器。本轮新算每个tick的角差、分支残差和相邻环节：

- 最大变化轴的|Δq|分位数[min,q25,median,q75,p90,max]：`9.107408, 10.171111, 10.536144, 11.889139, 12.238974, 12.485260` rad。
- 对每个轴差取最近2π整数倍，|残差|分位数同序：`0.081111, 0.677231, 2.030227, 2.395260, 2.852447, 3.118880` rad；仅1/37残差≤0.10rad。0.10只作为幅度描述容差，不是修改安全门。
- 最近整数倍计数：`{'-2': 9, '-1': 1, '1': 1, '2': 26}`。35/37最近的是±4π而非±2π，但残差一般很大；“±2π附近”指跳变前的分支位置，不能写成每次幅度精确等于2π。
- 全存档逐控制步的新层级计数：观测中clip(q)的单轴变化>2π为41次；相邻实发绝对PD腕目标的单轴变化>2π为26次（不是同一个时间定义）。已保存原始R输出的22回合/5405个相邻输出对，换算后的腕增量变化>2π为0次；S2e10回合原始R输出未单独保存，不能冒充全32回合的R=0。
- 已有37个q跳变事件中，36个事件之前的PD绝对目标变化≤2π；唯一前驱PD大跳是S2e/block1-wam的tick201，而tick200已先有q跳变。可见记录支持“大不连续先出现在native回传q、随后进入clip观测/后续PD”的层级定位；无法分辨该物理间隔内的积分器折返、驱动误差与碰撞谁先发生。

表中ΔPD为`PD[t−1]−PD[t−2]`最大单轴rad；ΔR为原始R腕**增量**相邻变化×scale，不是绝对角。S2e原始R被runner.py:140覆盖，保存raw_action实际是consumer输出，故记未落盘。τ列是Cartesian最大分量变化m/valid前→后，不能当角度。原事件发生于state[t]；产生它的命令是action[t−1]，而终止state[t]之后没有R/action[t]。

| 批次/回合/tick | 腕DOF | Δq rad | 最近2π倍数/残差rad | 观测角差最大rad | 前驱ΔPD rad | 前驱ΔR rad | τ变化m / valid | ‖e_new‖ rad | G_R前→后rad |
|---|---:|---:|---|---:|---:|---|---|---:|---|
|S2e/block0-geometry/106|4|10.423135|2 / -2.143236|9.152091|0.160744|未落盘|0.11245 / 1→1|0.679611|-0.131670→0.387471|
|S2e/block1-geometry/143|4|12.125659|2 / -0.440712|10.000000|0.039861|未落盘|0.06630 / 1→1|0.124879|-0.304559→-0.232383|
|S2e/block1-wam/200|5|9.402065|1 / 3.118880|8.134575|0.228755|未落盘|0.36355 / 1→1|1.338846|-0.098188→-0.099763|
|S2e/block1-wam/201|4|11.026612|2 / -1.539759|9.886428|9.741362|未落盘|0.20996 / 1→1|0.826652|-0.099763→-0.081426|
|S2e/block1-wam/232|3|-10.703209|-2 / 1.863162|9.510048|0.260573|未落盘|0.47155 / 1→1|0.833783|-0.298538→0.244309|
|S2e/block2-geometry/71|3|12.004446|2 / -0.561925|10.000000|0.117684|未落盘|0.04300 / 1→1|0.476699|-0.277214→-0.199005|
|S2e/block2-geometry/86|5|-12.461005|-2 / 0.105366|10.000000|0.156431|未落盘|0.01492 / 1→1|0.440071|-0.265741→-0.017269|
|S2e/block2-geometry/104|4|10.253683|2 / -2.312687|8.998642|0.224244|未落盘|0.18713 / 1→1|0.628515|-0.184646→-0.101653|
|S2e/block2-wam/179|4|9.877982|2 / -2.688388|8.690735|0.447043|未落盘|0.25157 / 1→1|1.757464|-0.158753→0.229398|
|S2e/block3-geometry/141|4|10.941588|2 / -1.624783|9.672350|0.085644|未落盘|0.11751 / 1→1|0.034119|-0.074547→0.446117|
|S2e/block4-geometry/107|4|11.563169|2 / -1.003202|10.000000|0.170524|未落盘|0.10759 / 1→1|0.677344|-0.160338→-0.139028|
|S2e/block4-geometry/124|5|-9.655626|-2 / 2.910745|8.380906|0.173836|未落盘|0.13122 / 1→1|0.684173|-0.096500→0.038735|
|S2e/block4-wam/78|4|10.442065|2 / -2.124306|9.303168|0.287262|未落盘|0.16107 / 1→1|0.996769|-0.310806→-0.075196|
|S2d/block0-c-live-gt/112|4|10.296071|2 / -2.270300|9.015487|0.074008|0.023834|0.20944 / 1→1|0.190386|-0.280794→0.135575|
|S2d/block1-c-live-gt/119|4|9.891746|2 / -2.674625|8.616122|0.215021|0.075412|0.20712 / 1→1|0.676447|-0.114925→0.240526|
|S2d/block2-c-live-gt/144|4|10.952674|2 / -1.613697|9.684702|0.105006|0.061126|0.11525 / 1→1|0.063700|-0.091658→0.335057|
|S2d/block3-a-original-wam/208|4|9.618713|2 / -2.947657|8.413124|0.613546|0.428726|0.18868 / 1→1|1.163153|-0.266375→0.158957|
|S2d/block3-c-live-gt/113|4|10.171111|2 / -2.395260|8.888322|0.140272|0.131714|0.17527 / 1→1|0.722556|-0.185095→0.179554|
|S2d/block4-a-original-wam/81|4|10.024086|2 / -2.542285|8.775777|0.235012|0.152177|0.26677 / 1→1|0.957916|-0.275051→-0.022958|
|S2d/block4-c-live-gt/112|4|10.296071|2 / -2.270300|9.015487|0.074008|0.023834|0.20944 / 1→1|0.190386|-0.280794→0.135575|
|BC/p0003-A1/219|3|-12.022941|-2 / 0.543430|10.000000|0.107006|0.147102|0.35817 / 1→1|0.873443|-0.237002→-0.300537|
|BC/p0003-A1/235|5|10.235479|2 / -2.330892|9.010830|0.295177|0.193619|0.21632 / 1→1|1.037132|-0.331514→-0.199026|
|BC/p0003-A1/286|4|11.041364|2 / -1.525007|9.837377|0.263500|0.400831|0.10523 / 1→1|0.650746|-0.282478→0.172601|
|BC/p0004-A1/97|5|12.485260|2 / -0.081111|10.000000|0.169664|0.121729|0.05874 / 1→1|0.465667|-0.245311→-0.254152|
|BC/p0004-A1/101|5|-9.940895|-2 / 2.625476|8.675600|0.207232|0.250301|0.30508 / 1→1|0.442414|-0.257386→-0.079092|
|BC/p0004-A1/103|4|11.923008|2 / -0.643363|10.000000|0.630822|1.202965|0.13715 / 1→1|0.706059|-0.135816→-0.265936|
|BC/p0004-A1/168|3|10.236561|2 / -2.329809|9.019807|0.422367|2.493164|0.13949 / 1→1|1.463711|-0.222238→-0.012743|
|BC/p0004-A1/181|3|-12.445549|-2 / 0.120821|10.000000|0.482680|0.900748|0.14371 / 1→1|1.561028|-0.235824→-0.247774|
|BC/p0004-A1/186|3|10.355730|2 / -2.210641|9.116287|0.286287|0.124707|0.22362 / 1→1|0.987104|-0.306813→-0.031045|
|BC/p0004-A1/231|4|10.536144|2 / -2.030227|9.423986|0.486003|0.357761|0.14486 / 1→1|1.367913|-0.263450→0.177206|
|BC/p0005-A1/211|5|11.902957|2 / -0.663413|10.000000|0.478295|0.441770|0.08941 / 1→1|1.096266|-0.271473→-0.263748|
|BC/p0005-A1/235|4|12.408946|2 / -0.157425|10.000000|0.107338|0.174530|0.05751 / 1→1|0.717532|-0.278060→-0.256365|
|BC/p0005-A1/374|3|-9.671587|-2 / 2.894783|8.472064|0.840308|0.624461|0.14763 / 1→1|2.738819|-0.329979→-0.142373|
|BC/p0005-A1/467|3|-9.107408|-1 / -2.824223|7.866863|0.946105|0.825494|0.14327 / 1→1|1.931480|-0.238085→-0.092650|
|BC/p0005-A1/534|3|-11.388033|-2 / 1.178337|10.000000|0.666258|0.536299|0.16483 / 1→1|1.540959|-0.291771→0.124020|
|PPO/p0001-A1/141|3|11.889139|2 / -0.677231|10.000000|0.318043|0.200751|0.11514 / 1→1|0.565938|-0.295103→-0.254099|
|PPO/p0003-A1/211|3|-11.433978|-2 / 1.132393|10.000000|0.531536|0.355809|0.25803 / 1→1|1.766219|-0.098267→0.103126|

## B/C. 同批交叉、PD误差分布与逐回合门重放

交叉答案：**是同批**。brief40所列15个终止腕跳变都包含在21个手指关节中止内，而且15/15的触门手指末次PD目标均合法；另外6个手指关节中止没有终止当步该腕签名。0/21是“非法目标计数”，不是与15例互斥的另一组事故。

定义t为终止state：`e_new=PD[t−1]−q[t−1]`是送入该控制间隔的腕跟踪误差；`e_held=PD[t−2]−q[t−1]`是跳变前状态对上一目标的误差。两者都不是未知的物理子步误差。

手指门值`G_R=max_j{lower_j−0.1−q_j, q_j−upper_j−0.1}`，j仅受限手指、单位rad；平移门`G_P`按DOF0:3计算、单位m。>0才触门。腕3:6不属于任何受限旋转门。

| 量（21回合） | min | q25 | median | q75 | p90 | max |
|---|---:|---:|---:|---:|---:|---:|
|‖e_new‖ rad|0.034119|0.255810|0.679611|0.939050|1.540959|1.766219|
|‖e_held‖ rad|0.070392|0.287815|0.491834|0.594083|1.030517|1.985303|
|G_R[t−1] rad|-0.298538|-0.278442|-0.185095|-0.096500|-0.074547|-0.015901|
|G_R[t] rad|0.001516|0.076653|0.135575|0.229398|0.335057|0.446117|
|G_P[t−1] m|-1.110311|-0.832931|-0.477399|-0.124398|-0.123917|-0.122302|
|G_P[t] m|-1.098127|-0.798531|-0.563049|-0.201282|-0.145843|-0.122560|
|事后raw腕PD误差范数rad|0.131222|0.746482|10.530277|11.369197|11.862734|12.415199|
|事后4π统一代表腕PD误差范数rad|0.131222|0.746482|1.269325|1.972762|2.147060|2.667506|

### 跳变事件本身的误差/门值分布（复用已有37个索引）

| 分组/量 | min | q25 | median | q75 | p90 | max |
|---|---:|---:|---:|---:|---:|---:|
|终止当步15事件/‖e_new‖ rad|0.034119|0.420566|0.684173|1.265533|1.670862|1.766219|
|终止当步15事件/G_R前 rad|-0.298538|-0.280794|-0.185095|-0.106596|-0.093595|-0.074547|
|终止当步15事件/G_R后 rad|0.038735|0.135575|0.177206|0.242417|0.366506|0.446117|
|未当步终止22事件/‖e_new‖ rad|0.124879|0.581582|0.850047|1.081482|1.551296|2.738819|
|未当步终止22事件/G_R前 rad|-0.331514|-0.290842|-0.261563|-0.225635|-0.138269|-0.098188|
|未当步终止22事件/G_R后 rad|-0.300537|-0.252518|-0.140701|-0.079675|-0.023767|-0.012743|

统一角度时，对提交目标使用同一时刻的分支偏移：`PD_U[t−1]=PD[t−1]+q_U[t−1]−q[t−1]`，故`e_new_U=e_new`（代码逐例断言）。事后腕误差中位数虽可从10.5303变为1.2693rad，但触门手指q完全没变；这不是执行变好，也不是新的物理反事实。不得只改q而不改PD坐标来制造误差改善。

| 批次/回合 | t | 终止腕跳变/触门手指PD合法 | ‖e_new‖ / ‖e_held‖ rad | G_R前→后rad | 原触门DOF | unwrap2π后 | unwrap4π后 | 是否撤门 |
|---|---:|---|---|---|---|---|---|---|
|S2e/block0-geometry|106|是/是|0.679611 / 0.491834|-0.131670→0.387471|8,10,12,17|8,10,12,17|8,10,12,17|否|
|S2e/block1-geometry|156|否/是|0.193893 / 0.070392|-0.015901→0.077161|17|17|17|否|
|S2e/block1-wam|232|是/是|0.833783 / 0.579054|-0.298538→0.244309|15,16|15,16|15,16|否|
|S2e/block2-wam|179|是/是|1.757464 / 1.985303|-0.158753→0.229398|10,12|10,12|10,12|否|
|S2e/block3-geometry|141|是/是|0.034119 / 0.110210|-0.074547→0.446117|10,12|10,12|10,12|否|
|S2e/block4-geometry|124|是/是|0.684173 / 0.577082|-0.096500→0.038735|16|16|16|否|
|S2d/block0-a-original-wam|93|否/是|0.255810 / 0.433076|-0.095665→0.076653|10|10|10|否|
|S2d/block0-c-live-gt|112|是/是|0.190386 / 0.150646|-0.280794→0.135575|6|6|6|否|
|S2d/block1-c-live-gt|119|是/是|0.676447 / 0.449445|-0.114925→0.240526|8|8|8|否|
|S2d/block2-a-original-wam|91|否/是|0.680975 / 0.573057|-0.267563→0.026950|16|16|16|否|
|S2d/block2-c-live-gt|144|是/是|0.063700 / 0.171479|-0.091658→0.335057|10,16|10,16|10,16|否|
|S2d/block3-a-original-wam|208|是/是|1.163153 / 0.536231|-0.266375→0.158957|15,16|15,16|15,16|否|
|S2d/block3-c-live-gt|113|是/是|0.722556 / 0.594083|-0.185095→0.179554|16|16|16|否|
|S2d/block4-c-live-gt|112|是/是|0.190386 / 0.150646|-0.280794→0.135575|6|6|6|否|
|BC/p0003-A1|286|是/是|0.650746 / 0.648167|-0.282478→0.172601|15,16|15,16|15,16|否|
|BC/p0004-A1|231|是/是|1.367913 / 1.030517|-0.263450→0.177206|10|10|10|否|
|BC/p0005-A1|534|是/是|1.540959 / 0.655017|-0.291771→0.124020|16|16|16|否|
|PPO/p0002-A1|62|否/是|0.371024 / 0.287815|-0.043573→0.001516|15|15|15|否|
|PPO/p0003-A1|211|是/是|1.766219 / 1.238755|-0.098267→0.103126|16|16|16|否|
|PPO/p0004-A1|211|否/是|0.939050 / 0.468973|-0.234715→0.044828|10|10|10|否|
|PPO/p0005-A1|49|否/是|0.560121 / 0.296461|-0.278442→0.045953|10|10|10|否|

## D. 修复与n=5门禁

**CONFIRMED条件未满足，因此无最小修复diff、无获证据支持的修复位置，本轮不实施修复，也不运行n=5。** 尤其不在continuous_wrist.py、PD转换或JointAdmission里加腕角上限/换PD增益来追求少报中止。前文代码位置只用于说明合同与取证入口，不是待提交补丁。

按预注册下一步仅申请补brief40三项证据：①native.task.step内部物理子步q/dq、实际drive target/error、drive/constraint impulse；②手/物体/桌面/地面/自身成对body/shape接触点、法向、穿深和冲量；③本机SDK实际腕关节类型及wrap行为（URDF continuous/hasLimits=False不能替代）。S2e还需保留actor原始输出再进入consumer，避免runner.py:140覆盖后无法核对这一层。这里只提出记录要求，不新增记录任务。

n=5的必要条件：补录后获得可定位单点的证据、明确唯一修法并获新授权，再按同冻结R/初态/PD/评分预注册。现在没有资格直接启动。原S2e为300控制上限；未来若采用600末态评分，旧600基线未测必须明示，不能把旧0/5直接当同600预算前值。

## 复现（CPU只读，无临时脚本/输出文件）

从仓库根执行：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=2 \
/home2/wyy/miniconda3/envs/graspenv/bin/python - <<'PY'
from pathlib import Path
import json
p=Path("src/task/wam-rl-policy-v3/docs/audit/wrist-2pi-wrap-hypothesis-20261023.md")
lines=p.read_text().splitlines()
begin=lines.index("<!-- brief44-code-begin -->")+2
end=lines.index("<!-- brief44-code-end -->")-1
code="\n".join(lines[begin:end])
ns={};exec(compile(code,str(p),"exec"),ns)
r=ns["analyze"]()
print(json.dumps(r,indent=2))
assert r["decision"]=="REFUTED" and r["removed_2pi"]==r["removed_4pi"]==0
PY
```

预期：decision=REFUTED；removed_2pi=0、removed_4pi=0；cross_terminal_branch_and_PD_legal=15；所有21例原/重放索引一致。程序只导入brief40 helper函数，**不调用collect.main，不重跑brief40死因统计，不运行模型**。只读原q/PD/obs、audit、canonical/table数据和各自冻结安全函数；缺输入即报错，不替换来源。

本报告源码块和结果只写本新文件，brief40目录未改/未重新入库；没有GPU、仿真或训练调用。预注册表保持计算前原文。

## Git边界记录

本轮仅显式提交本新报告。`git pull --rebase origin main`实测被共享工作区他人的未暂存修改拒绝（`cannot pull with rebase: You have unstaged changes`）；未stash、未覆盖脏文件，按brief44改普通`git push origin main`。终态HEAD/远端核对随交付回复提供。

## 附录：离线程序（只读；上文复现命令直接执行本代码块）

<!-- brief44-code-begin -->
```python
import importlib.util, json, types, sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

def analyze():
    root=Path.cwd()
    prior=json.loads((root/'tmp/s40-deathcause/evidence.json').read_text())
    spec=importlib.util.spec_from_file_location('s44_helpers',root/'tmp/s40-deathcause/collect.py')
    helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
    catalog=helpers.read(root/'outputs/wam-rl-policy-v3/inspire-teacher-rollout-audit-20261020-r3/manifest.json')['records']
    gates={};geometries={};events=[];cases=[];steps=[];mount_errors=[];readonly_inputs=[]
    distributions={name:[] for name in ('new_PD_error_norm_before','held_PD_error_norm_before','finger_gate_max_before','finger_gate_max_after','PD_error_norm_after_raw','PD_error_norm_after_unwrapped_4pi','prismatic_gate_max_before_m','prismatic_gate_max_after_m')}
    for record in prior['records']:
        folder=root/Path(record['source_files']['trajectory']).parent
        run=folder.parent;manifest=helpers.read(run/'manifest.json')
        audit=manifest.get('native_initial_audit') or helpers.read(folder/'native-audit.json')
        limits=audit['physical_joint_limits'];mask=np.asarray(limits['has_limits'],bool)
        assert not mask[3:6].any(),'wrist mask contract differs'
        flo=np.asarray(limits['lower'])[6:18][mask[6:18]];fhi=np.asarray(limits['upper'])[6:18][mask[6:18]]
        def event_finger_gate(value):
            v=value[6:18][mask[6:18]]
            return float(np.maximum(flo-.1-v,v-fhi-.1).max())
        with np.load(folder/'trajectory.npz') as archive:
            arrays={k:archive[k].copy() for k in ('q','dq','palm_pose','observation','action','raw_action','pd_targets_actual','tau','tau_valid','objects','table_pose','object_contact_force')}
            adapter=archive['adapter_raw_action'].copy() if 'adapter_raw_action' in archive.files else None
        q=arrays['q'][:,0].astype(float);pd=arrays['pd_targets_actual'].astype(float)
        observed=arrays['observation'][:,3:6].astype(float)
        assert np.array_equal(observed,np.clip(q[:,3:6],-5,5))
        rotations=Rotation.from_euler('XYZ',q[:,3:6]).as_matrix()
        palm=Rotation.from_quat(arrays['palm_pose'][:,0,3:7]).as_matrix()
        mounts=rotations.transpose(0,2,1)@palm
        mount_errors.append(float(Rotation.from_matrix(mounts@mounts[0].T).magnitude().max()))
        scale=np.asarray(audit['controller']['scale'])
        for t in record['wrist_branch_transition_state_ticks']:
            delta=q[t,3:6]-q[t-1,3:6];axis=int(np.argmax(np.abs(delta)));change=float(delta[axis])
            multiple=int(np.rint(change/(2*np.pi)));residual=float(change-multiple*2*np.pi)
            rvalues=adapter if adapter is not None else arrays['raw_action'] if record['batch']=='S2d' else None
            rdelta=None if rvalues is None or t<2 else float(np.max(np.abs((rvalues[t-1,3:6]-rvalues[t-2,3:6])*scale[3:6])))
            events.append(dict(batch=record['batch'],label=record['label'],state_tick=int(t),axis=axis+3,q_before=float(q[t-1,axis+3]),q_after=float(q[t,axis+3]),delta=change,nearest_2pi_multiple=multiple,residual=residual,PD_error_before_norm=float(np.linalg.norm(pd[t-1,3:6]-q[t-1,3:6])),finger_gate_before=event_finger_gate(q[t-1]),finger_gate_after=event_finger_gate(q[t]),observation_step=float(np.max(np.abs(observed[t]-observed[t-1]))),PD_target_step_before=float(np.max(np.abs(pd[t-1,3:6]-pd[t-2,3:6]))) if t>1 else None,R_increment_change_before=rdelta,R_raw_available=rvalues is not None,tau_cartesian_change_m=float(np.max(np.abs(arrays['tau'][t]-arrays['tau'][t-1]))),tau_valid_before=bool(arrays['tau_valid'][t-1]),tau_valid_after=bool(arrays['tau_valid'][t]),palm_rotation_step_rad=float((Rotation.from_quat(arrays['palm_pose'][t,0,3:7])*Rotation.from_quat(arrays['palm_pose'][t-1,0,3:7]).inv()).magnitude()),terminal=bool(t==record['terminal_tick'])))
        # New per-control checks in angular layers. Source event total is quoted
        # from brief40 rather than recomputing its death counts/detector.
        rvalues=adapter if adapter is not None else arrays['raw_action'] if record['batch']=='S2d' else None
        steps.append(dict(batch=record['batch'],label=record['label'],controls=len(pd),observation_q_steps_gt2pi=int((np.abs(np.diff(observed,axis=0))>2*np.pi).any(-1).sum()),absolute_PD_steps_gt2pi=int((np.abs(np.diff(pd[:,3:6],axis=0))>2*np.pi).any(-1).sum()),R_increment_changes_gt2pi=None if rvalues is None else int((np.abs(np.diff(rvalues[:,3:6],axis=0)*scale[3:6])>2*np.pi).any(-1).sum())))
        if record['signature']!='physical_joint_limit':continue
        if record['batch'] not in gates:gates[record['batch']]=helpers.package(run,'s44_'+record['batch'])[0]
        sequence=record['sequence']
        if sequence not in geometries:
            c=next(c for c in catalog if c['sequence']==sequence)
            with np.load(Path(c['source'])/'canonical.npz') as a:points=a['points'].copy()
            with np.load(root/'outputs/wam-rl-policy-v3/inspire-teacher-rollout-audit-20261020-r3/sequences'/sequence/'admission.npz') as a:bounds=a['table_bounds'].copy()
            geometries[sequence]=(points,bounds)
        points,bounds=geometries[sequence]
        state={key:arrays[key][-1].copy() for key in ('q','dq','objects','table_pose','object_contact_force')}
        state['q']=q[-1:,:].copy()
        gate=gates[record['batch']];original=gate.physical_safety(state,0,points,bounds,limits)
        assert original['reason']=='physical_joint_limit'
        converted={};raw_error=pd[-1,3:6]-q[-2,3:6];held_error=pd[-2,3:6]-q[-2,3:6]
        for multiple in (1,2):
            period=2*np.pi*multiple
            unwrapped=q.copy();unwrapped[:,3:6]=np.unwrap(q[:,3:6],axis=0,period=period)
            assert np.array_equal(unwrapped[:,:3],q[:,:3]) and np.array_equal(unwrapped[:,6:],q[:,6:])
            # Shift target by the SAME branch offset as the command-time state.
            target=pd[-1,3:6]+unwrapped[-2,3:6]-q[-2,3:6]
            assert np.allclose(target-unwrapped[-2,3:6],raw_error,atol=1e-12,rtol=0)
            corrected=dict(state,q=unwrapped[-1:,:])
            decision=gate.physical_safety(corrected,0,points,bounds,limits)
            converted[multiple]=dict(indices=decision['joint_violation_indices'],abort=decision['abort'],reason=decision['reason'],after_PD_error_norm=float(np.linalg.norm(target-unwrapped[-1,3:6])))
            assert decision==original,'wrist-only remapping changed the physical gate'
        lo=np.asarray(limits['lower']);hi=np.asarray(limits['upper'])
        def finger_gate(value):
            selected=np.arange(6,18)[mask[6:18]]
            return float(np.maximum(lo[selected]-.1-value[selected],value[selected]-hi[selected]-.1).max())
        before=finger_gate(q[-2]);after=finger_gate(q[-1]);assert before<=0 and after>0
        pselected=np.arange(3)[mask[:3]]
        def prismatic_gate(value):return float(np.maximum(lo[pselected]-.1-value[pselected],value[pselected]-hi[pselected]-.1).max())
        values=dict(prismatic_gate_max_before_m=prismatic_gate(q[-2]),prismatic_gate_max_after_m=prismatic_gate(q[-1]),new_PD_error_norm_before=float(np.linalg.norm(raw_error)),held_PD_error_norm_before=float(np.linalg.norm(held_error)),finger_gate_max_before=before,finger_gate_max_after=after,PD_error_norm_after_raw=float(np.linalg.norm(pd[-1,3:6]-q[-1,3:6])),PD_error_norm_after_unwrapped_4pi=converted[2]['after_PD_error_norm'])
        for key,value in values.items():distributions[key].append(value)
        cases.append(dict(batch=record['batch'],label=record['label'],tick=record['terminal_tick'],terminal_branch=record['terminal_wrist_branch_transition'],triggering_PD_illegal=any(j['target_nominal_excess']>1e-6 for j in record['joints']),original_indices=original['joint_violation_indices'],unwrap_2pi_indices=converted[1]['indices'],unwrap_4pi_indices=converted[2]['indices'],**values))
    assert len(cases)==21
    residual=np.abs([e['residual'] for e in events]);delta=np.abs([e['delta'] for e in events])
    quantiles={k:np.quantile(v,[0,.25,.5,.75,.9,1]).tolist() for k,v in distributions.items()}
    removed=[sum(not c[key] for c in cases) for key in ('unwrap_2pi_indices','unwrap_4pi_indices')]
    fraction=removed[1]/len(cases)
    decision='INVALID' if removed[0]!=removed[1] else 'CONFIRMED' if fraction>=.5 else 'REFUTED' if fraction<.2 else 'INVALID'
    return dict(decision=decision,scope='Wrist-coordinate-only gate replay; no inference of physical drive causality.',removed_2pi=sum(not c['unwrap_2pi_indices'] for c in cases),removed_4pi=sum(not c['unwrap_4pi_indices'] for c in cases),cross_terminal_branch_and_PD_legal=sum(c['terminal_branch'] and not c['triggering_PD_illegal'] for c in cases),event_abs_delta_quantiles=np.quantile(delta,[0,.25,.5,.75,.9,1]).tolist(),event_abs_residual_quantiles=np.quantile(residual,[0,.25,.5,.75,.9,1]).tolist(),events_residual_le_01rad=int((residual<=.1).sum()),event_multiple_counts={str(k):sum(e['nearest_2pi_multiple']==k for e in events) for k in sorted(set(e['nearest_2pi_multiple'] for e in events))},mount_invariance_max_rad=max(mount_errors),quantiles=quantiles,cases=cases,events=events,layer_steps=steps)
```
<!-- brief44-code-end -->
