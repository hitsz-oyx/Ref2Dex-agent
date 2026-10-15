# 本地 UniDexGrasp++ policy 数据与 DExplore 微调可用性

日期：2026-10-15。结论是文件/控制合同可行性判断，不是训练收益结论。
配套[官方审计](policy-data-finetuning-audit.md)，区分2023 UniDexGrasp/++与2026 UniDex。

## 结论

本地已有足量的真实ShadowHand仿真状态—动作数据：2066个不同mesh、33056条执行，
约为当前DExplore的41.3倍物体/50.1倍轨迹；统一30Hz后状态帧约19.9倍。
不必重新采Shadow rollout才能取得动作标签。原始文件保留300D观测和真实控制/PD targets；
现有forecast provider主动只输出几何标签，并不意味着原始采集没有动作。

它可直接服务已有WAM几何预测路线，或经独立Shadow状态/动作头进行动作辅助训练；
不能把24D动作/22D joint target改名当当前Inspire q_PD监督。手型、控制物理量和任务都不同。
当前v2三臂仍冻结原数据/输入，本文没有启动新训练或改变准入。

## 本轮复算证据

报告：[audit.json](../../outputs/3d-data-preparation/unidex-policy-feasibility-20261015-r1/audit.json)。
可重跑的纯CPU审计脚本同时存于该报告目录（原运行脚本在项目tmp）。
读取原20物体run与82个扩产batch的全部manifest/result，共83个完成replay根；
核实所有原始episode和33056个紧凑轨迹引用文件都存在。4个跨批次原始episode抽样检查
所有字段shape/finite、控制数=状态数-1、真实dt=1/60通过；不是本轮重审全部payload值。
既有全量文件/语义审计见[3D数据记录](../../src/task/3d-data-preparation/docs/STATE.md)。

| 口径 | 本地UniDexGrasp++ | 当前DExplore补采660 | 比值 |
| --- | ---: | ---: | ---: |
| 不同物体mesh | 2066 | 50 | 41.32 |
| 执行轨迹 | 33056 | 660 | 50.08 |
| TRAIN / DEV轨迹 | 26448 / 6608 | 589 / 71 | 划分不同 |
| 原始状态帧 | 6611200，60Hz | 166337，30Hz | 不直接跨频率比 |
| 统一30Hz状态帧 | 3305600 | 166337 | 19.87 |
| 原始实际控制帧 | 6578144，60Hz | 165677，30Hz | 控制时钟/物理量不同 |
| 每条平均观测时长 | 约3.3秒 | 约8.4秒 | 任务长度不同 |

UniDex原始episode约7.67GB、compact约1.61GB；不含共享CAD/模型/其他日志。
2066对象每个16次执行，仅取一档已准备scale；重复同对象/初始条件不能全部算成新任务。
DExplore是多参考任务跟踪，UniDex++是桌面抓取举升，轨迹数量不等于任务多样性。

### 窗口口径

正在远端v3使用的严格recipe source4为263486 TRAIN / 65883 DEV窗（合329369），stride8。
当前DExplore严格位置准入为93680 TRAIN / 10376 DEV（合104056），stride1。
两者质量标准/起始历史规则/时长都不同，不能把329369÷104056解释成“数据只多3倍”。
对每条100帧UniDex，H4/future24完整dense query原始可有73个，即2413088个候选；
此为质量过滤前的数学上限，不是已准入的actor训练数量，不应通过重叠窗夸大独立样本量。

来源：
[严格v3 recipe](../../outputs/ptv3-shared-wam-v3/nine-source-strict-quality-20261015-r1/recipe.json)、
[DExplore严格审计](../../outputs/ptv3-shared-wam-v2/rollout-admission-current-20261015-r1/report.json)。

## 原始动作没有丢，紧凑版与reader只保留了子集

真实原始文件样例：
`outputs/3d-data-preparation/unidexgrasp-expansion-20261014-r1/batch000/replay/episode-00.npz`。
200条时间状态×200个环境，每条199个60Hz控制：

| 字段 | 每环境shape | 含义 |
| --- | --- | --- |
| q / dq | 200×22 | Shadow关节实测位置/速度 |
| hand | 200×11×3 | palm/指根/指尖语义几何点 |
| hand_root | 200×13 | root位置、四元数、线/角速度 |
| object / goal | 200×7 / 200×3 | 当前物体SE(3)/任务目标 |
| observation | 200×300 | 官方actor当前观测，含状态、传感量、上一拍动作与shape feature |
| action / applied | 199×24 | canonical actor命令 / 坐标变换后的实际施加命令 |
| force / torque | 199×27×3 | 实际刚体力/力矩，根部索引需按控制源码解释 |
| target | 199×22 | 实际关节position target，不含可等同Inspire的腕部PD目标 |

compact保存100个30Hz手/物/q/dq/root状态，但动作仍是199个60Hz控制与action_time；
没有300D observation、target、force/torque。当前forecast loader进一步只返回统一几何字段。

源码：[采集器snapshot与step](../../src/task/video-data-preparation/tools/run/replay_unidexgrasp.py)、
[compact导出](../../src/task/video-data-preparation/tools/audit/audit_unidexgrasp_replay.py)、
[forecast reader](../../src/task/3d-data-preparation/src/pose_mesh_data/windows.py)。
未来做actor数据应从raw按状态→命令→下一状态取样，不把array index混用。
若保留30Hz WAM、60Hz Shadow执行，8个30Hz区间应保留对应16个60Hz控制；
不能只抽每第二拍force/torque就声称原物理控制等价。也不能把q_real下一拍替代为q_PD。

## 数据质量：总量不能全部当expert

33056条中：

- official_success=14671（44.38%）；是曾进入物体目标5cm范围的sticky flag。
- 最末帧物体距目标5cm内=12490（37.78%）；比曾经成功更严，但仍不保证接触或持续稳定抓持。
- 这两种计数不等于held45或本项目正式成功率。

若只按官方flag选，会有约1.47万条候选（相对660仍约22.2倍），实际高质量模仿数量还需
进一步排除高速飞离/掉落/瞬时接近，并核验保持。失败转移可以用于动态模型，不能因此
不加区分作为成功抓取BC teacher。当前几何清洗主要检查物理运动异常，并非成功示范筛选。

独立控制源码明确目标距离flag：
[compute_hand_reward](https://github.com/PKU-EPIC/UniDexGrasp2/blob/6c9e710b04769bf5d9765e3233820101cb3d718f/dexgrasp/tasks/shadow_hand_grasp.py)。

## 和目前路线的关系

当前三臂父模型80k的TRAIN统计sources为OakInk2、GRAB、ARCTIC，未包含UniDex。
证据：[stats_step80000.json](../../outputs/ptv3-shared-wam-v2/remote-parent-20261015-r1/stats_step80000.json)。
远端v3九源预训练已经包含清洗后的UniDex作为hand/scene forecast来源，未使用Shadow action loss。
因此不能说目前的Inspire动作微调已经利用了这三万多条动作监督，也不能重复声称forecast
数据尚未接入任何模型。

## 决策建议（推断，未实验）

1. 若要判定“660条是否不够”，固定Inspire手型、控制定义、模型与dev，对TRAIN做分层
   1/4、1/2、全量学习曲线；若全量仍显示明显随样本数改善，再优先增加同手型/同控制合同
   的teacher rollout和初始条件覆盖。仅增加相邻窗口或重放完全同样的轨迹不等价新数据。
2. 若要马上利用已有UniDex policy数据，保留共享WAM及Inspire头，另建Shadow状态投影与
   24D控制头，先在固定小面板检验原观测→action的可拟合性及动作重放，再做跨手共享表示
   预训练→Inspire微调。该比较同时涉及跨手/跨任务迁移，不能当“纯数据量”因果实验。
3. 若目标是扩大直接可训练Inspire q_PD标签，最终仍需在Inspire控制器中重新执行并记录，
   或经明确retarget+物理qualification后再采集。手姿重定向本身不保证原物体抓取仍成功。

本轮不改变进行中的12k三臂，不新增GPU采集或大下载。当前训练仍在早期下降，数据量不足
是合理候选解释，尚不能认定为主要原因；轨迹覆盖、模型拟合和控制语义仍需分别验证。
