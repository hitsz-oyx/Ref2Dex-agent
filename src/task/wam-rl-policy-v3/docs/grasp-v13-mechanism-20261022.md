# v13：机制分解、同接口几何/GT 和 17 例中止审计

## 结果解读表（运行前固定）

| 结果（Hermes 指定） | 结论 | 下一步决定 |
|---|---|---|
| (i) 腕抓姿单项有效 | 关键在“抓哪儿” | 优先检验可行抓姿的跨物体覆盖 |
| 单项不足，(i)+(ii) 有效 | “抓哪儿 + 怎么合手” | 保留腕姿和手指的联合接口 |
| (iii) 时机单项有效 | “何时闭手”是可用机制 | 比较预测时间与实测相位，避免误归因 |
| 单项都不行、完整意图有效 | 联合意图，不能归为单点 | 保留组合并做必要性检查，不能仅凭单项失败证明三阶交互 |

**事前范围澄清／仍不一致**：v10 的 35/40 实际使用腕姿+手指、实测相位；预测 contact_index/eta 没有进入相位切换。因此原始 A1 就是 (i)+(ii) 的联合臂，不能说“预测时机三项合起来才87.5%”。新 TIME 臂是明确增加预测时间控制的反事实。固定模板 T0 也不等于零τ的 R 输出：保留原 A2 0/40，额外 T0 控制模板本身的强弱。Hermes 的单项解释须以单项优于 T0 为前提；单项相对完整 A1 不劣的功效有限，不能把不显著等同于相等。

## A3/A4 结果解读表（运行前固定）

| 结果 | 结论 | 下一步决定 |
|---|---|---|
| A3 同接口达到高位 | 随物体移动的目标可能足够，WAM 不特殊 | 重估 WAM 增益与成本 |
| A3 低、A1 高且 A4 确实进入 close | WAM 包含纯时钟几何重定向没有的信息 | 结合成分消融归因抓姿/合手 |
| A4 没进 close 或未来无效 | GT 实现/准入仍阻塞 | 不把 A4 当上界，不做 WAM 优越性结论 |
| A1/A3/A4 在原未见子集均低 | 接口/控制/任务难度仍未区分 | 三臂 RL HOLD；按原始轨迹定位阻塞 |

来源：Hermes 提出机制和同接口对照；外部审计要求跨物体覆盖与实现可核查。旧 v9 的 A3/A4 不能直接解释新接口。仍不一致：A4 是源示范的闭环相位匹配目标，并非最优机器人策略上界。

## 中止审计结果解读表（读取轨迹前固定）

| 结果 | 结论 | 下一步决定 |
|---|---|---|
| 集中在 approach、触桌前无稳定抓取 | 接近/支撑几何是近端阻塞 | 修接近候选与支撑准入 |
| 集中在 close、接触且物体被压过桌面 | 接触闭合与桌面约束是近端阻塞 | 分离手部离桌与物体安全，检查闭合/恢复 |
| 集中在 lift/hold、已有持握后再入桌 | 抓后消费/滑移阻塞 | 检查相对姿态与保持控制 |
| 跨阶段/形状分散 | 不支持单一形状原因 | 保留逐例量，不用“其他”合桶 |

Hermes 要求17例逐例；外部审计要求几何量/形状关联。观察关联不升级为形状因果，源控制帧 n−1 与中止状态 n 分开记录。

## 冻结设计、预算与决策

Decision Probe；不训练。最便宜方法：复用 v10 已保存的 A1/A2 各子集40次、同初态/seed、同冻结 R/WAM 和控制代码，新增 seen WRIST/FINGER/TIME/T0 各40；之后在 seen/unseen 上新增 A3/A4 各40。新物理回合最多320、192000 controls；每阶段每worker墙钟≤4h，排队≤4h，磁盘剩余≥20GiB。只在空闲GPU4/5/6上运行，不能抢占正在运行的 n40。先机制、后 A3/A4；失败全部保留，技术故障不自动换seed补样。每阶段前5个 seen seed 是实现检查，仍纳入40，禁止按成功率改参数。完整 A1=腕姿+手指+实测阶段，沿用原始40/40配对数据，不重采。

模板在运行前固定为既有 `s2b-reference-r2/reference.npz` 的 contact30/held50 模板：物体坐标腕抓姿与六指预形/闭合；各臂保留 R 调用和真实 WAM 输入，随后仅交换指定输出字段。T0/TIME 不使用 R 的腕/手指输出；保留调用不代表输出因果有效。WRIST 保留 R 腕抓姿，手指模板；FINGER 保留 R 手指，腕模板；TIME 两者模板，首次有 contact_index 的 WAM 预测锁定接触截止时间，且只有该时钟允许 approach→close；后续 close→lift 仍必须真实稳定接触。T0 完全模板+实测阶段。模板腕依现有 template_intent 限速，六指模板同一份。

A3 沿用每一源未来帧的物体局部手形，通过当前物体姿态重定向，源时钟 t；A4 改用当前实测手的刚体基座与源物体局部手匹配，单调游标直到源接触帧，再取24帧、每帧映射当前物体，禁止以时钟跑过接近而永不闭合。源接触帧取reference真实接触力首次≥2指0.25N（缺失则明确拒绝，不能临时调阈值）。两者同 R、same v10 RepairedGraspConsumer、15mm/20s 终点、真实关节和桌面安全。不能混入新几何阈值 n40 以解释旧 A1 的87.5%。

统计：每臂 Wilson95、原始 strict/hold/close/physical abort、Fisher 对 T0 和 A1、配对 McNemar，探索性多比较逐项报原始p与Holm。高位预定义≥32/40；单项“有效”要求相对T0 Fisher双侧p<.05（多比较校正另外标注）。无新seed、无调参、无神经网络更新，不重训v6，三臂RL继续HOLD。

运行代码、确切命令、环境、输出与结果在运行登记和下文补齐；在结果完成前不回答可推广/成功机制为已证实。

## 17 例审计结果（原轨迹，无新增物理步）

17/17 都触发原协议的 `physical_table_penetration` 安全门，阶段为 approach12 / close5 / lift0 / hold0。但此安全门是**采样物体点低于桌面20mm且质心投影回到桌面范围**的保守代理，并非精确网格碰撞。细分后：

| 近端事件 | approach | close | 合计 | 结论 |
|---|---:|---:|---:|---|
| 桌面投影内跨过−20mm门槛 | 5 | 4 | 9 | 支撑/姿态违规，不能仅凭点集断言接触压穿 |
| 已完全落到5.5mm厚桌板下方，再进入桌面投影 | 6 | 1 | 7 | 掉落/失去支撑后的投影重入；不是桌面碰撞穿透的证据 |
| 已低于桌面，在边缘重新进入投影 | 1 | 0 | 1 | 边缘掉落/旋转后的投影重入 |

**修正初步归因**：不能把17例统一说成闭合把物体压穿。按同一512点代理，7例完全桌下重入的末状态最高物体点也低于桌面顶面40.7–963.2mm（亦低于底面）；这些物体已经落下，保守门才因投影重新进入而中止。原安全阈值、失败分母和87.5%/7.5%结果完全不改。17例全部曾启用恢复；16例最后一次手部支撑间隙为正，保护手部几何不能保证物体不掉落。

阶段取最后命令n−1，几何取中止状态n；候选距离是该状态重算的几何候选，不能伪称实际执行候选。指尖间隙列第二近手指，JSON保留五指向量。关节余量是12个有限位手指DOF的nominal最小余量；负数不等于超过额外0.1rad物理安全余量。

| trial | 物体 | tick | 阶段 | 近端事件 | 第二指间隙 mm | 腕距几何候选 mm | 最小关节余量 rad | 物体桌面间隙 mm |
|---|---|---:|---|---|---:|---:|---:|---:|
| p003-A1 | banana | 86 | close | 投影内跨门槛 | 16.46 | 63.39 | 0.19697 | -21.17 |
| p007-A1 | cup | 127 | approach | 完全桌下后投影重入 | 682.27 | 165.20 | 0.10853 | -720.38 |
| p011-A1 | flashlight | 75 | approach | 完全桌下后投影重入 | 92.21 | 43.08 | 0.20587 | -142.65 |
| p013-A1 | gamecontroller | 137 | approach | 完全桌下后投影重入 | 585.08 | 75.89 | 0.16199 | -694.04 |
| p015-A1 | hammer | 77 | approach | 投影内跨门槛 | 116.41 | 150.17 | 0.09877 | -35.76 |
| p019-A1 | banana | 86 | close | 投影内跨门槛 | 16.73 | 64.59 | 0.19174 | -21.44 |
| p023-A1 | cup | 140 | approach | 完全桌下后投影重入 | 637.23 | 72.99 | 0.13315 | -717.63 |
| p027-A1 | flashlight | 88 | approach | 完全桌下后投影重入 | 947.05 | 71.31 | 0.19744 | -1018.38 |
| p031-A1 | hammer | 83 | approach | 投影内跨门槛 | 117.50 | 143.38 | 0.09132 | -38.77 |
| p047-A1 | hammer | 80 | approach | 投影内跨门槛 | 157.49 | 173.76 | 0.09549 | -35.60 |
| p051-A1 | banana | 86 | close | 投影内跨门槛 | 16.73 | 65.69 | 0.18992 | -22.56 |
| p059-A1 | flashlight | 76 | approach | 低于平面后投影重入 | 43.82 | 40.51 | 0.17784 | -107.38 |
| p063-A1 | hammer | 76 | approach | 投影内跨门槛 | 108.20 | 146.47 | 0.10539 | -35.71 |
| p067-A1 | banana | 86 | close | 投影内跨门槛 | 16.95 | 65.63 | 0.19119 | -24.37 |
| p075-A1 | flashlight | 91 | approach | 完全桌下后投影重入 | 982.50 | 56.69 | 0.18465 | -1043.62 |
| p077-A1 | gamecontroller | 325 | close | 完全桌下后投影重入 | 626.37 | 74.26 | 0.00023 | -741.14 |
| p079-A1 | hammer | 78 | approach | 投影内跨门槛 | 104.37 | 140.39 | 0.10764 | -27.53 |

| 物体 | 中止/总数 | 最小宽 mm | PCA长短比 |
|---|---:|---:|---:|
| alarmclock | 0/5 | 45.28 | 2.78 |
| banana | 4/5 | 41.36 | 4.97 |
| cubesmall | 0/5 | 40.13 | 1.05 |
| cup | 2/5 | 80.98 | 1.25 |
| cylindermedium | 0/5 | 79.62 | 1.11 |
| flashlight | 4/5 | 32.26 | 4.27 |
| gamecontroller | 2/5 | 49.67 | 2.83 |
| hammer | 5/5 | 23.23 | 8.23 |

香蕉、手电筒、锤子合计13/15，其余五物体4/25；但香蕉4例和锤子5例属投影内门槛，手电筒4例均属掉落后重入，不能合称同一形状机制。8个物体各5个重复seed，物体/序列/形状混杂，不能把重复seed当形状因果检验。跨物体覆盖仍必要。

逐例五指间隙、候选旋转差、12关节及coasting/PD余量、末步位移、有效接触、初始与首次低于平面状态、物体最高点/桌板厚度、阶段和原始路径在 `outputs/wam-rl-policy-v3/grasp-v13-mechanism-20261022-r1/abort17-final.json`。早期 `abort17.json` / `abort17-detailed.json` 保留为逐步诊断产物，最终细分以 final 为准。

复现：

```sh
python3 src/task/wam-rl-policy-v3/tools/audit/audit_v13_abort17.py --run outputs/wam-rl-policy-v3/grasp-v10-stat40-20261022-r1 --output outputs/wam-rl-policy-v3/grasp-v13-mechanism-20261022-r1/abort17-final-replay.json
```

## 确切批量命令

```sh
python3 src/task/wam-rl-policy-v3/tools/run/run_grasp_components.py --output outputs/wam-rl-policy-v3/grasp-v13-mechanism-20261022-r1 --queue-wall-s 14400 --detached
```

重复改为新的输出目录即可；不覆盖检查点。`supervisor.json` 保存每个worker的确切命令、PID、提交和GPU启动状态；冻结新增干预代码在 `launch-code/`。worker 的 `runtime-code/` 保留实际消费/host/scorer，manifest保存Python/Torch/CUDA、模型路径和控制计数。原生物理host固定 `/home2/wyy/oyx_ws/.runtime_envs/dexplore_v120_train/bin/python`。

## 实现预检与排队

预注册提交 `fad72d0a4c3c527a77fb13991569e90dcc6a719c` 已推送并核对远端。原supervisor PID3701886 以独立进程排队；零worker/零物理步时停止，改为逐卡等待接管，旧收据保留在 `supervisor-original-queue.json`。当前PID见 `supervisor.json`，不会抢占覆盖n40。34项针对干预/阶段/消费/伺服的测试和全量278项相关测试通过（20.09s），日志 `preflight-tests.log`。17例审计重复运行逐字一致。

首次共同有效预测时，seen40对的非τ观测误差为0，R的腕部与手指动作都改变；归一化动作差范数中位数约0.03014/0.05841。它们单位不同，不能据此判定哪个贡献大；这只是干净输入效应检查，不是成分成功机制。40对是同一seen初态重复seed，不能看作40种物体覆盖。复现：

```sh
python3 src/task/wam-rl-policy-v3/tools/audit/audit_v13_common_state.py --run outputs/wam-rl-policy-v3/grasp-v10-stat40-20261022-r1 --output outputs/wam-rl-policy-v3/grasp-v13-mechanism-20261022-r1/common-state-replay.json
python3 src/task/wam-rl-policy-v3/tools/audit/audit_v13_phases.py --run outputs/wam-rl-policy-v3/grasp-v13-mechanism-20261022-r1 --output outputs/wam-rl-policy-v3/grasp-v13-mechanism-20261022-r1/phases.json
```

A4实现预检：9条源序列均有有效双指实测接触，接触帧30/41/40/48/37/42/33/39/45，匹配初始游标全为0；不存在缺失接触帧而默默回退。物理close是否真正进入仍待新批量记录，预检不代替实跑。

安全门细分另新增2项边界测试通过：桌板下投影重入与桌面内跨阈值分开，不能以大步下落自动判作重入。17例初始桌面间隙−4.8至+2.8mm且都在投影内，均不在初态就触发−20mm中止。

轨迹审计图（最终JSON导出，不是新实验）：[PNG](../../../../outputs/wam-rl-policy-v3/grasp-v13-mechanism-20261022-r1/abort17-figure.png) / [PDF](../../../../outputs/wam-rl-policy-v3/grasp-v13-mechanism-20261022-r1/abort17-figure.pdf)。空心点为首次低于−20mm，实心点为安全门中止；投影重入可能明显滞后，p077约3.27s已落下，10.83s才因投影重入中止。方形为close，圆形为approach。

```sh
python3 src/task/wam-rl-policy-v3/tools/audit/plot_v13_aborts.py --audit outputs/wam-rl-policy-v3/grasp-v13-mechanism-20261022-r1/abort17-final.json --output-prefix outputs/wam-rl-policy-v3/grasp-v13-mechanism-20261022-r1/abort17-figure
```

## 冻结接口回放检查

`replay_v13_legacy_consumer.py` 在CPU用保存的R输出与真实状态重放5条固定旧轨迹：p000-A1/A2、p003-A1、p015-A1、p077-A1，共1688步。v13可选时机钩子未启用时，命令最大误差0、阶段差0；结果 `legacy-replay.json`。没有神经网络CPU推理、没有新增物理步。它支持复用A1/A2对照，但不验证TIME分支的物理成功。

```sh
python3 src/task/wam-rl-policy-v3/tools/audit/replay_v13_legacy_consumer.py --run outputs/wam-rl-policy-v3/grasp-v10-stat40-20261022-r1 --output outputs/wam-rl-policy-v3/grasp-v13-mechanism-20261022-r1/legacy-replay-repeat.json
```

WRIST包括R的腕增量与τ关键抓姿（也供几何恢复作hint）；预测接触index会选择关键姿态，而非直接推进相位。因此“没有预测时钟切阶段”不等于预测contact_index对抓姿无影响。A3的几何τ也使用源示范的局部手形，不是只看物体表面就凭空生成完整手型。

## 排队操作调整（零实验步时）

最初supervisor等待三卡全空闲；覆盖n40三卡结束时间不同，改为每张卡≤64MiB才各自启动分配好的worker。第2阶段仍等待全部机制worker和原始重评分结束才启动。旧排队已确认零worker/零物理步，旧PID3701886已退出；冻结的 `launch-code/` 和两份registration不变，预算不重新领取。GPU准入测试通过，禁止0–3/7。接管确切命令：

```sh
python3 src/task/wam-rl-policy-v3/tools/run/run_grasp_components.py --output outputs/wam-rl-policy-v3/grasp-v13-mechanism-20261022-r1 --queue-wall-s 14400 --resume-empty-queue --detached
```

Hermes新重复仍使用前述普通命令与全新输出目录；`--resume-empty-queue`只接受已明确标记停止、从未启动worker的队列，不能恢复有实验结果的运行。

## 干预接线独立重放

新增 `replay_v13_interventions.py` 不调用生产swap函数：独立选择腕/手指字段、从记录的R输出和真实状态重建命令；A3/A4还独立重建源物体局部未来与闭环游标。逐步检查完整τ、抓姿hint、预测时钟、每步R调用且输出非零，故意篡改抓姿1cm的审计负例必须被拒绝。GPU6先完成13回合/7800步已PASS，命令/抓姿/τ/阶段差均0；`component-replay-g6-progress.json`只是进度审计，不代替全量最终审计。

```sh
python3 src/task/wam-rl-policy-v3/tools/audit/replay_v13_interventions.py --worker outputs/wam-rl-policy-v3/grasp-v13-mechanism-20261022-r1/mechanism/g6 --output outputs/wam-rl-policy-v3/grasp-v13-mechanism-20261022-r1/component-replay-g6-final.json
```

对两阶段g4/g5/g6分别运行；CPU只重放控制/几何，没有神经网络推理或新增物理步。TIME若没有contact_index则保持未排定闭合，必须报告其覆盖率；不能把“信号未产生”称为“时机准确但无效”。

## A3/A4执行前的可用性边界

A3按源时钟到 `t+24>=T` 时失效；A4按实测手匹配源游标，不随墙钟到达源尾而失效。该差异在运行前由实现已知，必须并列报告有效τ控制数、close进入数与游标范围。A3低不能单独证明“任何几何方法都做不到”；若A4也未进入close，它是实现/准入与R输入适配的阻塞记录，不是证明WAM优于最优GT策略。Hermes的A3高/低解读用于已登记算法之间的取舍，外部审计的可实现性要求仍须满足；二者不等同于几何方法的理论能力边界。

## 冻结R权重复算

`replay_v13_actor.py` 在空闲GPU从worker记录的base checkpoint批量复算R，并逐步检查observation的τ切片/valid mask；另在共同观测保持mask与全部非τ字段，将R输入τ清零做敏感性检查。这个反事实不执行物理控制，不能冒称新的A2全策略消融。GPU6已完成的52回合/31200步PASS：R动作最大误差7.246e−7，τ/valid一致；零τ使R最大输出变化0.4802。实际GPU计算1.066s，0updates，Python `/usr/bin/python3`、Torch2.4.1+cu121/CUDA12.1。启动前物理GPU6为2MiB。

```sh
python3 src/task/wam-rl-policy-v3/tools/audit/replay_v13_actor.py --worker outputs/wam-rl-policy-v3/grasp-v13-mechanism-20261022-r1/mechanism/g6 --gpu 6 --output outputs/wam-rl-policy-v3/grasp-v13-mechanism-20261022-r1/R-replay-mechanism-g6-repeat.json
```

各worker都可独立复算；只接受空闲物理GPU4/5/6，其他作业占用时排队。模型权重与保存观测在各worker目录/manifest中追溯，工具不生成checkpoint。

GPU5的52回合/31174步R复算同样PASS，最大误差9.537e−7，零τ敏感性最大0.3636。GPU6全52回合/31200步独立干预重放PASS，保存的R非零、每步调用、τ、关键姿与实际命令/阶段均匹配。

运行性能只读检查：同样1688步，OpenBLAS2线程13.54s、64线程13.49s，命令和阶段差均0。未见墙钟收益，故不改正在执行或后续批量的物理控制配置；记录在 `replay-threads2.time` / `replay-threads64.time`。

## GT抓取相位资格修订（正式对照重跑前固定）

**结果解读表仍按开头预注册，不改变成功判据或物理常数。** 新增资格反馈：真实源第49帧已接触但尚未建立抓取（第50帧才建立），旧A4接触cap=30，使R的8帧窗口最多看到38。预接触腕姿接口的截断被误用于需要持续合手的R接口。`GT-acquisition-red.log` 可复现失败，不依赖当前对照成功率。修复后原生11点同时匹配腕和手指，相位单调推进到源首次“有效力+近物接触+运动耦合抓取”；不使用单个初态的帧常数。

9条源序列均PASS：抓取cap分别50/55/61/65/51/54/54/54/82；在每条源的cap−1状态均匹配到cap−1，R8包含后续抓取，完整24帧全为真实源数据。`GT-acquisition-qualification.json`。新增实际源回归测试先红后绿，另用“腕姿相同、只有指尖变化”的合成例防止退回只匹配基座。

检查与排队存在并发：零回合守卫发现对照已启动，拒绝按空队列处理；随后仅停止自有supervisor与自有worker（SIGINT触发保存/清理）。旧provider保留9个已完成回合和1个部分回合，确认3668个控制步，至多再有1个在途步；原路径 `grasp-v13-mechanism-20261022-r1/comparators/` 全部保留，未选择性替换同一统计中的失败。详见 `GT-preflight-interruption.json`。这些是未通过资格的工程数据，不与修后正式对照合并。

机制160回合/95620步和原始重评分160/160 PASS保持冻结；6臂初态40对PASS；机制三worker全部干预重放与R权重复算PASS。修复后对照另用 `grasp-v13-comparators-20261023-r2`，同一固定80初态×A3/A4，共160回合，无新seed、无调参，仍同v10 R/consumer/15mm/20s。正式总样本仍320；包括旧工程在途步的本轮物理上限为95620+3669+96000=195289 controls，0updates。

修后确切命令：

```sh
python3 src/task/wam-rl-policy-v3/tools/run/run_grasp_components.py --comparators-only --mechanism-run outputs/wam-rl-policy-v3/grasp-v13-mechanism-20261022-r1 --output outputs/wam-rl-policy-v3/grasp-v13-comparators-20261023-r2 --queue-wall-s 14400 --detached
```

Hermes普通新重复仍可用同一runner、不加`--comparators-only`，在全新目录一次跑完整两阶段，使用修后GT来源。r2的`summary-final.json`将读取r1机制与旧v10A1/A2，明确列出两次执行及旧工程控制开销。

来源：Hermes指出旧GT未进入close；外部审计要求先排除实现/准入问题；当前回放确认并修复的是相位回退，不声称“源示范GT”是干预后的真实未来或最优策略上界。

## 完整源GT恒等性修订与有效A3复用（最终对照方案）

另一条资格检查揭示原实现错误：在源自身第49帧状态，GT必须还原源第50–73帧实际手轨迹；逐未来帧重锚定会抹掉物体后续运动，误差最大170.924mm。这是把A3几何臂的锚定误用于完整GT；`GT-anchor-red.log`先红。修为“在匹配到的当前源帧作一次SE3锚定”，保留全部源未来运动。原生11点/抓取相位修复同时保留。9条源均通过原状态未来恒等性和刚体变换等变性，见最终r3的 `GT-full-source-qualification.json`；GT依然是源示范参照，非当前干预的因果未来或最优策略。

只停止自有r2对照进程，保留所有完整/部分轨迹。为避免重跑有效A3，最终r3按**先r1后r2、每个trial首次有效完成**规则复用15个A3，包含物理中止失败；不读取成功标签作选择。A3生成器、消费器、R模型代码及swap函数均检查未变，原始目录直接软链接，原始重评分与R权重复算覆盖这些样本。专门的来源测试确保“首次失败、后次成功”仍保留首次失败。原生初态/模型seed继续原面板，不抽新seed。

旧两次对照执行合计确认10374步，上界10376；其中4510步属于复用的正式A3，额外工程开销5864–5866步（包含旧A4变体、重复A3和技术中断）。最终对照尚需145个新回合，正式样本仍A3/A4各seen40+unseen40。整轮物理上限95620+10376+145×600=192996步，未超过前一次修订的195289上限；全部0updates。

最终正式对照：`outputs/wam-rl-policy-v3/grasp-v13-comparators-20261023-r3/`。r1机制160不变，r1/r2对照原数据全部保留；旧A4两种算法作为工程变体，不混入正式“完整GT”统计。r3 `summary-final.json`会分列正式控制数、工程开销及总控制上下界。

```sh
python3 src/task/wam-rl-policy-v3/tools/run/prepare_v13_carry.py --source-run outputs/wam-rl-policy-v3/grasp-v13-mechanism-20261022-r1 --source-run outputs/wam-rl-policy-v3/grasp-v13-comparators-20261023-r2 --output-run outputs/wam-rl-policy-v3/grasp-v13-comparators-20261023-r3
python3 src/task/wam-rl-policy-v3/tools/run/run_grasp_components.py --comparators-only --mechanism-run outputs/wam-rl-policy-v3/grasp-v13-mechanism-20261022-r1 --carry-a3-index outputs/wam-rl-policy-v3/grasp-v13-comparators-20261023-r3/carry-A3-index.json --output outputs/wam-rl-policy-v3/grasp-v13-comparators-20261023-r3 --queue-wall-s 14400 --detached
```

上面prepare已执行，重复需全新输出目录。Hermes全新完整重复仍只需普通runner命令，自动使用最终GT来源，不必携带旧A3。A3每未来帧物体局部映射，A4在当前匹配源帧锚定完整真实源未来；两者区别不能再被抹掉。
