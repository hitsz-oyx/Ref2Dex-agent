# 当前状态

## 2026-10-18 GT-τ后期数据微调未过保持资格，停止这一配方

[GT-τ实验卡](experiments/probes/P-20261018-unidex-live-oracle-tau.md)的有界后续已完成：
原421 DAgger150行加453晚期60行，两臂同容量、同额外2000步微调，482各执行200步。
新τ臂和state-only均未达到目标或超过初始物体高度，终态距目标29.61/28.26cm；
原τ在相同前150步曾抬升18.32cm、连续达标13帧。200个新query回放误差全0，
三臂初始观测/物体状态逐值相同，末次动作后原始obs独立复算一致。训练/固定旧状态上
动作误差改善没有转化为闭环行为；该packet+额外更新配方UNPROMISING，停止追加。
保留固定421正向与旧checkpoint，不把一次失败归咎WAM，也不在该学生上直接换WAM-τ。
453已成为适应数据，471/482是开发评估，不是未触碰TEST。后续执行器应明确观测、
计划条件训练覆盖和恢复合同后再物理测试；并行调度执行器实现不由本结果自动认证。
代码d2c5de16，Python3.8.10/Torch2.0.1+cu118，35.36分钟、21.5MiB；32项相关测试通过，
本地GPU0及短诊断GPU1已释放，远端训练未动。结果与曲线在
`outputs/wam-rl-policy/oracle-tau-holding-adapt-20261018-r1/comparison/`。

## 2026-10-18 Viser已接入真实调度goal条件下的冻结WAM预测

[Viser Probe](experiments/probes/P-20261018-goal-wam-viser.md)固定修复版v4 step2000，
在相同H4状态下比较半倍前视／当前调度／双倍前视，保留检查点与跨度约束。
4个TRAIN物体68状态204组预测，远端RTX4090推理43.54秒，PyTorch峰值398MiB；
本地八卡有其他用户进程，本轮未占本地GPU。现有远端训练未停止或改动，推理GPU已释放。
原始手物状态、SE(3)预算、goal锚点、预测t0、原生τ转换及C概率独立复核通过。
模型手点末端相对调度臂平均差：短臂3.10–3.78mm，长臂3.49–5.74mm，仅条件敏感性。
端口8770的Viser只读本地缓存，当前实测与预测未来时间分开；尚无执行器或物理可达结论。
产物`outputs/wam-rl-policy/goal-wam-viser-20261018-r1/`；不修改WAM训练或调度算法。

## 2026-10-18 调度器加入手物历史并修复进度问题，真实3D审阅已就绪

[后续工程Probe](experiments/probes/P-20261018-interaction-goal-scheduler.md)修复持续慢速
误超时、低频跨过里程碑与确认时goal整段跳变，新增手物相对运动代理；弱proxy只影响
goal，不抹掉独立实测路径证据。17项单测通过，固定5ed8ae42的19片段/4811帧独立复算
通过；Clock10Hz仍未确认一个中间位姿，保留失败。4片段虽几何到达但末段proxy为LOST，
不能称操控成功。原速度4物体相对goal跳变p95约15mm→5mm，只是工程平滑性。
Firefox57帧及播放/鼠标拖动/时间轴检查通过，新3D显示手骨架、CAD、goal和判定依据。
入口`outputs/wam-rl-policy/interaction-goal-scheduler-20261018-r2/review.html`；WAM未改。

## 2026-10-18 独立goal调度器完成几何回放，尚未接WAM或策略

按用户最新范围，只实现有序SE(3)进度、状态相关前视、有界恢复与显式超时；不修改WAM。
[调度Probe](experiments/probes/P-20261018-goal-scheduler.md)的10项单测通过，固定代码
ee9a915d回放13片段/4197帧，独立几何复算PASS。12个运动重放完成有序参考位姿，
持续未准备片段按设计TIMEOUT；均为工程资格，不是抓取/任务成功或WAM收益。
readiness未知时明确unknown，不从物体位置匹配推断抓稳。下一步先审阅goal跨度和切换，
再决定真实任务参考上的后续检查；本轮未训练、未采集新物理轨迹。
交互回放：`outputs/wam-rl-policy/goal-scheduler-replay-20261018-r3/review.html`。

## 2026-10-18 UniDex 单环境 serial replay 通过；partial-obs oracle-τ 主要改善平移

[oracle-τ Probe](experiments/probes/P-20261018-unidex-live-oracle-tau.md) 先复刻了
consequence-evaluator 的窄合同：1 env、fresh process、seed 421、100 tick，四个进程的
`obs/q/dq/hand/object_pose/forces/dof_force/action/success` 逐 tick bitwise identical。
8-env 不是 twin（同 seed 只有 env0 lift 0.091 m），所以后续只用单环境 fresh-process
rollout，不用中途 restore。

32 个 200-tick 单环境 teacher rollout（bottle，scale 0.08）中 10 个最终 success，得到
6112 行；τ 取同一 rollout 的 teacher 未来 8 步手部 11 点。由于官方 teacher 是 300-D
前馈 MLP，full-state 下 `state` vs `state+τ` 没有信息增益，因此改用 partial observation：
`p1`=68D robot core，`p2`=139D robot+fingertip kinematics，均不含 object pose/vel 和
contact force。32-seed 分组 CV 下，`p2_tau` val L1 `0.1046`，优于 `p2` `0.1302`、
shuffle `0.1331` 和 full-state `0.1218`；wrist xyz 从 `0.1548` 降到 `0.0731`，但
rotation 几乎不变（`0.2307→0.2226`）。success-only 10-seed 子集 xyz 增益一致
（`0.1411→0.0607`），整体 L1 因 rotation/finger trade-off 只有边缘改善。

随后完成第一版在线闭环（seed 421，150 tick）：学生始终只执行自己的动作，每个 tick 由
fresh process reset + 重放学生动作前缀，再从学生真实状态 fork teacher 8 步得到 τ。
所有 query 的 replay error 都是 0.0，说明 fork 有效。纯 BC 时 `p2` 全程没有抬升
（max lift 0）；`p2_tau` 抓住并抬到 initial+0.057 m，但在 tick 80-90 附近掉落，
official success 仍为 0。

随后用同一次闭环的 oracle query 直接构造 DAgger packet（每个 query 的
`teacher_actions[0]` 就是学生状态下的 teacher action），τ 臂微调 2000 步、packet 10×
过采样：DAgger L1 `0.266→0.0073`，重跑同一 seed 得到 **official success 1、max lift
0.219 m、success 62 帧**，手与 teacher 偏差全程 0.014-0.016 m。matched 对照中，
state-only 用自己的 packet 只有 max lift 0.019 m、success 0；用 **同一份 τ-arm packet
去掉 τ** 训练时 max lift 0.082 m、success 0。架构、BC 数据、DAgger 规模、微调 schedule
和 seed 都相同，唯一差别是 τ 输入。

当前判断：在第一阶段固定 bottle/seed 421 上，闭环 GT-τ + DAgger 已经能稳定抓取、抬升
（0.219 m）并保持（62 帧）；state-only 的两个 matched 对照都失败，说明在这个任务上
live GT-τ 是起作用的输入，而不只是 DAgger。下一步进入第二阶段：用未参与 BC/DAgger 的
successful seeds/orientations 做同样两臂的有限泛化，再考虑 WAM-τ 或更多物体。

## 2026-10-17 WAM runtime 修正 Probe：旧退化被实现问题混淆，修复后仍仅 UNCLEAR

[runtime correction Probe](experiments/probes/P-20261017-wam-runtime-correction.md) 修正了
adapter 在 `act_inference` 重复加 residual 以及批次级 history 清空。64-env/32-iteration
smoke 完成 640 optimizer steps、冻结/有限性审计 PASS；旧有界 adapter checkpoint 在修正
runtime 上为 `58/128 success, 40 held, 63 lifted`，旧记录为 `65/128, 37, 70`，差异不能
单独归因于某一个修复。

修正 `gain=0.05` adapter 完成 64-env/1000 iterations、512,000 env steps、20,000 optimizer
steps，末值 EV=0.918、history invalid=0。固定 TRAIN 为 `67/128 success, 43 held, 70 lifted`；
同 runtime published control 为 `65/128, 39, 71`。单 seed/单 TRAIN 面板只记 `UNCLEAR`，不
宣称 WAM 收益；下一步不追加同配方长训，应采用相同可训练模块的 matched native、独立 DEV
和多 seed Validation。r1--r5 仅为远端环境准备失败，未进入结果统计。

## 2026-10-16：共同抓取起点上的协调奖励未改善保持，停止加码该接法

[协调奖励Probe](experiments/probes/P-20261016-wam-coordination-bonus.md)已完成：共同自训
BC起点、不加载官方actor、不新增BC训练，三臂各25轮/636800交互步。固定128条DEV
真实初态一致，BC/none/geometry/WAM保持28/26/23/25，到达42/51/46/51。WAM丢失
18条BC原本成功轨迹、新增15条；未过预设门槛，当前抬起后协调奖励接法UNPROMISING。
不把单seed负向Probe升级成WAM无效结论。动作、奖励真实next-state重算、冻结和学习
资格通过；最后critic EV中位数.912/.925/.920，没有旧死critic问题。

语义审查定位到：稳定阶段WAM/几何参考误差4.06/2.66mm；同一DEV实际转移中，两者
bonus平均仅差.001295（上限.2），主要提供抬起存活信号。抬起前总体预测略有改善，
但58个即将抬起的转折窗WAM/geometry为7.98/5.33mm，不能直接把该奖励前移。

下一步先在已有关键转折状态比较因果短goal与真实短终点oracle，区分目标条件错误和
关键交互先验不足，再决定改接口还是预训练覆盖。oracle严格不进入在线策略。
此后续尚未启动；停止扩大本奖励配方，不把BC改善设为前置任务。实验卡记录完整边界，
代码4b82bd84、28项相关测试通过，本轮产物892.6MiB，GPU已全部释放。

## 2026-10-16：无BC动作与运动参考已完成原生资格，抓取接触仍待突破

[独立PPO Probe](experiments/probes/P-20261016-independent-motion-ppo.md)：新增真正
不加载BC/官方actor、不加BC均值或正则的24D动作入口；WAM/几何参考按因果短目标生成、
每16步固定一次。三臂各8TRAIN物体、共64env，fresh25轮/318400交互步，代码01f9c80d。
24项单测、九类实际首帧一致、动作执行、参考保持、输入隔离与冻结审查通过。

三臂通过独立学习资格，范围PROMISING；后5轮native return为none -4.65、geometry -14.63、
WAM +8.79，训练chunk critic EV正常，WAM参考置零会改变动作。三臂到达/保持全程仍为0，
WAM实际任务增益UNCLEAR。最终指尖到物体中心平均距离113.55/126.41/94.58mm，主要是靠近。
WAM第20轮曾有单物体高度上升27cm，但终末距目标2.18m，不计有效抓取；不以此宣称收益。

下一步保留WAM手物协同先验，检验相对协调目标作为有界辅助学习信号是否改善接触与受控
抬升。先做实现/异常值检查，再单立短Probe；不继续扩大BC、不直接奖励预测物体运动、
不把本轮失败简单归因数据量。smoke+训练约291MiB，训练最长342.5s，三卡均已释放。

## 2026-10-16：学习短期终点未过门槛，下一步转向独立动作执行接口

[短期终点Probe](experiments/probes/P-20261016-wam-learned-short-goal.md)完成并审查。
原观测→刚体终点的raw-query、wam-mean两臂固定1000步，8个留出TRAIN物体endpoint48
误差73.32/67.61mm，零位移66.73mm，两臂均未通过预设门槛，当前方案UNPROMISING。
不能据此认定缺数据或WAM无用；当前监督目标仅是旧混合行为的未来，不是任务最优/可达目标。

局部正向信号：wam-mean预测终点下，冻结WAM hand32/scene32误差42.54/44.58mm，
比零目标改善10.7%/6.2%；同目标几何参考60.48/47.85mm。仍无原生任务收益证据。
10项相关测试、保存模型重放、对象隔离与误差重算通过。GPU51.69s、产物13MiB，已结束。

下一步不扩大终点回归，也不沿用“预测未来准确即可上线”的假设。先完成真正独立于BC的
PPO动作入口与运动参考接口，再单立有界原生执行Probe：同因果短目标下比较WAM参考、
几何参考和task-only，按真实抓取/保持收益判断。当前full-action实现仍带BC均值偏置，
需要显式拆除该依赖，不能把已有结果称作无BC的PPO。WAM定位为目标条件运动参考，
不是动作条件动力学；小机器人数据通用策略仍是方向，尚未证明。

## 2026-10-16：任务读出完成，下一步优先解决WAM短期目标语义

[任务读出Probe](experiments/probes/P-20261016-wam-task-readout.md)已完成，未新增BC或原生
采集。32TRAIN物体24/8整对象拆分、2048状态、五臂固定1000步：raw-query+aux留出回报
MSE771.38，WAM mean+aux594.18，query+aux614.03，query+forecast+aux665.94。
新读出未超过mean，没有直接扩大该接法PPO。WAM/raw差值不能单独归因预训练，参数量不同。

关键定位：预训练goal是t+24@30Hz真实物体终点，当前接口却给整任务goal、当前姿态；
后者不保证0.8s可达。在同448留出状态，仅改为真实短期终点，32原生步物体预测
145.56→14.29mm，手129.87→20.47mm。相同oracle终点的简单几何插值分别20.29/40.86mm，
提示WAM含有可利用的交互协同先验；oracle含未来信息，严格不能进入线上策略/价值拟合。
当前可部署的按剩余时间插值方案未胜零位移条件，尚不能宣称桥接已解决。

主线选择：先用现有TRAIN真实短期位姿学习当前观测→可实现短期目标，再以WAM提供
目标条件手物协同先验、PPO学习机器人动作实现。BC不是前置关卡；不直接把原forecast
当动作条件动力学、reward或实际未来。先过短目标整对象留出资格，再做最小原生执行
判别；没有在线保持提升或通用小数据策略结论，不改MISSION。7项测试通过、总模型
运行约134s，产物1.4GiB，GPU已释放。详细证据、条件边界与预算见实验卡。

## 2026-10-16 20:12：200轮未稳定复现v3收益，转入WAM价值学习资格

上一批全部结束、GPU已释放。本地WAM PPO200实际初态PASS，held80→78/256，虽lift124→144、
到达117→130，保持未改善。远端v3最终三个reset为BC→PPO 69→72（初态FAIL）、
58→72（PASS）、75→64（PASS），有效面板方向相反；第50轮小幅增益未稳定复现。
none/v3后期分别33/35个episode全部13段EV近零，训练程序/冻结审计通过不代表配方合格。
不能据此否定WAM，也不能宣称已收敛有效。

本地缩放25轮对照完成，EV .783→.875，TRAIN保持41.8→38.5/128，没有行为提升证据。
20:22远端四卡已启动none/v3 × value_scale1/100，共同remaining clock、2632D、大Tanh头、
同BC/噪声/hold奖励，fresh125轮覆盖旧86/105轮失稳区；四组2轮smoke+审计均PASS，已进入fresh训练。
代码87d14c7c，本地31项/远端3项测试通过，约5.24GiB/卡、利用率67–80%；训练预计85–95min，评估另20–30min。
连续8轮全部13段已知EV近零自动保存checkpoint并停止该配方，避免继续空耗；不视为WAM负结果。
固定终点及三reset评估，不按DEV选checkpoint。详见规模卡最新Decision Note。

## 2026-10-16 v3第50轮行为读出（最新）

固定64DEV×4=256条完整原生轨迹、reset249：BC持续保持69，PPO零WAM特征76，PPO正确v3特征85。
两组分别与BC及直接互比的九类实际首帧均逐值相同，审计PASS。v3比none多9条（3.52点），
但有none-only32/v3-only41，未达到预设10点，单训练seed/单reset；仍UNCLEAR，不宣称已验证WAM，
也没有v3胜v2的有效比较。none/aligned两臂继续固定200和249/250/251，不挑中期端点。
产物：`shadow-wam-feature-v3-{none,aligned}-u50-core-seed249-20261016-r1/`、
`shadow-wam-feature-v3-aligned-vs-none-u50-audit-20261016-r1/comparison.json`，JSON已同步本地。

critic短在线四组25轮均完成审计PASS；时钟轻度改善Tanh，但没有TRAIN保持提升，
ELU终末EV仍负，未将ELU推广为修复。已有轨迹scale100回归使Tanh+clock末层饱和率.657→0、
留出MSE572→445，终末MSE92→171却变差；只作为资格线索。
GPU5正在跑同机fresh value_scale1/100各25轮，物理bootstrap/GAE单位不变、critic loss归一化，
30项测试通过、scale1原生smoke审计PASS。主训仍GPU1–4，6/7留空；详见规模卡Decision Note。

## 2026-10-16 critic资格诊断与v3直接特征（最新）

远端small/large/free/hold四组200episode已完成，没有方向稳定的10点保持增益。
独立检查发现free最终critic在2048个state上输出逐值相同、深层Tanh近100%饱和；hold亦明显饱和。
各组确实有约1.9万–2.8万次actor更新，不是未训练；full-action确定性偏离BC始终<.14，未真正挑战旧cap。
因此这些负结果不用于关闭大网络、自由动作、保持奖励或WAM路线。当前结论UNCLEAR。
下一项最便宜Decision：固定已有free TRAIN轨迹，GPU做Tanh/ELU × zero/remaining clock四臂
fresh critic MC-return回归，同540D/初值/1000步；按物体留出，只判断价值学习资格。已完成：留出MSE依次830/572/833/435；时钟PROMISING，尚非行为收益。

远端四卡现在实际运行v3-none/v2-aligned/v3-aligned/v3-shuffled各200episode，公共MLP50k BC，
独立冻结WAM encoder。后续v2/shuffled分别连续8/10轮全部13段EV近零，且末两Tanh100%饱和，现保留产物并用owned STOP停止；
不能以它们否定WAM。none/aligned仍有学习信号，继续200。释放GPU1/3做四臂25轮在线价值资格对照。
v3物理预测新五域约35%改善尚未证明机器人收益；Shadow388DEV与v3源4 TRAIN对象无交集，
但参与过预训练DEV，不称独立TEST。本地WAM PPO继续至200；100轮core held80→75/256，
lift124→143、到达117→130、终点101→116；2条实际初态不一致，只作描述性读数。

## 2026-10-16 扩张配方与直接WAM特征（进行中）

远端四卡分别运行small、large、full-action、full-action+hold四臂，每16步5epoch×4mini，
各32TRAIN/128env/200episode；先smoke审计再fresh训练。原full199和chunk16对照已完成，
actor800→10400步仍无稳定大收益（held202→197/210/768），不据此评价WAM有效性。
本地WAM主训继续；50轮同初态审计PASS，held80→83/256、lift124→148、到达117→139，
终点101→114；尚无明显持续保持改善。26个BC-only保持病例退步、29个PPO-only进步，不忽略退步。

用户要求WAM z和新v3后：新增独立冻结特征源，公共MLP BC不改，拟四臂v3-none/aligned/shuffled
及v2-aligned，同2631D(完整raw300+WAM2304+goal3+base24)、同大头/full Gaussian/hold/mini。
v3首选固定final80k，基于既有物理预测面板；不宣称预测改善已经转化为抓取收益。
完整组合GPU5两episode原生smoke已通过（26批、109actor步、maxKL.019892、重放3.05e-5、冻结PASS）。
当前实现还没有自定义失败terminal，不能冒称已补齐终止；下一项需alive/valid/逐env bootstrap
与全局有效样本加权的独立资格检查。所有阶段仍UNCLEAR。详见规模卡最新Decision Notes。

## 2026-10-16 用户授权大规模持续推进（进行中）

[规模Probe](experiments/probes/P-20261016-shadow-scale-continuous.md)：本地GPU1–4同步PPO已启动，
128个TRAIN物体、64env/rank、256个全局环境，目标200次更新/10188800环境步，最多12h。
保留50/100/200轮checkpoint；GPU5自动完成全部388DEV基线和固定64DEV阶段读出。
本地GPU0外部作业保护，6/7始终空闲。四卡实际显存约5.2GiB/卡，实测约95s/update，
初步预计总训练约5.3h；资源持续写入run的resources.jsonl。不能把启动训练当有效结果。

代码加入显式all-reduce梯度、全局advantage归一化/重放误差/KL、共同rollback，
两进程CPU合并batch等价测试通过；两卡原生2轮smoke的actor/critic跨rank逐值相同，
两卡全部597个冻结字段未变、无grad、原生action/force/torque误差0。
训练恢复包括actor/critic/Adam及各rank RNG，限定episode边界重新reset，非PhysX精确续跑。
WAM BC省略未被actor使用的forecast readout，批量H4输入各字段与旧路径测试一致。

并行单拍BC MLP为fresh current300→1024→1024→512→512→24；采用相同strict源窗口，
完整TRAIN1536897/DEV354761条标签，TRAIN-only归一化；50k步/batch2048约163s。
离线固定6208窗DEV动作MAE=.01939，root=.03302、finger=.01485，不能当闭环误差。
全部388DEV×4起点=1552条真实native rollout：official679/1552、final590/1552、
held60+lift10cm396/1552=25.52%。全DEV WAM9k为511/1552=32.93%，比MLP50k高7.41个百分点；九类实际初态全部逐值一致。
两基座训练历史/目标不同，不能将差值单独归因shared结构。
另同一32DEV×4起点面板，旧WAM9k与MLP20k均held37/128，final64/55、official70/61；
九类首帧状态逐值相同。不是MLP50k的配对全DEV比较，不能混合宣称改善。

本地WAM r1第32轮因rank03单env越出BC支持域中断，所有共同checkpoint为31；
已在r2显式采用原控制器的范围外零BC回退后恢复，非有限仍硬拒绝，记录fallback病例。
r2持续运行中，50/100/200核心DEV读出队列已切换r2；r1记录保留。

远端4×4090原生四卡smoke通过，128TRAIN的MLP PPO完成200轮：core held69→71/71/66
（50/100/200轮），没有明显收益。64TRAIN标准/强探索各两卡200轮完成，三reset合计
202→226/768和209→231/768，均约3个百分点；不足预定10点且有reset退步。
BC100k constant/cosine core held68/74，对照50k的69；离线误差下降不能当抓取改善。
目前转入539D归一化current vs 冻结BC512隐藏特征的等容量对照，见规模卡Decision Note。
远端有4/256环境在第一次策略动作之前初态不同；五个最小工程排查未定位，保留实际审计、
暂缓继续追查，小幅变化不作严格配对结论。不能混合本地/远端Torch版本宣称单变量效果。

关键产物：`outputs/wam-rl-policy/shadow-scale-panels-20261016-r1/`、
`shadow-scale-wam-128obj-4gpu-20261016-r1/`、`shadow-current-bc-seed241-20261016-r1/`、
`shadow-scale-mlp50k-alldev-20261016-r1/`、`shadow-scale-eval-queue-20261016-r1/`。
新实验状态仍为UNCLEAR；正式研究claim和learned RL-token效用尚未得到验证。

## 2026-10-16 用户转向Shadow：全量既有轨迹行为复查

只读复查三对reset全部48轨迹/策略，无新训练或仿真。逐物体曾达到held60计数BC/PPO：
瓶子TRAIN4/5、相机TRAIN6/7、手机DEV0/1、书DEV1/2，各12条。
两策略总抬升10cm均29/48、到过5cm目标均26/48；曾保持11→15，末段持续60帧直到
记录结束11→14。PPO仍15/24条DEV未抬到10cm，主要失败集中于DEV抓取/抬升。
相同起点曾保持的转换为both8、BC-only3、PPO-only7、neither30；不能只看净增4忽略退步。
相机233/env6虽曾保持114帧，终点抬升仍11.49cm但goal距离9.26cm，只能说明离开目标区，
不能据此声称掉落。全部仍为UNCLEAR，未新增正式RL效用结论。

新增离线HTML同步回放，覆盖所有48对起点，可切物体/seed/env、拖动旋转和逐帧查看；
显示真实手部11关键点与按记录object pose变换的同scale CAD表面点，不是仿真相机视频。
目录`outputs/wam-rl-policy/shadow-bc-ppo-20261016-r1/behavior-review-verified/`，
含`replay.html`、`objects.png`、`behavior.json`、`cases.csv`和`viewer-audit.json`。
生成code57151f23；48case×5帧的Node DOM/canvas-stub检查通过、坐标有限，已查看统计PNG；
没有真实浏览器视觉测试。纯文件统计/CAD变换在CPU执行，无神经推理，不占GPU。

## 先前Stage-0训练与评估

2026-10-16：Shadow BC→RL Stage-0完成，UNCLEAR。代码在本Task，路线来自用户ref1；
冻结Shadow9k BC/WAM，只训练新327D actor/critic。20×199×32=127360 env steps，
约19.2分钟；80个actor epoch均接受，jointKL<=.01762，latent logprob重放最大5.73e-6。
全部597个冻结state_dict字段逐值未变、无gradient；canonical/yaw/force/torque误差0，
手指目标最大1.2e-7。

三个同初态16env reset面板216/233/234，BC/PPO保持分别5/8、4/3、2/4；合计11/48→15/48，
DEV1/24→3/24，official sticky均26/48。方向不一致；同一BC本身还有9/16与6/16的数值
重复波动，不能据合计+4宣布有效。保持为60帧goal<=5cm且lift>=10cm位置代理，非接触认证。
一个训练seed、仅两TRAIN物体，不作正式结论。

完全同初始输入/冻结权重/fixed RNG，GPU reference重复推理仍有约6e-6–1.12e-5 command差；
zero same-packet精确等于BC，但独立闭环不能要求逐值相同。原完整轨迹门FAIL保留，
未定位具体算子，也未排除额外PhysX波动。独立审查已复算全部raw分split指标与初态。

下一步先诊断有效探索和reward/保持关系，检查饱和边界是否吞掉残差，再决定是否扩大
PPO预算；不要因合计增益直接放开WAM。learned RL-token、aligned/shuffled/none和独立
动作路径仍是后续路线，当前尚未实现/验证其效用。

产物`outputs/wam-rl-policy/shadow-bc-ppo-20261016-r1/`；模型`train/latest.pt`，
评估`evaluation-pairs.json`，汇总`summary/results.json/csv`与`comparison.png`。
当时checkpoint保存actor/critic/optimizer/RNG，CLI仅evaluate加载actor；本轮规模实现已补齐train-resume。
全部有界作业退出，未停止外部GPU0任务。协议与证据见
[P-20261016-shadow-bc-ppo](experiments/probes/P-20261016-shadow-bc-ppo.md)。
