## 2026-10-20 Inspire主训保持暂停，三卡GPU物理短测完成

CPU后备三卡r2已正常保存累计215轮并退出，checkpoint完整校验通过，主训保持暂停。
GPU4/5/6各512环境的新诊断从215轮模型/optimizer显式迁移到GPU PhysX，完成16轮至231。
335次workspace终止/reset、PD/回合错配/同步错误0；GPU231模型和optimizer完整校验通过，
未重现PhysX崩溃，但未证明历史间歇故障已消除。所有测试进程退出、三卡释放，不自动恢复长训。
[实验卡](../src/task/wam-rl-policy-v3/docs/experiments/probes/P-20261020-inspire-measured-official-ppo.md)。

## 2026-10-20 Inspire measured686正式三卡PPO已开始更新

按用户指定GPU4/5/6各512环境，总1536，官方UniDex PPO的8步/5epochs/4minibatches及
1024/1024/512/512容量，global rollout12288与原Shadow六卡×256对齐。沿用完整501 TRAIN
序列/50物体、r3准入、Scheduled686 DEV-best和WAM80000；actor为BC初始化，critic新建。
GPU PhysX首步接触核间歇崩溃，r1无更新已停止；r2明确改用CPU PhysX后备，神经推理/优化仍在
三张指定GPU，不能称严格GPU物理复现。CPU异步reset通过，三rank各完成至少2次同步更新/
40 optimizer steps，参数同步误差0、source错配0，首个61MB checkpoint完整校验通过。
约32s/轮、10k预计89h，48h预算先保存停止，可后续续训；暂不能判断成功率或WAM净收益。
[实验卡](../src/task/wam-rl-policy-v3/docs/experiments/probes/P-20261020-inspire-measured-official-ppo.md)。
WAM完整24步的Gym/MANO近似预览也已完成，拟合残差均值22.83mm，不等于真实执行。
[可视化卡](../src/task/wam-rl-policy-v3/docs/experiments/probes/P-20261020-wam-mano-preview.md)。

## 2026-10-20 WAM-v6为用户重启正常停于23758，NAS末步已验证

三卡r2已own STOP正常保存step023758.pt（1186363152bytes），CPU完整读取、模型有限、568份
optimizer（原564计数23758/事件4计数8638）、三卡RNG/global192/elapsed齐全，已fsync。
新supervisor/torchrun/3workers均退出；旧驱动退出残留仍待节点重启清理，agent未执行系统重启。
NAS reboot-preparation目录额外hard-link固定末步并准备严格三卡恢复脚本；四卡恢复需另显式迁移。
本地完整备份传输较慢、交接时仍进行中，不能称已验证；以NAS已验证文件作为恢复源，不必等本地拷贝。
[状态及恢复入口](../src/task/ptv3-shared-wam-v6/docs/STATE.md)、
[停训验证](../src/task/ptv3-shared-wam-v6/docs/experiments/probes/P-20261020-wam-v6-mixed-pretrain.md)。

## 2026-10-20 goal/task10执行路线迁移到 wam-rl-policy-v3

用户要求的新增goal+BC已实际完成：unidex-task-bc-20261020-r1使用授权GPU5/6/7三卡DDP，
Current310随机训练50k，再接同源Scheduled575训练50k；全1891658行task10与τ缓存逐位对应，
新增goal/τ权重真实优化，直接载入且无零扩展、无PPO更新。64个TRAIN场景中，310原生曾成功32、
最终稳定21；575 DEV-best49000为33/24，预声明50k末态为38/29。575两模型64初态全同，
与310仅61/64全同；共同61场景最终稳定20→23/26。属于执行复现，不是WAM净收益或泛化证据。
149项测试通过，运行全部结束，产物4.3GiB；[实验记录](../src/task/wam-rl-policy-v3/docs/experiments/probes/P-20261019-inspire-task-contract.md)。

用户追加的最新DExplore goal+WAM BC也已完成：r3审计数据重建62605行τ，Current421
startup_half best(step7000)接Scheduled686训练50k，WAM80000冻结。配对6081行DEV父模型
0.04012、末态0.05318，DEV-best仍step0；四条TRAIN运动/300控制的抬升5cm持续30帧代理
由父模型2/4变为末态1/4，窗口内完整任务均0/4，三臂全部初态逐位一致。本轮未见改善，
不作为WAM净收益或全回合失败结论。GPU6/7已释放、无PPO；[结果卡](../src/task/wam-rl-policy-v3/docs/experiments/probes/P-20261020-inspire-scheduled686-bc.md)。

追加从头学习：UniDex Current300随机初始化50k，再训练WAM条件Scheduled565 50k，
同64初态原生曾成功35、最终稳定26（40.625%），无历史BC/PPO初始化；证明565训练可复现，
不代表新task10已学会或WAM独立收益。Inspire修正支撑v2的完整active数据56036/6081行，
uniform/startup_half均训练50k：四条预选TRAIN抓取运动各best有2/4抬高5cm连续1秒，
完整路径任务均0/4；启动加权降低TRAIN首动作误差但未解决完整DEV启动与闭环纠错。
补修packed状态重复解码与大矩阵加载问题，v3全套143项通过。四序列拟合为独立诊断，
未从这些结果恢复长PPO；详见[v3状态](../src/task/wam-rl-policy-v3/docs/STATE.md)。

用户澄清BC后已补纯BC评估：UniDex全量BC best46000＋原WAM12000，不载入PPO、不建optimizer。
两组64初态完整一致，BC565原生曾成功32/64、最终稳定25/64；接TaskController的零扩展BC575
分别35/64、24/64。可作TRAIN执行正对照，尚非泛化或goal增益证据。138项测试通过，任务已结束。

追加UniDex对照：按用户要求先排除数据因素。v3以PPO565零扩展为575（新task列未训练），
净接触力版64首回合原生曾成功47、最终稳定29，但23个稳定场景未进入TRACK，路径进度全0。
新增链路可执行，不能将DExplore零进度全归咎于数据；先校准阶段门控与8mm/原生5cm口径差异。
两组初轮仅53/64初态相同，共同场景均38原生成功/27最终稳定，不作全64严格配对优劣。
另外修正DExplore桌板局部Y薄轴被误作Z的支撑计算；660条全量重审完成，命令60529→62605，
208条前缀增长，schema v2防止混用旧τ/准入。137项Task测试通过，无新增长训；详见[v3状态](../src/task/wam-rl-policy-v3/docs/STATE.md)。

用户要求分离路线：Inspire及新增TaskController/measured421/686的BC/PPO、审计与实验卡迁至v3，
包名wam_rl_policy_v3，新输出outputs/wam-rl-policy-v3。v2保留UniDex/Shadow300/565及其官方PPO，
恢复Shadow-only MLP，保留reset/KL等有效修复。数据和权重不搬动、不覆盖；用户ref文档原位保留。
迁移前冻结1f9ae236的GPU6/7原生修复检查已通过：Current450拍在终止处正常结束，
Scheduled以banana/apple完成60拍，PD误差0、跨回合错配0；仅工程验证，无策略改善结论。
v2的59项、v3的127项测试通过。ref1_2确认的零扩展与双调度源问题已修复，追加真实父checkpoint
梯度/goal一致性及shape漂移审计。v3完整新τ缓存406秒完成，有效TRAIN52660/DEV5693，
点目标最大差7.61e-8m；真实8物体256行零扩展最大输出差2.38e-7，tau/valid首步梯度非零。
ref1新增的无效τ样本覆盖和掉落事件游标问题已修复，v3共133项测试通过。GPU6的Scheduled686 r3
使用所有active TRAIN54162/DEV5879行完成50k更新（597秒）；配对DEV最优仍为精确保留的
Current421 step0(0.038258)，final50k为0.054275。GPU6/7相同初始状态的两条TRAIN运动
300拍末均无路径进度；final50k首动作模仿改善、曾短暂TRACK，之后回ACQUIRE。
尚无稳定闭环或WAM净收益证据，下一步诊断耦合丢失；未启动PPO长训，两卡任务已结束。
实际最新产物与边界见[v3状态](../src/task/wam-rl-policy-v3/docs/STATE.md)。
[新入口](../src/task/wam-rl-policy-v3/README.md)。

## 2026-10-20 Inspire measured接口审计修复

已确认旧collector smoke使用Gaussian mean，撤回此前“已验证on-policy PPO采集”的表述；
旧数字只证明传输/张量兼容。修复采样动作、统一shaping/GAE γ=.96、终止后结束单回合批次、
在线实测支撑反馈、按环境绑定geometry，以及Current421→686新增列清零。
WAM与task10现共享TaskController决策，τ缓存/新Scheduled BC改为v2，旧产物不覆盖。
126项Task测试通过；准备冻结后原生检查与GPU6/7新τ缓存，无长训。
[修复记录](../src/task/wam-rl-policy-v3/docs/experiments/probes/P-20261020-inspire-task-reward-bridge.md)。

## 2026-10-20 WAM-v6已按用户要求改用三卡继续训练

GPU0–2独立CUDA及三rank NCCL通过，GPU3仍reset故障但已排除。冻结d43b662d从17000显式
world4→3迁移；local128→64/global512→192，保留568份optimizer、原LR/warmup、rank0–2 RNG、
模型与24步归一化。8步恢复到17008通过DEV/保存/三rank签名，CPU核对564原状态计数17008、
4事件状态1888和统计逐值不变。新长训threegpu-r2已核验17120，约2.034更新/s、九源均采样，
新进程allocated约10.14GiB；总物理显存含旧残留约38–40GiB，selected利用率83–100%。
到80k含DEV/存盘初估10–12h，保留原累计7日上限；batch减少，不是原512曝光量的等价续训。
56项测试通过；具名failed-run分享入口核对旧退出worker身份、空闲/健康和至少20GiB容量，拒绝未知进程。
未做系统reset，数据排除候选仍未应用。
[当前状态](../src/task/ptv3-shared-wam-v6/docs/STATE.md)、
[三卡协议及实测](../src/task/ptv3-shared-wam-v6/docs/experiments/probes/P-20261020-wam-v6-mixed-pretrain.md)。

## 2026-10-20 WAM-v6硬件故障停训，17000完整checkpoint可恢复

约北京时间00:35中断，最后日志17300；09:36核对GPU3明确requires reset/Recovery Action Reset，
旧supervisor/torchrun已退出，progress.json的RUNNING已过期。恢复原四卡需管理员修复GPU，未做系统reset。
完整17000已保留；DEV handEPE32.784mm、τ指标16.892mm，较16129小幅改善。监控对[N/A]解析
另有确定缺陷，已修复并通过3项测试；不能把修复监控等同恢复GPU。17301–17320四rank共10240
CPU样本已重放，检查输入均有限，尚不能证明非有限GPU观测报错的唯一原因。
[当前状态](../src/task/ptv3-shared-wam-v6/docs/STATE.md)、
[失败及恢复边界](../src/task/ptv3-shared-wam-v6/docs/experiments/probes/P-20261020-wam-v6-mixed-pretrain.md)。

## 2026-10-20 WAM-v2 measured TaskReward bridge smoke passes

The task-local measured `TaskController -> TaskReward` adapter is now wired into
both online BC executors. Bounded 60-control Current421 and Scheduled686 runs
completed with finite causal rewards (totals `[2.721500, 2.336822]` and
`[2.368382, 2.720846]`), while neither arm reached a terminal or task success.
Both native hosts preserved the physical248/PD/episode contracts: maximum
physical248 error `2.98e-7`, canonical-to-PD error `0`, and zero cross-episode
action mismatches. This admits only a small 421/686 actor/critic collector
shape smoke; reward calibration and long PPO training remain unvalidated.
See [task-reward bridge card](../src/task/wam-rl-policy-v3/docs/experiments/probes/P-20261020-inspire-task-reward-bridge.md).

A measured PPO actor/critic shape smoke then loaded both real BC checkpoints at
421/686 dimensions, generated a causal 6-by-2 transition batch, and completed a
finite update with behavior log-prob replay error `0` and retained KL about
`1.1e-9` ([shape-smoke manifest](../outputs/wam-rl-policy-v2/inspire-measured-ppo-shape-smoke-20261020-r2/manifest.json)).
This only qualifies the offline collector boundary; an online collector must
still record actual next-state bootstrap and terminal behavior before PPO.

A 30-control Current421 online collector smoke has since passed with two native
environments and 30 transitions each ([manifest](../outputs/wam-rl-policy-v2/inspire-current421-ppo-collector-smoke-20261020-r2/manifest.json)). It used the actual final measured state
for bootstrap, observed behavior log-prob error `0`, retained KL `0.01803` under
the `0.02` bound, finite value loss `0.15579`, and repeated native checks with
physical248 error `2.98e-7`, PD error `0`, and zero action mismatches. Neither
environment terminated or completed the task. The matching Scheduled686 online
collector smoke also passed with 30 transitions per environment, 56/60 valid
tau observations, finite critic loss, zero behavior log-prob replay error, and
actual final measured bootstrap; its actor candidate was safely rolled back by
the KL gate. Both online collector contracts are now qualified, but reward
calibration, task completion, and PPO improvement remain open, so do not launch
long PPO yet.

## 2026-10-20 WAM-v2 measured BC paired audit and Current421 closed-loop contract pass

Following the teacher-rollout correction, Current421 BC `inspire-current421-bc-20261020-r1` completed 50k updates on GPU6; best DEV command MAE was 0.038258 at step3000 and final was 0.055811, so only the best checkpoint is retained downstream. A fresh spatial WAM tau cache from the same-execution goals completed with two workers on GPU6/7: 53,042 TRAIN and 5,799 DEV rows passed workspace validity out of 60,529 admitted rows. Scheduled686 BC `inspire-scheduled686-bc-20261020-r1` completed 50k updates; best DEV command MAE was 0.036683 at step1000 and final was 0.052845. On the 5,693 active-and-valid common DEV rows, Current421 command MAE was 0.036321 and Scheduled686 was 0.036683, so tau has no stable offline advantage in this probe. Current421 completed 300 deterministic native controls over two fixed TRAIN motions: both environments stayed ACQUIRE/progress0. Scheduled686 completed the same 300 controls with 596 valid tau observations; airplane stayed ACQUIRE/progress0 and alarmclock reached TRACK but stayed progress0. Native physical248/PD/episode checks passed in both arms. A paired action-scale audit found teacher/current/scheduled absolute means 0.19095/0.18532/0.18784, so the policies are not output-collapsed; the zero-progress result is a closed-loop generalization/conditioning problem. The adapters are operational, but no long PPO run is admitted until a measured TaskReward/critic bridge is specified and a bounded smoke demonstrates it preserves this contract.

## 2026-10-20 用户选择继续WAM-v6，四卡切换完成并定位异常目标

v5 after-v6-probe-r3已正常保存/停止15723。v6从自己已学事件的16136完整checkpoint严格恢复，
冻结d37e2320、568份optimizer state、原LR/RNG/数据/24步归一化/global512不变，现已核验16520。
四卡99–100%，约1.114更新/s，剩余到80k含DEV/存盘初估17–19h。
8.925m目标定位为原Shadow rollout第36帧起飞离，原始/compact/cache位移逐值一致；当前继承
准入漏了未来失控段。39526序列全量67.28s扫描：Shadow>2m未来中心位移29 TRAIN/5 DEV；
按首次中心速度>5m/s截断的候选为60 TRAIN/10 DEV、25序列，整体抽样曝光0.00275%。
另HOT3D2个TRAIN窗口为跟踪候选。具体排除清单已备好但未准入/未改运行数据，不据此废弃ckpt。
[当前v6状态](../src/task/ptv3-shared-wam-v6/docs/STATE.md)、
[训练及排查记录](../src/task/ptv3-shared-wam-v6/docs/experiments/probes/P-20261020-wam-v6-mixed-pretrain.md)。

## 2026-10-20 WAM-v6四卡续训检查通过，按用户选择恢复v5

独立事件goal Task冻结d37e2320，实际完成8+1000+8更新至16136，DEV/完整checkpoint/恢复均通过，
四rank参数签名一致；564份原optimizer state到16136、新4份到1016，4份RNG及24步归一化保持。
稳态约1.115更新/s、allocated峰值18.81GiB；最新DEV handEPE33.167mm，关事件为33.285mm，
仅同模型诊断，尚无匹配训练对照或事件净收益结论。末步局部loss尖峰对应批次发现8.925m物体目标
位移异常，保留证据待上游追查，未静默过滤数据。
v6已停止并保留checkpoint；v5原ac951370从自己的15120恢复为after-v6-probe-r3，
已核验15260、四卡100%、约1.116更新/s。v5不消费v6更新。
[v6状态](../src/task/ptv3-shared-wam-v6/docs/STATE.md)、
[实验卡](../src/task/ptv3-shared-wam-v6/docs/experiments/probes/P-20261019-wam-v6-event-continuation.md)。

## 2026-10-19 按用户纠正改用teacher rollout目标，六卡补全660条并完成筛选

此前把DExplore训练reference当任务路径是实现错误；旧0/660不能用于判断当前路线可行性。
新的task/goal及计划事件只来自checkpoint teacher同轮实测object/contact rollout，保留原始帧号。
本地GPU0–5完成完整660/50物体、589 TRAIN/71 DEV回采，补齐DOF力18/有效位、连续接触力、
刚体状态/物体速度、canonical动作/PD/tick及CAD512。16环境的四批PhysX故障经8环境补采补齐；
有效596条通过只读链接继承，未覆盖任何旧数据，不能把失败run目录当作无引用数据删除。
最终raw=inspire-teacher-rollout-20261019-r5，audit=inspire-teacher-rollout-audit-20261019-r2。
660条字段/时钟/动作与PD工程检查全PASS；按实测失控/掉落、穿透、跳变及可对齐人工标注，
带50帧失败前缓冲筛选连续前缀，保留563条/60529动作。TRAIN501条/54544动作覆盖50物体，
其中46物体有前缀耦合段；DEV62条/5985动作覆盖42物体。不是机器人成功率。
新TeacherRolloutCorpus与可训练shape encoder+同Shadow容量MLP已实现，current421/scheduled686；
114项Task测试和GPU0八个真实episode输入检查通过，力字段保真误差0，尚未训练shape/新BC或PPO。
旧参考条件的物理/τ缓存拒绝复用，旧319入口拒绝静默丢弃新力/形状/阶段字段。
[实现与数据入口](../src/task/wam-rl-policy-v3/docs/research/2026-10-19-measured-task-interface.md)、
[完整记录](../src/task/wam-rl-policy-v3/docs/experiments/probes/P-20261019-inspire-task-contract.md)。

## 2026-10-19 WAM-v5事件数据与checkpoint迁移准备完成，原训练继续

用户确认下一CONTACT/RELEASE类型+相对30Hz帧数，并要求先准备。
事件候选侧表覆盖39526序列、58444 CONTACT/26929 RELEASE；复用冻结contact缓存，无KNN。
实际混合抽样发现HOI4D全在synthetic odd anchors，已修正为相邻真实已知观测间的计划事件时间查询，
physical C/state仍masked，原窗口/target/source-strata/24步归一化不变。
v2/e8d995c9全量193.41s完成；32768次原采样中条件有效98.71%，HOI4D98.50%，spawn2/batch128读取通过。
真实step10049按名扩展保留564份原optimizer state/step，只新增50304参数；九源GPU τ/C迁移差0。
44项v5测试通过。候选recipe/checkpoint仍为独立未准入产物，当前四卡ac951370未热改、继续训练。
下一步是显式绑定新事件recipe后做短条件续训对照，尚无事件收益结论。
[准备证据与边界](../src/task/ptv3-shared-wam-v5/docs/experiments/probes/P-20261019-wam-v5-event-preparation.md)。

## 2026-10-19 WAM-v5工程合同修复后，从10049恢复四卡长训

用户批准spawn loader、实际batch绑定和独立τ候选保留。原768在10049正常保存停止；
首次四卡恢复和旧实现回退均暴露Torch2.0.1 fused AdamW step留CPU，未完成更新。
ac951370修复标量设备恢复，保留动量/步数；单参数CUDA下一步与不中断训练误差0。
四rank真实恢复10050–10057、DEV/保存及参数签名检查通过，随后
`pretrain-nine-v5-width768-contracts-20261019-r2`从原10049继续，原数据/网络/loss/统计/global512不变。
恢复报告到10140、四卡100%、约0.90s/update，剩余含验证/存盘初估18–20h。
33项测试通过；另已实现可选CONTACT/RELEASE事件输入和零残差迁移测试，事件数据/optimizer
扩展尚未准入，当前生产没有热加事件。路径、失败和恢复证据见
[v5状态](../src/task/ptv3-shared-wam-v5/docs/STATE.md)、
[训练卡](../src/task/ptv3-shared-wam-v5/docs/experiments/probes/P-20261019-wam-v5-mixed-pretrain.md)。

## 2026-10-19 WAM-v5真实checkpoint批组成审计完成，继续原训练

固定6334e1ea/2000步权重、同远端Torch2.0.1/4090只读推理，真实batch使Hilbert depth9→10，
结构patch分组确会变化；九源小面板batch128相对单样本τ位移宏平均0.118mm、最大单点1.826mm，
重复单样本自身平均波动0.064mm，不将全部差异归因于depth，亦不外推全量/后期上界。
已确认NCCL后fork、有效batch未绑定resume、总目标best单一保留三项合同风险；没有观察到死锁。
主训未暂停或热改，继续推进到3000；DEV20→2000手EPE59.69→42.87mm、根点相对19.95→19.27mm，
不是仅C改善，但手型相对误差改善尚小。2000步审计副本独立保留；下一次恢复前应修复并验证合同。
[审计证据与边界](../src/task/ptv3-shared-wam-v5/docs/experiments/probes/P-20261019-wam-v5-batch-audit.md)。

## 2026-10-19 WAM-v5按用户要求切换768，四卡混合训练

远端v4 scale1024已停止，保留完整39000步checkpoint及历史产物；v5初始1024运行
也已own STOP正常结束于869步，完整末步checkpoint保留。按用户要求新开768运行
`pretrain-nine-v5-width768-20261019-r1`，冻结6334e1ea，从头训练，768宽/12头/12层。
只改变共享宽度及头数：继续完整39526序列、TRAIN3218740/DEV545459窗口，
每卡128/全局512、九源同步四卡、bf16、LR1e-4/500步warmup/80k上限；
21点τ保留24步逐horizon归一化，模型内无KNN。768已到120步、九源均采样，
首轮DEV及step1/20/best落盘；四卡100%、约24–28GiB/卡，近期0.889s/update，
80k含验证/存盘初估21–23h，短窗结果不构成精度或最终吞吐结论。
此前22项测试及1024四rank真实混合数据启动检查通过；尚无v5效果或相对v4增益结论。
[当前状态](../src/task/ptv3-shared-wam-v5/docs/STATE.md)、
[训练实验卡](../src/task/ptv3-shared-wam-v5/docs/experiments/probes/P-20261019-wam-v5-mixed-pretrain.md)。

## 2026-10-19 Inspire修复任务输入歧义，旧六卡已停于972

按ref1审计，复现相同状态/终点而局部路径不同、启动tau无效时旧输入逐位相同。
新增直接局部SE3/confirmed进度7维，无WAM分支同样提供；current319、条件584，
完整660序列/50物体和Shadow网络、两阶段各50000步保持，不重新生成tau缓存。
全量字段有限，50物体抽查离线/在线误差0，85项测试通过；两阶段BC已完成，
DEV选择仍是tau权重为零的current扩展点（MAE0.03892196），没有WAM收益或抓取改善结论。
删除静止贴路径的持续正奖励，成功改为有序完成后真实SE3终态保持1秒且无越界。
旧mixed6-r4完整保存972并停止，不能普通resume为新输入/奖励合同。
原命令恢复原16环境采集配置后，cup前158拍q/手/物体位置/PD逐位一致；
2环境重放则有长期漂移，不能把两者混作同一物理协议。
独立64环境原生PPO/reset检查已通过4轮/80次优化，episode/PD/均值/logprob误差0。
新BC确定性完整64首回合/50 TRAIN物体评估已完成：完整任务0/64、workspace失败17/64，
最终无越界高度保持1/64（仅代理）。仍是弱warmstart，保持旧六卡停止，暂不重开万轮。
下一步定位BC闭环及任务条件信息充分性；task7修复物体局部指令缺失不等于全部行为可辨识。
后续排查确认全部660序列physical248重建差≤1.1e-6；发现旧BC仍采样scheduler超时后的
13667条命令，已修正两阶段及DEV准入并保留全部原始序列，87项测试通过，未重开长训。
原示教在旧有序路径与8mm/.12rad终态协议下几乎不能完成，完美reference的50物体重放通过；
这提示任务/标签相容性问题，不能仅靠补输入或增加训练轮数解释。用户当前要求讨论
统一TaskSpec/阶段及补力、形状观测的方案，尚未按新方案实现或采集数据。
Shadow通用PPO入口已接actual KL并绑定优化身份；尚未做匹配续训因果验证。
[输入修订及结果](../src/task/wam-rl-policy-v3/docs/experiments/probes/P-20261019-inspire-task-contract.md)。

## 2026-10-19 Inspire在线吞吐约提升两倍（已停的旧577合同）

用户授权调优后，旧mixed6-r1在300、r2在339完整保存并停止；其后运行
`inspire-mixed6-20261019-r4`，冻结dd694523，从339续训，最终于972保存停止。
GPU0–5仍为1536环境/同一同步策略、50物体/589 TRAIN；原48h总截止不延长。
共享Transformer单独允许TF32、读取同一hand head、batch256；几何及PPO精度不变。
参考路径初始化批量校验，无损数组传输替换浮点JSON列表；公式、数据、动作和频率保持。
实际稳定窗口18.562→9.358秒/轮，约1.98倍、1313 env-steps/s；到10000剩余约25h。
短时GPU样本六卡均值约64%→68%，各卡59–72%、显存约12GiB；不保证持续满载。
剩余主要为WAM约5.7–6s及CPU scheduler约1.5s/轮，物理RPC已降到约0.5s。
50 TRAIN物体400真实H4，采用方案最大τ差0.428mm、batch/single0.239mm；
六卡在线首批≤0.373mm、同步/回放/PD误差0，79项测试通过。

packed的首个r3试跑在首个物理步触发PhysX非法访问、0优化更新；发现并修复首次模拟前
重复indexed setter及索引owner生命周期缺口：逻辑reset/FK仍立即生效，setter合并在
pre-physics提交一次，无隐藏物理步。修复后相同339 checkpoint实际运行≥168控制步通过；
这支持修复必要性，不声称已证明先前崩溃的唯一底层原因。推理/传输/reset提交身份和迁移均记录。
[协议及结果](../src/task/wam-rl-policy-v3/docs/experiments/probes/P-20261019-inspire-shadow-parity.md)。

## 2026-10-19 Inspire已改为六卡混合训练，发现KL校准缺陷

按用户请求停止旧GPU6/7两卡训练，保留iteration161。新inspire-mixed6-20261019-r1
在GPU0–5每卡同置模型/原生物理，共1536环境，同步一个策略，全部50物体/589 TRAIN
motion分摊覆盖。双rank工程检查及60项测试通过；六rank实际首轮162参数同步、回放/PD误差0。
BC/网络/奖励/动作/FP32 WAM保持，旧单卡optimizer继续；新增actual-KL优化合同及显式迁移。
约20秒/轮、9GiB/卡；10000上限估约54h，当前48h预算保护及滚动保存保持，不自动扩预算。

Shadow退化审计排除了checkpoint编号/normalization buffers变化，随机训练指标自身也在下降。
发现官方adaptive KL按exp(log_std)计算，但实际scale_tril为exp(2log_std)；真实方差探针
会产生相反的LR调整方向。新Inspire已校准KL，保留原采样分布和其他官方超参。
这是一项已验证的计算缺陷，尚未通过匹配Shadow续训证明它是退化唯一原因或已恢复成功率。
[六卡运行](../src/task/wam-rl-policy-v3/docs/experiments/probes/P-20261019-inspire-shadow-parity.md)、
[退化审计](../src/task/wam-rl-policy-v2/docs/experiments/probes/P-20261019-ppo-checkpoint-evaluation.md)。

## 2026-10-19 六卡Shadow终点表现回落

7000/8500/9500/10000轮、全96 TRAIN物体×8场景的确定性评估完成；四轮共同初态为
731/768场景，仍覆盖96物体。原生曾成功83.31/88.24/85.09/76.74%，
最终稳定保持60.88/66.48/64.71/56.77%。9500→10000分别−8.34/−7.93pp，
物体cluster配对区间均为负，全768原始结果趋势相同。结论为晚期退化，不是稳定平台。
8500是这四个候选中最好，保留候选、不自动延长训练；退化原因未定位。
原WAM12000/legacy协议、单seed TRAIN，不作泛化或WAM净收益结论。
[评估卡](../src/task/wam-rl-policy-v2/docs/experiments/probes/P-20261019-ppo-checkpoint-evaluation.md)。

## 2026-10-19 WAM-v5表面正确性修复与局部C读出

按用户ref1/ref1_1核对远端Trimesh4.12.2，确认Shadow capsule重复居中会偏移表面，
旧关节点对齐测试无法检出。已修复并停止旧数据冻结，独立新目录重建Shadow并继承人手；
H4准入与reader统一、cache上游绑定、同ID逐点C读出与细粒度指标已实现，21测试通过。
真实GPU B128短测1.104s/update、19.16GiB，主WAM暂停34s后恢复。全量新数据/比例审计
仍在跑，法线候选仅作小审计，未改BCE或C定义，未启动v5长训或接入PPO。
[v5当前状态](../src/task/ptv3-shared-wam-v5/docs/STATE.md)。

## 2026-10-19 Inspire转为全量Shadow BC/PPO路线

按用户要求，共享Shadow的1024/1024/512/512 ELU大网络、两阶段各50000步BC及原版
UniDexGrasp2 PPO引擎/超参，Inspire仅适配实测current312、12独立动作/18原生DOF、30Hz，
完整物体SE3参考路径经scheduler→WAM，不直接输入hand-ref或模仿误差，不用DExplore PPO。
全量660序列/50物体BC已完成（589 TRAIN、71 DEV），WAM条件1165个越界帧明确排除，
所有命令均进入current BC。DEV选择初始current扩展点，tau列初始为零；尚无WAM收益证据。
两环境错开reset及256环境实际官方PPO更新通过，动作episode错配/均值/logprob误差0；
GPU输入构造20轮计时已通过；26个已结束回合中23个workspace失败、0个路径成功，
不能将启动检查称为任务成功。此前在GPU6/7从iteration20续至10000上限，现已按用户要求停止，
首轮恢复到21、均值/logprob误差0；steady约16.1s/轮，预计约45h，未有Inspire成功结论。
另在GPU4/5完成确定性BC的64首回合/50 TRAIN物体检查：最终无越界高度保持0/64、
目标附近保持1秒0/64、完整路径0/64。6个高度保持代理中5个后掉落、1个末态越界，
不称抓取成功；初始q/物体状态与示教对应，尚需定位闭环误差。BC仍只是弱warmstart。
Shadow原六卡训练r4已正常完成10000终点、进程全部退出，不再处于训练中。
[当前实验](../src/task/wam-rl-policy-v3/docs/experiments/probes/P-20261019-inspire-shadow-parity.md)。

## 2026-10-19 后期PPO评估与Inspire初次自主闭环

GPU6/7完成5500/7000/8000/8483各256场景评估。5500/7000/8483完整初态逐位一致；
8000初态不同，仅作诊断。有效组原生曾成功202/208/219，最终稳定146/161/163。
7000→8483保持仅+0.8pp，成功仍+4.3pp；不能确认收敛，继续既定10000终点，不扩预算。
[评估卡](../src/task/wam-rl-policy-v2/docs/experiments/probes/P-20261019-ppo-checkpoint-evaluation.md)。

DExplore Inspire新current986/30Hz/command12接口已完成实际WAM80000+空间scheduler+BC原生闭环：
542拍、539拍真实预测τ、canonical动作/PD审计通过；单TRAIN飞机BC最大抬升6.1cm、保持代理0，
尚未成功抓取保持。官方DExplore PPO尚未接入（0更新），不将其称整套流程成功。
当前reactive teacher未复现旧批量采集，旧canonical重放可曾保持120拍但最终失守；差异仍待定位。
未生成全量缓存，GPU0–5主训练未改。
[Inspire实验卡](../src/task/wam-rl-policy-v3/docs/experiments/probes/P-20261019-inspire-scheduled-executor.md)。

## 2026-10-19 WAM密集点计时完成，主训练已恢复

独立4090/1024宽度/bf16：batch128下当前GPU训练步0.790s；778手表面+2048物体、
5mm编码且保留紧凑tau为0.893s，C查询增至2048后0.954s（较当前+20.8%），
峰值allocated16.64GiB。完整778点共享token/轨迹在batch64为1.86–1.89s，
同batch基线0.452s，慢约4.1倍；batch128在保留主训练显存的额度内训练OOM。
这是四个真实GRAB窗口的GPU驻留计时，不含全量输入/DDP，不构成效果提升证据。
主1024训练检查时已恢复到resume16k运行；本次经用户授权暂停累计4分20秒，
每轮均恢复，未改主模型/数据/config或写benchmark checkpoint。
最终检查同一组PID已推进至17560步，四卡98–99%利用率、约10.5GiB/卡。
[计算可行性Probe](../src/task/ptv3-shared-wam-v4/docs/experiments/probes/P-20261019-wam-dense-point-throughput.md)。

## 2026-10-19 远端WAM状态与contact/voxel数据检查

768宽度已完成80k；1024宽度因旧8GB运行产物guard在写step15000时退出，
latest仍为已发布step14000，step15000只有未核验.part；四卡已空闲，NAS约48TiB可用。
没有重启或清理产物。现有人手训练缓存主要为11点；表面contact需补原始参数派生标签，
不一定要新增数据集。36窗几何抽查中512物体点在10/5/3mm网格平均占306/439/484格，
仅支持把5mm列为候选，未测训练收益或吞吐；模型继续无需KNN。
[数据与网格审计](../src/task/ptv3-shared-wam-v4/docs/audits/2026-10-18-input-label-contract.md)、
[1024运行记录](../src/task/ptv3-shared-wam-v4/docs/experiments/probes/P-20261018-wam-v4-scale-1024.md)。

## 2026-10-19 执行边界修复与原生回归通过

新增corrected reset初始化隔离、position_only目标、最终/曾保持及失守指标、PPO checkpoint
外部条件绑定和显式迁移；共同workspace协议可用于匹配对照。45项测试通过。
GPU6/7单卡及双卡错开reset的官方PPO+WAM回归通过，参数同步误差0；新checkpoint正常
加载/续训，16场景实际评估输出新指标。主六卡r4继续使用原冻结代码，未切换或覆盖checkpoint。
这是工程修复验证，尚无新成功率或WAM净收益结论。[详情](../src/task/wam-rl-policy-v2/docs/research/2026-10-19-execution-boundaries.md)。

## 2026-10-19 Preview4官方专家获取受阻

用户授权仅评估对应物体专家。官方专家包浏览页及两种下载入口均404，作者fork无替代地址；
同代理官方prior数据页200，未发现可验证镜像。尚未拿到专家权重，实际评估0场景，无成功率。
GPU6/7未占用，主训练未改。[追加记录](../src/task/wam-rl-policy-v2/docs/experiments/probes/P-20261019-unigrasp-published-eval.md)。

## 2026-10-19 UniGraspTransformer发布策略兼容评估完成

官方state Transformer及PointNet严格加载，GPU6/7各256场景、200步、无训练更新。
最终jar4/256、camera46/256；使用Preview4和动态物体初态（保留训练手起点）。
官方静态初态包链接404，原环境为Preview3；因此尚未严格复现论文，不用于WAM优劣结论。
本机旧native库退出崩溃已通过完成仿真/销毁sim后显式worker退出规避；未定位底层库根因。
GPU6/7已释放，主训练未改。[实验卡](../src/task/wam-rl-policy-v2/docs/experiments/probes/P-20261019-unigrasp-published-eval.md)。

## 2026-10-19 官方baseline来源审计：发现复现差距

当前16个object-scale全部属于官方TRAIN；官方逐物体结果均值92.54%，本地发布权重仅46.88%。
权重逐字节同上游，原始/适配actor在64个实际观测上的动作完全相同。
绕开近期评估reset改动的旧原生入口仍29/64（45.31%），低分并非仅由该改动造成。
官方模型已经训练；“未追加我们微调”不等于“未经官方微调”。当前本地比较不能升级为胜过官方基准，
直接原始train.py --test、单jar256环境仍124/256（48.44%），官方该jar为99.6%；未用多物体扩容或评估reset改动。
资产提取核查通过（手/mesh/feature/prior）；显式同步和Torch2.2.2对照均124/256，初始及终止状态完全相同。
未找到issue现成修复，Preview3交叉结果与权重-结果对应关系仍待确认。主训练不改。
[审计记录](../src/task/wam-rl-policy-v2/docs/experiments/probes/P-20261019-native-ppo-comparison.md)。

## 2026-10-19 官方原始state_based_model重测完成

初始native观测/手物状态/root完全对齐的64场景：官方发布原始模型成功30/64、
保持20/64；当前PPO5500+WAM80000成功47/64、保持38/64，含1次workspace失败。
本地测得当前系统高于发布模型；但下述来源审计发现严重官方复现差距，
不能表述为优于官方基准，也不据此隔离WAM净收益。GPU6/7释放，主训练未改。
[实验卡](../src/task/wam-rl-policy-v2/docs/experiments/probes/P-20261019-native-ppo-comparison.md)。

## 2026-10-19 历史官方PPO同面板评估完成

固定16物体×4场景、seed246，初始原生观测及完整root状态逐位一致：
历史官方PPO10000成功45/64、保持33/64；当前PPO5500+WAM80000为47/64、38/64。
小幅正向Probe，微调物体覆盖、初始化、预算不匹配，不能隔离WAM净收益。
评估reset采用独立CPU RNG并规范桌面初态；37测试通过。主训练不改，GPU6/7释放。
[比较卡](../src/task/wam-rl-policy-v2/docs/experiments/probes/P-20261019-native-ppo-comparison.md)。

## 2026-10-19 最终WAM80000直接接入Probe完成

固定PPO5500及64 TRAIN场景，旧WAM→最终80000：成功46→51，保持35→42。
初始策略观测/手物状态相同，残余桌面物理差异，属于探索性结果。
用户授权本地几何不变，仅切权重；接口加载/执行正常。新权重已下载，
主训练未切换，GPU6/7已释放。
[接入评估卡](../src/task/wam-rl-policy-v2/docs/experiments/probes/P-20261019-wam-final-swap.md)。

## 2026-10-19 checkpoint确定性评估Probe

GPU6/7完成16 TRAIN物体×4场景的固定面板评估。初始状态严格匹配的
1000/3000/5500轮成功率57.8%/65.6%/79.7%，保持率40.6%/53.1%/57.8%。
仍有改善，不能认定收敛或泛化；主训练继续。5000轮完整物理状态不匹配，单列诊断。
修复评估首次reset初始化差异：预热后重设seed并再次reset，正式采样从第二次开始。
[评估卡](../src/task/wam-rl-policy-v2/docs/experiments/probes/P-20261019-ppo-checkpoint-evaluation.md)。

## 2026-10-18 六卡256环境优化版续训已启动

用户授权停止r3、改每卡256环境并采用checkpoint轮转。r3最后日志581轮，
最新完整checkpoint562轮，19轮未保存；r4从562恢复模型/optimizer，仿真重新初始化。
代码98c7d63f，GPU0–5/1536环境/96 TRAIN物体，两个优化选项all，继续9438轮至10000，
48小时上限。最近3个+500整点+最终checkpoint，原子保存；旧r3清理26个文件、保留3个。
36项测试通过，初始六卡同步误差0，初测约6.5秒/轮，剩余约18小时。
[官方PPO卡](../src/task/wam-rl-policy-v2/docs/experiments/probes/P-20261018-official-scheduled-mixed-ppo.md)。

## 2026-10-18 每卡256环境短测完成

GPU6/7最新优化版双卡12轮完成，每卡160→256环境：4.73→6.50秒/轮，
271→315 envsteps/s（+16.4%），同步误差0，显存约9.3–9.7GiB。
利用率采样64%/68%，未提高；此为吞吐Probe，不代表成功率收敛加快。
主六卡r3未改规模/代码，GPU6/7已释放。详见
[官方PPO卡](../src/task/wam-rl-policy-v2/docs/experiments/probes/P-20261018-official-scheduled-mixed-ppo.md)。

## 2026-10-18 CPU校验与同步等待的第二轮优化完成

保持160环境/PPO配置，GPU6/7短测合并版每轮5.20→4.63秒，吞吐再提高12%。
双卡检查完成（4.73秒/轮、同步误差0），稳定窗口利用率约69%/76%。
有限性检查仍全部执行，35项测试与GPU有效NaN拒绝检查通过。
主六卡r3未切换，新选项默认关闭，GPU6/7已释放。证据见
[官方PPO卡](../src/task/wam-rl-policy-v2/docs/experiments/probes/P-20261018-official-scheduled-mixed-ppo.md)。

## 2026-10-18 GPU6/7批量控制接口加速测试完成

向量化GPU输入构造/τ转换与CPU批量Scheduler connector已实现为opt-in。
单卡160环境短测每轮7.35→5.41秒，吞吐+35.8%；双卡12轮同步/动作路径检查通过，
稳定窗口利用率约65%/70%。31项测试及真实归档字段审计通过。
六卡主r3继续冻结旧版，本次不切换；预留卡已释放。结果及边界见
[官方PPO实验卡](../src/task/wam-rl-policy-v2/docs/experiments/probes/P-20261018-official-scheduled-mixed-ppo.md)。

## 2026-10-18 优化版续训与WAM越界保护

r2在第101轮后因rank4手点超出WAM固定空间而失败；最新完整checkpoint仅第1轮，
未保存的100轮不可恢复。用户要求切换优化版，r3从该checkpoint模型/optimizer续训9999轮，
目标累计10000，960环境/96 TRAIN物体/GPU0–5。每20轮保存，48小时上限。
新增越界环境显式终止并单独reset，记录workspace_failures；不裁剪或伪造预测。
详见[官方PPO卡](../src/task/wam-rl-policy-v2/docs/experiments/probes/P-20261018-official-scheduled-mixed-ppo.md)。

## 2026-10-18 用户确认565维输入，继续官方10000轮预算

六卡各160环境（总960）、96 TRAIN物体，保留最终位置误差+WAM tau条件，
从全量scheduled BC初始化。运行official-scheduled-mixed-20261018-r2，48小时预算，
预计约26小时；GPU6/7不分配本任务。此前1000轮训练及192环境计时已主动停止。
hand-only推理未通过真实GPU等价性检查，正式训练保留完整WAM推理。
详见[实验卡](../src/task/wam-rl-policy-v2/docs/experiments/probes/P-20261018-official-scheduled-mixed-ppo.md)。

## 2026-10-18 六卡混合训练改为复用官方 UniDex PPO

按用户要求复核旧官方PPO实验，v2现复用固定上游run/update/storage及原配置，
仅适配565维观测、自训练BC均值和六卡同步。96 TRAIN物体、192环境，GPU0–5，
GPU6/7不分配。六卡32iteration检查完成：各640 Adam step、参数同步/动作均值误差0，
已跨实际episode重置。1000iteration有界训练启动，预计约1小时，详见
[官方混合训练卡](../src/task/wam-rl-policy-v2/docs/experiments/probes/P-20261018-official-scheduled-mixed-ppo.md)。

## 2026-10-18 v2批量BC→PPO试跑完成

16环境×180拍×4次更新完成，11520训练交互，actor接受22步；严格likelihood回放
误差均0，KL<.01。修复了不同CUDA推理批大小导致检查假阳性，阈值未放宽。
耗时245秒，GPU6已释放。均值策略终态5cm内2/16→4/16，但重复BC本身也在2–4/16间
波动，尚不能声称PPO提升。允许下一步扩展TRAIN物体panel，仍需续训/异步reset接口。
结果见[批量PPO卡](../src/task/wam-rl-policy-v2/docs/experiments/probes/P-20261018-scheduled-batch-ppo.md)。

## 2026-10-18 新观测执行器迁入 wam-rl-policy-v2

按用户要求，新565维 scheduled BC/PPO链路使用独立Task和Python包 `wam_rl_policy_v2`。
已有全量缓存和BC checkpoint保持原路径复用；新输出写入 `outputs/wam-rl-policy-v2/`。
批量PPO已从v2目录启动GPU6/16环境有界试跑，结果尚待完成。见[新Task入口](../src/task/wam-rl-policy-v2/README.md)。

## 2026-10-18 全量scheduled BC完成，native闭环与真实PPO更新接口通过

条件缓存全部完成，TRAIN1536897/DEV354761行经独立逐行合同审计通过。
自训练BC继续50k步，best46000，DEV动作MAE0.0193933→0.0170210；改善不能单独归因于WAM，
尚无matched state-only续训对照。checkpoint为
`outputs/wam-rl-policy/full-scheduled-current-bc-20261018-local-r1/fit-fast/best.pt`。
父BC扩展与新BC各两次96步native loop完成；新BC终态4.45/3.15cm，但仅末尾5/7帧达标，
不称稳定保持。已记录每拍目标与48步后实测姿态；期间重规划，不能直接据此判定可达性。
真实64步PPO工程更新通过概率回放、末态bootstrap和KL回滚，actor接受1步、critic4步；
不是长期PPO训练或效果提升结论。后续推进批量rollout/持续PPO和控制对照。
见[全量BC卡](../src/task/wam-rl-policy-v2/docs/experiments/probes/P-20261018-full-scheduled-current-bc.md)。

## 2026-10-18 全量BC调度条件缓存五卡并行，正式Shadow控制接口落地

复用已有全量UniDex筛选池（TRAIN1567/DEV388对象）和自训练BC step50000，冻结WAM
step12000。条件缓存优化后单卡约322行/s；现已在本地GPU2/3/4/5/6按记录互斥分片续跑，
GPU1/7不分配本任务，GPU0已有任务不动。全量完成后自动启动BC续训，不能把缓存生成等同于策略训练完成。
当前状态与ETA见`outputs/wam-rl-policy/full-scheduled-current-bc-20261018-local-r1/cache-fast/parallel.json`。

正式接口分离TaskIntent、实测episode/tick/手点mask、外部readiness、空间path goal、
世界系60Hz tau以及raw PPO action/canonical action；拒绝缺测、布局不匹配和过期执行。
25项合同测试及真实WAM/BC的12拍归档接口smoke通过，PPO概率回放误差0。
phase暂不进入WAM/actor，空间waypoint不保证0.8s可达；565维native物理闭环和PPO训练仍待完成。
见[接口审计处置](../src/task/wam-rl-policy-v2/docs/research/2026-10-18-control-interface.md)。

## 2026-10-18 GT-τ有界后续失败；WAM-v4在更大DEV上主要改善C

复用421的150行与453晚期60行微调2000步后，482的GT-τ/state-only各200步均未抬过
初始高度或达标，终态29.61/28.26cm；原τ在同前150步曾抬升18.32cm、连续达标13帧。
200个新query回放误差0、初态逐值一致、终态raw obs复核通过。离线动作误差下降仍未
保住闭环行为，停止这个packet+额外更新配方，保留旧正向与checkpoint；不直接换WAM-τ
来解释WAM质量。运行35.36分钟，32项相关测试通过，本地GPU0/1均已释放。
详见[GT-τ实验卡](../src/task/wam-rl-policy/docs/experiments/probes/P-20261018-unidex-live-oracle-tau.md)。

WAM-v4修正run保持原配方。1024个不同于选模panel的DEV窗口、八源固定权重推理，
step0->13000 objective2.08745->1.87396(-10.23%)，八源总分均降；手轨迹36.90->37.09mm，
物体流11.16->10.86mm，分数下降约98.6%来自C项（算术分解，非因果贡献）。这些仍是
同DEV split中的不同窗口，不是独立TEST。20:04远端到step16040，常规1.6步/秒、四卡约
7.7GiB/98--99%，剩余约11小时；best step15000 objective1.76669，step16000为1.83189，
仍有波动。不凭标量loss下降宣称手轨迹或机器人收益。见[v4审计记录](../src/task/ptv3-shared-wam-v4/docs/audits/2026-10-18-input-label-contract.md)。

## 2026-10-18 GT-τ 留出保持结论修正，等容量对照完成

接续当天DSH的`ff7519bd`会话，重算了GT-τ闭环归档。固定seed421确有62个连续实时
达标帧；留出453/471/482只连续达标14/4/13帧，最后记录均已离开5cm目标区，旧19/17/44
是累计success标记，撤回“留出终末稳定保持成功”。GT-τ为3/3曾到达，对照0/3，但仍不是
跨初态稳定抓取结论。补完403维零填充state-only，同175384参数、同BC/DAgger配方，
四初态均未抬升/达标，实际初态与归档逐值相同；32项相关测试和四次原生
rollout通过。只占本地GPU0，已释放。teacher在部分近边界学生状态的8步端点也会离开目标，
下一步优先保持阶段覆盖/恢复资格，暂不把GT-τ换成WAM-τ或扩大物体。
详见[GT-τ实验卡](../src/task/wam-rl-policy/docs/experiments/probes/P-20261018-unidex-live-oracle-tau.md)。

v4旧过滤器全量过渡扫描已完成：TRAIN恢复2471窗(+.0768%，OakInk2454/UniDex15/HOT3D2)，
DEV恢复426窗(+.0781%)，均已扣独立quality排除。规模小，不因此重启当前修正语义run；
新filter单独保留，当前训练代码/recipe/normalization未改。见[v4审计记录](../src/task/ptv3-shared-wam-v4/docs/audits/2026-10-18-input-label-contract.md)。

修正run到step5000，DEV2.06847->1.98219(-4.17%)，C part accuracy60.31->64.37%，
手轨迹35.11->35.38mm、物体流12.42->12.37mm，尚无整体轨迹收益。真实权重三组54窗
的最后共享层梯度检查完成：C与hand+scene的cosine=.008/.029/-.002，未见持续明显对冲；
不凭标量loss大小改C权重，下一步看各源拟合/条件分布。probe9.79秒，无optimizer更新，
已退出；主训练继续。曲线图和梯度JSON见本地`outputs/ptv3-shared-wam-v4/filter-transition-20261018-r1/`。

## 2026-10-18 停止旧 v4 run，以旧 ckpt 做 model-only 初始化启动修复版

按用户指令先停止旧 `pretrain-nine-v4-fastinput-warm80k-20261018-r1`（step13560，
保留 step13000 best/日志/过滤器），然后从 step13000 做 `--init-v4` model-only 初始化，
启动 `pretrain-nine-v4-contractfix-warm13000-20261018-r1`。远端新代码目录
`training-code-contract-20261018-r1`（commit `2183eb46`，由本地 `812c26aa` 起的补丁生成），
四卡/global512/workers4，supervisor2638598。优化器、step、RNG、历史 best 全部重置；
data identity 记录新的 input/label semantics。正式 run 已完成80k（耗时51482s/约14.3h），
最终DEV objective 1.73703，best仍为step30000的1.68407；step30000后DEV在1.68--1.79
平台波动，没有持续改善，仍无表征质量结论。训练日志有14个loss>3的尖峰（scene项主导，
最大step39600=23.58、step43200=14.77、step80000=12.09），clip_grad=1.0生效但
step80000的最终更新受离群scene样本影响，后续应优先用best step30000而非final。
该 run 仍使用旧
window-filter manifests；`moving_object_noncontact` 的重新筛选后续已单独生成（见上文），
未切入本run。因此采样覆盖仍是旧的，但输入特征和 C 标签已切到修复语义。新旧曲线不可直接拼接，见
[v4 实验卡](../src/task/ptv3-shared-wam-v4/docs/experiments/probes/P-20261017-wam-v4-interaction-head.md)。

## 2026-10-18 WAM-v4 输入/标签合同修复（旧 run 不可静默续训）

按第二轮审计修复了 native 与在线路径的 target/static 18维特征不一致、C 体素原点依赖未来
t+1、无接触物体运动被当作 idle、GT-τ action prefix 被上游原地改写、oracle object_pose
取到 t+H、success_frames 使用累计标记等问题。新增 v4 runtime adapter（512/32/128、
dense overlay、summary 与 τ/C 分开）和 30Hz anchor→60Hz world 的 τ 合同。真实 native
窗口与在线 adapter 的 xyz/features/hand/flow/interaction 输入最大差 6.6e-17；20个真实窗口
上 current-origin 使 C origin 平均移动5.52mm、4.69% contact 标签变化。代码 `812c26aa`，
28项reader、31项online、19项模型测试通过。`data_identity` 已加入 input/label semantics，
旧 checkpoint 不能静默续训；可用 `--init-v4` 做 model-only 初始化另起新 run（优化器/step/RNG/
选模重置，曲线不与旧 run 拼接）。远端正在跑的旧代码/数据未改动，需跑完旧 run 或另起新 run。
Hilbert 动态 depth、scheduler goal 分布和 executor 对 v4 τ/C 的使用仍未验证。
见[合同修复记录](../src/task/ptv3-shared-wam-v4/docs/audits/2026-10-18-input-label-contract.md)。

## 2026-10-18 WAM-v4 精确输入加速与原状态续训

代码`b65872f8`保持数据、标签、目标和global512不变，C查询改为无碰撞int64查表，
native直接构造v4并缓存固定几何，DEV microbatch8输入缓存约42MiB；补齐分阶段计时与
TRAIN/DEV filter摘要的后续恢复检查。53项本地/远端测试通过，九源200个真实窗口全部字段逐值一致。
真实四卡同step3000/相同RNG及采样的短跑：原输入workers2=.346、workers4=.655，
优化输入workers2=.803、workers4=1.573步/秒，最佳约4.54倍；DEV 6.05->1.71秒且指标完全一致。

旧修正版run停在logged3260，保留step3000完整状态/历史best/日志；未保存的约260步会重算。
完整入口3000->3004恢复smoke通过，四rank RNG/optimizer恢复准确且较差DEV未覆盖best3000。
新`pretrain-nine-v4-fastinput-warm80k-20261018-r1`从该step3004续训，supervisor2622903，
远端四卡/每rank workers4，已至step3100、loss1.86196、梯度有限；近期1.572步/秒，
暂估剩余约13--14小时（含周期验证/存盘需继续校准）；未重新从v3初始化。
见[实验卡](../src/task/ptv3-shared-wam-v4/docs/experiments/probes/P-20261017-wam-v4-interaction-head.md)。

## 2026-10-18 WAM-v4 审计修复与新版80k

用户28361da审计指出的hand/object DDP局部分母、DEV按microbatch平均、未来mask进入forward、
恢复best重置和overlay内容未绑定均已修复。代码7b6d28a5，29项本地/远端测试通过，真实NCCL
两卡/四卡完整总损失梯度与合并batch一致（最大差5.82e-11/0）。dense文件已绑定SHA-256，
复用现有九源数据；C仍是共享上下文的并行头，不声称显式轨迹一致性或反事实预测。

旧992962ec运行已在logged step1200通过STOP停止，step1/1000与日志保留；不能将旧损失/选模
结果与修正版连成同一条曲线。新版本从v3 step80000重新model-only初始化，完成四卡2步smoke，
再精确恢复至step4；所有rank的RNG/optimizer恢复通过，历史best0未被更差的恢复评估覆盖，
step4改善后才切best4。新版`pretrain-nine-v4-auditnorm-warm80k-20261018-r1`继续这条修正轨迹，
supervisor2598776，远端四卡/global512/每rank workers2，已至step100，loss2.64132、梯度有限；
初段约.350步/秒，暂估剩余63小时，显存每卡约8.0--8.3GiB。七日/单run8GB/20GiB保护不变。
见[实验卡](../src/task/ptv3-shared-wam-v4/docs/experiments/probes/P-20261017-wam-v4-interaction-head.md)。

## 2026-10-18 WAM-v4 真实512点与C监督修复通过（后续损失审计已覆盖）

九源目标几何均已通过真实512个独立点和单位法向审计；从原始CAD补采样1,078个不足对象，
保留原32个flow query及v3数据/checkpoint。877个受影响的旧已筛窗口复查后恢复1个TRAIN窗，
派生配方为3,218,740 TRAIN/545,459 DEV窗，继承v3 normalization，未重新拟合。
C缺测mask、MCP语义、多卡独立分母及rank0验证误同步已修复；本地/远端18项测试通过。
完整98.7M模型四卡/global512两步smoke完成梯度审计、DEV验证和checkpoint保存；
这是工程资格，不是表征或机器人增益结论。

代码992962ec、独立NAS目录`ptv3-shared-wam-v4-dense-20261018-r1`的
`pretrain-nine-v4-dense512-cfix-warm80k-20261018-r1`已启动，远端GPU0/1/2/3、
每卡batch128/loader workers2、v3 step80000仅模型初始化，C模块新初始化；
supervisor2583130，80k新步数/7日/单run8GB预算。已完成step20和step1恢复点，loss2.8428、
梯度有限；约.376步/秒，初估剩余59小时，物理显存每卡约7.6--7.9GiB。监督器每10秒监控进度和GPU，
30分钟无更新则停止自己的进程组。见[实验卡](../src/task/ptv3-shared-wam-v4/docs/experiments/probes/P-20261017-wam-v4-interaction-head.md)。

# 2026-10-16 官方 UniDexGrasp PPO 远端复现与 WAM 接线

按用户要求使用远端 GPU 跑 pinned upstream PPO（commit `6c9e710b`）。fresh state-only
训练 1024 env、10000 iterations、81920000 environment steps，PPO audit 通过，但 TRAIN
和 core-dev 固定评估均为 `0/128` native success。官方发布模型在同 TRAIN 面板为
`65/128`，所以环境/评估和 checkpoint 加载链路可用；当前不能把失败归因于自写 PPO，
后续先核对官方 reward/episode 配方和训练面板条件。

冻结 WAM-v2 step80000 已接入同一官方 PPO，2607D observation smoke（64 env、32 iter）
完成 640 optimizer steps、audit 通过；固定评估为 `0/128`，仅作接线工程证据。详见
[官方 PPO 实验卡](../src/task/wam-rl-policy/docs/experiments/probes/P-20261016-official-unidex-ppo.md)
和本地证据目录 `outputs/wam-rl-policy/official-unidex-remote-evidence-20261016-r1/`。

## 2026-10-17 官方初始化 baseline 与 WAM-v3 对照

复核官方 `run_train_ppo_state.sh` 确认 `--model_dir=example_model/state_based_model.pt`
是默认 baseline 配方；此前 fresh 0/128 不应代表官方 baseline。**下面的 initialized PPO
均为官方 actor warm-start 诊断，不是 Objective A 的从零自训证据。** initialized PPO
1024 env、
10000 iterations 完成 81920000 env steps，固定 TRAIN `103/128 success, 85 held60,
105 lifted`，core DEV `98/128, 72 held60, 108 lifted`。WAM-v3 2607D 包装器的官方模型
正对照仍为 `65/128`，说明接线初始保持行为；WAM-v3 1000 iterations/64 env 为
`19/128 success, 0 held60, 35 lifted`，同 iteration official model 为 `83/128,
58 held60, 93 lifted`。WAM-v3 10000/64 正在运行，匹配优化 iteration 但不匹配官方
environment-step 预算；1024-env 试跑因约52小时 ETA停止，不作为科学负结果。

WAM-v3 64-env run 在7200秒边界到4734/10000，model_4000固定 TRAIN 为
`10/128 success, 6 held60, 17 lifted`。此前记录的同iteration official initialized
`103/128, 84 held60, 108 lifted` 缺少可独立核验的对应评估产物，现从匹配对照中剔除。
直接拼接WAM summary并整网更新仍暂记UNPROMISING；它不否定WAM表征本身，若继续应改成冻结
native路径的zero-init adapter或低学习率对照。旧小规模WAM PPO卡中的“学习资格/PROMISING”
只说明输入和独立策略能学，不等同于官方initialized state baseline上的任务收益。

冻结 native 路径的 zero-init WAM actor adapter 也完成 1000 iterations/64 env，
learning-audit=PASS 且仅 adapter 参数更新。published checkpoint 加零初始化 adapter 的
TRAIN 正对照保持 `65/128 success, 39 held60, 71 lifted`；直接评估训练目录的
`model-initial.pt` 也复现 `65/128, 39, 71`；adapter model_1000 固定评估
仅 `1/128, 0, 5`。因此当前两种 WAM 接法在官方 initialized PPO 上都记为
`UNPROMISING`（单 seed/短预算 Probe），下一步不应继续堆同配方长训；若要继续，应先
单独验证 reward/优势尺度、动作噪声和 adapter 的行为约束，再设计新的低风险干预。

随后完成的 WAM-v3 zero-init actor adapter 256-env 判别 probe（1000 iterations、2,048,000
env steps、audit=PASS）固定 TRAIN 仅 `2/128 success, 0 held60, 6 lifted`；相比 64-env
adapter 的 `1/128, 0, 5` 没有恢复。并行环境数不是当前退化的主要解释，已停止继续扩大
这条 adapter 配方。

新增的有界 zero-init adapter（显式 `gain=0.05`，执行残差为 `0.05*tanh(adapter)`）完成
256-env/1000 iterations、2,048,000 env steps，audit=PASS，固定 TRAIN 为
`65/128 success, 37 held60, 70 lifted`。同预算 native-only 官方对照为
`73/128, 61 held60, 80 lifted`。有界残差消除了无界 adapter 的灾难性退化，但相对 native
PPO 没有改善；当前只保留为低风险 WAM 接入候选，不作收益结论。

在官方并行规模下又完成 gain=0.05 的 1024-env/300-iteration matched Probe（2,457,600
env steps）。TRAIN 的 WAM/native 为 `58/128, 30 held, 63 lifted` 对 `56/128, 33 held,
59 lifted`；core-dev 为 `45/128, 23 held, 51 lifted` 对 `69/128, 45 held, 75 lifted`。
TRAIN 的小幅 success 差异未泛化到 DEV，因此有界 adapter 路线停止追加，当前仍为
`UNPROMISING`，不能称为 WAM 改善。
同一 checkpoint 在 TRAIN 上换用 seed250 重评估时，WAM/native 分别为
`50/128, 38 held, 53 lifted` 与 `69/128, 50 held, 76 lifted`，进一步排除了 seed249
小幅 success 优势的偶然性。

## 2026-10-17 WAM runtime 修正 Probe

修正 adapter 的重复 residual 和批次级 history 清空后，64-env/32-iteration smoke 完成
640 optimizer steps、audit=PASS；旧有界 adapter checkpoint 在修正 runtime 上为
`58/128 success, 40 held, 63 lifted`，旧记录为 `65/128, 37, 70`，差异不能单独归因于
某一个修复。修正 `gain=0.05` adapter 完成 64-env/1000 iterations、512,000 env steps，
末值 EV=0.918、history invalid=0；固定 TRAIN 为 `67/128, 43, 70`。同 runtime published
control 为 `65/128, 39, 71`，本轮只记 `UNCLEAR`，不宣称 WAM 收益。r1--r5 是 Isaac Gym、
gym/ninja 或 JIT cache 环境准备失败，未进入科学结果统计。详见
[runtime correction Probe](../src/task/wam-rl-policy/docs/experiments/probes/P-20261017-wam-runtime-correction.md)。

## 2026-10-15 DexYCB全被试已补齐，扩充版九源数据本地/远端就绪

按用户要求补齐10被试1000段/72,746帧；官方s1划分，清洗后DexYCB3,096 TRAIN/382 DEV窗，200段TEST独立。九源新配方4,805,826 TRAIN/821,791 DEV、source8采样1%，TRAIN-only归一化重拟合；本地和远端mixed-reader/端点/集合/数值统计一致性验收通过，其他源既有隔离无回流。Viser8769已切新数据。现有远端80k继续使用此前冻结严格配方，本次未切训练。当前资源以最新CAMPAIGN为准（本地保持两卡空闲，远端可全占）；本次只用CPU处理数据。详见[扩产记录](../src/task/3d-data-preparation/docs/research/2026-10-15-dexycb-full-expansion.md)。

## 2026-10-15 九源严格清洗已切入远端四卡训练

用户授权直接隔离模棱两可数据并收紧阈值。全39,528条运动审计完成，九源额外排除2,943 TRAIN/204 DEV窗口，保留4,803,076 TRAIN/821,409 DEV；原始数据和旧checkpoint保留。`pretrain-nine-strict-remote4-80k-20261015-r1` 已从完整step8000恢复并更新至8080，固定代码3709c966、supervisor1892360，远端0/1/2/3/global512；normalizer数值、optimizer及四rank RNG完整保留。约1.465步/秒、四卡82–99%利用率/17.7–36.1GiB，剩余暂估14–16h，无OOM/非有限量。仍为80k/48h预算内同一训练轨迹；DEV面板已改变，前后不作匹配比较。全局6卡上限下远端4卡占用，其他项目任务最多2卡，本地v3不恢复。Viser8769已使用严格recipe，原始隔离片段仍可回看。详见[v3状态](../src/task/ptv3-shared-wam-v3/docs/STATE.md)和[复核记录](../src/task/ptv3-shared-wam-v3/docs/motion-followup-20261015.md)。

## 2026-10-15 14:07 远端四卡九源motion全量训练已启动

用户已授权直接开始。`pretrain-nine-motion-remote4-80k-20261015-r1`使用清洗后4,806,030 TRAIN窗、全新初始化，远端GPU0/1/2/3（supervisor1877848、固定代码6e535bc1），每卡128/global512，80k步/48h/8GB预算。已至step120，四卡利用率98–100%、显存约22–32GiB，loss/gradient有限，step1完整恢复点保存成功；约1.4–1.47步/秒，预计16–18h。当前项目远端占4卡，本地启动时8卡均空闲；6卡全局上限下其他新增项目任务最多再使用2卡。首个训练后DEV在step1000，不形成效果结论。状态和快照见[v3 Task](../src/task/ptv3-shared-wam-v3/docs/STATE.md)。

## 2026-10-15 九源预训练前的UniDex motion清洗

用户要求先筛查清洗。33,056条全量审计后，隔离25条中的114 TRAIN/22 DEV窗口，保留原始数据及可信前缀，其余八源窗集合不变；新九源为4,806,030 TRAIN/821,613 DEV，24-horizon统计已精确重算并独立复核。远端原数据和独立`data-motion-r1/recipe`均CPU验收通过，随后已按用户指令启动上述远端四卡；本地v3保持停止。Viser8769已切换新recipe，整段隔离记录仍可回看。细节和边界疑点见[Task当前状态](../src/task/ptv3-shared-wam-v3/docs/STATE.md)。

# Ref2Dex 当前研究状态

## 2026-10-16 完整参考下的控制底座：同序列隔离插值也失败

三条件Probe已完成（Inspire、cup/mug族、相同完整教师输入/单步标签/MLP，24k更新）。
训练状态/同序列隔离片段/原DEV新序列的序列等权腕部PDXYZ mean为0.564/32.652/42.436mm；
逐帧mean1.334/30.371/41.623mm。训练与留出[t-3,t+24]支持不交叠，新进程独立复核PASS。
45/47插值序列mean>10mm，同47序列seen-later仅0.662mm；当前不支持只归因新序列缺示范。
本轮局部控制泛化UNPROMISING，下一步优先检查控制表示/物理目标与优化归纳偏置，
先于追加整套固定数据训练或闭环动力学归因；无新的正式控制/抓取结论。
代码ce823f98、单GPU5/77秒，已释放。详见
[实验卡](../src/task/ptv3-shared-wam-v2/docs/experiments/probes/P-20261016-teacher-sequence-generalization.md)。

## 2026-10-15 v3九源统一预训练启动

用户最新要求：本地WAM v3已停止并释放GPU1/3，最后日志step440；保留完整启动step4
checkpoint，未到1000步DEV，不形成有效性结论。远端现已按最新明确授权启动上述清洗后四卡run。viser完整序列浏览已启动127.0.0.1:8769（转发8769），支持直接排查UniDexGrasp++
原仿真高速飞离；详见v3 Task STATE。

用户明确第一轮合并九源。新Task `ptv3-shared-wam-v3` 独立拥有训练/采样/归一化入口，
清洗产物和v2运行源码不修改。TRAIN共4,806,144窗；共同TRAIN统计、九源batch审计和
双卡每卡128/global256启动4步通过，已保存完整恢复点。`pretrain-nine-80k-20261015-r1`
从step4恢复，GPU1/3，supervisor1584518，固定代码67f7af78，目标80k步、48h/8GB预算；
与原GPU4/5/6/7运行合计6卡。已恢复并持续更新至step60，稳态约0.92步/秒、
预估80k约24小时；GPU利用率接近100%，loss/grad有限。新增远端九源落盘请求并行推进。
尚未形成模型收益结论。
详情见 [v3 Task STATE](../src/task/ptv3-shared-wam-v3/docs/STATE.md)。

2026-10-15用户要求三卡全量补采，已用GPU0/1/2重新rollout660条，包含原生当前721D、
实测contact、完整body state与同次q_PD；supervisor1548597，run dexplore-current-rollout-20261015-r1。
首48条独立审查通过，全量结束后由CPU auditor1553774逐序列核对再准入。用户临时允许
合计7卡。原四卡微调因并发src修改已在3340触发guard退出；现已用固定代码fd83aebd
从step3000恢复为r5（supervisor1544990/GPU4/5/6/7/global512），实际已继续更新。
当前原生多源模型修改保留，不影响隔离续训；详情见v2 Task STATE与两张实验卡。

用户范围纠正：此前R当前输入Probe只补了可重建的参考误差和几何近邻特征，未补
DExplore实测contact/body完整观测，不是所请求的原生DExplore当前输入对照。
该结果不能代表DExplore信息无收益；原问题仍未回答，需要同次执行补采后再比较。
原四卡WAM微调未因此改变。详情见v2 Task STATE与当前信息实验卡。

## 2026-10-15 native七源版本化清洗已完成

用户已授权实际数据处理。`native3d-clean-mixed-20261015-r1`保留37,591段/2,638几何，
verified TRAIN427,707、DEV105,952；HO-Cap另8,933个时钟待核实DEV不在默认reader中。
法线、QA/clock-gap、完整target horizon与单手有效性过滤已执行，HOI4D的一致15Hz子集已显式插值且保留原观测标记；
H2O TEST220文件归属已对账，不是orphan。35项合同测试、全几何检查、96窗独立端点复算、七源正反混合batch通过。
所有新增native数据仍`training_allowed=false`，本工作没有启动新的多源训练；既有WAM-v2机器人微调独立运行。
TRAIN全量统计`native3d-train-moments-20261015-r2`已完成，不读取DEV/TEST payload或DEV-only CAD。
另建`native3d-clean-encoder-20261015-r1`：按当前观测输入域排除UniDex41 TRAIN/6 DEV，保留427,666 TRAIN/105,946 verified DEV；
对应24×XYZ hand/scene充分统计同步扣除TRAIN排除项，原完整数据/清洗版保留。全parent窗口点数≤3,072，未触发默认4,096 cap。
详见[Task清洗记录](../src/task/3d-data-preparation/docs/research/2026-10-15-native-cleaning-results.md)。

## 2026-10-15 10:21 当前信息增益的四臂离线 Probe

按用户授权补当前参考目标/误差及手物表面几何，不新增未来参考。四臂R-only
Probe已在GPU0启动，四组统一GT τ、step500 R快照、数据抽样和3000更新；主四卡
WAM微调保持运行。该诊断先区分信息缺失是否值得继续投入，不形成抓取结论。
详见[实验卡](../src/task/ptv3-shared-wam-v2/docs/experiments/probes/P-20261015-R-current-information.md)。

首轮3000步四臂完成：合用两输入的dev腕部XYZ p95改善14.5%，平均误差不变，
判为UNCLEAR。四臂尚在下降，因此在GPU0/1并行进行一次12000步同配方检查；
保留首轮全部证据，维持主训练四卡、项目最多六卡与离线Probe结论边界。

12000步对照已完成：baseline49.06/168.37、reference52.78/187.03、geometry49.99/
162.11、both54.42/176.04mm（dev首控腕部XYZ mean/p95）。新增两类输入没有整体
收益，停止追加、保持UNCLEAR；原始PD独立复算和输入生效检查通过。未测试的接触/
body动力学不能据此否定。GPU0/1已释放，主四卡训练继续。

## 2026-10-15 10:01 远端WAM-v2完成后，四卡q_PD微调启动

远端80k实际正常完成，step80000完整权重已取回。按用户明确要求，微调无R warmup、
用q_PD监督，GPU4/5/6/7每卡128/global512、12000更新；owned supervisor1436540，
run `ptv3-shared-v2-robot-finetune-20261015-r2`，实际代码dccd77ee。四卡工程门发现
308个TRAIN窗口的实测手点超出parent固定体素输入域；保持原corpus和坐标不变，新增
显式observed-fixed-grid.v1准入，TRAIN110469/DEV11908窗，完整排除清单/旧新索引保存
在run中。17项测试、独立审查及真实四卡四步通过，正式训练已开始，暂无学习或抓取结论。
这是既有RL660微调，不是新增native3D多源混合训练；后者的准入状态仍独立维护。
详见[v2 Task状态](../src/task/ptv3-shared-wam-v2/docs/STATE.md)与
[实验卡](../src/task/ptv3-shared-wam-v2/docs/experiments/probes/P-20261015-ptv3-shared-v2-robot-finetune.md)。

step2220平台检查：固定dev1000→2000总损失5.202→5.262，但动作/GT-anchor仍各降
约11%；forecast残差回升抵消动作收益。暂不能按整体已到平台停训，原配方继续。
best按总损失仍选1000，不能单看best不更新判断动作平台；详见主run的plateau快照。

## 2026-10-15 多源 native→Shared-WAM v2 准入审查仍阻断

约 06:00 CST 完成全 source-pair mixed-collate 只读矩阵：对 source4–9 各取一个真实窗口，枚举全部 15 个不同源无序 pair；当前 `ptv3_shared_wam.egodex_data.forecasting_collate` 均可完成 padding，source 顺序、`point_valid`/`object_valid`/`goal_object_delta` 重算和浮点有限性检查均通过。每个 pair 的首窗 local `sample_id` 都是 `[0,3]`，且 `source_profile`、`anchor_world_to_sample`、`training_allowed` 均未进入 batch，进一步确认身份必须使用 `(source_id, sample_id)` 复合键，provenance 必须由 dedicated adapter 明确携带。source4 样本只来自 320-sequence standalone index，source7/8 仍走 direct lower-level path；未覆盖反向顺序、同源 pair、较大 batch、截断或 invalid-mask，因此只属于 wiring smoke，不改变 admission。JSON/Markdown/PNG 已由接口与可视化子代理独立复核一致，见 `tmp/goal-supervision-20261015/mixed-collate-pair-matrix.{json,md,png}`。

约 06:05 CST 完成只读 mixed-collate stress：六源正序、反序（batch=6）及 HOT3D 同源重复（batch=3）均保留 source 顺序，派生字段和浮点有限性通过；真实 HOT3D `[0,99]` 部分无效窗口与 source4/source7 混合后，fixed/padded masks 与 masked targets 全零均保持。额外用 `scene_cap=512` 压测同一 6-object/3072-point 窗口，观察到前 512 点只来自 object 0，而 `object_valid` 仍声明 0–5；这是单样本的 order-biased truncation 风险，不是默认 cap 的全量缺陷。adapter 应保持当前 4096 cap（该样本 3072 点不截断）或在获批设计中加入按对象均衡采样。详见 `tmp/goal-supervision-20261015/mixed-collate-stress.{json,md}`；子代理复核通过，未改变任何准入或训练状态。

约 04:47 CST 新增只读 field-contract audit：[`native-v2-field-contract.md`](/home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/tmp/goal-supervision-20261015/native-v2-field-contract.md) 对 source4/5/6/9 各取一个真实 engineering window，在不改动 `ptv3_shared_wam.egodex_data.forecasting_collate` 的前提下逐字段比较 shape、dtype、有限性、mask 和派生值。四源均保留当前 v2 所需模型字段，`point_valid`/`object_valid`/`goal_object_delta` 的 collate 派生值分别与原始长度、对象 mask、goal−start 一致；source4/source5 一对混合窗口也保留 source/sample 顺序，但两源的 local `sample_id=[0,3]` 相同，且 `source_profile`、`anchor_world_to_sample`、`training_allowed` 在该 collate 中全部丢失。该结果把“需要 dedicated adapter、且必须在 collate 前 fail-closed 携带 provenance”从单一 metadata probe 扩展为逐字段合同证据；不改变任何 admission gate、manifest 或训练状态。PNG 为 `native-v2-field-contract.png`，脚本和结果均只在 `tmp/`。

04:23 CST source4 全量 live 文件审计完成：`tmp/goal-supervision-20261015/source4-live-file-audit.json` 对 33,056/33,056 manifest-listed compact NPZ（3,305,600 frames；standalone 320、expansion 32,736；TRAIN/DEV 26,448/6,608）逐条检查路径、身份、完整 payload 数组、连续30Hz、reset 和严格 bool admission，错误为 0；16/16 跨批次语义样本复放通过。仅忽略 watcher 易变 `checked_at` 比较内容指纹，快照稳定。因而 `admission-dry-run` 仅将 source4 `file_consistency` 提升为 PASS；source4 reader 仍 UNVERIFIED，clock/provenance/geometry/QA/training flag 及全源 metadata boundary 仍阻断，整体不准入。审查脚本和图文由子代理复核；未修改 manifest 或启动训练。

04:27 CST source4 唯一几何 blocker 的 provenance probe：raw `decomposed_012.obj` 与 live geometry 对齐；Piano 511 个采样点中 2 个法线为零，均不在 32 个 query 点，但落在同一局部面片，incident face normals 成相反方向并抵消，不能安全选择符号。受影响 16 条 TRAIN 行暂保留隔离建议；没有写入修复法线或改变准入状态。详见 `tmp/goal-supervision-20261015/source4-piano-normal-probe.{json,md,png}`。

04:42 CST live-manifest drift probe 双读 5 秒：去除易变 `checked_at` 后内容 hash 未变；36,351 sequences/2,353 objects、source4 33,056 唯一路径及 `training_allowed=false` 均与既有审计一致，当前无需重复全量 NPZ 审计。详见 `tmp/goal-supervision-20261015/live-manifest-drift.{json,md}`。

04:57 CST 在监督快照后再次进行 5 秒 semantic drift probe：hash、source4 行/唯一路径、总行数/对象数和 `training_allowed=false` 检查仍全部通过；因此不重跑全量文件审计，继续沿用现有准入证据边界。

约 05:02 CST 对 source7 (HO-Cap, 仅 dev) 与 source8 (DexYCB, 仅 train) 做了 direct-only lower-level contract probe：同一 `validate_sequence`/`pose_mesh_sample` 加 `egodex_data.forecasting_collate` 可生成有限 batch，并在各自一个全有效窗口上复算 `point_valid`、`object_valid`、`goal_object_delta`；source7/source8 混合 pair 的 source 顺序保留，但两源的 local `sample_id=[0,3]` 碰撞、三项 provenance 仍被丢弃。脚本还显式确认正常两边 split 的 `PoseMeshWindows` 对两源均按预期拒绝；split 门禁未放宽，manifest `training_allowed` 仍为 false。该 probe 仍只属于 bounded wiring smoke，不构成 reader 或训练准入。详见 `tmp/goal-supervision-20261015/direct-source-v2-contract.{json,md}`。

05:10 CST 监督器触发的 live-index mtime 更新后，重新执行 5 秒 semantic drift probe；hash、source4 对账、总行数/对象数和 `training_allowed=false` 仍全部稳定，因此不重跑全量 NPZ/geometry 审计。

05:25 CST 监督器新快照后再次完成 5 秒 semantic drift probe；语义 hash、36,351/2,353 总量、source4 唯一路径和全局 false flag 仍稳定，继续不重跑全量审计。

05:40 CST 新监督快照后的 5 秒 semantic drift probe 仍为稳定内容；未触发 full-file/geometry 重审计，准入状态不变。

05:55 CST 监督器新快照后的 semantic drift probe 仍稳定：总量、source4 对账和全局 `training_allowed=false` 未变，继续不重跑全量审计。

06:10 CST 监督快照仍显示远端 WAM-v2 stale `RUNNING` 33,520/80,000、SSH 不可达；native live index 仍为 36,351 sequences/2,353 objects、`training_allowed=false`，本地 RL660 已完成 19,000，GPU 空闲，磁盘约 401.5 GB 可用。watcher mtime 更新后重跑 5 秒 semantic drift 双读，去除 `checked_at` 后内容 hash、source4 33,056 行/唯一路径和总量均稳定，未触发全量 NPZ/geometry 重审计；详情见 `tmp/goal-supervision-20261015/live-manifest-drift-0610.{json,md}`。

06:25 CST 自动快照仍为远端 33,520/80,000、SSH 不可达；native 36,351/2,353、`training_allowed=false`，本地 GPU 空闲、磁盘约 401.5 GB。watcher 新 mtime 后的 `live-manifest-drift-0625.{json,md}` 仍为 `STABLE_SEMANTIC_CONTENT`，所有对账检查通过，未触发全量重审计。

06:40 CST 自动快照无变化：远端仍 stale 33,520/80,000，native 36,351/2,353 且全局 `training_allowed=false`，GPU 空闲、磁盘约 401.5 GB；`live-manifest-drift-0640.{json,md}` 语义双读稳定，未触发全量审计。

06:55 CST 自动快照仍无转变：远端 33,520/80,000、SSH 不可达，native 36,351/2,353 且全局 false flag，GPU 空闲、磁盘约 401.5 GB；`live-manifest-drift-0655.{json,md}` 仍为稳定语义内容，未触发全量审计。

07:10 CST 自动快照仍无转变：远端 stale 33,520/80,000、SSH 不可达；native 36,351/2,353、`training_allowed=false`，GPU 空闲、磁盘约 401.5 GB。`live-manifest-drift-0710.{json,md}` 语义双读稳定，未触发全量审计。

07:25 CST 自动快照仍无转变：远端 stale 33,520/80,000、SSH 不可达；native 36,351/2,353、`training_allowed=false`，GPU 空闲、磁盘约 401.5 GB。`live-manifest-drift-0725.{json,md}` 语义双读稳定，未触发全量审计。

07:40 CST 自动快照仍无转变：远端 stale 33,520/80,000、SSH 不可达；native 36,351/2,353、`training_allowed=false`，GPU 空闲、磁盘约 401.5 GB。`live-manifest-drift-0740.{json,md}` 语义双读稳定，未触发全量审计。

07:55 CST 自动快照仍无转变：远端 stale 33,520/80,000、SSH 不可达；native 36,351/2,353、`training_allowed=false`，GPU 空闲、磁盘约 401.5 GB。`live-manifest-drift-0755.{json,md}` 语义双读稳定，未触发全量审计。

08:10 CST 自动快照仍无转变：远端 stale 33,520/80,000、SSH 不可达；native 36,351/2,353、`training_allowed=false`，GPU 空闲、磁盘约 401.5 GB。`live-manifest-drift-0810.{json,md}` 语义双读稳定，未触发全量审计。

约 08:15 CST 完成只读跨源 cleaning ledger：`tmp/goal-supervision-20261015/cleaning-ledger.{json,md,png}` 将 source4–9 的 native contract、admission、provenance、clock、split、target/window 和 source4 全量文件审计逐项对账；source ID 覆盖、总行数、全局/逐源训练关闭、source4 行/路径/语义复放、source9 的 220 个 orphan 交叉核对及 gate↔failed-list 一致性全部为 `true`。该 ledger 明确是 proposal-only：所有 source 仍 `admissible_now=false`，没有执行 quarantine/normal repair/provenance sidecar/clock 转换，没有改 manifest 或 production loader，也没有启动混合训练。子代理复核 JSON/Markdown/PNG 一致；下一步仍只在获批后实现版本化 adapter/quarantine manifest。

只读审计已覆盖 live source4 legacy geometry 与 source5/6/7/8/9 manifest-listed NPZ：五源 direct
shape/finite/30Hz/mask dtype 检查通过，但行级 provenance/时钟字段、法线质量、HOT3D QA 窗口过滤和
对象 split policy 尚未满足准入。H2O manifest 仅 524 序列/67,358 帧，目录外多出的 220 个 orphan
NPZ 已隔离计数；DexYCB 法线约 `2π` 尺度，H2O 有混合法线尺度，HOT3D 有 55,361 个 QA-false
帧、19 个非单调边和 4,717 个残差超过2ms的帧。当前 `native-contract-audit.json` 的 admission
verdict 为 `BLOCKED`，所有 provisional manifest 仍 `training_allowed=false`。

03:16 CST 按 live watcher 最新快照重跑了 provenance、native geometry 和九门 admission 审计：live
manifest 读前后稳定，36,351 sequences 与 2,353 objects 对账 exact，source 读取无错误；native
audit 后重新生成的 admission 仍为 `BLOCKED`，source4–9 全部 `admissible_now=false`。子代理
复核确认计数和时间顺序一致，未修改 manifest 或启动训练。

03:25–03:27 CST 监督与重审计没有发现状态转变：远端仍为不可达的旧 33,520/80,000 快照，live
仍为 36,351/2,353 且 `training_allowed=false`；最新顺序化审计后 source4–9 仍全部不准入。

03:33–03:35 CST 的 target/window 只读审计显示 HOT3D target-valid 为 810,374/999,865 帧；QA-false
只是其中 55,361 个 target-invalid 帧，另有 134,130 个 QA-true 但 target-invalid。stride=8 的
HOT3D TRAIN/DEV target horizon 全有效比例为 98.46%/98.13%，仍有 27/4 个含 QA-false 窗口；
HO-Cap/DexYCB future hand validity 中位数约 0.5。target-slot bookkeeping 通过不等于语义准入，
adapter 必须同时保留 target-valid、QA 和 hand-valid 门禁。

03:33 CST 的 target/window 只读审计补充了接口边界：HOT3D target-valid 为 810,374/999,865 帧，
QA-false 只是其中的 55,361 个 target-invalid 帧，另有 134,130 个 QA-true 但 target-invalid；
其多对象 validity columns 在当前产物中完全相同，属于 frame-level availability 证据而非对象特异
质量。当前 stride=8 窗口中 HOT3D TRAIN/DEV 的 target horizon 全有效比例为 98.46%/98.13%，
仍有 27/4 个含 QA-false 窗口；HO-Cap/DexYCB 的 future hand validity 中位数约 0.5。adapter
必须同时保留 target-valid、QA 和 hand-valid 门禁。

03:24 CST 的 stable-read split/clock probes 进一步确认：当前 group split 无交叉；source5/6/9 若按
object-disjoint 会分别失去全部 DEV/整源被隔离，因此暂不自动重切。clock 证据只有 source6 有
`native_timestamp_ns`，source5/7/8/9 仅有 frame-id 与生成的 `time=index/30`，source4 仍待 legacy
raw reader。详见 [split probe](../tmp/goal-supervision-20261015/object-split-policy-probe.md)
和 [clock probe](../tmp/goal-supervision-20261015/clock-provenance-probe.md)。

真实 PoseMeshWindows/native collate 到 v2 objective 的 bounded plumbing probe（source5/6/9）和
48 窗 sampled semantic replay 均通过；旧 `shared_collate` 明确因缺少 `points` 拒绝。probe 使用
stand-in geometry/fallback stats，只证明 reader/collate 合同，不证明生产 sparse geometry、混合
adapter 或训练收益。远端 WAM-v2 status 仍 stale 在 step 33,520/80,000、SSH tunnel 不可达；本地
RL660 geometry-lift 与 native live-index 由独立 supervisor 每15分钟记录，未启动新的混合训练。
新增的只读 fail-closed adapter 原型以 source-local row→live row 唯一映射并独立重算 target anchor；source5/6/9 的 baseline 均为 `VALIDATED_PROVISIONAL`，而不是准入，缺失/伪造 source/profile/training flag、非有限或有限但错误的 anchor 均在 native collate 前拒绝。sidecar 仅由原型在外部暂存，现有 collate 会丢弃 provenance；所有 manifest 仍为 `training_allowed=false`，source row 也尚无显式 row-level admission 字段。
最新 read-only admission dry-run 将九类门禁（含 metadata boundary）逐源展开，source4–9 均仍 `admissible_now=false`；成功
reader smoke 的严格判定与 JSON/Markdown/热图已由子代理复核一致。产物见
`tmp/goal-supervision-20261015/admission-dry-run.{json,md}` 和 `admission-gates.png`，不修改
任何 manifest 的 `training_allowed`。
另一个独立边界 probe（`tmp/goal-supervision-20261015/native-v2-interface-audit.md`）证明旧 native
collate 会丢弃 `source_profile`/anchor/provenance，且接受缺失或伪造元数据；因此 dedicated adapter
仍是准入 blocker。live watcher 当前约36,351行并持续更新，而旧
`shared-interface-audit-r1.json` 为过期快照，不能作为当前 live 的 ENGINEERING_PASS 证据。
01:55 监督快照：本地 geometry-lift step 16,400、GPU5/7 约100%利用；远端仍 stale 33,520/80,000
且 SSH 不可达，live index 仍 `training_allowed=false`。
当前 live provenance 对账为 36,351/36,351 sequence、2,353/2,353 object exact，但 source-level clock/lineage
元数据未完整聚合，旧 interface audit 已标为 STALE；详见 `tmp/goal-supervision-20261015/live-provenance-audit.md`。
03:44–03:47 CST 在相邻 stable live snapshots 上重跑 target/window、split、clock、provenance、native contract
和九门 admission：source4–9 仍全部 `admissible_now=false`，native contract verdict 仍 `BLOCKED`。
fail-closed 原型的 baseline 只标为 `VALIDATED_PROVISIONAL`，并明确现有 collate 丢弃 provenance；不构成可训练或
production adapter 已安全的证据。随后独立扫描 sources 5–9 的全部 sequence rows，均没有显式
`training_allowed` 字段；原型另核对 `sample_id` 与重算 anchor，并拒绝相应篡改案例。
03:55 CST 自动监督仍显示远端旧快照 33,520/80,000、SSH 不可达；live index 36,351/2,353 且
`training_allowed=false`，本地 RL660 已完成 19,000，GPU5/7 空闲，磁盘约 402.1 GB 可用。
随后 fail-closed metadata probe 扩展覆盖 source4–9：source4 使用 standalone 320-sequence compact
index，5/6/9 用 windowed reader、7/8 用同一底层 canonical direct replay；source4 live 全量 clock/
provenance 仍未覆盖。admission dry-run 因此将 source4–9 metadata gate 明确标为 FAIL，source4 reader
仍 UNVERIFIED，整体仍不准入。
03:02 监督更新：RL660 geometry-lift 已自然完成 19,000 updates、exit code 0，GPU5/7 已释放；该
engineering Probe 保持 `UNCLEAR`，未触发 native 多源混合训练。
详见 [Task 状态](../src/task/3d-data-preparation/docs/STATE.md)和
[清洗/准入研究笔记](../src/task/3d-data-preparation/docs/research/2026-10-15-multisource-cleaning-sharedwam-admission.md)。

## 2026-10-15 RL660 q 三维复核与 v2 远端微调阻塞

三条 dev 序列的 raw `q_ref` 与 nearest-live branch replay 均完成；独立命令/PD
审计通过，但 q_ref 直接执行手/物体误差很大且 held=0，故保留 q_ref 为几何意图，
不替换 RL660 的反馈补偿 `q_pd` 监督，`q_real` 只作实际执行状态。v2 Task 新增
显式 `q_ref_next/q_pd/q_real` adapter，并用本地 v2 checkpoint 完成 4-step
engineering smoke；远端 v2 80k checkpoint 当前无法读取（22023 refused、直连
100.64.0.23 timeout），两卡科学微调未冒充启动。GPU4/6 保留给远端权重恢复后，
GPU5/7 既有 geometry-lift 训练不干扰。实验卡与适配器详见
`src/task/ptv3-shared-wam(-v2)/docs/experiments/`。

## 2026-10-15 TACO 原生 3D 全量工程链路完成

TACO 三个官方几何包已完整取得；15,231,790,511B 手部 ZIP 的 223/223 个 Range、2,317 个序列/363,198 帧/162 个几何对象已转换，0 个转换失败。单源接口审计为 `ENGINEERING_PASS`（TRAIN 30,642 窗、DEV 7,946 窗），live engineering index 已追加 source_id=5 的 1,826 TRAIN 和 491 DEV 序列。产物与索引仍为 `training_allowed=false`，TACO 原生时钟、目标语义和跨源准入待后续核实；详见 [Task 状态](../src/task/3d-data-preparation/docs/STATE.md)和[架构报告](../src/task/3d-data-preparation/docs/user/架构.md)。

## 2026-10-14 用户解除400GB限制，原生3D已全量并行扩产

按用户新授权更新CAMPAIGN并替换旧全局磁盘停止watcher；六张项目GPU、外部只读和身份边界不变。
HOT3D全部849包完成：290段/999865时间步转换，4段无QA时段跳过，130记录无监督GT。
UniDex排除旧20与官方TEST，2046新mesh分82批采集与逐批审计已完成并追加索引；
TACO模型、位姿和15.23GB手包均已完成断点Range下载，2317段已逐序列转换并通过单源接口审计。
旧Task已有HO-Cap64段72944帧和DexYCB subject10的100段7282帧，本轮只读复用；
HO-Cap仍全部DEV不进入TRAIN，时钟未核实的来源继续工程状态。
实时查看：`python3 src/task/3d-data-preparation/tools/audit/status_geometry.py`；
完成分片自动汇入`outputs/3d-data-preparation/native3d-live-corpus-20261014-r1/`，不重复展开窗口。
H2O账号与HOI4D网盘下载访问已向用户提问；没有等待它们而停止其他处理。详见
[扩产状态](../src/task/3d-data-preparation/docs/STATE.md)和[报告](../src/task/3d-data-preparation/docs/user/架构.md)。

## 2026-10-14 用户暂停视频路线，原生3D样例已接入

按用户授权，EgoDex下载及远端GPU1/3旧重建已正常STOPPED，输入/产物保留；既有训练继续。
新Task PoseMeshWindows复用UniDex3200窗并接入HOT3D115窗、TACO25窗，共3340工程窗口，
全部字段/共享collate通过，UniDex与旧接口最大差2.98e-8；5项合同测试通过。
HOT3D清单已授权，全量纯几何下载2.70GB；TACO几何15.69GB、全解压26.99GB。
当前不做正式多源训练：HOT3D目标语义、TACO原生clock及source准入仍待核实。
建议本地处理和审查、远端48GB GPU训练、NAS冷归档，遵守全局400GB。
详见[原生3D报告](../src/task/3d-data-preparation/docs/user/架构.md)和[Task状态](../src/task/3d-data-preparation/docs/STATE.md)。

## 2026-10-14 UniDexGrasp刚体多对象数据接口续接完成

有效20对象/320条++回放已生成TRAIN2560/dev640按需窗口，16/4对象不跨mesh。
实际3200窗合同、20对象future隔离和spawn双worker通过，相关回归76项通过。
累计目标命中119，原始末帧5cm内100，末0.5秒30Hz采样持续距离83（TRAIN67/dev16）；
距离不等同稳定抓持。完整CAD/ShadowHand暂设source_id4，约15MiB compact共享CAD，
仍禁止自动训练准入、不发出Inspire控制batch。现场本地5/7与远端0/2训练、远端1/3
重建用满六卡，下一步GPU双头工程smoke待卡释放。详见
[UniDex实验卡](../src/task/video-data-preparation/docs/experiments/probes/P-20261014-unidexgrasp-local-admission.md)。

## 2026-10-14 PTv3共享WAM v2本地Probe完成，远端80k已启动

本地1000步局部PROMISING：三域hand_all24为23.83/87.16/55.14mm，均优于step0；
scene在GRAB/ARCTIC改善，OakInk2略差，不形成抓取/Cm结论。
远端GPU0/2已从头启动 `pretrain-main3-remote-80k-20261014-r1`，代码1650ffce，
每卡128/global256，80000更新、48h/8GB上限。固定384窗dev每1000步评估，
不同于本地96窗面板。step460快照约0.682秒/update、375.5样本/s，纯训练剩余15.1h，
dev/保存另计；30秒GPU0/2平均利用率96.8%/96.3%，物理显存最大29107/23733MiB。
完整恢复更新预检通过，采用进程nofile65536与Torch2.0 fused AdamW恢复修复。
本地预算watcher每120秒汇总已声明scope，393GB主动停止本run，400GB硬限，
初始合计344.48GB；旧Task GPU5/7微调继续。详见
[v2实验卡](../src/task/ptv3-shared-wam-v2/docs/experiments/probes/P-20261014-ptv3-shared-v2-pretrain.md)。

## 2026-10-14 残差 PPO 预算/课程探索完成，起步仍未抓稳

同一 corrected update20 checkpoint 配对续训：A旧0/32/48/62课到60，
B接触0/48/50/52课到60，再单独续到100。40/60/80/100完整frame0评估均held0。
B60近接触抬高64mm但仍有支撑，B100近接触抬高23mm，不能当抓持成功。
B60最后20轮接触50/52的new_hold45为45%/60%；到100为30%/45%，并未
持续改善或迁移到起步。实际接触手点与教师最近姿态仍有29–60mm RMS差距。
三轮训练独立bootstrap/context审计通过，20项相关合同测试通过。
补做exact-base拇指±/手指合拢冻结诊断，四份也held0，动作时段/实际命令审计通过。
目前不能称PPO已收敛；停止这个固定教师课程预算块，下一优先级是实际访问接触
状态的覆盖与手腕/手指执行配合，不直接把残差方法关闭。详见
[课程卡](../src/task/trajectory-policy/docs/experiments/probes/P-20261014-wam-residual-curriculum.md)与
[固定手指卡](../src/task/trajectory-policy/docs/experiments/probes/P-20261014-wam-residual-basin.md)。

## 2026-10-14 PTv3共享WAM停止预训练，转全660机器人联合微调

ref2_2后7组GT-q重放与独立审计完成，GPU6已释放（约196s/3.62MB）。3条dev的原PD held63/89/74；直接q手点误差约61–86mm，固定阻尼补偿降至15–19mm，但mug仍held0；换回oracle手指PD仍held0，接触附近腕部约32mm误差。原native精确校准门未过，整体UNCLEAR，不切换大规模R_kin/PPO；现有PD训练保留，优先资格化接触阶段腕部跟踪。详见[GT-q重放卡](../src/task/ptv3-shared-wam/docs/experiments/probes/P-20261014-rl660-realized-q-replay.md)。

用户追加授权扩大训练：2π修复83c06492用15维geometry+continuous-lift保存分支；122685窗全量标签无明显分支误差/新增clipping，full-batch双卡2步与K4真实执行/独立审计通过。新run `rl660-joint-geometry-lift-20261014-r1` 已在GPU5/7运行（supervisor82326，workers82548/82549），首段20步loss2.38有限、约1.43秒/步、GPU均100%、physical约11.1/11.2GB，预计7.5–8小时；global512、512D×16、原LR/权重/19k新更新；从旧14k仅继承forecaster，R/optimizer重置。run预算8GB、剩余空间20GiB保护；旧latest/best与远端v2保留。新run卡：[geometry-lift联合卡](../src/task/ptv3-shared-wam/docs/experiments/probes/P-20261014-rl660-geometry-lift-joint.md)。无收敛/抓取结论。

当前GPU预约：用户明确要求新geometry-lift联合大训练，GPU5/7已用于full-batch工程门，随后同卡global512/19000步；保守计远端WAM及既有重建最多4卡，已用满本项目6卡预算。新增GPU数据任务应等待本项目额度，不仅检查本机空闲卡。源码83c06492，旧latest/best保持保留；新训练就绪/进度见geometry-lift-joint实验卡。

17:52 CST按用户ref2_1要求已停止旧raw-native微调（最后进度约14040）；保留latest=step14000和best=step4500，各1.22GB。旧自动部署/GT队列均取消，远端v2继续。物理目标chunk及live-q转换已提交00a7e7ee，70项测试通过；三臂2k更新已完成：dev mean/p95分别56.31/180.70、51.96/184.84、51.25/186.15mm，未过改善门（UNCLEAR）。relative-PD可逆，geometry有2π的Euler分支差异并禁止执行；新PD训练2步和K4真实执行/独立审计已通过。大规模联合微调保持停止。协议见[R物理目标卡](../src/task/ptv3-shared-wam/docs/experiments/probes/P-20261014-rl660-R-physical-targets.md)。以下15:44/16:10为历史快照。

按用户要求停止三域80k运行，保留原20k/38k，best/latest为best available 38k；
历史22k最低dev权重已被旧清理规则删除，不能恢复，现已修复best保护。
用户确认全660条同执行补录手点、状态、实际命令，并排除明显失跟窗口。
全量补录r2的660条/50类已完成，当前至未来24步10cm质量门排除18.48%窗口；
TRAIN/dev为589/71条、110777/11908窗口，每条序列均有准入。
原global256/52M运行在完整1000步后保留best/latest并停止。按用户要求，
共享trunk512D×16层，参数101.6M；向量化padding/token、重计算、融合AdamW
和TF32接入。函数保持迁移、48项测试及降低分组LR后的短更新稳定性门通过。
`rl660-joint-large-20261014-r1`已在GPU5/7、global512启动剩余19000步；
optimizer显式重置，原权重不覆盖。15:44 CST快照扩容phase9460/19000，累计10460/20000，
约1.45秒/步、剩余约3.84小时。训练分段日志均值4.73→1.80；固定dev1000→9000步
PD xyz命令误差71.48→57.27mm，但hand_all24为111.48→111.43mm、scene40.72→41.95mm，
轨迹泛化进入平台。15:44快照best为step4500、latest为9000，尚无新增闭环抓取证据。
16:10 ref2只读审计确认all8等权监督和总objective选点缺少控制侧重；10500的GT命令
xyz mean/p95=52.95/177.93mm优于4500的61.75/233.06mm，但10500未保存。另有RL660
逐序列数据与旧native单文件入口不兼容的确定blocker。默认执行K4，不能照搬ref2的
K1前提或把全部误差归因loss；16窗时序/坐标/PD复算通过。保留当前训练为基线，
执行前须修选择/入口。R-only已在六卡满额时用CPU4线程完成：control-aware相对all8
dev mean/p95=50.78/161.68→45.80/148.05mm，局部PROMISING但不够执行。256窗
能拟合到TRAIN8.64mm，dev反而63.30mm，优先查扩大覆盖后的优化/泛化。修复
9a9b9ad1在隔离工作树通过56项测试与真实输入预检；物理执行未验证。旧训练仍跑原
all8，守护4007948待peer退出且目标无并发修改后应用补丁，不热改源码。详见
[660联合微调卡](../src/task/ptv3-shared-wam/docs/experiments/probes/P-20261014-ptv3-shared-rl660-joint.md)。

全TRAIN覆盖对照（68747480）同预算下将B的dev mean/p95从45.80/148.05降至43.02/137.14mm，固定小池续训未改善；独立复算通过。候选固定step1500，startup仍81.28mm，物理执行未验证。已准备原训练自然退出后的有界GT执行交接，最多480秒/1GB，仍不启动新联合训练。

[R-only修复卡](../src/task/ptv3-shared-wam/docs/experiments/probes/P-20261014-rl660-R-control-repair.md)。

## 2026-10-14 残差 PPO 按 ref6 修复基础合同

已复现非逐级 KL 回退和 λ=.95；修复非终止截断的真实末端 V bootstrap、
教师实测前缀的 H4/previous-command/物理状态重建、任务起始 lift 与初始 held、
λ=1 和 optimizer/RNG checkpoint。四个 phase 的零残差诊断和逐 DOF 饱和率已接入。
相关 17 项合同测试通过。r2–r5 的失败轨迹保留，但因实现缺陷，不能用它们关闭
残差方法。修复提交 b201396b/a2f85d4b；r6 的教师 batch 初始化失败保留，r7
在 GPU6 完成 fresh 20-update Probe，492s、18832MiB。四 phase 零残差门通过；
22 次 reset 与 20 个末端 V 独立复算通过，bootstrap 最大误差 1.91e-6。
训练 held 均值23.5%，主要来自已抓持的 phase62，平均 DOF saturation0.289%；
frame0 评估仍4/4 held0，物体都留在桌面，局部 UNPROMISING。不能把后段持有
当成起步抓取进展，不继续加更新。下一决策先查这套 exact pair 的 frame0 小残差
basin；不据此否定 residual RL，也不直接归因于 WAM。详见
[残差卡](../src/task/trajectory-policy/docs/experiments/probes/P-20261013-wam-residual-ppo.md)。

## 2026-10-13 残差 PPO 二百次更新后仍然抛掷

`r5` 在 GPU0 跑完 200 次更新，2562 s。Frame-0 评估的残差 rms 是 0.136。
最好的一份保持 5 帧、贴住时抬高 93 mm，然后掉到 −765 mm。其余几份留在
桌面，几乎没有抬起。保持 45 帧没有出现。奖励和保持比例从第 51 次到第 200
次都贴在地板上；残差幅度从 0.155 慢慢爬到 0.209。局部 UNPROMISING。
同一套 α 和奖励不再加更新。详见
[残差 PPO 卡](../src/task/trajectory-policy/docs/experiments/probes/P-20261013-wam-residual-ppo.md)。

## 2026-10-13 残差 PPO 改成一次 batch 前向

串行 4 环境的 WAM 查询被收成同一个模型的一次 batch。GPU0 上 batch 4 是
92 ms、43.7 samples/s，串行 4 次是 262 ms、15.2 samples/s。batch 32 峰值
661 MiB，没有 OOM。同一输入的轨迹最大差 1.46 mm，base action 最大差
5.2e-4。`r4` 停在第 70 次更新，保留的权重是 `update0050.pt`。`r5` 用
8 个环境从这次权重继续，每次物理步一次 WAM+R。零残差诊断间隙 0.42 mm。
显存仍约 18.7 GB，其中 Isaac 约 18.0 GB，WAM 约 0.6 GB。还没有新的抓取结论。

## 2026-10-13 残差 PPO 短探针没有离开冻结底座

冻结 context 演员和旧 Chunk-R。PPO 只加 12 维残差，系数 0.15。奖励是物体
进度、接触、保持和滑落，不模仿专家动作。GPU0，295 s，显存约 18.6 GB，
利用率约 20%。零残差诊断最近间隙 0.60 mm。六次更新后，frame-0 的平均残差
只有 0.012，保持 0。四份同时重置的闭环从同一步命令分开：最好的一份在
10 mm 内把物体抬高 17 mm 并留在桌面，有一份掉到 -757 mm。局部 UNCLEAR。
不据此换 R，也不把 World 放进 actor。详见
[残差 PPO 卡](../src/task/trajectory-policy/docs/experiments/probes/P-20261013-wam-residual-ppo.md)。


## 2026-10-13 学生状态动作适配一轮没有形成抓取

三域初始化、408 维示范参考（t+1/t+16）和冻结的成功基线 R，effect loss 关闭。
学生状态只监督专家当下的 PD，不把失败后的手运动标成轨迹。训练 1000 步，
GPU0，127 s，显存 478 MiB。晚段学生腕部误差 615.5 mm 到 611.7 mm；起步状态
仍约 304 mm。

自主闭环在 GPU0，125 s，显存约 18.6 GB，执行审计通过。教师保持 435 帧并放下，
同一套 R 上的真值轨迹保持 438 帧并放下。适配后的演员保持 0，裁剪 9.6%，
最近间隙 2.4 mm，物体最高 19 mm，全程停在桌面上。局部 UNPROMISING。
不给这一对追加 epoch，也不单独更换 R。详见
[学生动作卡](../src/task/goal-point-wam/docs/experiments/probes/P-20261013-goal-point-wam-student-pd.md)。


## 2026-10-13 进度目标闭环跑完，前缀在改索引之前就分开了

已准入的 realign actor 加 visited R，只改演员的物体目标索引。物体还在
reference[tick] 的 5 mm 内时，目标仍是 t+8/t+16/t+24；落后之后停在抬起起点
（这次参考上是第 62 帧，+15 mm），不再指向空中的演示帧。GPU2，542 步，149 s，
显存约 18.6 GB。执行审计通过。教师保持 437 帧并完成放置。演员保持 0，
裁剪 41%，最近间隙 0.24 mm，物体后来被抛到 680 mm。

目标索引从第 50 步才离开演示时钟，但动作从第 6 步起就和上一次时钟闭环不同，
那时目标还完全一致。后面的抛掷不能归因于这次索引。局部 UNCLEAR，不据此改训练标签。
详见
[进度目标卡](../src/task/goal-point-wam/docs/experiments/probes/P-20261013-goal-point-wam-progress-reference.md)。


## 2026-10-13 pointwam-repro：冻结编码器上一步优化已跑通

GPU0 上，合格 clip 的 5848 个场景点经过冻结 Mosaic3D 和 Recap-CLIP，随机初始化的 Volt-H 按 Table 4 完成一步 AdamW。损失 7.903，梯度范数 5.592，峰值 13028 MiB，步耗时 0.604 秒。标准化统计只来自这一条 clip。60k 和 batch 192 没有启动。见
[冻结优化步卡](../src/task/pointwam-repro/docs/experiments/probes/P-20261013-pointwam-repro-frozen-step.md)。


## 2026-10-13 pointwam-repro：冻结编码器小规模前向已接通

本机 Mosaic3D 全数据 checkpoint 已下完，平均 1.8 MiB/s。GPU0 上 1024 个真实场景点经过冻结 Mosaic3D 和 Recap-CLIP，随机初始化的 Volt-H 输出有限，峰值 6479 MiB。这只说明接口通了，不是预训练，也没有启动 60k。见
[冻结前向卡](../src/task/pointwam-repro/docs/experiments/probes/P-20261013-pointwam-repro-frozen-forward.md)。


## 2026-10-13 pointwam-repro：论文 Volt-H 一步能跑，编码器还没接上

可训练 forecaster 按论文 stem、5³ patch、Volt-H、解码头和 Table 4 优化器写好，参数量 655,527,092。GPU0 上 12000 点、bf16 一步峰值 12776 MiB，损失有限。特征是同维度随机张量，不是 Mosaic3D / Recap-CLIP，所以不能当成预训练。60k 和 batch 192 没有启动。见
[Volt 前向卡](../src/task/pointwam-repro/docs/experiments/probes/P-20261013-pointwam-repro-volt-forward.md)。


## 2026-10-13 锚定联合更新未守住起步和留出接触映射，不进闭环

从已过离线门的 realign actor 和 visited R 出发，场景/World/R 归一化冻结，
无 World loss。损失为手轨迹 MSE、预测轨迹动作损失（同时更新 actor 和 R）、
以及真实手轨迹动作损失（只更新 R）。GPU2，3000 步，609s，显存 1.8GB，已释放。
首步梯度符合约定：预测动作进入 actor 与 R，锚只进入 R。

12 个检查点全部未过预注册门，因此没有 GT 资格，也没有 542 步闭环。
选中的 step2000 上，训练接管预测 PD 均值 1.33→2.49mm，没有下降；
起步 GT p95 1.52→2.49mm、起步预测 p95 1.42→3.48mm，超出 1.1 倍+.1mm。
名义 GT p95 14.60→10.73mm 改善，接触训练均值 22.4→13.8mm 改善，
但留出精确接管 GT p95 228→288mm，并随步数继续变差。锚保住了名义轨迹，
没有保住起步精度和留出接触状态。局部 UNPROMISING，不部署这一对。
下一步改参考时钟合同，不再加长这次更新。详见
[锚定联合卡](../src/task/goal-point-wam/docs/experiments/probes/P-20261013-goal-point-wam-visited-joint-anchor.md)。


## 2026-10-13 pointwam-repro：缩小 forecaster 只改善 fit 手部

48 个 clip 的坐标合同仍然有效。GPU0 上的 1000 步零初始化拟合把 fit 手部 EPE 从 54.8 mm 降到 29.5 mm，同时 fit 场景从 9.0 mm 升到 15.1 mm，dev 手部从 69.3 mm 升到 87.1 mm。按静态损失平衡两项后结果相同。这套 1 cm patch 替代网络先停止加长；GPU3 全量 EgoDex 重建未动。见
[patch fit 卡](../src/task/pointwam-repro/docs/experiments/probes/P-20261013-pointwam-repro-patch-fit.md)。


## 2026-10-13 pointwam-repro：48 个 clip 合同通过，缩小模型只做本地测试

只读登录 `deepcybo-4-4090-48`。GPU3 上的全量 EgoDex 重建仍在运行，未停止：
`reconstruct_egodex_queue.py` 输出到 `egodex-full-20261012-r1`，当时完成 4563、
失败 17479。已有 PointWAM 扩展包 48/48 通过形状、z-up、相对当前帧位移和
`training_allowed=false` 核对，场景点 2856–12000，手末端位移中位数约 79 mm。
新任务加入 1 cm / 5³ patch 和论文运动加权损失的缩小实现，不含 Mosaic3D、
Recap-CLIP 或 Volt-H，也没有用这 48 个 clip 开训。见
[pointwam-repro](../src/task/pointwam-repro/README.md)。


## 2026-10-13 视频物体轨迹goal审计：先修目标部件覆盖，不启动扩训

用户选择非语言物体运动goal。远端旧合格视频池22段各抽时钟中位窗口，
先固定12个当前帧可见表面polygon，再检查+8/+16/+24原始点轨迹。
仅3 TRAIN/0 held通过固定几何门；通过的是桌面/底座，末端位移中位数
1.13–3.37mm，手实际在操作独立桌腿。其余9个标注只有2–12目标点，
运动瓶子4点到末端全部失效，灯泡5点受遮挡。局部UNPROMISING仅指从
现有稀疏场景点直接截取goal的配方，不否定物体goal或原始视频。
已实现显式物体时序goal数据包、掩码/点身份/手独立/存储隔离测试（4通过），
未接新actor编码器、未训练、未改World或远端。下一步明确目标刚体部件并
做目标内补点跟踪；保留静态支撑goal，不能冒充运动意图。详见
[视频物体goal审计](../src/task/goal-point-wam/docs/experiments/probes/P-20261013-goal-point-wam-video-object-goal.md)。


## 2026-10-13 参考手腕goal matched Probe：无近期收益，停止该配方扩训

相同ref2_8 step4500底座、原合格R/旧control adapter冻结、相同数据与抽样，
仅B新增计划人手腕+8/+16/+24条件；A同容量adapter输入为空。各1000步，
GPU2共329s、显存2.1GB，已释放。三个精确C接管状态TRAIN first8均值
A57.082/B57.667mm，未达10%改善，t32 contact16首PD p955.906->9.140mm
防退化也失败，其余门通过。局部UNPROMISING，不扩训、不接入闭环。
首批输入/预测/loss严格一致，冻结state/R差0，保存数组独立复算通过。
同物理查询batch1/batch16编码有约1–2mm有效预测差异；用contact16首行
复核A57.144/B57.642mm，仍无收益，不能把微小差异解释成科学结论。
手腕输入屏蔽/错时只带来小幅预测变化，human严格不受该输入影响。
这不是否定手goal：仅固定示范、稀疏腕、短适配、3个旧C状态；当前actor
真实访问状态与接触/手指语义仍未覆盖。下一步优先真实状态专家纠偏和
已知参考条件基线，不继续清洗或盲目扩训。见[参考手腕goal卡](../src/task/goal-point-wam/docs/experiments/probes/P-20261013-goal-point-wam-reference-wrist-goal.md)。

## 2026-10-13 ref3.1 两个独立Probe完成：R拟合改善但直接替换失败，actor降权信号不足

冻结既有actor，以50% nominal/25%精确C接管/25%专家后续修复Chunk-R，
GPU2/3000步55s。3个接管瞬间首PD32.82/58.95/159.93->1.74/1.84/1.37mm，
nominal DEV18.01->18.68mm，20离线门通过；仍仅3个拟合C状态+192后续。
首个GT运行因其他会话进入GPU2安全中止，转GPU4，128步GT held75、零裁剪。
显式新R执行身份接入同一adapter4500，完整542步actor held0/最高2.7mm，
裁剪50.37%（首次tick252，前128零）；GT held438并放置，但全程裁剪1.845%
未过严格<1%校准门。13执行合同通过。固定相同actor tau交叉计算，接触16
PD p95在三组都恶化；GT tau反而都改善，说明修R映射不自动修好actor/R配合。
保留修复候选，默认旧R不动，不扩训adapter。见[卡40](../src/task/goal-point-wam/docs/experiments/probes/P-20261013-goal-point-wam-ref31-own-R.md)。

独立人类数据A/B，各300步，同底座/9600采样/随机流，仅OakInk几何mismatch
actor权重.1；保留全部World监督、原训练模块。确定性GPU设置修复首batch
配对不一致，修后输入/预测/World loss严格相同。244s完成，384 VAL中目标
附近交互81窗EPE13.189->13.167mm（仅.169%），未达5%门；三域/World基本持平，
结论UNCLEAR，无扩大或删数据依据。见[卡41](../src/task/goal-point-wam/docs/experiments/probes/P-20261013-goal-point-wam-ref31-actor-weight.md)。

下一优先级：修复R固定后的有界actor语义重对齐和真实访问状态覆盖；GRAB/ARCTIC
far-to-contact覆盖仍待独立处理，未并入本轮。无稳定自训练抓取或Cm因果结论。
本轮GPU3/4已释放；outputs+tmp约296.7GB，低于400GB，保留所有原始失败记录。

## 2026-10-13 三域actor降权短对照无实质收益；参考手目标来源已核实

原始WAM step10000、同9216 TRAIN池/96k抽样、decoder-only各1000步：
静止目标远手窗降权.1，六个交互panel first8为A14.67368/B14.66057mm，
仅改善.0893%，未过5%门；三域防退化及goal作用门通过。局部UNPROMISING，
停止该配方扩训，不否定从头清洗或共享编码器适配。冻结state差0，保存数组
独立复算通过；GPU2共187.56s，已释放。用户允许参考示范，已从已知计划
导出+8/+16/+24人手腕/物体目标，几何合同通过但形态/时序尚未接入WAM。
历史reference+PD DAgger两次固定条件成功可作对照，不能归因于参考输入单项。
见[降权对照卡](../src/task/goal-point-wam/docs/experiments/probes/P-20261013-goal-point-wam-actor-curation-ab.md)。

## 2026-10-13 ref2_8 本体/temporal-goal适配完成，接触改善但仍未稳定抓持

保留几何底座/goal32/raw-hand/World/合格Chunk-R，WAM新增48D q/dq/上一
实际ACTIVE12动作及27D自身当前object相对计划+8/+16/+24位姿，零门控初始
C输出严格相等；human屏蔽新条件。当前C t32/40/48实测专家接管分别held
75/74/71，195自有expert future窗口，其中仅3个是真正C接管瞬间，其余为
专家继续执行状态。adapter-only2000后decoder小LR5000（GPU2，659s），
按contact选4500；冻结模块/R差0，35保存面板指标独立复算通过。
原严格stage1门失败，按公开Decision Note修订探索门后继续；不是原门通过。
精确C查询首PD t32/40/48为51.76/119.76/351.89 ->1.98/83.49/33.69mm；
contact16 p95改善，但t32 first8仍有4.77%退化。真实GT tau进入固定R在新
contact16仍有34/172/223mm误差，仅名义轨迹资格不能覆盖这些恢复状态。

4步联调通过，完整542 controls/147.54s，teacher/GT held434/441，actor
held0；最小gap .913mm后物体飞起至276.6mm（此时gap363.7mm），随后掉落。
t48腕偏174.21->46.77mm、t56 393.97->82.76mm，t32反而7.21->32.57mm；
裁剪9.225%->1.476%。13项执行/新增输入检查通过，SE3数值差<.9e-6。
局部闭环UNPROMISING，接近/接触有部分改善，不证明缺观察是唯一原因，
也不否定条件输入或Cm；未做matched input/Cm-off训练，未扩大到20k。
保存候选与实测重建视频，下一步先辨别当前访问状态覆盖和R恢复映射局限。
本轮GPU2/3已释放，outputs+tmp295.18GB<400GB。
见[本体控制条件卡39](../src/task/goal-point-wam/docs/experiments/probes/P-20261013-goal-point-wam-ref28-control-context.md)。

## 2026-10-13 C最终模型实际rollout未抓起；执行链审计通过

用户要求将C step50500接入真实仿真闭环。保留Probe32 warm scene/goal/hand
query，加载C decoder，只用自己的实测H4/计划object goal；固定训练时相同
合格Chunk-R，每步预测8动作/执行首步，teacher不干预actor。
GPU2/seed292完成542 controls，native/server正常exit0，组128.89s。
教师/GT-Chunk分别434/441连续held（14.47/14.70s），C actor held0，最高
抬升0mm、没有任何有效抓持帧，裁剪9.225%。GT后段放置掉落，完整任务不合格，
但本次局部抓起校准通过。actor腕部距teacher t32=7.21mm、t48=174.21mm、
t56=393.97mm，接触阶段漂移迅速扩大。实际commands/PD/首步/当前状态/GT帧/
自身H4/计划goal/outcome独立数组审计10项max0；首次live直接模型等价差0。
单reset固定checkpoint闭环局部UNPROMISING，不否定C表征或证明Cm无效。
原离线改善不能当闭环抓持收益；保留候选/基线/新实测负结果，暂不扩大训练。
视频用实测位姿点云重建，非相机录屏；两次早期图形工程失败已隔离记录。
GPU已释放，未新采集DAgger/训练/RL。见[闭环卡38](../src/task/goal-point-wam/docs/experiments/probes/P-20261013-goal-point-wam-ref27-rollout.md)。

## 2026-10-13 ref2_7 C四卡global400完成C0 500+C1 50k

B研究冻结，old raw-hand/Probe32 warm和合格Chunk-R保留；scene/goal/raw/
action/world/R/buffers全冻结，只训trajectory decoder。4卡GPU0..3，global400，
60%三域/25%nominal/15%实际expert recovery；human PD loss0，lambda_a1、
lambda_E.1、AdamW1e-5。r3从r2 step500连同optimizer/context接续，05:55:31
完成总50500更新/C1额外50k，曝光20.2M重复窗口，约.348s/update，GPU已释放。
三域human all24和effect gap下降；等权8panel gap10.576->2.928mm。
nominal/startup首PD XYZ p9571.416->14.538mm、21.503->.740mm，recovery
TRAIN t8/t16/t32为43.197->9.247、44.177->6.959、171.827->148.081mm。
全部零裁剪、rotation/finger保留，冻结state/R差0，GT oracle固定数组变化0。
53评估的保存数组/optimizer50500/四rank context审计通过，非新推理或标签复算。

C0数值合同合格，但root额外10%轨迹门有startup first8失败10.107->11.384mm；
按ref2_7长训练意图明确作事后协议修订：严格36项继续报告，停止门保留原first-PD
限，轨迹改用2x+.05mm严重退化限。C1两次连续停止门失败仍退出；20k有gap信号
自动扩到50k。最终停止门36/36，原严格诊断34/36：startup h1+16.34%、t16
recovery h1+12.02%仍退化，不能称全面不退化/原始预注册门通过。

局部离线适配PROMISING；负gap是冻结模型差距而非不可约误差，recovery是TRAIN，
t32仍148mm，不能声称可靠抓持或Cm因果收益。无Cm-off长训练/物理执行，本阶段
仅保留候选/20500/final与旧A/B。首轮fork/LLVM及DDP/scene数值工程错误已修复，
原记录保留，8h截止未延长。r3约.655GB，全局294.51GB<400GB。
root按30分钟观察，10点前静默。见[C实验卡37](../src/task/goal-point-wam/docs/experiments/probes/P-20261013-goal-point-wam-ref27-effect-consistency.md)。

## 2026-10-12 ref2_6 B2门控接口保持初始行为，通过不退化门；手特征收益仍UNCLEAR

按ref2_6将硬加改成raw+alpha P(LN(current PTv3 hand))，alpha0；保留A2
合格Chunk-R、goal32和世界接口。完整模型初始五组tau/首步动作/goal扰动
输出与旧WAM逐值相等，最大差0。冻结scene/goal/raw-hand/R/action/world，
两臂只训decoder，新臂另训LN/P/alpha；各200更新100.69s，21/21门通过。
nominal首步PD p95旧19.60/新17.39mm，startup11.09/9.09mm；startup first8
11.46/10.83mm。全部零裁剪，final alpha+.00017116，投影从第二步获梯度。
但alpha0消融仍有nominal17.49/startup9.30mm，大部分改善来自已适配decoder
路径；手特征本身效用仍UNCLEAR。PROMISING只指初始行为保持及当前离线
筛查，不构成抓持或表征结论。t32 h1恶化9.33%仍在预设宽容门内，不能说
全指标改善。180独立指标/标签复算、相同采样及保存的冻结模块比较通过。
保留B2候选和旧接口，两份checkpoint，不开启C或actor闭环。11相关测试
及仓库verify通过。新增433.84MB<500MB；全局293.70GB<400GB，GPU2已释放。
见[门控手读出卡36](../src/task/goal-point-wam/docs/experiments/probes/P-20261012-goal-point-wam-ref26-gated-hand.md)。

## 2026-10-12 B显式手token读出已实现，200步匹配比较未过非退化门

在ref2_5合格Chunk-R之后，按ref2_4/ref2_5进入B：完整PTv3 packed输出
按当前语义提取2×11hand feature，与原raw-hand query相加；不受256scene
采样影响。两臂同Probe32 warm policy/固定新R/同采样各200更新，129.87s。
nominal h1旧6.51/新8.59mm、first8旧16.06/新25.00mm，startup控制p95
6.41->4.44mm改善，但10个轨迹门及t8恢复控制门失败，总10/21通过。
全部零裁剪，goal sensitivity保留。144独立指标/标签复算通过，tau差
<=1.2e-7m，其他指标/动作标签差0；R/action/world/统计buffer变化0。
当前固定短适配局部UNPROMISING，初始新读出h154.51->8.59mm说明需区分
适配不足与接法/尺度，不否定手交互表征。保留两臂，按gate暂不进入C或
新actor闭环；A2的GT Chunk-R资格仍有效。B产物430.23MB<500MB，
outputs+tmp293.27GB<400GB，仓库verify通过，GPU2已释放。
见[手读出卡35](../src/task/goal-point-wam/docs/experiments/probes/P-20261012-goal-point-wam-ref25-hand-readout.md)。

## 2026-10-12 ref2_5首步控制目标让Chunk-R通过GT抓持门

保留Chunk Transformer和同初始化/同采样，固定3000更新48.38s；改用
4×首步standardized L1+后7步L1+首步physical PD loss，checkpoint按
first-PD XYZ p95优先选择。report首步XYZ p9531.09->19.51mm，恢复TRAIN
94.80->12.78mm，代价为all8 L1.16397->.23493。实际128步GT比较22.73s，
teacher73held/旧PD-R75held/新Chunk-R74held，全零裁剪；60held门通过。
8接口测试通过，标签与实际执行复算max0，objective复算差<9.4e-8。
局部PROMISING，仅取得GT执行资格，不能当自主WAM、完整放置或Cm收益。
已保留候选，下一步按ref2_4/ref2_5进入B显式PTv3手token读出离线比较，
C仍以B不退化为条件。累计wam-v2产物34.17MB<80MB，GPU2已释放。
见[控制目标卡34](../src/task/goal-point-wam/docs/experiments/probes/P-20261012-goal-point-wam-ref25-control-aware-R.md)。

## 2026-10-12 ref2_4 Chunk-R已实现，GT执行第一关未通过

按用户ref2_4先做A：24个tau token+state token、2层128维Transformer、
8个12D native applied action；每tick重新规划、只执行首步。合格teacher
与259实际恢复窗训练3000更新49.45s，report首步PD XYZ p95为31.09mm，
recovery TRAIN为94.80mm。修复新对照入口中旧PD-R的Torch2.0.1单样本
推理异常：同查询batch2精确复现既有合格首步动作，原无效比较记录保留。
修正后128步独立GT回放33.20s，teacher73held/旧R75held/新Chunk-R0held，
均零裁剪；对照资格通过但新R未过60held门。实际标签、当前状态、GT坐标、
native PD转换、只执行首步的独立复算均max0，5项接口测试通过。
结论仅为当前固定配方局部UNPROMISING，不能否定Chunk Transformer；
按ref2_4暂停B手token显式读出和C预测tau world一致性，下一决策优先解决
GT Chunk-R首步精度与执行，而不是继续改WAM。旧WAM/goal/checkpoint保留。
新增产物17.70MB<80MB，outputs+tmp292.82GB<400GB，GPU2已释放。见
[Chunk-R实验卡33](../src/task/goal-point-wam/docs/experiments/probes/P-20261012-goal-point-wam-ref24-chunk-R.md)。

## 2026-10-12 ref2_3多接管训练降低后期漂移，自主持物仍未建立

按用户ref2_3，Probe32从Probe30精确WAM/R继续固定1000更新，加入
Probe31的259实际expert future+384当前PD-only；nominal/startup回放、
K24和GT-only R梯度职责保持。GPU2用时471.29s、峰值2149MiB。
13/13原离线门通过；643行全部实际抽到，256指标/PD标签max0，
tau1.19e-7m，初始参数精确承接Probe30，冻结buffer/action/world不变。
nominal predicted PD22.67->31.21mm、startup8.31->13.56mm有保留代价；
t32恢复TRAIN predicted422.17->254.79mm、near8轨迹60.72->22.79mm，
但GT-tau R误差仍257.80mm，不可认为后期控制问题已解决。

仅一次新128步闭环97.67s完成，2316物理检查max0。t24 PD265.83->
108.10mm，t32腕偏差354.10->63.17mm，t48腕522.77->105.09mm；
纯actor裁剪5.47%->0，但held仍0。新GT-R75held/零裁剪获得局部短抓持
资格；t16专家接管75held但107/108两步裁剪，严格local recovery=false。
预设闭环组合门4/5：t16 PD28.28mm略超原2x限27.80mm，其余通过。
后期漂移幅度降低局部PROMISING，整体UNCLEAR，不宣称漂移起点推迟、
自主抓取或正式证明DAgger有效。保留候选与Probe30基线，停止本轮固定
实验；下一决策围绕新分布的更晚恢复监督及早期保留，暂不改变架构/goal。
未启动额外训练、world/evaluator/RL。ref2累计804.52MB<850MB，
outputs+tmp292.80GB<400GB，GPU2已释放。见
[训练及新闭环卡32](../src/task/goal-point-wam/docs/experiments/probes/P-20261012-goal-point-wam-ref23-multitakeover-fit.md)。

## 2026-10-12 ref2_2保住R映射并改善早期闭环，后期恢复监督已聚合

按指定历史会话/ref2_2继续，暂冻结结构修改/world/RL。Probe29五个
K1/4/8/16/24同初始化/同采样GT-R窗口比较1000更新16.14s/GPU2完成。
K16 worst-panel36.21mm vs K2439.99mm，未达预设20%改善；K8 nominal
15.40/t8TRAIN5.34/t16dev21.66/t32dev42.19mm，无明显短窗口优势，保留K24。
240指标/实际PD标签独立复算max0，不能由GT结果排除预测far-tau敏感性。
Probe30保持原startup WAM/原R/同1000更新流，预测action梯度只更新
WAM，R只用真实expert tau更新；纯student动作查询不更新R。404.40s/
GPU2、峰值2159MiB，13/13离线保留门通过；nominal GT-R13.35mm
（原无锚定43.96mm）、predicted22.67mm。160指标/标签独立复算通过。
新128步实际闭环t12 PD103.44->21.98mm/腕偏差30.80->9.33mm；
GT-R重新获得75held/零裁剪局部资格，但actor仍0held，t24 PD265.83mm，
后期漂移未解决。旧校准代码哈希兼容/终端相对路径问题已修复，原FAILED
记录保留；完整物理trace经身份/快照一致审计形成派生记录，无重复仿真。
停止loss-only调参，Probe31同一learned pair完成t8/16/32实际专家接管。
三源各2316物理检查max0，接管held74/75/41；t32短迹不足45帧持物门，
仅是部分恢复信号。聚合643行=259实际专家future+384纯actor当前PD，
7073独立重建检查通过（state/tau浮点差<=1.19e-7，其他标签误差0）；
纯actor未来仍禁止用于轨迹/R监督。此阶段结束时新数据尚未训练；后续固定
GT-only R职责的多接管训练与新actor闭环结果见上方ref2_3记录。
整体UNCLEAR；未开始world/evaluator/RL，不把训练标签当自主成功。
outputs+tmp约292.58GB<用户已更新Campaign上限400GB，GPU2已释放。详见
[窗口卡29](../src/task/goal-point-wam/docs/experiments/probes/P-20261012-goal-point-wam-ref22-r-horizon.md)
、[锚定卡30](../src/task/goal-point-wam/docs/experiments/probes/P-20261012-goal-point-wam-ref22-gt-anchored.md)
和[聚合卡31](../src/task/goal-point-wam/docs/experiments/probes/P-20261012-goal-point-wam-ref22-multitakeover.md)。

## 2026-10-12 完整WAM策略链与R已联合微调，R映射保留仍是问题

按用户要求参考PointWAM，a1bd58e8/GPU2完成1000更新458.44s、batch64、
峰值2159MiB。scene/PTv3、goal、trajectory、hand_query、R.net均有实际
action-only梯度和参数更新；action/world参数与所有统计buffer冻结。起步PD
XYZp9512.08->2.76mm、early pure TRAIN587.93->60.89mm，但late purged诊断
940.48->955.48mm；nominal近8手轨迹17.20->14.76mm，PD p9525.63->35.26mm，
均值14.10->11.63mm改善，尾部主要暴露tick32局部退化，不能称正常预测全部变差。
nominal真实tau->新R10.59->43.96mm，未过预设30mm门，未执行物理回放。
同状态交叉审计：newWAM+oldR nominal24.94mm/recovery32.56mm，exact learned
pair35.26/49.12mm，局部暴露R更新损害部分映射；不是关闭联合训练路线。
160metrics重算max0/当前PD标签max0/真实future max1.19e-7m，scope与源身份通过。
当前UNCLEAR；下一决策保留端到端梯度、加入真实tau->R动作监督以保护R物理
映射，另需后期真实偏离状态覆盖。原R物理资格不转移，无新的自主成功证据。
见[实验28](../src/task/goal-point-wam/docs/experiments/probes/P-20261012-goal-point-wam-ref2-joint-retargeter.md)。

## 2026-10-12 几何拟合改善但即时控制退化，优先锚定起步

真实expert future监督/217恢复训练窗把nearest8降到约11mm，nominal
17.22->14.06mm，但同一frame0 PD12.12->24.72mm。实际192步新actor
仍0held/8.33%clip，t8腕部偏差由旧9.2变28.9mm；独立review未见
标签/anchor/order/head/R/native错配。整体几何门掩盖即时控制退化，
当前geometry-only recipe局部UNPROMISING，不关闭聚合路线。native完整
3468/max0审计PASS，但group触发100MB产物限FAILED，原状态/数据保留。
只清理root-owned可再生成Triton缓存恢复99.53MB。原R在新t8专家真实
轨迹上的all/earlyXYZp95为25.10/30.12mm，仍存在恢复覆盖误差。
下一步原startup头+新t8 true recovery+固定startup输出锚定200更新，
增加frame0/repeat-H4控制保留门，暂不扩大backbone解冻/RL。固定200更新
100.45s通过起步/nominal门：frame0 PD12.046->11.855mm，恢复TRAIN
nearest8 18.094->14.175mm；新128步回放COMPLETED/2316checks全0，但自主仍0held/7.81%clip；
frame0 PD11.852mm/t8腕部9.413mm已恢复，t12偏离仍30.798mm。
此固定修正局部UNPROMISING，不关闭聚合路线；下一步优先早期visited-state
控制覆盖，停止平均几何/解冻继续搜索。新Probe
增量20MB/累计120MB且global292.10GB<300GB，不改变旧失败预算。
见[实验27](../src/task/goal-point-wam/docs/experiments/probes/P-20261012-goal-point-wam-ref2-startup-anchor.md)。

## 2026-10-12 数据聚合后发现真实恢复轨迹的R覆盖误差

实验24固定500更新112.41s：pure-WAM训练PD XYZ p95636.50→143.26mm，
但nominal25.56→59.48mm，未过50mm新actor采集门，不部署该head。
late pure诊断1194.21→1186.27mm；不能由训练改善推断恢复收益。
随后用role3实际专家接管的137个完整24控制窗独立检查冻结R：all XYZ
p9547.17mm，早期32:48为126.64mm，同ticks nominal为16.01/9.71mm。
输入是真实测量专家future，不是无约束tau优化；原actor future禁令保留。
优先补齐R物理恢复覆盖，原六个normalization buffers冻结并混合nominal
replay；固定1000更新64.58s，恢复训练53.22→6.05mm但held17.99→19.74mm，
startup旋转超原1.5倍保留门，未进入物理校准。唯一alpha=.5插值held16.72mm，
nominal/startup旋转仍未过原门，也不部署。下一步用已有t16实际专家接管轨迹
跨接管时刻诊断，区分局部拟合与覆盖迁移，再决定真实恢复数据/监督方式。
WAM head仍用原startup，显式记录其训练旧R与执行新R的身份差异。
见[实验25](../src/task/goal-point-wam/docs/experiments/probes/P-20261012-goal-point-wam-ref2-recovery-retargeter.md)。

## 2026-10-12 ref2_1机器人数据聚合启动，持续工作到19:30

用户新目标按goal-point-wam/ref2_1持续推进到今天19:30。先验证更远自主偏离状态
的真实恢复：best startup head、四角色、takeover32/controls192，独立审计后聚合
unclipped pure-WAM0:63和recovered32:160的真实PD。旧val32:48/64:80显式退役
为训练，不能再称validation；用新训练actor产生的新状态查询检验收益。所有真实
applied action与measured next state一起保留作actual consequence，mixed-policy
future手轨迹不能作expert标签。先head-only检查coverage，不扩大PTv3解冻。
短采集不判断完整placement；默认542/16评价保留。阶段预算/准入/停止条件见
[实验24](../src/task/goal-point-wam/docs/experiments/probes/P-20261012-goal-point-wam-ref2-robot-aggregation.md)。

## 2026-10-12 局部解冻改善部分拟合，未学会更远偏离状态的恢复

按用户建议完成PTv3最后decoder dec0+trajectory/hand_query的100更新，
其余scene/goal/action/world/R冻结；固定代码c5df7848、运行HEAD d56cfa31，
GPU2用时109.46s/峰值758MiB，已释放。live train PD XYZ p95 44.24→33.32mm，
validation384.76→398.67mm；nominal nearest8 EPE19.53→17.45mm，nominal
PD82.69→46.82mm，但未保住original startup要求38.335mm。两项门未过，
局部UNCLEAR，不运行这个head的闭环，不据此继续盲目扩大解冻范围。
独立数组/标签/输入身份/scene补丁重建hash审计通过，metric与label误差0。

大validation误差集中在pure-WAM ticks32:48（PD p95 433.06→435.20mm）；
专家接管后的validation ticks64:80为31.93→33.77mm。训练pool中pure-WAM
0:32也未改善45.94→50.23mm，整体改善主要来自专家接管状态39.35→24.29mm。
下一步优先核对自主偏离状态覆盖、恢复监督与nominal replay冲突，再决定是否
扩大解冻；保留固定holdout，不把训练拟合改善当作恢复能力。严格补丁桥接已实现，
17项CPU测试通过。视频仍由另一个agent负责，当前head不声称保住human/world质量。
[实验23](../src/task/goal-point-wam/docs/experiments/probes/P-20261012-goal-point-wam-ref2-scene-thaw.md)。

## 2026-10-12 早期专家PD头部修正可拟合训练点，未改善留出偏离状态

固定b002588c完成300更新/77.01s，live train PD XYZ p95 136.61→44.28mm，
留出404.15→384.46mm，未过减半门；nominal nearest8 EPE17.22→19.51mm，
但nominal PD25.56→82.83mm，局部UNCLEAR，不运行这个head的闭环。
livePD梯度实际到达actor，scene/goal/action/world/R冻结身份一致。下一步先排查
冻结R的局部输出可达性，再决定用户提出的PTv3输出层逐步解冻；不能把训练
拟合不足直接归因backbone容量。全量recovery仍因末段放置失败而未过门。

## 2026-10-12 真实偏离状态可恢复抓持，完整放置仍失败

固定7f516d59的专家接管Probe完成542步，teacher/GT-R持物438/251帧并放置；
WAM仍0。第16步从命令回放产生的真实偏离状态接管，专家持物435帧、零裁剪，
但最终放置失败，原完整recovery门未过，保持UNCLEAR。原full-phase PD微调
实验21未运行。root与独立代理分别复核全部live PD/H4/goal/current state/源身份，
数值检查全0；两行PhysX状态仅近似配对，不是隐藏状态fork。
下一步限定早期approach/抓持PD监督（实验22），排除后段失控样本；先检查冻结
backbone的头部适配训练/验证误差，按用户建议再决定是否逐步解冻PTv3输出层。
尚未开始新的PD训练，不把专家接管成功持物当自主WAM收益。

## 2026-10-12 新head完整闭环完成，执行仍未建立

代码ed53315d、冻结startup head 2b7eff74991a8ef306b192747d716210d4348d3259f628fa73c78dfc7b617fc2、
seed292的r3完成542 controls/543 measured states，整组115.25s，GPU2已释放。
teacher和GT-R各held441帧、controlled-placement PASS、clipping0，完整calibration
PASS。WAM held0、placement FAIL、clipping22/542=4.059%，物体最终落到地面。
按预定门槛，此精确初始化的physical Probe为UNPROMISING；startup离线局部Probe
保持PROMISING。没有据一次nominalepisode否定WAM/backbone或形成正式科研结论。
[执行结果](../outputs/trajectory-policy/wam-ref2-execution-20261012-r3/native/result.json)。

实际PD纯逆解、current R state、live自有H4、planned goal、applied action均max0；
冻结R独立回放encoded max2.38e-7、requestedPD max4.77e-7，所有源身份核对。
root亦独立核对全数组、实际PD/current-state/H4/goal均max0并复算clipping。
[独立全回合审计](../outputs/trajectory-policy/wam-ref2-execution-20261012-r3/root_full_audit.json)。
首步实际PD差225.27→11.71mm，state1 wrist差25.32→1.65mm，startup修复进入
真实执行；之后state9 wrist>10mm，tick12 PD>50mm，tick16 wrist53.68/PD58.51mm，
tick32 wrist89.86/PD260.09mm；此时三角色物体仍在桌上，问题已出现于接触之前。

相同实际WAM状态t8/12/16/32，将预测tau换为source未来GT时立即PD相对WAM差
49.66/105.87/130.56/318.70mm；仅替换前4步后差降为2.76/7.77/17.11/3.76mm。
source GTtau在这些learner状态所得PD与source nominal teacher PD差仍15.36/
44.55/75.08/106.37mm。短前缀forecast误差仍实质影响动作，且执行分布逐渐偏离；
source未来不是learner状态的专家recovery标签，不能据此单独指认R错误。
[依赖诊断](../outputs/trajectory-policy/wam-ref2-execution-20261012-r3/R_dependency_audit.json)。
下一决策优先获得实际learner接近阶段的专家PD监督，检查小偏差下的可恢复性；不再
继续同一nominal teacher-offline recipe或扩大人类训练步数。PD-DAgger采集/训练需
单独最小协议、source准入与停止条件，当前尚未启动。视频由用户另一个agent负责。

## 2026-10-12 Startup continuation通过预定局部门槛

固定代码ed53315d、GPU2、model293/draw294完成300追加更新，68.25s。
现有head共600更新，原humanbase/backbone/goal/action/world/R未改；5项冻结hash
一致，25%first16 draw9600/38400精确。startup validation PD XYZ p95
199.878→18.341mm，frame0 GT-R差13.890mm、实际teacher差12.116mm。
held real-H4最近8步20.769→16.478mm（预定上限25.966mm），两项局部信号通过，
本startup Probe判PROMISING；不是物理执行成功或sampling因果Validation。
held全panel最近8步约17.22mm、PD XYZ p95约21.14mm；原10mm/.05rad离线
screen仍FAIL且未改。root已对保存的预测/PD arrays独立复算并核对身份、冻结hash。
[结果](../outputs/trajectory-policy/wam-ref2-head-startup-20261012-r1/result.json)。

按预先protocol启动唯一一次r3闭环：同3角色、native配置/R、seed292、frame0、
actor period1，只换固定startup head；剩余整组deadline<=530s计入训练+仿真600s
总预算，单GPU2/当日全部ref2输出+scratch<=100MB。若动作/坐标/身份/资源异常
保存部分trace退出；完整行为由校准teacher/GT-R与WAM保持/放置一起判断。
不从执行结果选checkpoint，不追加训练recipe。

## 2026-10-12 可审计闭环前缀与启动修复方向

用户最新明确专心联调，视频由另一个agent接手；本会话不再修改视频代码或标签。
修复07bf1e6b的相同模型/seed292 r2仍在query172触发原PTv3范围保护，保留172步
实际控制/173帧状态和失败请求。独立pure-array核对实际action/PD逆解/current
R state87/本角色live H4/planned goal/失败请求均max0；冻结R replay encoded
max2.98e-7、requestedPD max4.77e-7。GPU已释放，没有放宽输入工作区。
[原始前缀审计](../outputs/trajectory-policy/wam-ref2-execution-20261012-r2/root_prefix_audit.json)。

已执行前缀teacher/GT-R最长held118/119帧，WAM0；没有完整放置/calibration verdict，
实验仍UNCLEAR。WAM第0步实际teacher PD XYZ差225.27mm，GT-R2.28mm；
state1 wrist偏差25.32mm。WAM基地到物体距离tick64/96/172为.224/.406/1.837m，
其物体留在桌面，teacher/GT在已记录部分已抬升并保持。相同frame0状态仅将WAM
预测前4/8/16步替换GT，立即WAM-vs-GT-R XYZ差225.80→38.53/14.59/4.61mm。
这是privileged离线定位，不是执行成功；偏离状态后整条GT也不能恢复nominal PD。
[输入依赖诊断](../outputs/trajectory-policy/wam-ref2-execution-20261012-r2/R_dependency_audit.json)。

下一最小Decision：沿用同一humanbase和300-update机器人head，冻结backbone/R，
只再训练现有trajectory/hand_query300更新。训练env0:16，固定25%抽tick0:16，
75%抽tick16:519，保持原损失/batch128/lr3e-4；不新增架构/对照或多seed。
固定final，验证startup PD p95<=20mm且real-H4轨迹误差不恶化>25%才视为下一次
短闭环的正向信号。单GPU2训练+最多一次同protocol仿真总<=10min、当日ref2
输出+scratch<=100MB；不选held checkpoint，不改原checkpoint或早期producer。
检验短前缀PD错误是否能由机器人startup coverage修正；不能将继续训练后的改善
直接归因于sampling因果效应。若仍失败，定位真实visited-state教师监督而非扩大
人类主训练步数。初始研究问题/claim和权限资源边界不变。

## 2026-10-12 首次WAM-R闭环触发输入范围保护

固定代码2fb7a8b9、seed292、3角色teacher/GT-R/WAM-R、GPU2原生仿真已启动。
初始q/dq/hand/obj/velocity角色差exact0；planned native reference误差exact0；
首query完整原模型与actor-only桥接预测max0。共返回172次live WAM预测后，
下一次当前手点相对物体超出原PTv3固定工作区（grid y_min=-4<0），server主动
FAIL，native因EOF停止；整组76.90s保护退出，两owned进程已释放GPU。
原运行保留[状态](../outputs/trajectory-policy/wam-ref2-execution-20261012-r1/group_status.json)。
这是闭环状态范围问题的观察，尚无完整episode/calibration或成功率判断。

实现记录缺陷：native异常路径未持久保存内存中的已执行部分轨迹，因而当前无法
独立审计172步的PD和visited states。按AGENTS第14节保留r1，在原实验卡注明
证据受影响；只修partial trace/失败request保存，不改模型、R、采样、seed或PTv3
范围保护。修复review/commit后允许同recipe一次r2，仍单GPU2<=600s、当日所有
ref2输出+scratch<=100MB；资源/身份/有限值冲突停止。用可审计的实际状态确定
是否需机器人startup/visited-state监督，不能把r1日志缺失当作方法负结论。

## 2026-10-12 机器人head校准与最小仿真Decision

固定代码7f053ac9、GPU2完成300更新（61.42s，8304 train windows）。仅训练
trajectory decoder/hand_query；scene/PTv3、goal、action、world、R权重hash保持一致。
冻结64窗held report最近8步86.530→28.025mm，全24步205.414→38.213mm；
WAM-vs-GT-R PD XYZ p95 288.487→199.834mm。真实H4窗口最近8步
64.741→20.773mm、PD XYZ p95 155.074→34.877mm；startup repeated-H4
最近8步仍59.453mm、PD XYZ p95 225.526mm。所有窗口clipping=0。
预先要求两项误差均减半未全部满足，Probe保持UNCLEAR；不能将head拟合或
零clipping当成自主机器人成功。
[校准结果](../outputs/trajectory-policy/wam-ref2-head-calibration-20261012-r1/result.json)。

Decision Note：当前需要区分剩余离线误差是否导致逐tick重规划闭环失效，尤其是
启动阶段。ref2明确要求最小闭环判断控制价值，mapping/frame硬工程门已通过；
10mm/.05rad是保守offline screen，保持FAIL记录，不把它当作仿真禁令。
root选择实现一次最小native仿真Probe，从frame0使用真实测量H4（不足时重复当前
已有首帧）、当前object与已知reference endpoint，不引入未来手/关节状态，不用
teacher prefix隐藏启动问题。固定冻结R/校准head、actor period1；同次保留teacher/
GT-R执行校准。先review接线再启动，不接物理机器人。预计单GPU2不超过10min，
所有当日ref2输出+scratch合计仍<=100MB，不占远程或他人卡。非有限值、错误
applied PD/坐标/身份或资源界限触发停止；校准臂失败则结果UNCLEAR，校准通过
但WAM失败则据真实visited-state轨迹与PD定位下一步，不直接否定共享backbone。
当前授权覆盖可逆仿真，不需要新增外部权限。

## 2026-10-12 短时域审计与优先机器人联调

用户要求先统计最近8步，再并行推进ref2，并最新明确优先WAM→冻结R→PD联调。
本地final10000原1536窗审计：最近1--8步宏EPE13.460mm（0.267s），
中间9--16为50.407mm，最远17--24为85.629mm，全24仍49.832mm。
第1/8/24步2.847/27.585/98.420mm；最近8静止31.574mm、匀速21.911mm。
平均50mm主要受远期拉高，不等于已经达到不可约难度，也不等于机器人能用。
[分时域结果](../outputs/goal-point-wam/goal-memory-ddp-20261012-r3/final_horizon_audit.json)，
[曲线](../outputs/goal-point-wam/goal-memory-ddp-20261012-r3/final_horizon_curve.png)。

ref2初次64窗机器人shadow：active-hand顺序与当前物体frame数值门通过
（max3.662e-7m），PD native往返1.192e-7。人腕/机器人基座等仅语义角色对应，
不宣称解剖位置等价。原人类模型直接用于机器人h1/8/24为12.10/172.40/311.24mm，
最近8为86.53mm；WAM与GT喂给同一冻结R后的PD XYZ差p95=288.67mm，未通过
闭环screen。零clipping受native XYZ1m量程影响，不能等价为控制可用。
进一步shadow混合GT前8/WAM后16，立即PD差p95仍129.88mm；WAM前8/GT后16
为219.54mm；真实H4窗口最近8仍64.74mm。当前优先冻结scene/PTv3与R的机器人
域trajectory-head/observed-hand adapter小Probe，不执行未过门的WAM动作。
[联调卡](../src/task/goal-point-wam/docs/experiments/probes/P-20261012-goal-point-wam-ref2-shadow.md)。

视频点监督接口已实现共享同一PTv3参数，原生可选SE3 head解码到统一点损失，
视频直接逐点轨迹；旧WAM显式权重迁移，新optimizer/checkpoint身份。工程混合
一步/双rank smoke有限且video WORLD梯度进入共享backbone。标签准入仍engineering：
只用过去/当前4帧重新构造输入，但与离线教师当前点位置median33.575mm不一致。
已发现并修复video未来action单位合同（native为位置/当前位移/单步位移，
不是速度/加速度）；原pre-fix日志保留。视频额外研究暂让位联调，不因缺SE3
拒绝点监督，也不将有几何偏差的标签直接投入正式训练。
[视频卡](../src/task/goal-point-wam/docs/experiments/probes/P-20261012-goal-point-wam-video-points.md)。

远程原独立四卡EgoDex-native混合训练保持运行，仍是小subset手部辅助监督，
不是全量EgoDex或新视频点world训练。以下2600步记录为较早运行快照。

## 2026-10-12 本地10000完成与远程约2600步快照

本地r3完成10000步、exit0，四rank final参数SHA一致，optimizer/RNG完整保存，
GPU已释放。1536-window actor宏均值7000/7300/8000/9000/10000为
49.779/49.589/49.740/49.674/49.833mm，后段呈平台，暂不追加训练步数。
最终checkpoint固定同scene的goal/static/zero-hand审计已完成：真实goal相对
static的EPE改善OakInk2/GRAB/ARCTIC为7.77/27.43/23.66%，相对zero-hand为
44.50/50.03/31.27%，三源均通过预定>=5%门槛。本地candidate判为Probe
PROMISING，仅说明该goal-conditioned表示有用，不证明机器人控制或Cm/RL因果收益。
[DDP-scale卡](../src/task/goal-point-wam/docs/experiments/probes/P-20261012-goal-point-wam-ddp-scale.md)。

远程EgoDex混合训练实际RUNNING约2600/10000，global1024，主更新1.42s、
aux.36s，剩余约2.4h加验证保存。2000步EgoDex held-session EPE63.280→48.924mm
（初值下降22.69%），但1000步47.646mm更低，尚无持续改善趋势；static为59.815mm。
main64/source actor macro46.028→45.149mm，cached true goal优势三源仍保持。
远程与本地panel大小和训练条件不同，不横向比较上述绝对误差。

外层phase chain189914因一次NAS du返回1退出，实际scientific launcher190557、
模型四rank一直继续，没有重启训练/重置optimizer。恢复独立CPU监控198540，
combined watcher4161832改为直接依据实际launcher状态判断和保存停止，避免
误把chain FAILED当作训练结束；原失败日志保留。近期combined296.40GB<298GB
主动阈值/300GB硬边界，NAS4.406GB<5.1GB。独立监控能容忍单次统计失败。
[实时快照](../tmp/egodex-training-20261012-r1/status_update-20261012-1430.json)。
以下是较早快照；当前状态以本节为准。

## 2026-10-12 本地7300续训与远程原生EgoDex混合训练

本地主训练r2因他人PID4084729在GPU0--3建立CUDA context，保护性保存停在7300，
优化器/四rank RNG完整，四rank最终参数SHA一致；从未干预他人进程。用户先确认
他人任务，后明确告知临时占用结束并要求继续。复查所有本地卡空闲，r3现GPU0--3
从final7300恢复同一个模型/优化器/RNG，保持原始绝对deadline；实际推进至7410，
梯度范数四rank一致，约1.07s/update，余下约47min加验证/保存。GPU4--7保留空闲。
固定1536-window actor macro4000/5000/6000/7000/7300分别52.516/51.391/50.387/
49.779/49.589mm，仍下降但减速，尚不能判定平台期或替代最终goal-ablation gate。
[DDP-scale卡](../src/task/goal-point-wam/docs/experiments/probes/P-20261012-goal-point-wam-ddp-scale.md)。

用户要求先本地小测后加入远程EgoDex。新流程只接入官方TRAIN原生HDF5手部/相机/
置信度，按session分割、固定当前camera frame、缺失关键点mask；不使用离线全片
teacher，不伪造object goal/world标签。3次原主三域完整WAM更新+1次EgoDex actor-only
更新，共用同一模型，EgoDex跳过world/action forward。局部200步单卡通过，EgoDex
单held-session EPE108.168→99.366mm（8.14%），main小panel宏均值基本保持，
ARCTIC约恶化1.9%；这是局部正向信号，未证明数据因果增益或机器人收益。

远程原三域训练已在2100步完整保存。原生EgoDex24/26记录通过，6484帧、
4476train/1305dev windows，5held sessions。远程新code0fd399e3，4×4090单模型
每rank256/global1024/10000步，parent为远程原main final2100，optimizer reset。
初次四卡工程测试训练4步后因rank0-only metrics误用于全rank最终验证分支，
导致NCCL collective错序；失败日志保留，修复为所有rank一致的last_evaluation_step。
修复后完整batch测试4步通过，四rank最终参数SHA一致；mixed正式训练实际通过10步
并到18步，四rank梯度一致，auxiliary world loss为0，GPU快照全4卡100%，
显存35211--45569/49140MiB。初步主更新1.41s/aux.35s，余下约3.2h加验证保存。

这只是现有短片原生EgoDex子集，不是完整EgoDex大语料；大batch暴露重复4476窗口。
完整327GB归档FAILED，未用于训练，未新增下载。object-goal/world监督继续只来自
OakInk2/GRAB/ARCTIC；native aux不改变global claim。所有结论仍Probe/UNCLEAR。
[原生EgoDex卡](../src/task/goal-point-wam/docs/experiments/probes/P-20261012-goal-point-wam-egodex-auxiliary.md)。

资源：用户授权本地主4+远端4，不恢复额外方法；新本地段与远程混合模型均有guard。
保留所有科学checkpoint，记录SHA后只删除本地4步DDP engineering smoke的可重现
checkpoint，释放2.513GB。跨主机watcher4108465覆盖local r3与remote chain189914，
近期合计294.46GB，298GB主动保存停止/300GB硬上限不变。新NAS子预算由root记录
DecisionNote调为5.1GB（原5GB为root分配），代码sparse checkout降低原子保存峰值，
没有改变用户总存储限额。远程科学launcher190557；后台训练不依赖SSH保持连接，
监控访问需要Windows隧道。以下旧时间快照保留作历史，当前状态以本节为准。

## 2026-10-12 本地额外卡释放与远程四卡训练

按最新指示，额外方法在GPU6/7按两波恢复optimizer/RNG，GPU4/5释放；四方法
均完成固定3000步，CPU哈希绑定比较已完成。自有额外GPU4--7作业均已释放；
GPU5已有他人作业，不干预。主训练仍GPU0--3，12:50快照5520/10000步，
最近100步中位1.076s/update，余下训练约80min加验证/保存。固定1536-window
panel第1000→5000步macro actor EPE57.272→51.391mm（降低10.27%），
属于同一次训练内的趋势，不是方法因果结论；不再为占利用率启动额外作业。
[P-ref12-methods](../src/task/goal-point-wam/docs/experiments/probes/P-20261012-goal-point-wam-ref12-methods.md)
四候选局部PROMISING，actor三个候选只在OakInk2/GRAB通过完整门槛，ARCTIC
相对parent不足5%；ROI world三来源通过，但joint→world-only和单物体归一化
混杂未消除。Change head在OakInk2比静态差14.95%；不能据actor增益声称aux有效。
额外训练/新头/共享已看panel没有matched-duration control，因果与正式claim仍未验证。

用户明确授权尝试远程大规模训练。已在NAS独立root迁移1.8776GB、24479文件，
逐文件SHA通过，4,378,478 train窗口/源indices/stats/vendor不变；仅新manifest
的路径与panel绑定改变，原字节保留。远端既有项目和数据未修改、依赖未重装。
四4090/48GiB，graspenv torch2.0.1+cu118；一个四rank模型，strict PW50000初始化，
hand+memory goal/world1，每rank256/global1024/10000步（10.24M暴露），同seed275。
这是额外远程更大batch执行，不是多seed/control，也不迁移本地optimizer。

remote-memory-ddp-20261012-r1 初次preflight因NAS TMPDIR使Unix socket路径110B
而阻塞step0；保留日志，已停止所有自有worker。最小Listener复现长路径失败、
同NAS scope/t短路径通过。代码f120642b新增remote专用launcher，修复preflight和
正式worker TMPDIR，不修改本地live guarded源码。run-r2四rank全batch4步通过，
final参数SHA一致，正式训练推进到560步且loss/梯度有限，最近100步中位
1.426s/update（global1024），约718样本/s；本地global512约476样本/s。
四卡此前20s采样平均利用率85.5--87%，显存约37--46GiB/48GiB；
预计余下约3.7h加评估/保存。实际NAS数据加载CPU-only同batch/4worker
稳态平均0.478s/batch，供给能力快于1.426s/update；不是训练内实测等待时间。
训练不经Windows SSH读数据。四卡直接P2P不可用、跨两NUMA，GPU2当前PCIe4x8；
这些是同步开销候选，尚无kernel/NCCL分段时间，不能量化计算/通信占比。
输出：`/mnt/ugreen_nas/storage/oyx_storage/goal-point-wam-remote-20261012-r1/repo/outputs/goal-point-wam/remote-memory-ddp-20261012-r1/run-r2/`。
[本地远程运行快照](../tmp/remote-training-20261012-r1/remote_observation-r2.json)。

资源Decision：此次用户临时允许本地主4+额外2与远端4；额外已结束，现主4+
远端4。远端训练<=6h/1440GPUmin、整个新NAS scope<=5GB；preflight保存数值/
身份/parameter/checkpoint哈希后删除自有可重现工程cp，为科学rolling cp留余量。
跨主机watcher4026316计入本地outputs+tmp和整个新NAS scope，combined298GB保存
停止自有supervisor；远端独立source/foreign-GPU/deadline/5GB守卫持续运行。
最近combined~294.94GB<300GB。SSH断开不会结束后台训练，状态获取仍需Windows隧道。

## 2026-10-12 ref1_2 文献方法并行探索

用户明确授权使用剩余卡继续ref1_2正文的方法。临时最多8卡：主训练独占GPU0--3；
新探索只用空闲GPU4--7，不修改主训练监控中的源码、配置、checkpoint或进程。
新增预算<=90min wall / 360GPUmin、<=8GiB产物，所有任务合计仍<300GB；
298GB触发自有任务保存停止。今日这轮探索结束后恢复Campaign默认4卡边界。
问题：动作条件world能否区分候选、区域加权/多达成时刻goal重标注/冻结world闭环/
未来变化辅助监督是否值得继续。以旧memory+hand/world1 final1200暖启动，单seed；
不启动新的多seed或主训练对照。新时间面板生成因未看stratum不足而失败，未读取模型
结果；改为固定既有DDP面板的768窗口子集，明确作为共享探索评估而非独立Validation。
action→effect核对：cached repeat noise0、梯度非零，moving预测比identity好，但
GRAB/ARCTIC误差仍113/79mm，五候选中真实动作模型排序第一仅10--14%，没有
未执行候选的真实标签。ROI forward、合法prefix增广、中间几何plan、future-change
四方法已各batch64/3000步暖启动旧final1200；冻结world的L_goal仅通过梯度smoke，
不升级为完整训练。四方法/修正aux smoke均完成，标签不进入cached actor。GPU4--7正式作业已RUNNING，
约.44--.48s/步；CPU后处理在完成后自动绑定final3000、parent/panel哈希生成comparison，
科学解释仍待实际结果复核。原四卡DDP主训练继续RUNNING。

## 2026-10-12 单模型四卡规模训练运行中

用户明确取消多seed/对照，要求一个模型四卡分布式训练；最新指示直接训练，
不再为提高利用率追加性能调优。
[P-20261012-goal-point-wam-ddp-scale](../src/task/goal-point-wam/docs/experiments/probes/P-20261012-goal-point-wam-ddp-scale.md)
以严格PW50000初始化，训练代码cc434265，memory-goal+当前hand XYZ/history+
world1；每rank batch128、global512、10000步/512万样本暴露，seed275/抽样276/
评估277。四rank小batch/完整batch smoke均通过同步参数SHA相等；独立实现复核
已修复DataLoader iterator创建前恢复CPU RNG导致resume序列偏移的问题。

r1在第400步保存后暂停；Hilbert融合无收益，采样优化草稿也未采用。按最新指示，
r2已在GPUs0/1/2/3从原checkpoint恢复同一个模型、优化器及all-rank RNG，保留
原6h绝对deadline，停止性能测试。第616步loss/梯度有限，四rank最近共享步610梯度范数一致，约1.14s/步，预计剩余
训练约3h，另加评估/保存时间。单次GPU快照四卡100%，显存约18--22.5GiB；
不把瞬时利用率当稳定性能结论。aggregate artifacts283.10GB<300GB。

1536-window面板固定且避开本Task历史已看窗口共享帧，仍复用val序列/PW选择
数据，不是独立新受试者评测；每1000步计数加权分布式评估。取消的6臂没有训练，
panel449未观察结果且不用。上限6h/1440GPUmin、新产物<=6GiB，监控仅管理
自有进程，每30s核对source/foreign GPU冲突、298GB主动停止。训练结束前保持
UNCLEAR，不形成多seed/因果或RL结论。
[实时状态](../outputs/goal-point-wam/goal-memory-ddp-20261012-r2/group_status.json)
与[训练日志](../outputs/goal-point-wam/goal-memory-ddp-20261012-r2/console.log)。

## 2026-10-12 ref1_2 扩大预算与手部条件结果

按用户授权并行使用8张空闲GPU，完成原始memory/query/none和5个显式hand/world-loss
对照，每臂batch64、1200更新、76,800样本暴露、384平衡val；同数据/统计/train-val
indices，旧新核心初值完全匹配。正式训练代码a90b57eb/fbb33460。
[goal路径卡](../src/task/goal-point-wam/docs/experiments/probes/P-20261012-goal-point-wam-goal-injection.md)
中query准确率没有超过memory，原query优越性门槛 **UNCLEAR**；较大预算下memory
相对no-goal在GRAB/ARCTIC改善10.3%/9.5%，不再是几微米goal响应。

[手部/goal/world-loss卡](../src/task/goal-point-wam/docs/experiments/probes/P-20261012-goal-point-wam-hand-goal-interaction.md)
显式当前手XYZ/历史动态query使memory EPE从25.755/79.587/61.866降至
24.056/73.958/56.594mm，改善6.60%/7.07%/8.52%，当前手信息路径局部
**PROMISING**。固定hand/world1后memory goal在GRAB/ARCTIC相对no-goal改善
14.56%/14.51%，相对自身静态goal改善15.88%/12.34%，goal utility局部
**PROMISING**；OakInk2仍未达到goal门槛。query goal也有收益但三个源均稍差于
memory。关闭world loss收益仅约0.3--2.5%，未过5%门槛，保持 **UNCLEAR**。

原whole-forward隔离门槛失败，原结果保留；相同输入重复forward也发生数值变化。
8个补充final-checkpoint诊断固定scene/执行action后world变化0，goal梯度边不存在。
这只支持信息隔离，不能作为物理可靠性或原门槛通过证据。冻结goal评估缓存同一scene，
EPE与训练final一致<=1e-5m。保留“显式hand+memory goal+world1”用于后续独立
多seed Validation，停止query-only/world-off调参；L_goal仍待forward action可靠性。
训练总105.74GPUmin，约10.1GiB/card，outputs+tmp279.873GB<300GB。
所有本轮自有GPU任务已结束；后来GPU2/3出现外部作业，补充诊断避开。临时8卡
授权本轮已停止使用。既有15 tests passed/1 optional CUDA skipped；没有RL/Cm结论。

## 2026-10-12 等待实验期间的远程与数据核对

Windows反向隧道恢复，严格SSH已能只读访问远端NAS；无需重配密钥，保持隧道窗口。
[远程库存笔记](../src/task/goal-point-wam/docs/research/2026-10-12-remote-data-access.md)
记录限定两层目录未见touch/EPIC包，不能推广为整个NAS无数据。已有48个EgoDex
工程clip明确training_allowed=false，手点10/11、时钟插值、整段离线教师非因果输入、
目标对象/刚体标签均需适配；未下载、写远端或使用远端GPU。
[Touch/EPIC适配笔记](../src/task/goal-point-wam/docs/research/2026-10-12-touch-epic-data-fit.md)
建议优先对T-Rex做少量episode合同审计用于独立辅助表征；它不提供现行对象运动
完整合同。原EPIC本地34clips有0个合格H4+K24手窗口，扩大batch无法补齐监督。
数据研究不改变本轮主线结论，也未启动外部数据训练。

## 2026-10-12 显式手几何学习诊断

用户确认可能没有新修改并授权继续自主探索。
[历史外推筛查](../src/task/goal-point-wam/docs/experiments/probes/P-20261012-goal-point-wam-hand-query.md)
当前速度/加速度/三步位移与raw重建误差0；常速度只在OakInk2有效，GRAB/
ARCTIC更差，多来源门槛失败，保持UNCLEAR，原条件式full-WAM query训练未启动。

[紧凑预测器](../src/task/goal-point-wam/docs/experiments/probes/P-20261012-goal-point-wam-compact-forecast.md)
用直接当前手XYZ/历史动态/当前物体几何，两臂同23,851参数/初始化、1200个真实
train窗口、600次batch2更新和既有96val。H-only相对不动手改善24.24%/7.50%/
9.40%，局部 **PROMISING for observed-input prediction**；H+object-goal相对
H-only变化-1.38%/+1.79%/+3.05%，未过goal增益门槛，goal utility仍UNCLEAR。
这支持先定位WAM的信息接线/学习差距，不能单独归因PTv3或world loss，也不形成
机器人/Cm/RL结论。下一步匹配WAM显式hand XYZ/history query开关，随后再隔离
world loss；不追加单位或常速度prior调参。现有val已多次观察，正式Validation
需独立证据。代码b24dc66a，13 tests passed/1 skipped，GPU2诊断总17.38s、
峰值采样373MiB、新产物约10.4MiB；本轮自有GPU任务全部结束。


## 2026-10-12 Goal 几何条件自主探索

[受控多goal诊断](../src/task/goal-point-wam/docs/experiments/probes/P-20261011-goal-point-wam-controlled-goal.md)
保持六train场景不变，用解析平移标签迫使模型使用goal；只训练goal/trajectory，
world冻结且无loss。米单位响应增益约.09--.12，.05m单位约.98--1.04，但
完整<=5mm门槛均失败，**UNCLEAR**。这只是工程条件学习信号，人工标签权重
从未进入真实训练。[配对位移对照](../src/task/goal-point-wam/docs/experiments/probes/P-20261012-goal-point-wam-paired-geometry.md)
同单位下endpoint与start/delta仅约2.3%误差差异，未达门槛，停止解析表示调参。

[真实尺度对照](../src/task/goal-point-wam/docs/experiments/probes/P-20261012-goal-point-wam-real-goal-scale.md)
从严格PW迁移+新头重新初始化，两臂相同参数/数据/seed/96val窗口，仅goal单位
不同；各600步，代码1310c47e。原始/缩放trajectory EPE为35.59/35.67、
141.64/141.69、71.86/71.88mm，均稍差于不动手基线；替换为静态物体goal
只改变预测约3--7微米。物体平移/刚体随动参照没有多来源明显收益。整体
**UNCLEAR**，停止单位与epoch堆叠，下一步先区分当前手空间/历史表示与条件
学习问题，不据此放弃物体goal/WAM/Cm或形成RL结论。保留可选诊断代码及原始
结果，未修改默认配置。11 tests passed/1 skipped；GPU2真实训练总约604s、
峰值采样1513MiB，所有自有训练与评估完成；新增产物<5GiB，总产物<300GB。


## 2026-10-11 ref1_1 物体点集 goal Probe

按用户 `goal-point-wam/docs/user/ref/ref1_1.md` 指定的语义，在 `39700aba` 增加
object start/goal 点集合同：当前 target flag 选择物体，32对点始终位于单一当前
anchor frame，trajectory 不读取 hand endpoint goal；world 仍 goal-blind，PW
50,436,224参数迁移和528动作tokens保留。历史hand proxy配置仍可读取。
[物体 goal 实验卡](../src/task/goal-point-wam/docs/experiments/probes/P-20261011-goal-point-wam-object-goal-response.md)：
7合同测试通过、1跳过；首个smoke的局部/全局sequence ID检查错误由 `66f42def`
修复，失败保留且未进入训练。后续4步smoke与300步六train窗口Probe完成，原始
物体endpoint重建最大2.98e-8m。trajectory拟合2.16/2.60/3.08mm，但 +/-10cm
物体goal干预仅约.034mm末端RMS响应，静态物体goal替换后EPE几乎不变。
冻结checkpoint重载、真实object输入接线和有限非零goal坐标梯度核验通过；缓存
同一scene的world变化为0，whole-forward重复噪声超1e-5仍记录，不冒充gate通过。
整体 **UNCLEAR**，仅六训练窗口的机制筛查，不是物体goal方案无效或RL结论。
停止扩大训练，下一步区分几何尺度/decoder条件表示/同scene目标变化不足。
GPU2用时62.8秒，新增约406MiB，所有自有GPU任务结束；Mission/Campaign与用户
草稿未修改，没有接入机器人、DAgger、RL、视频或启动mixed-source规模训练。

## 2026-10-11 WAM 学习能力与目标响应诊断

[固定训练窗口诊断](../src/task/goal-point-wam/docs/experiments/probes/P-20261011-goal-point-wam-learnability.md)
在六个 train 窗口完成独立标签重建（误差0）、共享梯度与 oracle loss 核对。
300步 lr1e-3 在 eval/train 两种 forward mode 均通过三来源拟合门槛，局部
**PROMISING**；这只表明固定输入可拟合，不证明条件信息使用或泛化。
[单变量学习率 Probe](../src/task/goal-point-wam/docs/experiments/probes/P-20261011-goal-point-wam-optimizer-scale.md)
复用原始初始化/seed/96个 val 窗口完成600步，best500；moving-world 相对静止
改善为 -.429%/+.017%/-.157%，未过5%门槛，仍 **UNCLEAR**。线性 start-goal
轨迹明显优于当前学习预测，因此停止只加学习率/epoch 的路线。

冻结六训练窗口的 +10cm X goal 干预响应很弱，包括低拟合误差模型。固定fit
checkpoint 的 whole-forward world 隔离1e-5门槛失败，同时不改输入的重复误差
也越界；缓存同一scene后，三个模型 world goal-change/repeat 均为0。原失败
标记和日志保留，不将数值重复噪声直接归因为goal泄漏。token范数和attention
诊断未支持scene幅度压制goal/action的猜想；没有据此改归一化或架构。
下一步应检查goal几何/边界合同；本次按用户要求先提交，未启动后续实验。
所有自有GPU任务结束；未改变最终claim、资源边界、Mission/Campaign或用户草稿。

## 2026-10-11 Goal-Point-WAM ref1 修订与短 Probe

接续前会话已授权的修订，在 `main` 的 `412c6aef` 复用修复后 PW 的 PTv3/scene
projection（454 tensors、50,436,224 parameters），world decoder 接入全部528个
逐点执行动作 token 与 mask。头和不同结构的 object projection 保持新初始化；
H328/R/PD/video 仍属后续阶段。训练未修改 Mission/Campaign 或用户草稿。

[ref1 alignment 实验卡](../src/task/goal-point-wam/docs/experiments/probes/P-20261011-goal-point-wam-ref1-alignment.md)：
合同测试5 passed/1 skipped；4步GPU smoke 的两分支共享梯度、goal isolation、
checkpoint replay 通过。GPU2 的600步训练完成，用时273.4秒，best为400步，
验证窗口与旧run exact；显存峰值采样1.48GiB，利用率3--26%。
评估入口读取不存在的 `MixedWindows.source_names`，由 `f1eb97b7` 修复为manifest
字段后完整补评，原始失败日志与训练checkpoint保留，未补跑训练。

同一96-window/source panel上，OakInk2 test/GRAB test/ARCTIC val diagnostic 的
moving-world EPE 相对静止基线只改善1.14%/0.80%/0.55%，没有来源达到预注册5%
门槛；trajectory 相对重放旧模型变化+0.20%/-0.43%/-0.26%。结论 **UNCLEAR**：
结构偏差已修正，但短 Probe 尚未形成 world/trajectory 收益证据，不反驳完整WAM
或Cm/RL claim。停止追加epoch与机器人/视频接入；下一步优先训练集固定小batch
学习能力诊断，分清目标/优化问题与短预算/来源覆盖问题，出现正向信号后再做
matched world-loss-off。新实验、仿真和Validation未启动。

## 2026-10-11 完整合并与 PW 工作树退役 Decision Note

用户授权合入 `cm-pointflow-effect-pretrain` 全部分支内容并删除旧工作树及
`epic-scene-flow`/`ref8` 两个软链接。固定源提交 `23241463`，新增两条实验记录；
5 份未提交草稿保留到主工作区，outputs 全量核对同步，tmp 和独立 PointWorld clone
归档。源 HEAD/内容漂移、进程占用或来源校验失败时停止删除。Git 分支与历史保留。
共享 corpus 原 manifest 使用旧绝对路径，因此创建独立 `data-main-worktree` 副本，
只迁移路径，保留数据 hash、split 与 normalization identity；历史产物不改写。
详见[整合和复现路径说明](archive/2026-10-11-worktree-retirement.md)。不改 Campaign，
不新增训练或改变最终 RL claim。

整合提交 `862a2ee5` 已包含源分支全部提交，工作树和两个别名已删除；Git 仅登记
主工作树，源分支仍保留。58,510 个保留路径核对通过，文档合同 PASS，
Goal-Point-WAM 测试 3 passed / 1 skipped。新 corpus 的六次 train/val 源样本读取
通过，原始 artifact manifest 和 checkpoint 不改写，用户草稿保持未提交状态。

源分支的[Goal-Point-WAM 600-update Probe](../src/task/goal-point-wam/docs/experiments/probes/P-20261011-goal-point-wam-three-domain-probe.md)
已完成，整体 UNCLEAR：小型 held-out panel 尚未出现明确 world-motion 静态基线收益，
ARCTIC 无 test windows，其 held-out 数字属于 val diagnostic。下一步仍需 source/
representation 和 held-out contract 诊断；本次整合不将结果升级为正式结论。

对照用户 ref1 的实现复核补充：该 Probe 未迁移 PW 预训练权重，world decoder
实际使用 24 个每帧平均动作 token，而非构造出的 528 个逐点语义 token；这两项
尚未落实 ref1 的关键选择，不能用当前无收益结果反驳完整方案。Goal 当前是 hand
endpoint proxy，H328/R/PD/video 按阶段暂缓。实验卡已纠正 trajectory EPE 为
24 步平均值（world EPE 才是 h24），保留原数值和 UNCLEAR 标签。先对齐上述
初始化与动作记忆，再评价是否追加训练；world loss 对轨迹分支的收益还需要 matched
world-loss-off 对照。未启动新的训练，也未修改模型行为。

## 2026-10-11 存储清理 Decision Note

用户明确授权删除已审查的第一批：30 个旧版展开数据文件（65.3 GB）与 ref5
pip 缓存（0.84 GB）。数据对应的 18 个原始 transition 分片仅 1.18 GB，
此前 SHA256 均匹配；关键正式实验数据、qfix 版本、原始分片与 checkpoint 保留。
删除前归档完整生成元数据、参数、来源 hash、代码身份与清单，复核文件身份和
进程占用。只处理清单路径，身份漂移、来源 hash 不匹配或发现占用时停止对应操作。
这是用户授权的不可逆本地清理，不改变科研结论或 Campaign 上限，也不启动训练。
完整记录位于 `outputs/archive/cleanup-20261011/`；原始输入匹配不意味着已证明旧
展开数据能逐字节重建。历史生成实现仍可从 Git 与归档的生成记录追溯。

执行已完成：删除 30 个展开数据文件及 204 个缓存文件，实际释放 66.154 GB。
18 个原始分片重新核对 SHA256 通过，16 个保留数据文件身份核对通过。
两个 worktree 的 outputs/tmp 按硬链接去重后约 251.35 GB，已回到 300 GB 上限以内。
[删除清单及核验记录](../outputs/archive/cleanup-20261011/manifest.json)
与生成元数据、代码历史、sidecar 摘要一起原位留存；用户 Mission/Campaign 未改动。

## 2026-10-11 PW 主实现整合与旧路线归档

按用户授权，以 `cm-pointflow-effect-pretrain` 固定提交 `5f12ae31` 为 PW
主实现合并，保留本页已有 DExplore/DAgger 进展。旧 PW 数据、checkpoint 和
pilot 原位归档，历史实现由 `archive/pw-premerge-20261011` 保留；
详见[归档与复现说明](archive/2026-10-11-pointworld-premerge.md)。
新实现不能直接复算旧 voxel 语义的 checkpoint；来源校验不放宽。

修复后 sharedstats 基线及 PointWAM 的局部状态见
[预训练 Task 状态](../src/task/cm-pointflow-effect-pretrain/docs/STATE.md)。
[Goal-Point-WAM 工程 smoke](../src/task/goal-point-wam/docs/experiments/probes/P-20261011-goal-point-wam-three-domain-smoke.md)
已通过有限梯度、checkpoint 和 world 分支 goal 隔离检查，尚无泛化或 RL 收益结论。
最终目标仍为 RL；本次只整合代码/证据和同步产物，不启动训练，不改 Mission/Campaign。
既有两个 worktree 产物去重约 313.6 GB 已超 300 GB；硬链接同步不新增大文件占用，
后续新增大规模产物前需处理既有超额。

## 2026-10-11 官方 DExplore corrected12 实际表现筛查

用户暂缓 WAM/PW，优先评估当前参考库。
[官方 actor 跨参考 Probe](../src/task/trajectory-policy/docs/experiments/probes/P-20261011-official-reference-screen.md)
完成同checkpoint/seed191、每条8个frame0完整回放：12条共90/96稳定持物、68/96
受控放置。s3/s9飞机、牙膏、水瓶均8/8持物+放置，是较好教师候选；但位置RMSE
仍5–7.5cm，不能称精准参考跟踪。Retake飞机/鸭子放置1/8和2/8；手机持物/放置
均2/8、位置RMSE1.56m。整体质量UNCLEAR，仅当前固定初始状态Probe。
完整有限轨迹、参考来源、native实际PD和指标重算审计PASS，PD误差0；新评估数据
training_allowed=false。GPU任务已结束，新增272MiB。未启动WAM训练；用户确认最终
目标仍为RL，当前不改Mission/Campaign或最终claim。

## 2026-10-11 ref3：competent teacher 真实 PD 分层路线

用户已确认官方 DExplore actor 作为 teacher、监督实际下发的 PD target，R 每步读取
真实测量反馈。后续已明确允许 G 使用 t+1/t+16 人手和物体计划参考，并允许 DAgger；
早先仅物体终目标 G12 是 agent 的限制假设，现已纠正。保留完整24步手轨迹，当前不接 PointWorld。
新训练数据是明确可训练的新采集，旧工程 packet 的禁训标记保持原样。

[P-20261011-competent-pd-retargeter](../src/task/trajectory-policy/docs/experiments/probes/P-20261011-competent-pd-retargeter.md)：
32/32 teacher 完成 hold45 和受控放置。实际 PD R 留出误差比 state-only 降低59%，
原5mm/.05rad离线门槛仍失败，原卡 UNCLEAR。另行冻结的
[物理 Decision](../src/task/trajectory-policy/docs/experiments/probes/P-20261011-competent-pd-physical.md)
`eaeb81f` 比较 teacher、source-PD、GT-tau/R，各8/8 hold45及受控放置；R median436帧，
裁剪0.0461%，独立实际PD/状态/轨迹/预测审计 exact，局部 PROMISING。
这只是 oracle实际未来手轨迹+learned R，不是自主高层策略。

[联合 BC](../src/task/trajectory-policy/docs/experiments/probes/P-20261011-competent-joint-bc.md)
`b00a533` 完成 matched H328/G12->24x33 轨迹监督和轨迹+真实PD联合监督，冻结 R。
3000updates/22.97s/GPU0，启动 first8 coordinate RMS0.922/0.693mm；联合 PD val MSE
比轨迹监督低29%。独立历史/目标/标签/预测及梯度审计通过，33个 Task 测试通过。
完整自主物理评估 r3 `6a75a26` 已完成：teacher/GT-R各4/4，两个高层各0/8持物，
clip4.36%/8.86%，独立H/G/计划/R/actualPD/outcome审计通过，局部 UNPROMISING。
r1相对checkpoint路径错误在控制前失败（`f4e1881`修复），r2因外来GPU上下文中断，
均非科研负结果。冻结验证状态诊断 `8cce545` 显示 joint 接触段PD XYZ p95由每步查询
29.33mm增至8步计划复用60.56mm，尚未证明闭环原因；当前按
[单变量 cadence Probe](../src/task/trajectory-policy/docs/experiments/probes/P-20261011-competent-actor-cadence.md)
保持checkpoint/seed/判据，仅把查询周期8->1。`c897d745` 完成74.72s并独立审计通过：
teacher4/4，GT-R4/4hold+3/4放置，纯轨迹0/8hold，联合PD4/8hold（最长311/309/427/427帧），
但完整成功0/8、clip2.7445%、median159帧，固定整体gate仍 UNPROMISING。
这是联合监督+每步反馈的局部抓持线索，不是合格 baseline；部分失物，部分静态持物
而未完成参考波形/放置。所有自有GPU任务已结束，route产物约240MiB。
冻结H/G覆盖诊断 `9cb2686b` 发现fresh teacher接触状态本身也偏离旧训练域，联合后段
偏离更明显，不能只归因为student covariate shift。下一步优先最小接触/后段数据覆盖
对照，再评价实际完整任务，不直接加epochs、用时钟/参考泄漏或接PointWorld。
教师末端物体位置相对 G 平均误差143.98mm；当前 hold/放置指标不证明精准目标位姿
或完整参考轨迹跟踪，更没有 Cm-on/off 训练收益。本路线预算共用20GPUmin/1GiB，
不改 Campaign，不新建分支、不push。

[参考条件/PD DAgger Probe](../src/task/trajectory-policy/docs/experiments/probes/P-20261011-reference-goal-dagger.md)：
H328/G408参考位姿与 native hoi_data 比较 exact。BC3000updates/28.10s，joint val
PD MSE .02297、启动RMS .618mm，独立审计通过。首次自主物理执行完整测量已保存，
汇总数组维度错误在7f67d38f修复；原FAILED记录保留，另行恢复汇总并独立审计通过。
自主0/8hold/放置；teacher4/4hold但1/4放置未过校准，GT-R4/4hold、3/4放置，
首轮 UNCLEAR。随后完成一轮 beta=.25 训练期PD查询与matched continuation：4336查询、
实际干预24.42%、采集7/8放置，训练3000updates/36.83s；完全无干预 seed186 评估
PD DAgger8/8hold+放置、median376帧、clip0%、物体参考RMSE7.57cm，teacher4/4、RMSE6.97cm，
GT-R4/4、median436。仅offline joint续训7/8hold、6/8放置、median92帧、RMSE75.69cm。
完整输入/模型/R/下发PD/outcome审计通过，局部 PROMISING。保持固定R与参考anchors，
下一步冻结matched重复/Validation，不继续aggregation扫描或接PointWorld。单motion/
单评估seed和teacher首次live变动限制了结论；query-val失败env23恢复仍未学会。
新产物452.1MiB，35 Task tests通过，所有自有GPU任务已结束，不改 Campaign。

[冻结双seed Validation](../src/task/trajectory-policy/docs/experiments/validations/VAL-20261011-reference-goal-dagger-repeat.md)
已完成：holdout400/401的DAgger均8/8hold+放置、零裁剪、median376/405帧、
参考位置RMSE7.58/7.52cm；offline continuation放置5/8、6/8，median均92帧，
参考RMSE80.78/76.74cm。每轮teacher/GT-R均4/4，完整GPU执行审计和独立gate复算通过。
原六项门槛在两轮均通过，局部固定模型/参考/初始化重复性 claim SUPPORTED；
不是DAgger训练seed泛化或Cm收益。不同seed的真实初始q/物体/手/速度全部exact相同，
所以这是独立native启动的重复验证，尚未测试初始化扰动。新增156.8MiB，未再训练，
所有自有GPU进程已结束。下一步保留该固定任务baseline、进入Cm/PointWorld接入设计；
扰动/多motion/训练seed证据延后，不继续无界优化无Cm策略。Campaign仍未由agent修改。

## 2026-10-10 用户转入 trajectory-policy 新架构设计

用户要求consequence-evaluator先做到这里，不再运行其后续实验，改看新Task
[ref1](../src/task/trajectory-policy/docs/user/ref/ref1.md)。新路线采用独立绝对
trajectory latent actor、固定D/R、真实奖励PPO baseline，之后检验WM动作条件
physical token是否改善策略训练。根级Mission/Cm claim不变，新Task已训练BC初始化及两轮短PPO，尚未获得可用高层策略或RL收益。
已实现固定48维D，2个微型合同测试通过；首轮
[D/R覆盖Probe](../src/task/trajectory-policy/docs/experiments/probes/P-20261010-trajectory-decoder-coverage.md)
ce503af native16env/542步完成57.29s：原GT3/4长时终末held，dense FK4/4；
四节点48D两组均0/4终末held（其中一行曾持有397帧），局部UNPROMISING。
压缩前8步手点RMS13.97mm、掌部max84.25mm，最大误差在启动窗口；
独立审查未见SO3/时序/FF接线错误，输入2.86e-6、命令2.38e-7、PD0。
5fe3d05仅修复audit JSON序列化，未重跑仿真。下一步先核对c拟合是否能保留
前缀位置和速度，再决定增加D表达能力；不据此否定48D或整体trajectory-policy。
本轮没有高层actor训练/WM，GPU2已释放。
后续7a0a21d前缀最优拟合：四节点位置RMS下界最差32.20mm、FF30.67mm，
只证实当前轨迹复现限制，不排除别的成功策略。772190e固定PCA D48离线fit完成：
first8手点RMS5.61mm、掌部max23.62mm、FF最差RMS18.50mm，未过预声明几何screen，
按协议没有启动仿真。后续7e7e909 pose/FF metric D48离线手点RMS0.919mm通过，
55.17s原生执行GT/dense各4/4，metric两组2/4和0/4长时终末held，局部UNPROMISING。
八行均先抓住，六行随后失持；在线live q/object前缀RMS8.017mm，与离线输入域不同。
4be7d90仅修复audit accumulator，独立实际c/输入/命令/PD审计通过，未重跑仿真。
按预声明停止48D重建搜索，下一步D288接口等价与H→c学习；actor/Cm仍未训练。
457c65f D288已对68x4条既有dense计划完成工程回放：q max5.96e-7、hand3.58e-7m、
velocity1.97e-5、wrist FF2.35e-6，GPU4约0.975s；8个Task合同测试通过。
纯接口核对，无新物理执行、无actor学习；下一步纯测量H→独立c初始化，再真实奖励RL。
证据：[metric D](../src/task/trajectory-policy/docs/experiments/probes/P-20261010-metric-trajectory-decoder.md)。
证据：[prefix fitting](../src/task/trajectory-policy/docs/experiments/probes/P-20261010-decoder-prefix-fitting.md)、
[lowrank D](../src/task/trajectory-policy/docs/experiments/probes/P-20261010-lowrank-trajectory-decoder.md)。
首轮独立H328->c288初始化68fccf9：2500update/14s，val目标比常量低80%，但
完整54.96s执行GT4/4、dense3/4、两个H组均0/4稳定抓持。启动H在训练域且输入exact，
首查询手点RMS56.36mm；初始预测错误早于反馈，不只是假设闭环分布偏移。
4af8345 audit匹配rl_games实际TF32后H/c/D/FK exact、Rfeatures3.81e-6、command2.38e-7、
PD0，保留原precision mismatch失败记录，无物理重跑。局部UNPROMISING，未PPO/WM。
c7d991b startup-balanced初始化13.14s完成：启动手点RMS7.395mm、掌部max15.61mm、
FF RMS5.464mm。改善但仍未过固定5/10mm门槛，局部UNPROMISING；按协议无新增仿真。
停止BC权重搜索，下一步真实奖励trajectory PPO，固定D/R、纯H，balanced500仅未验证
warm start；不以oracle数据/监督loss替代实际RL或Cm收益。
证据：[history actor](../src/task/trajectory-policy/docs/experiments/probes/P-20261010-history-trajectory-actor.md)、
[startup balanced](../src/task/trajectory-policy/docs/experiments/probes/P-20261010-startup-balanced-actor.md)。
首轮真实任务[PPO](../src/task/trajectory-policy/docs/experiments/probes/P-20261010-trajectory-ppo.md)
`4da53a9` 完成24updates/48992交互/264s，KL受控；训练随机采样有一次486帧且终末
持有，最终 `054a78c` 冻结完整比较GT/dense各3/4，warm/PPO均0/4且未形成held。
全历史/采样概率/价值/bootstrap/任务奖励/GAE、随机c独立物理解码及R/native/终末
结果审计通过；本轮局部UNPROMISING。初始同H最终XYZ mean变化仅.04964mm RMS，
已有rollout独立审查发现7条>=45held探索行启动normalized A均负；
`12a0f7c` GPU固定快照credit回放只改lambda.95->1，启动样本实际概率提升由1/7
变5/7，meanlogpchange-.086->+.194，KL<=.00876。机制线索PROMISING，未新物理收益；
后续 [lambda1任务Probe](../src/task/trajectory-policy/docs/experiments/probes/P-20261010-trajectory-ppo-long-credit.md)
`0487233` 完成24update/48992交互/265s；首轮128controls与lambda.95完全相同，
9/12长held探索行启动Apositive，但冻结GT/dense各3/4，warm/final仍0/4形成held。
完整H/概率/GAE/随机c解码/R/native/outcome审计通过，局部UNPROMISING。FP64validator
误差原样保留，独立FP32 A/return复算exact，未重跑训练/物理。
`77b0e9e` 固定warmH诊断：lambda1接触前XYZ均值改动最大.03893mm（per-coordinate
RMS），探索std1mm，未形成足够mean策略变化的线索。下一步先冻结warm/final同条件
随机采样执行，区分stochastic部署收益与探索偶然成功，再决定optimizer诊断；
不直接增加epochs/改LR。核心目标与WM收益仍未完成。
`959cd74` [同噪声冻结随机对比](../src/task/trajectory-policy/docs/experiments/probes/P-20261010-trajectory-stochastic-execution.md)
85.72s/64env完成并通过H/采样/D/FK/R/native/outcome独立审计。GT4/4、dense3/4；
warm/final各2/28长时终末held，而且同pair8/20成功，未显示稳定成功增量。
clip .9818%/1.0741%，局部UNCLEAR；当前没有可用trajectory baseline，Cm收益仍未证。
`eb3e96d` 已有rollout真实Adam重建5.39s完成，25个actor/value checkpoint bitwise一致。
单update LR恢复在19/24批次带来>=2倍启动mean位移、更好surrogate及同KL上限，机制
PROMISING；没有新任务训练或物理收益。用户随后提供ref1_1，root先核对learned
retargeter原文与actual hand/action数据合同：支持真实动作监督的小型R Probe，
不能说当前R已被定位为主故障。两轮已有PPO可构造93,376合法窗，但完整稳定held
仅1行、同seed，旧成功teacher packet不可训练；当时未新采集/训练。详情见
[结构取舍](../src/task/trajectory-policy/docs/research/20261010-ref1_1-retargeter-design-review.md)。
步长真实训练留作备选，WM尚未启用。

当前按 ref1_2 将执行边界收敛为共享的
`tau -> q_des/velocity_des -> measured-q feedback + bounded residual -> native`
合同。`goal_executor.py` 已接入 trajectory-policy 的 PPO 训练、PPO 冻结评估、
history 评估和 decoder 评估入口；18 个 Task 测试通过。对既有 8,672 条完整 PPO
控制记录的逐步重放，新执行器与旧公式的 requested command 最大差为
`2.38e-7`，因此这是接口统一而非行为改动。下一步应在同一执行器下做 GT tau 与
生成 tau 的对照，再决定是否补训练；当前没有启动新训练或物理仿真。
2026-10-11 已审计用户提供的只读 `third_party/regrind`（commit
`38347a9e30184620cae103`）。源码确认 REGRIND 的可迁移核心是
`reference target + bounded residual + measured-state feedback`；其 IsaacLab 的
root-wrench 数值、16/20-DoF action scale、phase 输入、reference-state reset/RSI 和
object-tracking reward 都绑定另一套资产与任务语义，不能直接复制到 DExplore 或当前
纯测量 H 合同。当前 executor 因而可称为 REGRIND-inspired closed-loop executor，不能
称为完整 REGRIND 实现。Decision Note：当前问题是是否先改成 REGRIND 全量训练；关键
证据是 executor 已通过 8,672 行等价回放，而 GT/generated tau 的物理差异尚未隔离；root
选择保留现有 H/D/R 与真实 frame-0 reset，先做同一 executor 的 GT/generated tau 对照，
再决定是否借用 failure-bin sampling。成本为一次既有协议的最小物理 Probe；若 GT 失败先修
执行合同，若 GT 成功而 generated 失败则继续查高层 tau coverage，不启动 REGRIND 训练配置。
审计记录见 [REGRIND source audit](../src/task/trajectory-policy/docs/research/20261010-regrind-source-audit.md)。
随后按该 Decision Note 在当前提交 `7748c42` 完成
[同执行器 GT/generated tau 物理对照](../src/task/trajectory-policy/docs/experiments/probes/P-20261011-regrind-executor-gt-generated.md)：
真实 frame-0、16 env、542 controls、68 次重规划在 GPU0 完成，GT 与 dense 校准均为
4/4 长时终末 held；metric48 为 2/4，重复臂为 0/4。独立保存轨迹审计通过，897 条
控制输入的 features 最大误差 `2.86e-6`、requested/applied command 最大误差
`2.38e-7`、PD 误差 0。因校准成功而生成 τ 两臂未通过预声明门槛，该固定 metric
decoder/executor 组合记为 `UNPROMISING`，生成失败不能归因于当前 executor 接线。
因此继续保留 H/D/R、真实 reset 和无 phase/future-reference 合同，不启动 REGRIND 的
4096-env/reference-tracking 训练或 failure-bin sampling；下一步只在新卡片约束下查高层
τ coverage/conditioning 或 GT-centered 扰动 basin。
后续`653f884` [learned R inverse](../src/task/trajectory-policy/docs/experiments/probes/P-20261010-learned-retargeter-inverse.md)
双臂1500updates完成36.42s/GPU5，独立原始标签/PD/normalizer/全部val预测审计PASS。
τ val动作误差比state-only低7.82%，同tick τshuffle提高56.74%，但首PD XYZ p95
49.71mm/startup40.04mm，未过5mm门槛，局部UNCLEAR，无新物理执行或可用R。
冻结train精度诊断4.44s：train也有38.78mm p95，val L1仅高6.95%，优先精度/conditioning。
`1e9782c` [局部inverse标定](../src/task/trajectory-policy/docs/experiments/probes/P-20261010-retargeter-step-inverse.md)
3.44s完成：hand恢复腕姿p95<=.00158mm/1.84e-5rad，但hand与privileged next-q
标定均未过screen，val XYZ约32.6mm，局部UNCLEAR。原r1 angular周期差缺陷已修正并
原样保留，不影响先前neural fit。后续`49013f6` [local-motion retargeter](../src/task/trajectory-policy/docs/experiments/probes/P-20261010-retargeter-local-motion.md)
固定双臂1500updates完成40.02s/GPU5，独立审计5.55s/PASS：对齐局部分支的
τ/state-only L1为.241886/.332077（ratio.7284），同tick shuffle ratio2.1482，
说明信号被使用；但all/startup XYZ p95仍44.15/51.21mm、rotation.3688/.1101rad、
fingers.1019/.1210rad，未过5mm/.05rad screen，局部UNCLEAR。保留离线条件化证据，
不进入competent物理采集。随后CPU冻结分解显示local对腕部normalized residual改善
43.9--51.9%，但首步wrist rotation p95仍.288/.116/.296rad、后续8步误差
.230→.260；fit audit已证明label→PD转换无接线错误，native preload/闭环语义仍未测。
当前保留为动作识别/conditioning gap，先不增加epochs、seed、数据、物理或接WM。
入口：[trajectory-policy](../src/task/trajectory-policy/README.md)。

旧Task35093fb corrected full r3已完成107.01s：GT4/4longheld+terminal、median479、
clip0，生成三组仍0/4抓持、clip100%/97.094%/94.649%；全历史/未来速度/输入/命令审计
通过。接口错误已修复，生成proposal执行问题未解决；checkpoint/失败证据保留，
没有启动bootstrap训练，GPU2已释放。完整ref8和Mission最终收益仍未完成。

## 2026-10-10 生成τ首次完整执行：当前组合 UNPROMISING

纯历史proposal/T/fingerfit executor冻结，PointWorld和E必要性继续冻结。原300step
投影经XYZ-only FK和单迭代CUDA graph加速，保留普通Adam/两起点/所有迭代；保存输入、
重复输入和变化输入均与原拟合点一致。7840f92 seed298完成16env/542步，107.71s，
68次投影中位.835s。GTτ对照4/4长时且终末held，median479、无裁剪；persistence、
位移、评分各0/4形成抓持，裁剪100%/92.435%/95.018%。只是当前冻结组合的局部负Probe，
不单独否定proposal/scorer/τ路线。旧短迭代精度失败和中断运行原样保留。

c8280e3全轨迹审计：初始state与参考exact，所有纯H/当前手重算error0；实际897输入
3.81e-6、原生命令3.05e-7、PD0、选择/物理outcome一致。独立review未发现接口遗漏，
但训练最早tick8而部署tick0，以及在线重标定/分段末帧padding均未被隔离。
8330b07短诊断确认GT在线链路只1/4 held45而原GT4/4，前8步GT handoff仍未恢复生成
抓持。在线几何q接近原目标，但首目标速度差分混入live q跟踪误差，重规划FF偏移最高
7cm/.30rad。e3f5a17同预算内future-only速度/原始τ对照均4/4 held45且终末held，
修复35093fb后纯H完整执行仍未抓持，详见上方最新状态。真实τ仅明确的privileged
对照，不加入纯H部署；本Task后续实验按用户要求暂停。完整ref8、同状态排序与Mission Cm策略训练收益仍未完成，GPU2空闲。
协议：[原生执行](../src/task/consequence-evaluator/docs/experiments/probes/P-20261010-generated-tau-native-execution.md)、
[接口诊断](../src/task/consequence-evaluator/docs/experiments/probes/P-20261010-generated-tau-interface-diagnosis.md)。

## 2026-10-11 ref2 Stage 1 goal-conditioned low-level route

按用户提供的 `ref2.md`，trajectory-policy 新增了 REGRIND-inspired 的 goal-conditioned
low-level 合同：live `q/dq/hand/object/velocity`、previous native command 与 8 步未来
hand/object goal 输入 policy，禁止 future q、teacher command、phase、clock、contact label
和 Cm。REGRIND 源码审计仍限定迁移边界为 measured-state feedback + bounded native
correction，不复制其 asset-specific gains、reference reset、phase、4096-env 配置。

`P-20261011-goal-conditioned-bc` 在自有 tracker 执行轨迹上做 train-only normalization 和
按环境留出，17,344 rows 的 validation active-action L1 为 `0.00672288`，标为
`PROMISING`，但只代表 offline action distillation。旧 reactive packet 保持
`engineering_only=true, training_allowed=false`，这里只读取其 hand/object goal，丢弃
actions/q；没有把它升级为 teacher dataset。

物理 BC Probe `P-20261011-goal-conditioned-execution` 使用 GPU0、16 env、542 controls，
16/16 在 3 帧内失去 hold，long-held terminal `0/16`。独立 audit 的 feature replay max
`8.45e-3`、policy replay `6.41e-7`、requested/applied `0`、PD finite，说明当前失败是
direct-action BC 的闭环 covariate shift/隐藏 preload 问题，不是 executor 接线错误。

随后按 ref2 的 BC→PPO 顺序完成 `P-20261011-goal-conditioned-ppo`：24 updates、49,152
transitions、KL<`0.015`、actor parameter change `0.0529`，训练状态 `UNCLEAR`；5 个完整
episode 的最大 hold 仅 3 帧。冻结评估 `P-20261011-goal-conditioned-ppo-execution` 同样
为 `0/16` long-held terminal，独立 audit 通过，因而 direct native-action BC/PPO 在这条
单 motion、单 seed、固定预算下记为 `UNPROMISING`。

仓库中已有一条更接近 REGRIND action 分层的独立 tau-only residual 路线：几何名义目标经
`GoalConditionedExecutor`/native adapter，再叠加 learned bounded residual，并且每步重读
live `q`。`P-20261010-tau-geometry-tracking` 的 fixed-final r2 恢复了 `16/16` 的 45-frame
hold，但只有 `4/16` 达到 near-teacher/terminal，clipping 仍为 `17.90%`；feature、command
和 PD 审计通过，卡片严格标为 `UNPROMISING`。这说明 residual 分层比 direct action 更有
希望，但当前瓶颈是可行的 thumb preload/非饱和目标，而不是 REGRIND 的训练规模或 phase
配置；不会把这条单 motion 结果升级为完整 REGRIND 结论。

### Fixed tau residual-basin diagnostic

为区分“残差分层本身不可行”和“拇指预载方向尚未落在可行 basin”，新增了固定
offset 的冻结诊断 `P-20261011-tau-residual-basin`（代码提交 `00859cf`）。它沿用
verified GT tau 几何、实时 q/dq/hand/object/velocity、native PD adapter 和
future-only wrist feedforward；每个 control tick 在冻结 canonicalized actor 的
12-D latent 后持续注入一个预先固定的 thumb offset，并把扰动后的 `tanh(latent)`
作为下一 tick 的 previous residual。没有训练、没有 phase/clock/contact/future-q
输入，也没有使用 packet actions/q 作为标签。

GPU2 的 16-env/542-control full wave 已完成。无外部 offset 的 retained actor 为
`4/4` long-held terminal、0 clipping；`thumb_pitch_minus (coord11=-0.05)` 同为
`4/4`、clipping `4/2168=0.1845%`；`thumb_pitch_plus` 为 `4/4` 但 clipping
`55.35%`；`thumb_yaw_plus (coord10=0.05)` 为 `0/4` terminal-held，median held
131 帧、object-position RMSE `0.4197 m`。完整 controlled-placement `success` 仍为
`0/16`，所以这不是完整抓取/放置成功结论。按预注册的 baseline 与扰动门槛，Probe
标为 `PROMISING`，含义仅是该单 motion 上存在一个低裁剪的 thumb residual basin。
独立 audit 对 delta/latent/action/PD 和 tau/live-state feature replay 均通过
（最大 feature replay error `2.86e-6`）。

Decision Note：当前关键选择是直接把这个 fixed offset 接入 ref2 Stage 1，还是先验证
它的可重复性。root 选择保留 `thumb_pitch_minus` 为候选低层 adapter，只允许下一张卡
做一次 matched robustness Probe；在该 Probe 前不把它写成 goal policy 已解决，也不启动
REGRIND 原始 4096-env/failure-bin 配置、H→tau 或 PointWAM。该正向结果不能抵消 direct-action
BC/PPO 的 `0/16` 物理失败，也不能替代新的 `training_allowed=true` teacher rollout。

该 matched repeat 已按新卡完成（seed287，GPU2，65.7s）。baseline 为 `3/4`
long-held terminal，`thumb_pitch_minus` 为 `3/4`、clipping `5/2168=0.2306%`，
独立审计最大 feature replay error `1.91e-6`；因此重复仍通过预注册门槛。yaw 在这一
launch 为 `3/4`，而首轮为 `0/4`，说明 separate-live 单 motion 结果存在 launch variation，
不能把 yaw 方向写成稳定改善；两轮完整 `success` 都是 `0/16`。root 只保留
`thumb_pitch_minus` 作为固定残差 basin 候选，并据此完成了一次共享 executor 的
goal-conditioned residual integration Probe；仍不启动 REGRIND 原始训练规模或 H/PointWAM
高层路线。

### Goal-conditioned residual integration

新卡 `P-20261011-goal-conditioned-residual-integration` 使用 24 步 hand/object goal、
`tanh(previous raw residual)` 历史和共享 `GoalConditionedExecutor`，严格筛选 retained
r3 的 16 个 `role==tracker` 环境。offline BC 在 2,168 个环境留出样本上的 raw-latent
L1 为 `0.0249544`，因此允许运行 privileged nominal-reference physical Probe；nominal
q/velocity 明确来自 geometry cache，只作集成诊断，不是可部署的 no-future-q planner。

有效 r4 运行使用 GPU0、16 env、542 controls，全部完成且无 early reset；10/16 达到
long-held terminal，15/16 达到 hold45，median held `483`，但 native clipping 为
`6.6075%`，超过预注册 `<1%` 门槛，Probe 标为 `UNPROMISING`。所有 controlled-placement
success 仍为 false。独立 audit 为 `PASS`：feature `2.86e-6`、latent `7.75e-7`、
executor target `3.58e-7`、command `4.77e-7`，clipping flags exact、PD reconstruction
exact；负结果不是 residual/executor/PD wiring 错误。r1-r3 是仿真前的历史 hash、rl_games
compatibility 和变量遮蔽 setup failures，不计入科学波次。

Decision Note：当前关键选择是是否在该 residual student 上继续 PPO。由于 physical gate
同时未达到 12/16 long-held terminal 且 clipping 超阈值，root 关闭这条 student integration
变体，不启动 PPO；固定 residual basin 只作为控制诊断，并等待明确
`training_allowed=true` 的 fresh teacher 或新的 preload 参数化。该结果仍不 refute
measured-state residual control、REGRIND 思路或 H/PointWAM 路线。

### Fixed residual offset on the residual student

为检验 `thumb_pitch_minus` 是否能从冻结 tau basin 迁移到 residual student，完成了
`P-20261011-goal-conditioned-residual-offset`：在同一 16-env、542-control、同一
student 上只加入 raw residual coordinate 11 的 `-0.05` 固定偏置。结果为 `0/16`
long-held terminal、`12/16` hold45、median held `79`，clipping `0.8187%`，且
controlled-placement success 为 `0/16`，所以预注册的 `12/16 + <1%` physical gate
只通过 clipping 半边，Probe 为 `UNPROMISING`。独立 audit 为 `PASS`（feature
`2.86e-6`、latent `7.15e-7`、target `3.58e-7`、command `4.77e-7`、clipping
exact、PD exact）。这关闭当前 residual student 的 PPO 和继续调固定 offset；不能把
冻结 tau 的局部 basin 解释成 learned student 的可行性，也不 refute REGRIND-inspired
measured-state residual 接口。证据卡片见
[goal-conditioned residual offset](../src/task/trajectory-policy/docs/experiments/probes/P-20261011-goal-conditioned-residual-offset.md)。

### Decision Note: close direct-action Stage 1 variant

当前问题是 direct native-action goal policy 是否继续投入。关键证据是 offline fit 很低但
BC/ppo frozen physics 均在启动后失持，且两次 executor/feature/PD audits 通过。root 选择
关闭本 motion 的 direct-action BC/PPO sweep，保留 ref2 Stage 1 接口，下一步只考虑
`nominal goal + bounded residual` adapter 或重新采集并明确 `training_allowed=true` 的
teacher rollout；不启动 REGRIND 4096-env/failure-bin training，也不进入 H/PointWAM/ACT
高层训练。成本为保留现有 outputs 和代码；若新 residual/teacher Probe 仍失败，再决定是否
暂停 Stage 1 并转向执行合同或 fresh teacher provenance。

证据卡片：[goal BC](../src/task/trajectory-policy/docs/experiments/probes/P-20261011-goal-conditioned-bc.md)、
[BC execution](../src/task/trajectory-policy/docs/experiments/probes/P-20261011-goal-conditioned-execution.md)、
[goal PPO](../src/task/trajectory-policy/docs/experiments/probes/P-20261011-goal-conditioned-ppo.md)、
[PPO execution](../src/task/trajectory-policy/docs/experiments/probes/P-20261011-goal-conditioned-ppo-execution.md)、
[goal-conditioned residual integration](../src/task/trajectory-policy/docs/experiments/probes/P-20261011-goal-conditioned-residual-integration.md)、
[goal-conditioned residual offset](../src/task/trajectory-policy/docs/experiments/probes/P-20261011-goal-conditioned-residual-offset.md)。

## 2026-10-10 ref8：H→候选τ，纯历史位移预测 PROMISING

用户ref8把近期工作转向H→候选τ，并确认H只用当前/过去手物状态，不含任务phase/clock。
PointWorld和GT E/I→Y必要性重复实验冻结；保留τ→A的12/16候选，不继续其手指PPO。
旧1442维actor observation含t+1/t+16参考与contact差，旧H→τ结果不能当作纯历史proposal。
固定旧权重诊断复现.3028m；清零训练近恒定列没有改善，clip10反而.4360m。
旧测试episode误差中位47mm但两条超过1m，不能简单归因为所有episode都迁移失败。

新输入300维只读四个实测states t-3:t：手几何、物体相对SE3、手指q/dq、物体速度，
不读actor obs/真实未来参考/接触力/action/phase/clock。相同初始化、batch和1200steps
对比位置与相对当前手的位移目标，两臂分别由val L1选权重。完整held Probe数据中，
位置point/H24 RMSE为.46330/.51888m，位移为.21266/.33152m，persistence.26214/.40719m；
位移通过原90%/H24预测门槛，标记PROMISING。仍保留全部64episode，包括异常长尾；
这是探索性held split，不是新Validation，也不能把target normalization变化忽略后作纯anchoring归因。

纯训练集K=8检索每候选来自不同episode，距离只读实测H：top1.24278m，未过门槛；
GT best-of-8覆盖上界.17191m，不能称为可部署selector。独立重算/候选重建/来源合同通过。
下一决策转向生成τ的评价分布与候选可执行性，不立即扩大ACT/CVAE/diffusion或重做E必要性。
正式same-state ranking、生成τ→A→Z、Cm收益仍未证；本次计划内最小比较已结束，GPU2空闲。

checkpoint：outputs/consequence-evaluator/measured-history-tau-proposal-20261010-r1/displacement-best.pt；
SHAde98077c51ee7c8de3ffe604d91b8fa02459c80ef40b276f34c946f7df813d3c。
协议：[history-tau-proposal-diagnosis](../src/task/consequence-evaluator/docs/experiments/probes/P-20261010-history-tau-proposal-diagnosis.md)；
方法核对：[candidate-tau-proposal-review](../src/task/consequence-evaluator/docs/research/20261010-candidate-tau-proposal-review.md)。

## 2026-10-10 τ-only 手指命令重新拟合：PROMISING，仍有失持

保留τ、去掉真实未来q/物体参考和触觉输入的合同不变。冻结旧τ策略的腕部与编码器，
用其实际执行的手指PD命令重新拟合六个手指输出；覆盖全部16行及首轮学生的实时状态，
没有按成功挑样本。目标留出原生范围0.5%余量，并惩罚最差1%的越界；执行端仍用原来的
残差尺度、native clamp和coupling，没有通过新增裁剪隐藏越界。

固定1000update最终权重（6dc24eb，fit279/eval283）完整542步评估：16/16 hold45，
12/16达到433帧且末帧持有，保持中位479帧，裁剪6/8672=0.06919%；纯几何0/16。
原强门槛通过，标记PROMISING。独立输入/command/PD重构通过；剩余三行无支撑重力下落，
一行近桌面分离，参考当时仍holding。同轮冻结τ持有15/16但裁剪40.15%，不能声称新策略
抓持稳定性优于冻结策略。相对先前短PPO的4/16与17.90%只是不同Probe的描述性进展。

后续仅更新手指的64update PPO（be738cc，train284/eval285）仍退化：0/16近teacher或
末帧持有，保持中位11帧，裁剪0.438%。腕部/编码器/logstd保护及实际输入/PD合同通过；
该局部适配UNPROMISING，保留原始失败记录，不以此否定τ路线，也不继续参数/seed sweep。
保留预声明的手指重新拟合候选，下一决策是如何在接触适配时保护有效的手指预载。

保留checkpoint：outputs/consequence-evaluator/ref7_4-tau-finger-fit-20261010-r3/final.pt，
SHA79ee282e6b2287375c6480eb8f793864c858311b968ffa97b95c65842502fa45；
控制配置/复现argv：outputs/consequence-evaluator/ref7_4-tau-finger-controller-20261010-r1/controller.json。
仍是single-motion GT-τ、旧自训oracle知识warmstart的Probe；未完成稳定新τ泛化、原始放置
或Cm收益验证。本轮受控预算内已停止，GPU2空闲。
协议：[tau-finger-canonicalization](../src/task/consequence-evaluator/docs/experiments/probes/P-20261010-tau-finger-canonicalization.md)。

## 2026-10-10 ref7_4 τ几何 / 无真实未来q与物体参考

用户确认保留τ，去掉真实q_ref和未来物体参考，触觉暂不引入。沿ref7_4两层路线：
11点τ+初始q/静态URDF -> 几何q_hat与速度 -> learned closed-loop residual ->
full native command。学生输入897维，删除未来物体12列，关节误差/PD基底/前馈
均来自τ几何；奖励也改为τ手跟踪、当前物体lift/hold，不读未来物体目标。
从旧自训练oracle tracker迁移权重，不声称从未使用oracle训练信息。

几何拟合GPU14.31秒，整体坐标RMSE2.47mm、拇指尖13.80mm，其余指尖0.51–2.79mm。
冻结迁移权重完整对照：tau tracker保持中位478帧、13/16近teacher且末帧持有；
但41.01%控制步裁剪，主要拇指，强门槛仍为UNPROMISING。old oracle16/16近teacher，
tau nominal0/16，shifted tau近teacher0/16（末帧15/16，不能只看末帧持有）。
actual student897输入独立重构误差3.81e-6、command2.38e-7、native PD0。

第一次固定128update微调后0/16稳定保持，但查明奖励目标错误：手掌高度变化
不能代表物体lift或抓持阶段。原奖励把不抓轨迹排在稳定holder前面，不能作为
τ方法负证据。真实模式回归测试先失败后通过；现有轨迹重评分确认修复后的
实际lift/force-pair holding + 弱τ tracking奖励排序正确，独立审查通过。
52b5b88修复后从原始warmstart完成同seed277/128update的r2；固定seed278评估
16/16 hold45，中位保持292帧，4/16达到433且末帧持有；old oracle16/16，
tau nominal0/16。裁剪1552/8672=17.90%，强门槛UNPROMISING，仍未训练好稳定执行器。
12条末帧失持均在teacher仍holding时无支撑脱手/重力下落，不是原始任务放置。
实际897输入重构误差2.86e-6、command2.38e-7、PD0。修复恢复抓持但不优于冻结迁移。
保留两层路线的可行信号，不据此否定τ；本轮预算结束。下一决策是可行的拇指
target/预载参数化与短PPO适配不足的区分，暂不扩展seed/model sweep。
geometry只拟合τ/initial q，不按执行结果重新挑选；仍无触觉输入与未来物体监督。
协议：[tau-geometry-tracking](../src/task/consequence-evaluator/docs/experiments/probes/P-20261010-tau-geometry-tracking.md)。
单motionGT-τ上界与缓存几何，实时新τ retarget latency/泛化/原始放置/Cm仍待后续证据。

## 2026-10-10 wrist velocity feedforward 修复

冻结轨迹核算发现 position-only PD target 缺少参考速度阻尼补偿。原生腕部
Kp200/Kd20，使用 next-frame q 的中心差分速度和真实 control dt，PD参考加入
0.1*dq_ref；不使用 teacher command 标签，仅补偿腕部6维，手指仍由闭环残差处理。
固定旧策略的随机角色四臂完整542步对照：old tracker held中位285帧、近teacher
0/16、末帧0/16；tracker+FF中位479.5帧、近teacher13/16、末帧13/16、零裁剪。
原始预声明门槛通过，结论仅为此单motion干预 `PROMISING`，不是正式因果Validation。
剩余3/16仍有无支撑分离/自由落体。初始状态bitwise一致，原生PD重建误差0，独立
reference/residual/feedforward command合同最大误差2.38e-7。

修复后seed274微调128updates/262144转移完成，固定最终checkpoint经过seed275
完整542步评估：teacher median484；nominal+FF median92、近teacher4/16；
trained tracker+FF median483、31/32达到433且末帧持有，32/32 hold45。
评估裁剪76/17344=0.438%，原始强门槛 `PROMISING`，未降低标准。
剩余env38在451脱离后无支撑自由下落；reference当时仍抓持，不是受控放置。
训练r1的变量shadow defect已修复，原冻结对照不受影响；所有失败/成功运行均保留。
固定checkpoint：outputs/consequence-evaluator/ref7_3-tracker-feedforward-train-20261010-r2/final.pt。
控制配置、哈希与复现argv：outputs/consequence-evaluator/ref7_3-tracker-feedforward-controller-20261010-r1/controller.json。
运行必须显式启用wrist FF，权重单独加载不等同于修复控制链。正式Validation、
其余掉落、原始motion放置与τ retarget留作后续决策；当前robot-q/object-future
oracle尚未移除，仍未完成τ→A。此次单seed结果不升级为全局claim或Cm收益。
实验卡：[reference-tracking-feedforward](../src/task/consequence-evaluator/docs/experiments/probes/P-20261010-reference-tracking-feedforward.md)。

## 2026-10-10 ref7_3 reference-tracking control Probe

已实现并训练原生 Inspire/Gym 小残差 PPO：实测 robot-q/11点手/object reference
加 live state -> residual PD target -> native18维 command。使用明确声明的 robot-q
与 object-future oracle，不用 teacher future command 作训练标签或 tracker 输入。
GPU2 单 seed128更新/262144转移，约454秒；冻结 final checkpoint 后完整542步评估。

teacher16行的保持中位数483帧，nominal16行0帧，tracker32行286.5帧；tracker32/32
均达到45帧，但0/32达到参考481帧的90%，末帧仍持有也是0/32，最终回到桌面。
初始 q/dq/hand/object bitwise 对齐，native PD目标重建误差0，tracker评估零裁剪。
因此预声明近 teacher gate 为 `UNPROMISING`；保留短时保持的正向控制信号，不降低
gate，也不解释成方法无效。旧 loss-event 计数排除了supported状态，不能把其0写成
没有掉落。这里只是单动作/单seed/固定角色布局的Probe，未完成τ→A，未证明Cm收益。

下一步优先区分后段接触保持的训练覆盖与控制/reward问题，再替换robot-q oracle为
11点τ retarget；不继续仅压inverse L1，不启动新的sweep或正式Validation。协议与结果见
[`P-20261010-reference-tracking`](../src/task/consequence-evaluator/docs/experiments/probes/P-20261010-reference-tracking.md)。

用户追问放置语义后，冻结轨迹审计确认：原始motion确实有放置/松手（末端物体高度
回到初始+0.495mm，contact标签最后活跃tick498），但本轮PPO用的是teacher实测
reference，它末端仍抬升0.832m、离桌面0.786m。31/32 tracker行在无支撑分离后
出现接近-9.81m/s²的下落；其余一行在近桌面分离。故本轮行为不是受控放置，原始
任务的放置成功也不能用末帧持有判定。后续须明确原始任务目标与teacher控制上界
两种语义；本轮仅对实际使用的teacher-reference tracking gate作判断。诊断图/JSON：
`outputs/consequence-evaluator/ref7_3-tracker-drop-diagnostic-20261010-r1/`。

## 2026-10-10 主线 τ-conditioned evaluator Probe

为继续 `docs/user/完整链路.md` 的第 7–8 步，新增真实 hand-trajectory 接口的
离线 evaluator，而不是复用旧 `[24,18]` native-action branch。C0 输入 `H+τ`，
C1 输入 `H+τ+E_GT`，C2 输入 `H+τ+E_PW`；τ 使用 `pw_hand_future` 当前查询物体坐标系，
对象 effect 单独取 GT/PW 的前12维。GPU2 的 1200-step fit 在约31秒完成，输入、同-H
25x7 panel、train/val/test split 和 checkpoint hash 均记录在
`outputs/consequence-evaluator/trajectory-utility-fit-20261010-r1/`。

结果为 `UNCLEAR`：C0/C1/C2 严格 pair accuracy 为 .6282/.6410/.7692；C0 的 τ
shuffle 从 .6282 降至 .3718，说明新 τ 分支确实被使用；但 C1 增益仅1.28pp（gate
要求3pp），C1 τ-shuffle drop 也仅1.28pp。C2 的 τ-shuffle 为 .7821，高于未打乱的
.7692，不能把其 raw 分数写成 PW 保留能力。panel 仅5/25锚点有严格 label 差异、78
strict pairs，当前不扩 epoch、不改 gate、不进入可部署 planner；保留为主线第7–8步
的接口与证据，下一步需要非平凡同-H candidate bank/H→τ proposal 审计。

随后审计发现 r1 的 C0/C1/C2 是三个独立随机初始化，`initial.pt` 是未参与训练
的第四个实例，因此 r1 的 C1−C0、C1 shuffle 和 C1/C2 gate 不能作 matched-arm
归因。按 Decision Note 保留 r1、不覆盖产物，只修复初始化并以同数据/seed/步数重跑
`trajectory-utility-fit-20261010-r2`（GPU2，32.2s）。共享初始化后的 panel 为
C0/C1/C2 `.6282/.6538/.5897`，C1 增益 `.0256`、τ-shuffle drop `0`；GT
information 与 PW retention gates 均为 false。该实现修复后的结果仍只覆盖5个
informative anchors/78 strict pairs，故不扩 candidate bank、不启动 C2/PW 在线链路。
随后在 clean commit `15ca96f` 复播同一配置为 `trajectory-utility-fit-20261010-r3`；
初始与三臂 checkpoint hash 和 r2 byte-identical，确认 r2 数值不是脏工作树偶然产物。

随后对冻结 `H.pt`/`HA.pt` 做只读 candidate-bank audit（commit `e0b8e2f`）：H-only
七候选完全相同（candidate RMS 0），HA 虽有动作条件，但 panel 候选差异仅约
1.09 mm mean / 1.40 mm max，observed-τ point RMSE 约59.53 mm。将 HA 生成 τ
喂入冻结 evaluator 后 C0 严格 pair 仅29.49%，且选择集中在 candidate 2/4；这是
evaluator 训练于 observed τ、生成 τ 属于 OOD 的诊断，不是任务控制结果。它把下一
个低成本 blocker 具体化为“先获得有足够候选多样性且非 tie 的 H→τ bank”，而不是
继续加 evaluator epoch。

为区分 panel 太小与候选本身塌缩，随后只读重建 ref13 recovery 的单一路径
`initial + new-o8 + new-o16` bank（输出
`outputs/consequence-evaluator/trajectory-rolling-panel-audit-20261010-r7/`）：
75 个同-H panels、13 个 informative anchors、163 strict pairs；每个候选使用
自己的 post-query `trace.pt` geometry，H/prefix/valid32/zero-clipping 合同通过。
观测 τ 的 candidate spread 为 initial 9.37/61.35mm、new-o8 8.48/65.02mm、
new-o16 10.31/225.73mm（mean/max RMS），所以该 bank 并非 HA bridge 的1mm塌缩，
但三个 cohort 是同一 reconstructed e260 actor/motion 的 treatment-conditioned
时间视图，不是75个独立环境。

随后发现 r7 manifest 实际加载的是未匹配初始化的
`trajectory-utility-fit-20261010-r1/C0.pt` 与 `C1.pt`；其 .5706/.6319 及
6.13pp 增益只能作为 provenance-confounded engineering evidence，不能作为
matched-arm 结果。保留 r7 产物，不覆盖、不删除。

在相同 bank 和合同上，以 matched-init r3 C0/C1 重播
`outputs/consequence-evaluator/trajectory-rolling-panel-audit-20261010-r8/`：
C0/C1 strict pair accuracy 为 .5706/.6503（+7.98pp），mean regret
.03528→.03153，但 C1 τ-shuffle 仍为 .6503，drop=0pp；C1<.70 且
`screen_gate=false`。因此扩大 observed bank 仍未证明稳健 τ-conditioned ranking，
GT-information screen 继续为 false；没有启动 C2/PW inference、selector 或在线控制。
该 bank 的 parent/offset、r3 checkpoint hash 与审计脚本 hash 已写入 r8 manifest，
r1–r6 的构建工程失败继续保留，未形成科学结果。

r8 的只读 score-margin 审计没有 exact tie，但 C1 top-two margin ≤.01/.03 的
panel 分别为 35/75、53/75，teacher top margin ≤.02 的为 70/75；shuffle 改变
raw score 却不改变 strict accuracy。因此没有定义 deadzone 或 online selector，
第 9 步继续冻结。

随后完成冻结离线 `H→τ→PointWorld→C2` planner-chain wiring audit（输出
`outputs/consequence-evaluator/trajectory-planner-panel-audit-20261010-r4/`，GPU2）。
每个同-H anchor 只取一份当前状态，再由 HA bridge 展开7个候选；PointWorld
重复推理 bitwise 稳定，shape/frame/hash 合同通过。生成 bank 的候选 spread 仍为
1.09/1.40 mm，和观测 τ 的 point/H24 RMSE 为34.37/44.97mm；OOD C2 在25×7
panel 的 strict pair accuracy 56.41%、top-1 agreement 20%、mean regret .01625，
且选择集中到 candidate5（10/25）。20/25 anchor 属于 teacher tie set（tie-set
coverage 92%），因此这些数不能写成排序或控制收益；它只证明完整离线适配器可执行，
并再次定位“独立且足够多样的 H→τ candidate bank”是当前 blocker。r1/r2/r3 的参数、
索引和断言工程失败均保留，未形成科学结果。

为区分 HA 候选塌缩是 plan 分支失效还是 plan 幅度过小，完成了冻结 bridge 的
CPU-only plan-sensitivity audit：输出
`outputs/consequence-evaluator/trajectory-bridge-plan-scale-audit-20261010-r1/`，
固定同一 25x7 panel 和 HA checkpoint，仅将输入 plan 缩放为 0/.25/.5/1/2/4。
bridge 对 plan 有近似线性响应，非 baseline candidate spread（panel 24x11x3
RMS）约为 0/.30/.60/1.18/2.31/4.39 mm，最大值约 0/.56/1.11/2.18/4.20/7.82 mm；
scale 4 仍未达到 observed bank 的量级且 H24 error 从 77.90mm 略升至 78.28mm。
因此此前的 1mm bank 不是 wiring failure，但也不能用放大系数补成可部署候选；该
审计为 `UNCLEAR`，停止 amplitude sweep，不改变 OOD/无 ranking 或 control claim 的
边界。下一步若继续只应构造有独立、非 tie consequence 的 H→τ proposal bank。

另用同一 panel 只读检查已有 `physical` plan-unit bridge
(`hand-execution-fit-20261009-r2`，输出
`trajectory-candidate-bank-audit-20261010-r6`)：HA spread 仅升至
1.59/2.12mm，point RMSE 59.45mm；C1/C2 strict 为 .6795/.5256，仍低于
.70 且只有 5 个 informative anchors。该量纲对照关闭了“uniform plan unit 单独导致
候选塌缩”的解释，不解锁 selector、PW online、R execution 或 MPC。

为判断现有 ref7_2 结构化 rollout 是否至少能提供后续候选库，完成只读审计
`P-20261010-structured-candidate-bank-audit`（clean replay 输出
`outputs/consequence-evaluator/structured-candidate-bank-audit-20261010-r2/`）。
96/64/64 train/val/test episode 的 reset frame 在 object pose、hand keypoints、q、dq
和 pair proxy 上逐项 bitwise 一致，三 split 零 clipping。以 hand RMS≤3mm、q RMS≤.03
作近当前 proxy，在 tick16–32 多个 split 同时有非塌缩的 current-object-frame tau
spread 和严格 U32 label 差异：train tick16 为 18.69mm/410 near pairs/133 strict
pairs，val tick32 为 19.20mm/46/41，test tick32 为 25.62mm/57/52；tick8 的 label
仍全部为 tie。这个结果支持“值得补采集候选 panel”的工程信号，但不构成 H-matched
candidate bank：原 trajectory packet 没有保存 actor observation/history，近当前规则
也只检查 hand 与 q，未覆盖完整 H/隐藏 PhysX state。Probe 保持 `UNCLEAR`，不启动
evaluator fit、deadzone、selector、PW online、MPC 或 R native gate；若继续，只做一次
有界采集/回放，保存每个 query 的 exact H、q/dq、future hand、command 和 episode ID，
然后按完整 H 合同重审候选库。卡片见
[`P-20261010-structured-candidate-bank-audit`](../src/task/consequence-evaluator/docs/experiments/probes/P-20261010-structured-candidate-bank-audit.md)。

按该正向数据屏幕补采了一次 history-preserving structured rollout（seed411、64
env），再扩成 train/val/test 的 96/64/64 episode split（seeds412/413/414）。新
字段 `actor_observation[t] = obs["obs"]` 与 trajectory frame 逐项对齐，三 split
542 controls、requested=applied、zero clipping，runtime 仍为 CPU tensor/GPU PhysX。
完整 H RMS≤.03 且 hand RMS≤3mm、q RMS≤.03 的近 pair 在每个 split 都保留了非 tie
U32 label：例如 train tick16 为 369 pairs/43 strict，val tick32 为33/30，test
tick32 为46/43。该结果只说明有必要做一次离线 approximate-H panel，不等价于隐藏
PhysX state fork。

基于这些 split 构造了 25/12/18 个 train/val/test 七候选 panel（输出
`history-candidate-panel-20261010-r2/`，label 未参与候选选择），并完成 matched-init
B/T/C0/C1 trajectory utility fit（输出
`history-panel-trajectory-utility-20261010-r4/`，GPU2，1200 steps，约40s）。18个
test panels 中只有7个 informative anchors/137 strict pairs；H-only B strict
为 `.7226`，tau-only T 为 `.7007`（shuffle `.4599`，drop24.09pp），C0 `H+tau`
为 `.6569`（shuffle `.4234`，drop23.36pp），C1 `H+tau+E_GT` 为 `.5839`，shuffle
`.5693`（drop1.46pp），C1 相对 C0 为 -7.30pp，approximate-H evaluator screen
为 false。B/T ablation 说明 tau-only 仍有 exploratory signal，但与 approximate H
合并后 C0 反而低于 H-only；C0/C1 combined arms 在该 panel 上 `UNPROMISING`，不能
升级为正式 ranking claim；C1 GT-information 不成立。按 Decision Note 停止 C2/PW、deadzone、
selector、online planner/MPC 和 native R 接线，保留 panel/fit 作为后续若需扩大
数据或改进 H 合同的证据。

### Decision Note — history panel evaluator screen (executed)

当前需要决定的是：history-preserving split 是否足以把主线推进到 PW/C2 或在线
selector。关键证据是三 split 的近 H pair 仍有 label variation，但 held panel 只有
7个 informative anchors；matched H-only B 为 `.7226`，tau-only T 为 `.7007` 且
shuffle drop24.09pp，C0 为 `.6569` 且 shuffle drop23.36pp，C1 加 GT effect 反而低于
C0 且 shuffle drop 仅1.46pp。root 选择保留 tau-only 的弱 exploratory signal，但把
C0/C1 combined arms 降级为不能归因的 `UNPROMISING` engineering result，关闭
C1/PW/selector/online/native expansion。理由是 full-H data screen 说明候选库不是
塌缩，但 approximate-H/稀疏 informative panels 让 H-only 胜过 combined tau arm，尚不足
以支持 effect information 或可部署排序。下一步若继续，只能先扩大严格 H-matched
panel 或修正 H/label 合同；不追加 evaluator epoch 及不把单 seed/small panel 结果写成
科学结论。

随后对冻结的 matched-init C1 做只读 GT-effect branch audit
(`P-20261010-history-utility-effect-audit`)：同一 18 个 held test panels、137 strict
pairs、7 个 informative anchors 上，nominal `H+tau+E_GT` 仍为 `.5839`；将 `E_GT`
置零（或显式 `use_effect=False`）均为 `.7007`，比 nominal 高 11.68pp；panel 内
shuffle `E_GT` 为 `.4526`，shuffle `tau` 为 `.5693`（nominal drop 1.46pp），两者都
shuffle 为 `.4453`。零 effect 与禁用 effect 的预测逐项 bitwise 一致，输入/权重/面板
hash 合同通过。该结果表明当前 effect branch 被模型使用但会伤害该小型 approximate-H
panel 的排序，不能把 C1/PW 失败归因成“GT effect 不存在”，也不能直接修 selector。
Probe 标为 `UNPROMISING`（当前 branch/分布的 engineering screen），不启动 C2/PW、
deadzone、selector、online planner/MPC 或 native R；若重开，先修 effect contract 或
取得更大 exact-H panel，再做新的 matched audit。产物见
`outputs/consequence-evaluator/history-utility-effect-audit-20261010-r2/`。

### Decision Note — frozen C1 effect-branch safety (executed)

当前需要决定的是：冻结 C1 的 GT-effect 分支是否足以进入主线第 8–9 步。关键证据是
nominal `.5839`，`E_GT=0` `.7007`（+11.68pp），effect shuffle `.4526`，tau shuffle
drop 仅1.46pp，且零 effect 与禁用分支完全相同。root 选择关闭当前 effect-to-selector
路线，保留 audit 作为实现/分布 blocker，而不是对 GT consequence 的方法负结论；理由是
当前分支未通过 nominal、ablation-safety 和 tau-use 三项 screen，继续接 C2/PW 或在线
selector 不能改变已观测问题。预计成本为零；未来若修复 effect 表示或获得 exact-H 数据，
再按同一合同重跑，成功才重新审查 C2/PW。

随后完成 observed-τ 的 baseline-preserving selector abstention audit
(`P-20261010-tau-selector-abstention`)：冻结 T-only 的 18×7 held panel 上，nominal
pair accuracy `.7007`，τ shuffle `.4599`（drop24.09pp），7 个 informative anchors
的 T argmax 全部为 teacher maximum。以 candidate 0 为 baseline，margin
`epsilon=.01--.03` 时非 baseline 选择由15/18降到11/18或8/18，mean regret
`.00521/.00463`，informative mean regret 仍为0；`epsilon=.05` 才开始丢失 informative
rows，regret 升至`.04398`。该结果是 `PROMISING` 的 observed-τ/offline selector
engineering signal，不绕过 H→τ 的 held split 失败，也不接 C1/PW、native 或 MPC。

### Decision Note — observed-τ selector abstention (executed)

当前需要决定的是：第9步的 selector 接口是否值得保留，还是连 selector 也应冻结。关键
证据是 T-only 在7个 informative anchors 上 argmax 全部命中 teacher maximum，且固定
`epsilon=.01--.03` 的 baseline abstention 保持 informative regret 为0；但 panel 仅18个
近似-H panels，τ 是 observed 而非 H→τ 生成。root 选择保留一个 baseline-preserving
selector contract，把下一预算优先用于 exact-H、非 tie 的 trajectory proposal bank；
不启动 C2/PW/online/native。预计成本为未来一次数据采集与同一离线审计；若 exact-H
重审不再保留 signal，则关闭 selector 路线。

基于该 Decision，启动新的 exact-H serial candidate-bank Probe
(`P-20261010-serial-exact-h-candidate-bank`)：复用已通过的单环境 deterministic prefix，
以 seed421、envs=1 串行重放同一 reset/actor 前缀，在 tick120--151 仅对 finger6/8/10
施加 `0、±.08` 六条候选脉冲。候选0为 zero control；每条运行保存 actor observation、
hand/object/q/dq/action/PD target，要求 tick120 前跨候选 bitwise 一致。该 Probe 只判断
是否能形成 exact-prefix、非塌缩 candidate bank，不宣称 hidden-PhysX fork 或因果 effect，
也不提前接 C1/C2/PW/native；卡片和运行目录见
[`P-20261010-serial-exact-h-candidate-bank`](../src/task/consequence-evaluator/docs/experiments/probes/P-20261010-serial-exact-h-candidate-bank.md)。

该 Probe 已完成：7 条 serial rollout 全部 `COMPLETED`，543 个 state frames、542 个
commands、无 clipping；actor observation、hand、object pose/velocity、q/dq、action 和
native PD target 在 tick120 前跨候选均 bitwise 一致，且 7 条 post-pulse hand trajectories
均不同。可是 query tick120 的 `current_pair=false`，冻结 U32 utility 七条全为 `0.0`，
strict pairs=0、informative label rows=0；pulse 后 pair proxy 也没有恢复。因此该卡标为
`UNPROMISING`：serial exact-prefix 的工程合同成立，但当前 reset/阶段没有产生可标注的
consequence panel。这是 data/phase blocker，不是 selector 或因果 effect 的方法负结论；产物
见 `outputs/consequence-evaluator/serial-exact-h-candidate-bank-audit-20261010-r1/`。

### Decision Note — serial exact-H candidate bank (executed)

当前需要决定的是：是否继续在同一 reset 上扩大 serial pulse bank，还是回到
`ref7_2.md` 指定的 execution upper-bound。关键证据是 exact-prefix 和 7 条轨迹多样性均
通过，但 query-time `current_pair=false`、U32 七行全 tie，扩大同一阶段的候选不会改变
标签 blocker。root 选择关闭这条 exact-H serial candidate collection，优先按
`ref7_2.md` 回到真实 GT future-hand trajectory 驱动的 full-action execution model：先审计
现有 teacher-action packet/runner 的标签与执行合同，再做最小 GT-τ→full-action Probe；不
启动 PointWorld、selector、C1/PW、online planner、MPC 或 native R。预计成本为一次只读
代码/产物审计和一个 bounded execution Probe；若 teacher-action clone 已满足 held 90%
合同，再进入 episode-split 结构化 rollout 与 R 训练，否则先修复数据/执行接口。

随后按主线后续的 H→τ 链路做了一次独立的 bounded offline Probe
(`P-20261010-history-to-tau`)：输入 history-preserving rollout 的
`actor_observation[t]` 与当前 hand，目标为当前物体坐标系下的实测
`t+1:t+24` hand trajectory。width-256 MLP 在 GPU2 以 1200 steps 完成，训练/验证/测试
窗口为 6144/4096/4096，三 split actor hash 相同，test 未参与 checkpoint 选择，所有
输入和输出有限。训练/验证 point RMSE 为 `.04328/.06698 m`，相对 persistence 的
`.20065/.19587 m` 有改善；但 held seed-414 test 为 `.30280 m`，高于 persistence
`.26214 m`，H24 为 `.47161 m`，也高于 `.40719 m`，test screen=false。该结果更像
episode split 下的迁移不足，当前没有证据把它归咎于输入/执行实现错误；单次 Probe 保持
`UNCLEAR`，不写成 H→τ 方法的正式负结论。

### Decision Note — H→τ offline predictor screen (executed)

当前需要决定的是：H→τ baseline 是否足以继续接入 PW/selector 或 native control。关键
证据是 bounded fit 的数据、hash、finite 和 episode-split 合同均通过，但 held test 的
point/H24 RMSE 分别比 persistence 高约15.5%/15.8%，而 train/validation 的改善没有
迁移到 test。root 选择保留 packet 作为失败边界和后续数据审计依据，关闭这条 H→τ
baseline 向 PointWorld、selector、online planner、MPC 或 native R 的接线；理由是
当前没有跨 split 的可用预测信号，继续加 epoch、换 gate 或在线放大都不能解决已观测的
split transfer 缺口。预计成本为零（仅保留已有 7 秒 fit 产物）；若未来获得新的、独立且
合同一致的 H→τ 数据，再另起有界 Probe，仍需重新通过 held split screen。

随后完成 ref7_2 后续的 commanded-finger label coverage audit
(`P-20261010-finger-command-coverage`)。现有 history-preserving structured packet 的
96/64/64 episode 均完整有限、零 clipping，且保存的 action 与
`clip(actor_action + structured_residual)` 逐项一致；saved `pair` 也与 native force-pair
proxy 完全一致。三阶段都包含全部五种 structured residual mode，finger action 标准差
约 `.0374--.0516`，所以“没有 commanded-finger label variation”被排除。

但按预注册的 current-object-frame hand≤3mm、q≤.03、dq≤.10 且 same-pair 的 near-state
合同，contact 阶段 train/val/test 只有 `0/0/12` 对，hold 阶段只有 `26/5/38` 对；其中
非平凡 finger-action 差异（RMS>.05）最多2对，远低于每阶段50个 near/10个 nontrivial
门槛。也就是说当前 packet 有 command coverage，却没有 matched contact/hold state 来
检验 preload 是否可由 geometry+q/dq 识别。该 Probe 保持 `UNCLEAR`，这是证据不足而非
对所有 state-conditioned decoder 的正式反证；不启动新 decoder 训练或 native execution。

### Decision Note — commanded-finger label coverage (executed)

当前需要决定的是：现有结构化 rollout 是否已经足够支持下一次 commanded-finger/contact
decoder。关键证据是 action composition、force proxy、episode split 和 finite 合同均通过，
但 contact/hold near-state 对几乎不存在。root 选择保留 packet 作为 command-variation
工程证据，同时关闭当前 decoder/native 扩展；理由是继续训练只会在未匹配的物理状态上拟合
混合标签，无法改变已知的 contact-preload blocker。若以后重新打开这条路线，最小新增数据
必须在保持 contact/state 的条件下施加成对的 finger command perturbation，并保存对应
的 contact/state provenance；在此之前不接 H→τ、PointWorld、selector、MPC 或 Cm。

随后完成了一次最小 serial contact-stage paired-capture Probe
(`P-20261010-serial-finger-pulse-prefix`)。新增的 diagnostic profile 不改变默认
structured random rollout：它在单环境、同 seed 的 retarget world 中做 zero replay，或在
tick50 对 independently commanded finger coordinate 6 施加一个单 tick `±0.08` residual，
并同步保存完整 action、actor action、`task.real_pd_tar`、state frame 和 actor observation。
GPU2 上两次 zero replay 和正负 pulse 均在约41s完成，543 state frames/542 commands、finite、
requested=applied、zero clipping 和 input/actor hash 合同通过。

zero replay 的全部状态、action、actor observation、contact-force proxy 和 native PD target
逐项 bitwise 一致；±pulse 的 tick50 前缀也逐项 bitwise 一致，pulse residual/action 精确为
`±0.08`，对应 active PD target 改变量约`.064`。这证明了 serial deterministic prefix 和
command/PD-target capture 合同可用，但没有提供 hidden PhysX fork，也不等价于同状态反事实。
单 seed 的 post-pulse pair rate 仅作诊断：zero 为 onset/contact/hold `0.65/0/0`，+pulse
为 `0.60/0.033/0`，-pulse 为 `0.875/0.208/0.033`；不写成 success、preload 或因果效应。
结果卡与审计产物见
[`P-20261010-serial-finger-pulse-prefix`](../src/task/consequence-evaluator/docs/experiments/probes/P-20261010-serial-finger-pulse-prefix.md)
和 `outputs/consequence-evaluator/serial-finger-pulse-audit-{zero-repeat,plus,minus}-20261010-r1/`。

### Decision Note — serial paired-capture contract (executed)

当前需要决定的是：是否继续投入，先构造接触/保持阶段的 matched-state command
perturbation 数据，而不是把现有普通 env 行当作 counterfactual。关键证据是 zero twin 与
pulse prefix 均 bitwise 通过，requested/applied/PD target 可审计；但 post-pulse 只有单 seed
单 tick、隐藏 solver state 未克隆，且当前还没有每阶段20条 matched row/10条 nontrivial
差异。root 选择保留该 serial capture contract 为 `PROMISING` engineering evidence，
不训练 decoder、不跑 native success、不重开 H→τ/PointWorld/selector/MPC。预计下一步成本
是一次有界的多状态 contact/hold pilot；若 visible hand/q/dq/same-pair gate 或 twin gate
失败，则立即回到 `UNCLEAR/evidence-insufficient` 并关闭该数据路线；若达到预注册行数门槛，
再另建实验卡审查是否值得 decoder Probe。当前仍需把科学效应与工程可采集性分开，不能由
本 Probe 升级全局 claim。

随后执行了预注册的多状态 paired-triplet contact/hold pilot
(`P-20261010-paired-triplet-contact-hold`) 的 contact 半臂：96 env/32 triplet
groups，seed211，zero residual 加 triplet-shared initial q-jitter，pulse ticks
120--151。rollout 在 GPU2 上 99.3s 完成，542 commands、finite、zero clipping、时间
全局 `structured_phase` 合同和 residual overwrite 均通过；但 parallel env 的状态与
actor/action 前缀在 tick1 即漂移。严格 audit 为 `0/32` prefix-valid、`0/32`
applied-action-delta-valid、`21/32` same-pair，matched/nontrivial rows `0/0`，状态为
`INVALID_IMPLEMENTATION`（contact audit 见
`outputs/consequence-evaluator/paired-triplet-contact-audit-20261010-r2/`）。这不是
phase 维度 bug：审计已将 time-global phase 独立校验，并把 `done` 按 command transition
对齐；first-drift 位于 pulse 之前，且 ordinary state/action 已失配。

### Decision Note — parallel paired-triplet contract (executed)

当前需要决定的是：是否继续用该多环境 triplet 当作 contact/hold matched-state 数据源。
关键证据是 zero residual、重复 initial jitter、相同输入下仍在 tick1 出现 backend/env-layout
差异，无法满足 hidden-state twin 或 visible same-state 前缀；而 serial 单环境只是工程
capture contract，不提供多状态 counterfactual。root 选择立即停止 hold 半臂，将本 Probe
记为 `UNCLEAR`/contract invalid，不训练 decoder、不做 native execution，也不把它外推为
finger/preload 方法负结果。预计后续若重新收集，唯一保守方案是多 seed 的 serial
control/+pulse/-pulse 重放；其成本高且仍非 hidden-state fork，当前不占用探索预算。

## 2026-10-10 ref7_2 full-action hand retarget follow-up

按 `src/task/consequence-evaluator/docs/user/ref/ref7_2.md` 完成路线 B 的最小
判别 Probe：GPU2 采集 96/64/64 条结构化 retarget rollout（每条 542 步），
覆盖 wrist 平移/旋转、finger 开合/预载、联合动作及 approach/contact/hold；
action capture、q/dq/hand 对齐、episode split、残差边界和零 clipping 全部通过。
第一轮离线 full-action fit 的 test L1 为 0.01637；加入三条 train、 一条 val 和
一条 held-out teacher motion anchor 后为 0.01596，仍仅是 engineering fit 证据。

GT future hand→R→native Gym 的首个对齐复核（teacher held481）中，R 三角色
held10/30/21、hand RMSE44.68/43.59/57.71mm，未达到 held≥90% 且 RMSE<40mm；
native requested/actual 完全一致且零 clipping。query_period=1 的 receding 诊断
同样失败（teacher482，R held0/0/8，RMSE177.12/226.69/207.27mm）。这排除了
“只要缩短 24 步 dispatch 就能通过”的充分解释，但不把结果升级为所有 hand
geometry 无效的正式结论。Probe 总状态 `UNCLEAR`，当前 retarget execution gate
`UNPROMISING`；不进入 H→hand、PointWorld、evaluator 或 Cm claim，返回
ref7_1 的 finger commanded-PD/contact-preload 诊断。证据卡：
[P-20261010-hand-action-retarget-data](../src/task/consequence-evaluator/docs/experiments/probes/P-20261010-hand-action-retarget-data.md)，
产物位于 `outputs/consequence-evaluator/ref7_2-hand-action-*`。

随后做了 source/backend isolation：同 CPU tensor pipeline、同 4-env direct actor
layout 的 teacher packet 使 contextual R 在 env0 future broadcast 下 held=6/9/76、
hand RMSE=61.4/75.8/77.4mm（冻结 GPU source 为0/0/0、109–114mm），确认旧执行失败
混入 source/backend mismatch，但仍未过 gate。用每个 source env 自己 future 的单次
诊断反而 held=0/0/0、RMSE=117.2/101.2/81.3mm；同 CPU teacher env 间本身已有26–37mm
hand variation，不能把它解释为同-state验证。当前停止 query/model sweep，保留隔离
产物与失败 run，后续接触/preload 工作需 source-matched replay 和 contact-force proxy。

同一 source-matched CPU packet 上完成只读 contact-proxy audit
(`ref7_2-contact-proxy-audit-20261010-r14`)：R 三角色的 `surface_gap<=10 mm`
首段均从 tick43 开始，早于 teacher `pair` proxy 的 tick45；几何近段分别在
67/70/136 结束，随后 gap 为 15.87/14.07/11.76mm，而 teacher pair 仍为真。
全轨迹 hand RMSE 为61.43/75.83/77.40mm，近段退出时 wrist/finger误差为
5.56/17.01、6.03/24.72、30.99/51.09mm。source trajectory 与 source packet
字段逐项 bitwise 一致，执行 packet 没有 R-side pair、contact force 或 impulse，
故这些只是 gap/table-support 与 q/dq/action proxy，不能解释 preload 或宣称
contact-loss 机制；同-env q/dq/action 对比也不是 same-state 因果验证。
该审计只收紧了后续 source-matched replay 的观测需求，没有恢复 retarget gate，
不启动新的 R sweep、H-to-hand、PointWorld 或 evaluator 训练。

随后按该 Decision Note 完成了唯一一次 bounded source-matched native force
capture：`ref7_2-same-cpu-teacher-contact-20261010-r1` 与
`ref7_2-hand-action-context-contact-exec-20261010-r2`。两份数据均有限、543
帧状态/542 条命令、requested=applied 且零 clipping；source trajectory 与
packet 的 hand/object net-force、pair、state、action 字段逐项一致。新审计为
`outputs/consequence-evaluator/ref7_2-contact-force-audit-20261010-r2/`。

source teacher/env0 的阈值 pair 为真 498 帧（首个真帧 tick45）。三个 R 角色的
几何 near-gap 首段仍从 tick43 开始，但 R-side force-pair 只持续 23/27/93 帧，
分别在 tick66/70/136 变假；几何 near-gap 首段在 tick68/71/137 才采样退出，
而此时 R hand/object net-force 已为零、source pair 仍为真。这只说明 live R 与
teacher source 的观测时序不同，不能构成 same-state 因果比较。

接触字段是按配置顺序 `index_intermediate, middle_intermediate,
pinky_intermediate, ring_intermediate, thumb_distal` 五个 hand body 的 net force
与可能包含非手部接触的 target net force，pair 只是
`hand norm>.1 any AND object norm>.1` 代理；reset
frame0 标为无效。它没有恢复碰撞 pair、法向力、冲量或 preload，也不改变
`ref7_2` Probe `UNCLEAR` / native retarget gate `UNPROMISING` 的判断。当前仍
不启动 H-to-hand、PointWorld、evaluator 或 Cm 集成；相关运行代码的 future
anchor 广播维度修复和布尔一致性审计修复已由测试覆盖，并在 `053a4bb` 提交。

随后完成唯一一次 matched offline contact-conditioned decoder Spike
(`P-20261010-hand-action-contact-context`, run
`ref7_2-contact-context-fit-20261010-r1`)：同一 96/64/64 split、stride 2、
reset-excluded windows、seed406、GPU2、3000 updates 下，v2 contextual baseline
与零初始化 contact branch 的 v3 共用初始化和 batches。10 个 query-time 字段只含
当前 `pair/surface_gap/support_gap/world object velocity/table footprint`，窗口数为
24864/16576/16576；没有 future contact 或 teacher-anchor 混入。test finger MAE
从 .0190543 到 .0188270（1.19%），pair=true/near-gap 分层分别改善 1.29%/1.34%，
但 contact-onset 恶化 0.25%，未达到运行时 10% gate，故 Probe 为 `UNPROMISING`。
mean-contact 与 shuffled-contact ablation 说明分支使用了输入，但不足以支持继续
contact-proxy sweep。r1 的 manifest 保留旧 HEAD `19a646d` 与 trainer hash；后提交
`221af42` 只补 fail-closed contract/test/metadata，未改写该数值结果。已停止该路线的
更多 epoch、threshold sweep 和 native execution，不解锁 H-to-hand、PointWorld、
evaluator、selector、MPC 或 Cm；卡片见
[`P-20261010-hand-action-contact-context`](../src/task/consequence-evaluator/docs/experiments/probes/P-20261010-hand-action-contact-context.md)。

为落实 ref7_2 “先固定 wrist、再隔离 finger/contact command decoder”的下一最小
问题，完成 `P-20261010-hand-action-fixed-wrist-finger` 的 corrected r2 native
Probe。v2 contextual R 保留 `6:18` finger 输出，只有 `0:6` 替换为当前 source
packet reset-calibrated 的 11-point root inverse + train-only one-step PD；当前每个
dispatch tick 显式读取 live `q/dq`，不读 future joint/force/command label。source-matched
CPU packet、seed282、GPU2、四角色、542 controls 全部完成，requested=applied，整体/
wrist/finger clipping 均为零；finger coordinates 与 v2 输出逐项一致。live teacher held480、
source teacher481，reference floor432.9，但三个 hybrid 角色只 held4/6/8，hand RMSE
7.87/7.82/7.86mm。结论是 fixed-wrist/v2-finger 组合的 wrist 几何跟踪准确但未保持
接触；该单 source/seed Probe `UNPROMISING`，不能把失败单独归因于 learned finger
branch，也不把它升级为所有 finger representation 的正式反证，不再做同一 decoder 的
epoch/threshold/native sweep。r1 因 runner 未显式保证
chunk 内 live-state contract 而保留为 `INVALID_IMPLEMENTATION`，r2 才是可用记录；卡片见
[`P-20261010-hand-action-fixed-wrist-finger`](../src/task/consequence-evaluator/docs/experiments/probes/P-20261010-hand-action-fixed-wrist-finger.md)。
当前仍不解锁 H-to-hand、PointWorld、evaluator、selector、MPC 或 Cm；下一步若继续，
只能转向明确的 commanded-finger/contact state labeling 或审计后续链路，不能把该结果
当作 wrist 或完整 R 的部署上界。

随后完成唯一一次同进程 teacher-finger attribution audit：保持相同 fixed-wrist
decoder，把 env0 当前 live teacher 的 `6:18` action 广播给 env1--3。四个角色均完成
542 controls，requested=applied、各类 clipping 均为零；teacher held `482`，三个
fixed-wrist/teacher-finger 角色仍仅 held `8/10/9`，hand RMSE `7.81/7.87/7.85 mm`。
独立 packet 审计确认 test env finger dispatch 逐 tick 与 env0 teacher 完全一致，wrist
仍由解析 decoder 替换。因此 r2 的失败不能归因于 learned finger branch 单独失效，当前
应把它视作 fixed-wrist/source-state 组合级负结果；不再做 native finger sweep、系数或
阈值 sweep。卡片见
[`P-20261010-hand-action-fixed-wrist-teacher-finger`](../src/task/consequence-evaluator/docs/experiments/probes/P-20261010-hand-action-fixed-wrist-teacher-finger.md)。

### Decision Note — fixed-wrist teacher-finger attribution audit (executed)

当前需要决定的是：r2 的接触失败是否至少可归因于 learned finger branch，还是 fixed
wrist/source-state 映射本身已足以破坏接触。关键证据是 r2 wrist RMSE 仅
`7.87/7.82/7.86 mm` 但 held `4/6/8`，而 offline finger action 差异不足以单独定位
接触原因。root 选择做一次最小同进程对照：保持 fixed-wrist decoder，把 env0 当前
live teacher 的 `6:18` action 广播给 env1--3；不训练、不改系数、不加 future label。
该 542-step run 已完成，oracle 仍只 held `8/10/9`，因此关闭“learned finger branch
单独导致失败”的解释，并保留组合级负结果。该 oracle 仅用于归因，不解锁 H-to-hand、
PW、evaluator 或 Cm；仍无外部授权边界。

随后完成 full teacher-action same-process clone audit。r3 运行时实际为
`cpu` tensor device、GPU PhysX、direct environment rows；source packet 的
`tensor_device: cuda:0` 被记录为 legacy metadata，不再作为拒绝条件。env0 teacher
held `482`，完整 `0:18` action 广播到 env1--3 后 held `482/318/482`，无 clipping 且
requested=applied；但四个 env 在 reset 相同后于 tick1 的 q/hand/object 已分叉。该结果
只证明部分 row 可复现 teacher，不是严格 same-state A/B；结合前一轮 teacher-finger
`8/10/9`，fixed-wrist replacement 仍是主要实践嫌疑，但不能作隐藏 PhysX state 之外的
因果归因。native route 在组合/合同层面继续关闭。

### Decision Note — full teacher-action same-process clone audit (executed)

teacher-finger 对照仍失败后，当前需要区分 fixed-wrist replacement 与多环境执行合同。
root 选择一次更便宜的 bounded control：不替换 wrist、不使用 R 输出，把同一 tick 的
env0 teacher 完整 `0:18` action 广播到 env1--3；仍为 542-step/GPU2，只看 teacher hold、
requested/applied 与 finite/clipping 合同。r3 已完成，结果为 `482/318/482`，且 tick1
即出现 row-level state divergence；因此只保留 fixed-wrist 的实践嫌疑，不作严格因果归因，
也不再做 native sweep。

## 2026-10-09 ref7_1 物体系 GT servo 与接触控制诊断

按用户授权持续自主推进两小时，完成16组4env/64copies/seed282原生GPU Probe，
每组542步，源码/运行manifest/失败packet完整保留。分支main，旧Y与全局Cm claim
保持；旧R、ACT/G/H->V训练仍暂停。所有16组独立执行审计PASS，18项合同测试通过。

历史tick160..176世界GT手误差7.60mm、手物相对误差65mm，失抓后相对462mm。
但物体系每步完整SE3反馈会放大运动：命令臂held105、物体抬升峰6.63m。围绕
world nominal有界20mm/.15rad修正两次held484/483，无裁剪/中途失抓，仅说明
保留source PD预载的命令级控制PROMISING。固定世界命令fresh结果31..484，
角色即使执行同一绝对目标流，接触后也能held483对250；不存在严格same-state因果结论。

11点fixed-root解析腕逆解位置误差<1um。GPU审计coupled12 Jacobian满秩，但
实际18关节rank17、受载关节偏离理想耦合；静态coupled逆解thumb tip约5mm误差。
校准train3前40帧的两系数一步PD逆控制，只读当前q/dq与未来几何。保留source
finger命令时，冷启动腕控制两次held483，worldRMSE2.91/3.77mm，无裁剪；腕部信号
PROMISING。完整11点控制首轮held485却live teacher17未过行为门槛，重复238，
因此完整几何gate仍未通过。源未来q/命令污染的实际分支依赖测试通过，不是输入泄漏。

最终将finger目标换成特权实测next-q，未加/加因果过去PD负载EMA均held0，手误差
3.57/3.61mm、378/391次裁剪。该简单逆控制负信号有效，不能升级为几何表示无效。
离线接触阶段审计发现free-approach模型食指目标误差由.013rad升至.578rad，
它不是精确接触动力学逆解。当前完整路线UNCLEAR，关键缺口是finger几何/实际受载姿态与commanded PD
目标/接触预载的映射。下一最小问题：固定解析腕部，单独检验finger命令解码与
当前机械/手物状态；保留geometry-only和past-command对照。未重训旧R或推进H->V。

一次teacher被旧代理指标记为放回（held305/settled151），已核对桌面末态但未
迁移新控制器；不是完整任务可靠性证据。所有重复角色、不同布局及弱teacher运行
均保持原始gate，不计为独立seed或Cm增益。产物新增约0.35GB，GPU2已释放。
证据：[物体系GT上界Probe](../src/task/consequence-evaluator/docs/experiments/probes/P-20261009-object-relative-gt-servo.md)，
汇总`outputs/consequence-evaluator/object-relative-gt-campaign-20261009-r1/`。

## 2026-10-09 ref7 GT-hand retarget/control upper-bound Probe

用户暂停 ACT 推理和 G(H,A)->hand 路线，改按 ref7 先做几何 action retargeter。
旧Y/U及全局Cm claim保持；尚未训练H->V或接PW/evaluator。main实现两层decoder，
输入GT24x11未来手位移+当前q18/dq18，预测相对查询状态的commanded PD targets。
显式intrinsic XYZ/原生DOF索引、6独立finger及逐horizon归一化；原生wrist为增量，
固定24步绝对目标每tick用机械q反馈转回控制，R仍仅每24步query。

复用完整身份的5条teacher持握轨迹（3train/1val/1test，390/130/130窗口）；
不是完整任务success数据。GPU2拟合2400步27.97秒，val选step1600，test相对位置
目标RMSE27.17mm，不能代替真实手跟踪。首次native preflight因描述性ring/pinky
名称颠倒中止，c28d9eb修复；数值索引/标签/权重不受影响，原checkpoint保留。

修复后4env/64copies/seed282完整542步81.48秒：teacher held484；直接GT PD目标
servo held117/手coordinate RMSE4.35mm；learned R held42/135.89mm，重复目标臂
held0/136.05mm。held>=90%teacher且RMSE<40mm门槛未过，所有完整任务success=false。
GT servo无裁剪，commanded target误差<=5.96e-8，仍在tick176后失抓；说明这次
固定source轨迹贴近不保证抓持。learned前120帧RMSE70.25mm且无裁剪，首次裁剪
tick265，不能把初期失败归因于后期349次finger裁剪。重复臂并非独立seed。

执行审计PASS：23次query，actual=requested bitwise；未来几何t+1、当前q/dq/手
对齐；只读几何和当前状态的checkpoint GPU重放误差6.11e-7；目标转换/重复流一致。
当前冻结learned inverse为UNPROMISING，整条几何retarget/control仍UNCLEAR，
GT command servo也未过持握gate，不能将全局负结果仅归因于R。下一优先级是直接
GT servo失抓窗口的接触/执行诊断及已有packet上的source/live-state inverse复核，
暂不扩大训练或推进H->V。13项相关测试通过，GPU2已释放，新输出<25MB。
证据：[GT-hand retargeter Probe](../src/task/consequence-evaluator/docs/experiments/probes/P-20261009-gt-hand-retargeter.md)。

## 2026-10-09 ACT newest-action receding1 follow-up

用户提出每次只执行一步。新增`receding1`每tick重新预测，执行最新chunk的horizon0，
不混合历史预测，区别于`temporal1`。冻结同一checkpoint/归一化、seed282和4env
native GPU合同，单次542步67.16秒：ACT最高抬升0m/held0、重复ACT held0；
teacher0.827m/held484，repeat held483。542次query、最多1个chunk，逐tick
`action[t]=proposal[t,0]` bitwise成立，native/requested差0，无early done。
14项相关测试通过，GPU2已释放。该冻结checkpoint的one-step remedy为UNPROMISING，
未解决ACT抓取失败，不据此确定唯一根因；下一步仍优先deployment-history诊断。
证据：[receding1 Probe](../src/task/consequence-evaluator/docs/experiments/probes/P-20261009-act-receding1.md)，
产物`outputs/consequence-evaluator/act-receding1-20261009-r1/`。

## 2026-10-09 ref6 ACT inference diagnosis

冻结历史ACT checkpoint `ed26abd5`与其原归一化，不重训。`6751843`补齐按绝对tick
聚合的`overlap8`与`temporal1`，显式时间戳有效性、过期/reset和合法零动作保留。
四个seed282原生GPU group完整542步：open_loop24 held478/最高抬升0.595m，
receding8/overlap8/temporal1均held0；teacher分别held484/297/484/484。
前120步的8步边界wrist控制变化由33.56mm降到19.23/8.58mm，仍未恢复抓取。
四组actual-requested控制差均0，独立加权重建误差至多2.24e-8，query H因果对齐，
各自最多1/1/3/24个chunk、调用23/68/68/542次，重复ACT控制流一致。

真正stride8查询网格的四条val episode上，前8/后16 token MSE分别1.5994e-4/
1.6104e-4，没有前8平均误差特别差的证据；逐帧网格的7%差异不能用于解释receding8。
真实查询网格的同绝对时刻wrist预测差15.40mm，与teacher自然控制变化15.48mm
同量级。时间融合缺失已补齐，但当前checkpoint的抓取失败未解决：本次仅融合
remedy为UNPROMISING，根因仍UNCLEAR，不再默认把硬切换或部署分布说成唯一原因。
open_loop24仍只是抬升/持握基线，最终任务success=false，不能写成完整任务成功。

保留推理/审计工具与失败证据，下一优先级为deployment-history诊断，不扩融合
仿真、不在本轮重训或进入candidate/Cm/Gate1。严格same-state边界仍未改变。
13项相关测试通过，GPU2已释放。证据与命令见
[ACT时间融合Probe](../src/task/consequence-evaluator/docs/experiments/probes/P-20261009-act-temporal-ensemble.md)，
产物`outputs/consequence-evaluator/act-temporal-ensemble-20261009-r1/`、
`act-temporal-offline-20261009-r4/`和`act-temporal-final-audit-20261009-r1/`。

## 2026-10-09 ref5 prospective hand execution bridge and rolling control

用户ref5取代继续扩fork-panel/C2a路线：普通e260随机干预rollout训练
`G(H,A)->24x11hand`，再用冻结PWbest46000和同C1step1000在线选7候选。
旧U32继续冻结，只作teacher；真实目标为任务成功和持续稳定，不是旧U排序。
本轮没有新fork、future-GT部署输入、actor/PW/C1训练或RL收益声明。

自训重建e260SHA8882fabd/s3/full-frame0采集96train293、64val294、64test295，
3360决策窗口；首次桥HA/H各2000步，val选step200，测试点RMSE63.05/63.10mm，
persistence116.73、nominalFK102.71mm。Zero-A活跃窗口MSE仅增加.2747%，
未过额外1%动作gate。仅改变固定native请求单位的匹配r2，用新64episode298测试，
HA/H67.45/67.29mm，zero-A变化-.0403%；量纲调整没有建立动作敏感性。
旧/新test不同，不能把两次RMSE作matched比较。总数据4320窗口，所有episode542步。

Root按ref5的实际任务目标缩小端到端Probe，保留未过动作gate，不用代理指标
宣布桥有效。只按val误差冻结原r1HA200(.008722vs.008818)，同C1三层普通单计划
waterfall teacherRMSE.34452/.39038/.38288，C2a/C2b与C1GT分数差.15058/.16290。
GT本身存在e260迁移误差，不能把控制结果全归因于G/PW。没有新same-H排序结果。

初始两seed296/297完整轨迹各臂bitwise重复，16env重复不能算32独立episode；
已保留并纠正记录。随后只修复评估支持：固定1mm/5mrad/10mrad初始q扰动，
匹配baseline/planner且不同seed实际轨迹不同。最终32episode/arm任务均0，
stable45baseline24/planner26，稳定后失抓1/8，any recovery9/14，干预0/55.97。
稳定抬升匹配IDrescue/harm7/5；完整任务0/0因全失败没有判别力，不是零收益证明。
baseline有23final-placement-not-settled、1中途丢失和8未稳定；planner18/8/6。

2d8c3ba实际路径通过初态actor/RMS/q/object匹配、评分argmax/请求计划、四帧
过去几何、PW重复性、零裁剪和完整542步检查。状态UNCLEAR，不扩百级episode。
代码首次实现无futureGT的G->PW->C1每8步真实replan，但任务收益未证明。
下一步是动作敏感/显式wrist rotation与相对手形预测、C1/source迁移和受控放回
能力边界的区分，避免继续改Y、同样加epoch或扩fork。未放弃核心Cm假设/claim。
14相关合同测试通过，GPU2已释放。实验卡：
`src/task/consequence-evaluator/docs/experiments/probes/P-20261009-hand-execution-bridge.md`；
最终结果：`outputs/consequence-evaluator/hand-execution-control-20261009-r2.json`。

## 2026-10-09 ref4_4 old-U evaluator and actual one-shot screen

用户明确冻结旧`U32=Y7+.25Y3-Y6`作为teacher，不再定义新Y，推进C0(H+A)、
C1(H+A+GT24)、C2(H+A+PW24)。contact仅生成32步标签，future只有24步几何。
C2按用户确认使用实测未来手轨迹作为PW条件，只替换物体未来；是离线oracle，
不能表述为可部署native-control planner。现存192完整扰动episode用于监督，
25x7恢复ref13候选整体不进入拟合/归一化/val选择。历史删除数据未被假装复用。

提交edccfe1完成三臂各1200步，val选800/1000/1200，耗时31.50秒。冻结same-H
候选排序C0/C1/C2为60.26%/75.64%/69.23%，C1 future shuffle为51.28%；
GT增益15.38pp、regret .07083→.05667，GT ranking screen PROMISING。
PW比GT低6.41pp，未过原retention screen。仅78严格pairs/5有效anchors，
不是正式支持或Gate1。PW best46000推理39.38秒；修复CUDA原子group-mean的
重复漂移，采用数学等价的确定性CSR求均值，冻结权重/源码不改；失败运行保留。

继续执行三条冻结one-shot真实90步混合路径（各约40秒）：baseline20/25，
C019/25（harm1），C120/25（rescue0/harm0），C221/25（rescue1/harm0）。
所有评分/执行H exact、几何prefix exact、原prefix合同、实际计划、零裁剪和完整Z90
检查通过；saved scores GPU重放误差0。C2 gain CI含0，控制收益UNCLEAR。
这不是旧teacher的三次replan23/25，也不是完整受控放回或RL学习收益。

同一C1权重直接换PW future的post-freeze离线诊断为73.08%，优于独立拟合C2的
69.23%；未用这个诊断改selector或补跑第四控制臂。因此不能把6.41pp差距全归因于
PW精度。下一步保留旧U和共同C1，在新独立候选/cohort上冻结比较GT/PW信息保留；
不自动加epoch、重训PW、定义新Y或进入可部署planner/RL。Rolling仍属后续证据。
GPU2已释放。实验卡：
`src/task/consequence-evaluator/docs/experiments/probes/P-20261009-old-utility-evaluator.md`；
产物：`outputs/consequence-evaluator/old-utility-evaluator-20261009-r1/`。

## 2026-10-09 ref4_3 learned temporal evaluator

按用户ref4_3授权恢复 learned `Q(H,A,Zgt)` 路线，先用最终任务结果和
`±24/(T-t)`作弱监督，不再直接规定value为TCC/reference delta-progress。
用户明确：最终重新抓稳并受控放回的episode为success；其局部失抓/下落/recovery
窗口mask，重新稳定抓持后恢复正监督。物理阈值只用于binary outcome/mask审计。

复用192完整episode，重新标注train41/23、val38/26、test37/27（成功/失败），
其中test一条恢复成功轨迹由failure改为success；成功recovery窗口mask共48个。
提交5d13cf1在GPU2完成匹配HA/HAZ各1200步，26.66秒、显存约561MiB，
val均选step400。冻结test同时间跨episode排序88.42%/89.67%，future打乱后84.32%；
future gain1.24pp未达预设3pp screen，保持UNCLEAR，不进入完整任务GT control/Gate1。

关键数据限制：训练3968窗口仅128含非零请求计划（3.23%），没有成功recovery训练样例。
模型利用future的信号存在，但这还是结局预测，未验证同H候选排序的任务收益。
旧ref13七候选缓存的25锚点评分审计只用于检查迁移：HA全选grip+，HAZ改变7个选择；
候选A有72/3024个计划元素越出训练坐标范围，涉及5/7个候选臂；分布覆盖不足，
不能解释为控制成功或失败。
下一步信息应来自决策窗口、同H候选和恢复样例，而非延长训练或提前接PointWorld。

提交d02f45b补齐确认失抓前5帧的mask边界，10项相关测试通过。重新准备r2后，
所有数组及windows SHA与r1完全一致，模型/目标代码未变；因此保留冻结权重和测试，
不重复相同训练。原始运行、修正规则与等价审计都保留。GPU2已释放。
实验卡：`src/task/consequence-evaluator/docs/experiments/probes/P-20261009-weak-temporal-value.md`；
产物：`outputs/consequence-evaluator/weak-temporal-value-20261009-r1/`。

## 2026-10-09 ref13 whole-world replay recovery

按用户要求重新检查历史 ref13 方法。近期 host/group 诊断的 Torch2.0.1、强制
frame0/hybrid probability1 与跨 env broadcast 控制，不等价于原先 Torch2.4.1、
hybrid probability0.5、每个 env 自身反馈、跨 fresh world 对应 env 的重放。
因此下文的 host 关闭判断仅适用于那些具体实现，不能排除完整的旧 ref13 方法。

本次恢复 Torch2.4.1+cu121、96env、GPU PhysX/CPU tensor pipeline、单线程、
seed263 和原 r7 配置。原 checkpoint SHA16fd261b 与完整 panel 未找回；使用固定
的自训重建 e260 SHA8882fabd 和目前保留的三条转换 motion，不宣称原权重复现。
baseline 采到45锚点，fresh zero repeat 的45/45均通过旧筛选，记录的 before、
history、actor observation、control、PD target、height/pair 完全一致，Z90均33/45。
repeat未保存完整trace，不能把这些字段的一致升级为所有隐藏状态/几何的exact证明。
当前只恢复了旧执行筛选，不代表正式Gate1通过。

冻结的current-only同步s3组为tick71的25个env，重采baseline Z90为20/25；
reanchor的记录前缀误差为0。七候选与每条实际滚动路径都已完成：one-shot
baseline/原U/新P[t+24]-P[t]为20/25、21/25、22/25；三次真实8步决策后分别为
20/25、23/25、22/25。新Y对baseline救回2/harm0，原U滚动救回3/harm0；新Y
继续重规划没有增加成功，少于原U滚动一个救回。所有新增成功都在frame0组，
不是hybrid中途初始化造成：该17env组baseline12、原U滚动15、新Y14；其余8env
各臂均成功。首轮新旧选择21/25不同，原U选3次非baseline、新Y选22次。

35候选panel与6真实mixed执行均通过旧full-world前缀合同，候选评分所用的object/
11hand-point query前轨迹bitwise一致，初始动作无裁剪且实际进入运动。冻结bank/
TCC、actor、Y和Z判据没有调参。工程旧方法可继续使用；新Y one-shot有小幅局部
正向信号，但不能视为优于原U滚动或正式Gate1通过。结果保持UNCLEAR：单seed、
单motion、25锚点，收益CI下界为0，且只观察原Z90而非完整episode。本轮不扩跑、
不训练evaluator/PointWorld。下一步可在恢复后的合同中预先固定selector，做完整
episode的new-Y/baseline最小对照；更广reference bank和cohort是后续证据。
条件实验约2231秒、产物约1.37GiB，均在预算内；GPU2已释放，17个相关测试通过。
实验卡：`src/task/cm-interaction-oracle/docs/experiments/probes/P-20261009-ref13-progress-recovery.md`；
产物：`outputs/cm-interaction-oracle/ref13-progress-recovery-20261009-r3/`。

## 2026-10-09 GPU synchronous group Value-noise follow-up

在提交`4258adf`中修正了同步GPU group的角色合同：baseline、zero、positive、
negative现在由显式`role_index_map`绑定，`candidate_calibration_valid`不再在
reactive packet失败时无条件为true；runner和离线审计同时记录p99、峰值tick和首次
非零tick，避免接触脉冲被展平p95掩盖。旧r18的非默认zero pair没有角色元数据，新的
审计会拒绝它，之前按env0 baseline解释的r18 Value摘要不再采用。

按用户建议在原生`gpu_physx_gpu_pipeline`中做了两个4-env、seed282、query48、72-step
工程Probe。默认布局`zero=[0,1], candidate=[2,3]`的短窗baseline最高抬升0.2585m、
held15，物理bank Y为`[.043592,.043215,.022880,.024176]`，zero-pair标量中位噪声
`.000378`，正/负对baseline为`-.020713/-.019416`。交换布局
`zero=[2,3], candidate=[0,1]`的短窗baseline为0.1530m/held9，按角色映射后的Y为
baseline`.035543`、positive`.039070`、negative`.020899`，zero中位噪声`.002624`，
正/负对baseline为`+.003527/-.014644`。positive对比随角色布局变号且量级明显变化，
因此不能视为稳定的candidate effect；两包都保持engineering-only，没有进入Gate1。

两次zero pair的raw contact-force差异都在tick44首次出现；flattened p95可为0，但峰值
约8.1/26.2，说明p95-only gate不足。结论：关闭这条role-permutation scalar-Y
noise route，保留原生GPU group作为行为容器；不扩rolling Probe、不fit evaluator、
不进入PointWorld，不改Y、reference bank、TCC或policy weights。实验卡与产物：
`src/task/consequence-evaluator/docs/experiments/probes/P-20261009-gpu-group-value-noise.md`、
`outputs/consequence-evaluator/gpu-group-value-noise-audit-20261009-r1/`和
`outputs/consequence-evaluator/gpu-group-noise-audit-20261009-r1/`。

随后做了一个仅改变`physx.num_threads=1`的原生GPU pipeline Probe（提交`98cec12`），
仍为GPU PhysX、GPU tensor、GPU actor、seed282、4-env、72步。baseline最高抬升
0.2740m/held13，object zero p95为`8.36e-4m`；raw contact差异仍在tick44首次
出现，contact p95为0但max约19.5。物理bank Y为
`[-.011617,.011970,-.016655,.015730]`，zero标量噪声`.023587`，噪声大于候选
对比并把negative误选为最高。单线程调度没有恢复执行合同，关闭该backend变体，不跑
full542确认；strict Gate1、evaluator、PointWorld和policy/reference/Y仍保持冻结。

## 2026-10-08 Gate1 首轮完成：工程通过，任务收益 UNCLEAR

64522f1在CPU-PhysX/GPU2模型执行1个paired episode（seed282），314.94秒完成。
12候选、重复零分支、4次chosen8step混合prefix及全542步rolling均通过严格
canonical/RNG/控制/观测重放。tick48选positive，Y0.031013对baseline0.018941；
其余3次维持baseline。独立成功baseline0/1、rolling0/1、rescue0、harm0；
最高抬升0.74cm/2.48cm、held0，未过Gate1，不训练evaluator。

关键限制：CPU baseline未能抓起，而同seed/同权重的GPU预检连续hold483步，
最高抬升81.51cm；首步控制几乎相同，首步物理/观测即开始偏离。CPU后端解决
严格重放，却改变策略表现，不能据此否定原GPU策略或新Y。保留原e260的
64env历史56/64 lift、24/64 hold45、18/64 hold后无laterdrop结果（不同指标）。
单样本GPU推理异常已由d603153修复；GPU接触force/history重放仍待解决，
不放宽容差。下一步恢复代表性GPU执行与同状态GT合同，暂不扩大CPU seed。
GPU2已释放。完整结果、实验动机与限制见
[Gate1卡](../src/task/consequence-evaluator/docs/experiments/probes/P-20261008-gate1-gt-progress.md)。

随后按历史 ref13 合同加入并验证 `gpu_physx_cpu_pipeline` backend：GPU PhysX、
CPU tensor pipeline、`num_threads=1`、GPU actor。seed282 的 fresh72-step baseline
与 fresh48+24 zero candidate 在73个状态、73个RNG、72个control及contact/history
字段上全部 exact；但单环境 host full baseline 只抬升15.49cm/held8，仍远低于
匹配 GPU pipeline 的81.51cm/held483，因此没有进入 Gate1，也不能把该差异归因于
Y。详见 host audit
`outputs/consequence-evaluator/gate1-host-engineering-20261008-r3/host-backend-audit.json`。
当前 Decision Note：停止单环境 host Gate1，下一步在相同 host 合同下做一次同步
多环境 group replay，检查历史 ref13 的 solver 排程是否是行为保持所需条件。

同步 group probe 已完成：4-env host full542 最高7.91cm/held2；按 ref13 主组规模
96-env 的 bounded72-step 最高7.38cm/held3，均未恢复原 GPU baseline 的81.51cm/483
held。两组都保持同一控制流，但不同 env 间初始/动力学字段并非 bitwise exact，
tick44 contact-force drift 约5.7e-5。结论是 host 单环境与同步 group 都不能承载
当前策略的 Gate1，同步路线关闭；不跑正式 Gate1、不扩 CPU seed、不改 Y/标签/权重。
完整 group packet 与 mismatch audit 保留在
`outputs/consequence-evaluator/gate1-host-group-engineering-20261009-r6/` 和
`gate1-host-group-engineering-20261009-r7/`。下一步只保留原 GPU 行为下的执行合同诊断
或新的可证明重放设计。

作为对照，同一4-env synchronous harness切回原 GPU PhysX/GPU tensor pipeline 后，
baseline 恢复到最高81.28cm、held480、first stable tick107，说明 group 排程本身
没有毁掉原策略；但初始 state/history 已不同，tick44 contact-stage 仍不满足严格
same-state contract。因此保留该行为对照，但不放宽 exact replay，也不进入 Gate1。
审计见 `outputs/consequence-evaluator/gate1-gpu-group-engineering-20261009-r1/`。

随后在原生 GPU PhysX/GPU pipeline 内完成了4-env同步 zero-pair 噪声校准：env0
baseline、env1 zero repeat、env2/3在t=48 transition起的24个action tick分别加入
正/负固定残差，所有角色在query前共享env0控制。四次fresh full-baseline分别为81.28cm/480、82.39cm/484、
75.35cm/484、13.01cm/7，说明reactive policy的跨启动行为本身有大幅波动。最新
r4的24步候选效应相对同窗口zero displacement p95比值在object/hand/q位置约
1.34--2.87，但q速度、history和contact多为约0.36--1.18；没有形成一致的
candidate\(\gg\)solver-noise margin。该组只作为工程校准，不是正式Gate1或候选
价值证据。审计见
`outputs/consequence-evaluator/gate1-gpu-group-engineering-20261009-r4/`
及实验卡的 Native GPU group noise calibration 小节。

当前不跑正式Gate1、不fit evaluator、不扩CPU/host seed，也不改Y、reference bank
或policy weights。runner已补上flattened native tensor按env reshape、q/dq/contact
字段、初始semantic role guard（不一致即停止校准）及query-relative noise ratio；
当前4-env只有一个zero pair，candidate env2/3各自的query前漂移仍未被独立配对。
下一步在新执行合同前
先评估冻结ACT-like 24-step proposal baseline，或提出新的native GPU replay contract。

随后补做原生GPU group完整行为 gate：4-env、542 steps、seed282、同一
`gpu_physx_gpu_pipeline`。env0 baseline达到0.799920m、first stable tick106、
连续held481帧并保持到终点，行为保持 gate 通过；env0/env1控制完全相同，初始
object/q/dq/hand/gap/history/contact semantic gap全为0。状态仍非exact twin：
hidden hash首帧不同，derived geometry从tick1漂移，tick44附近contact/support
差异放大。候选相对zero displacement的后查询p95比值仍不一致（object pose
0.55/1.49、history0.68/0.70正/负示例），因此该结果仅是engineering/noise
calibration，不能进入GT scoring。完整产物见
`outputs/consequence-evaluator/gate1-gpu-group-engineering-20261009-r9/`。
原生GPU group保留为行为容器；exact same-state Gate1仍关闭，下一步落实冻结
ACT-like native-action chunk，再在该容器中重测。

## 2026-10-09 ACT-like native action chunk engineering

新增 `consequence_evaluator.action_chunk` 和
`tools/run/train_action_chunk.py`：输入 raw `history[t]`，一次输出未来24个
18-D native executed controls；目标取 `pre_physics_step` 捕获的 `action`，不取
`residual_plan`，不生成hand-flow。合同测试5项通过。labeled hold-audit中21条
同一`s3_airplane_lift`/`airplane_base` clean `expert_success`仅作engineering
数据（manifest仍`training_allowed=false`，显式`--allow-audit-only`，输出强制
`engineering_only`）；冻结当前checkpoint `running_mean_std`。17/4 episode holdout
的stride8 fit把chunk MSE从均值基线9.38e-4降到1.61e-4，first-action MAE0.01090，
只构成action-space PROMISING信号。

GPU group 542-step行为 screen（seed282）将reactive teacher、ACT、reactive repeat、
ACT repeat放入同一原生GPU进程。`open_loop24`中ACT最高0.7939m/held478，teacher
0.8134m/held483，均完成受控末端几何；23个chunk均一次生成后原样送入native
pre-physics。产物
`outputs/consequence-evaluator/act-native-chunk-engineering-20261009-r2/`。
但最终所需`receding8`在同一checkpoint下held0、未抬升；将21条全部用于fit把离线
MSE进一步降到4.53e-5仍然held0（r5），说明降低该离线拟合误差不足以恢复行为；
当时尚未区分chunk硬切换与部署分布误差。后续ref6推理诊断见
[时间融合Probe](../src/task/consequence-evaluator/docs/experiments/probes/P-20261009-act-temporal-ensemble.md)。r3/r5均仅engineering，
不进入GT scoring。

因此当前只保留`open_loop24`作为ACT-like行为基线，关闭该直接模仿checkpoint的
4--8步receding route；Y、reference bank和policy weights保持不变。下一步可在
query处用带prefix provenance的记录nominal chunk测候选效应，或先采集专门的
reactive deployment distribution后再训练route-conditioned proposal。详见
[ACT chunk Probe](../src/task/consequence-evaluator/docs/experiments/probes/P-20261009-act-native-chunk.md)。

随后补做了单环境 native GPU `open_loop24` 行为 screen（seed282、72步、tick48
一次生成24步chunk）。runner在`706f0a2`补齐了proposal manifest认证、实际/请求
control逐tick审计、prefix/RNG provenance和engineering-only边界。修正后的r2 packet
`outputs/consequence-evaluator/gate1-act-single-open-loop-20261009-r2/`控制合同完全通过
（actual-requested最大差`0.0`、proposal只调用1次、无未来observation feedback），
但对象72步最高抬升`0.0m`、held`0`，终点高度`-0.001388m`，未过预注册的`0.20m`
行为阈值。相同seed/backend的reactive-only 72-step fresh control
`outputs/consequence-evaluator/gate1-act-single-reactive-20261009-r1/`达到`0.2915m`、
held`13`；不过两次fresh process在tick44起contact/history和state/RNG hash已分叉，
不能把它们当作same-state ACT对照。首次r1因requested-control shape广播bug在运行时中止，
已保留并标为实现错误。当前单环境 reactive-prefix+frozen-chunk 执行合同关闭，不跑
serial ACT cluster；结果保持UNCLEAR engineering evidence，不升级为策略、GT value或
Gate1负结论，reference bank/TCC/Y和policy weights保持不变。详见
[单环境ACT Probe](../src/task/consequence-evaluator/docs/experiments/probes/P-20261009-act-single-open-loop-behavior.md)。

在该 screen 之后做了只读的 Isaac Gym API feasibility audit（无 simulator、无 GPU）。
归档 Torch2.0.1 runtime 的 `gymapi.Gym` filtered surface 只有 actor root/DOF/rigid-body
state、net contact-force/force-sensor tensor、rigid-contact query 和 sim rigid-body
getter/setter；没有 contact-manifold、warm-start、solver-island、hidden-cache 或
serialize/snapshot/restore API。Task twin contract 的 `SOLVER_CONTRACT` 也固定为
`fresh_simulator_prefix_replay`，拒绝 warm PhysX restore；公开 tensor 加 Python/NumPy/
Torch RNG restore 不能冒充 hidden-state twin。审计产物在
`outputs/consequence-evaluator/hidden-physx-api-audit-20261009-r1/audit.json`，卡片见
[PhysX state API audit](../src/task/consequence-evaluator/docs/experiments/probes/P-20261009-hidden-physx-state-api-audit.md)。
因此当前没有值得重复的 synchronous group、serial noise、CUDA synchronisation 或
deployment-fit 仿真；严格 Gate1 blocker 已收窄为需要新的可验证执行合同，或在
Decision Checkpoint 明确改变 Gate1 claim。

后续 candidate engineering probe 已保持原生 GPU PhysX/GPU pipeline，并把四个
env的 tick0--47 controls 统一为同一已成功 packet，tick48--71 广播一个冻结 chunk，
再只给 env2/env3 加正/负残差。学习 proposal 的 r6--r8 在 reactive query state
上没有保住 nominal grasp（最高约0--0.012m），且 zero pair 在 query 前已经出现
约0.0786m object drift、20级 q velocity 和26--32级 history drift；这不是可解释的
candidate effect。修正原生 RMS `sqrt(var+1e-5)` 与 `[-5,5]` clipping 后的拟合和
542-step screen 保存在 `act-native-chunk-engineering-20261009-r10/r11/`，ACT
仍未抬升。用成功 r9 reactive prefix 的 recorded nominal chunk 重放（r15）最高
0.240m、held11，zero-pair object displacement p95仍约0.338m；显式
`zero_pair_gate` 和 nominal behavior gate 均失败。r6--r15 全部保持
`engineering_only`，不进入GT scoring；candidate effect不作价值证据。下一步先
解决 recorded prefix/chunk 的行为与zero-pair gate，再决定是否收集部署分布重训，
不跑正式Gate1、不fit evaluator、不改Y/标签/权重。

随后用空闲GPU2完成r8的72-step runner guard validation（18.2秒）：初始
object/q/dq/hand/gap/history/contact semantic gap全为0，env0/env1 control exact，
且state packet确实包含tick49--72的完整candidate窗口；但hidden world-frame hash在
tick0即不同，之后仍有contact/trajectory drift。短窗口baseline最高24.17cm、held10，
不能判断full-episode抓取行为。r6/r7仅是native worker相对路径保存错误，已修复且
不产生仿真结论；r8审计见
`outputs/consequence-evaluator/gate1-gpu-group-engineering-20261009-r8/`。

随后做了16-env native GPU multi-zero follow-up。r17的72-step nominal group只达到
0.2536m/held10；r18用同一16-env prefix录制的reactive nominal chunk，在选定zero
pair env1/env4下达到0.2743m/12和0.2617m/10，positive/negative约0.253m/13与
0.259m/9。行为gate在72步没有评估为full542，`candidate_calibration_valid=false`，
所以这些数不能称baseline保持或candidate成功。选定pair的query前p95较小
（object1.83e-5、hand3.29e-6、q位置8.89e-6、q速度1.43e-3、history3.91e-5），
但max/contact仍有尖峰。runner随后改成同一个common pair的p95汇总，并要求至少
80%的zero pairs同时通过各字段阈值；离线重算r18为62/91=68.1%，multi-zero gate
失败，只作诊断。当前pairwise仍只含query前统计，尚未形成所有zero pair的post-query
displacement noise floor；因此不跑正式Gate1、不fit evaluator、不改Y/reference bank/
policy weights。完整packet见
`outputs/consequence-evaluator/gate1-gpu-group-engineering-20261009-r17/`和
`...-r18/`，实现提交为`429d9ff`（common-pair p95为`e4a97d6`）。

按用户建议再做了当前代码下的4-env复核：r19原生GPU group完整542步恢复了行为，
env0达到0.8262m、held481，behavior gate通过；但env0/env1 query前p95在hand、
q位置、q速度和object velocity仍超过robust阈值。用r19 prefix和tick48--71
recorded nominal chunk做candidate r21时，fresh72-step nominal只达到0.2607m/held11，
behavior gate未评估且`candidate_calibration_valid=false`。positive/negative相对
selected zero displacement的p95比值在object pose为2.52/1.10、hand为2.06/1.33、
q位置为5.20/1.72，但q速度为1.07/0.79、history为1.14/0.69，contact也低于1，
effect margin gate失败。结论是4-env process可以保留full reactive behavior，但不能
把fresh candidate window当成稳定nominal或GT候选证据；下一步若继续，只补齐所有zero
role的post-query displacement分布，或改造replay contract。r19/r21 packet分别在
`outputs/consequence-evaluator/gate1-gpu-group-engineering-20261009-r19/`和
`...-r21/`，仍不跑正式Gate1或evaluator fit。

随后新增只读审计入口`tools/audit/audit_gpu_group_gt_values.py`，用冻结的物理
reference bank/TCC encoder复算r19/r21的24步GT progress。两次progress-start
range都为4.90e-6，超过严格same-state的1e-9；r19四角色Y为
0.041369/0.042313/0.017920/0.041562，r21为0.042281/0.044033/0.040786/0.016099。
两次raw argmax都是zero-repeat，按固定0.01 deadzone都选择baseline。审计只作为
engineering contract evidence，不进入strict score worker或Gate1 utility claim；输出
在`outputs/consequence-evaluator/gate1-gpu-group-engineering-20261009-r22/`，工具提交
为`10ae238`。

随后补齐了同步GPU group的query-relative post-query noise审计。只读工具
`src/task/consequence-evaluator/tools/audit/audit_gpu_group_noise.py`（提交
`110cd13`）拒绝非candidate-role packet，检查所有role的prefix controls/done、状态
时间轴和zero-role集合，并对每个zero pair报告query-relative displacement。r18的14个
nominal role共有91个zero pair：object pose的post-query displacement p50/p90/p95为
0.2522/0.5375/0.5645m，q velocity为5.47/9.85/9.87，history为0.853/1.424/1.524。
候选增量相对所有zero pair median p95的描述性比值只有约1--2倍（object pose正/负
p95为1.96/1.74，history为1.55/1.81）；r21只有一个zero pair，object pose正/负
为2.02/0.86、history为1.21/0.87。该比值口径与runner selected-pair effect-margin
不同，不能称为candidate≫noise，也没有形成Gate1证据。审计输出在
`outputs/consequence-evaluator/gate1-gpu-group-engineering-20261009-r24/`。

对保留完整hold的r19再做同一审计（r26）：zero-pair object pose displacement p95为
0.1195m，正/负candidate增量比为4.28/1.95；但history为1.42/0.81、q velocity为
1.27/0.93，且current state仍不exact。因此即使成功hold的单次launch也没有跨字段的
candidate≫noise分离，不能作为Gate1候选排序。

为隔离group-size actor GEMM的可能影响，曾在提交`949a163`临时把group actor改成
固定env0/64-row推理后广播；r25使用原生GPU PhysX/GPU pipeline、seed282、4 env、
542 steps，但baseline最高抬升只有0.0473m/held4，而旧r19为0.8262m/held481，
behavior gate失败。该改动已由`b0c4b33`/`ecc4def`回退；r25只作为负的工程Probe，
不能解释旧r17/r19差异，也不能作为新的执行backend。当前保持reference bank/TCC、Y、
policy weights不变，不跑正式Gate1、不fit evaluator、不进PointWorld。下一步需要真正
的同一隐藏状态fork，或有足够重复数的baseline/candidate统计设计；同步group仍只作
工程容器。

在旧group actor合同下又做了一个seed283、4-env、542-step复核（r27）。该次env0最高
抬升仅0.0891m/held5，negative role为0.5663m/398；zero pair的post-query object
pose displacement p95为0.8247m，正/负candidate-vs-zero incremental比值为0.82/1.29，
history为1.11/1.68。它与旧r19的0.8262m/held481形成鲜明差异，说明原生GPU接触执行
分布在fresh process/seed间本身不稳定，candidate effect没有超过这种运行级波动。r27
packet和noise audit在`outputs/consequence-evaluator/gate1-gpu-group-engineering-20261009-r27/`；
不把r27或r19升级为Gate1、GT utility、evaluator或PointWorld证据。

用冻结physical-reference/TCC bank复算r19、r21、r27的GT progress后，r27四角色Y为
0.016152/−0.000431/0.020753/0.036116，progress-start range为1.72e-6；raw argmax和
固定0.01 deadzone都选择negative residual（相对baseline +0.019965），而r19/r21都选
baseline。三组都超过1e-9 strict same-state容差，因此该choice flip只说明执行敏感性，
不是GT utility或candidate成功。审计输出在
`outputs/consequence-evaluator/gate1-gpu-group-engineering-20261009-r28/`。

## 2026-10-08 ref4_2 成功机器人 reference bank

用户ref4_2明确替换原始运动作为唯一value时间轴的选择，先用已有source230成功
机器人rollout bank检验物理执行gap。340b847完成固定8参考、1000步TCC、同8条
source261开发标签对照，未调temperature/max_step/epsilon。四条clean tick176
Y由−0.0233..−0.0251变为+0.0432..+0.0442；负窗口5/6/6/8→全0，累计回退
0.249..0.368→0..0.002428，全部既定开发检查通过。Probe **PROMISING**，保留
物理reference路线；不是正式任务价值/策略收益结论。旧原始R结果保留。
失败e1末尾P0.0233；其tick51→75窗口仍在抬升，严重落下在之后，不能用later
failure污染局部Y；窗口内已开始分离，延迟风险和OOD仍待同状态候选排序验证。
GPU1完成训练10.12秒/标签27.38秒并释放，44测试通过。下一步按完整链路做
fresh simulator完整prefix replay的GT-value Gate1；当前未运行Gate1，未fit
evaluator，标签training_allowed=false。结果与限制见
[物理reference bank Probe](../src/task/consequence-evaluator/docs/experiments/probes/P-20261008-physical-reference-bank.md)。

## 2026-10-08 Reference-conditioned delta-progress 监督

用户ref4_1替换airplane-specific S/P/M：Y=P[t+24]-P[t]，P来自原始成功参考
与实际3D历史的因果clip匹配；不是episode的t/T。用户确认reference使用原始
成功动作，按Inspire URDF重建同定义11机器人手点。复用固定Google XIRL的
距离/因果context helper，增加允许前进、停滞、回退的有界历史prior；不使用
全episode OT、反向cycle或窗口之后的恢复。旧数据/标签不改，S恢复问题不再
阻塞新路线。六项因果/重复阶段/停滞/回退工程测试通过，先做train-only8条
离线标签Probe再判断是否训练；未启动evaluator。当前在main开发，不push。
[标签Probe](../src/task/consequence-evaluator/docs/experiments/probes/P-20261008-reference-progress-labels.md)。
几何标签r1/r2已完成8条/496窗口，因果截断误差0；修复早期prior支路丢弃后，
四条clean最终P仍仅0.262..0.437且有较多回退，UNCLEAR/不可训练。固定R不变，
转向[TCC对齐编码器Probe](../src/task/consequence-evaluator/docs/experiments/probes/P-20261008-tcc-phase-alignment.md)，
只用source230train拟合geometry embedding，source261仍作已暴露开发检查。
TCC在ef9a770完成固定1000步（GPU1/10.84秒），8条496窗口标签重放9.69秒。
四条clean末尾P提高到0.932..0.994；self/static/因果检查通过，但累计回退
0.249..0.368均超过预设0.2门槛，局部Y仍UNCLEAR。再次抬升与末尾放回附近
出现负Y；固定checkpoint和标签保留、training_allowed=false，GPU1释放。
下一步核查已有局部阶段证据，不以低cycle loss或末尾P代替候选排序正确性。
用户完整链路要求标签过后先直接GT-value同状态滚动Gate1，不先训练evaluator；
允许fresh simulator完整prefix replay，禁止mid-state PhysX restore。Gate1成功
定义为完成稳定抓取并正常放回，允许中途恢复；中途失抓另统计。旧32anchor脚本
仍在但原输出目录缺失；旧23/32→27/32不能替代新Y验证。未运行新Gate1。

## 2026-10-08 DExplore 官方数据生成器筛查

用户授权尝试归档官方策略。官方 `inspire.pth` SHA8f6823db… 已核对并跑通；
可用完整运行时为 `.runtime_envs/dexplore_v120_train`（Torch2.0.1+cu118、
rl_games1.1.4），conda Torch2.2.2 环境存在依赖/扩展ABI缺口。采用进程内旧版
factory、单次外部RMS及严格compiled-key适配，不修改外部权重或环境。
同s3 airplane输入/seed239各64条完整episode：官方lift5/hold45为64/64、64/64，
自训为56/64、24/64；保持后直到episode结束不掉落为官方0/64、自训18/64。
原reference末帧回到初始高度+0.5mm，完整片段包含放回，与全程保持的S标签
存在任务终点冲突。源数据定义的放回前tick481仅作事后prefix诊断（63/64 vs18/64），
不能替代原门槛或新固定reference评估。用户明确选择完整reference并把受控正常放回
计为Task成功；新几何标签另行验证，不裁剪reference、不把落桌都计为成功。未启动fit，
官方权重仅为数据生成候选，不替代Mission的自训练最终策略。评估已结束且GPU释放。
记录：[官方生成器Probe](../src/task/consequence-evaluator/docs/experiments/probes/P-20261008-official-generator-screen.md)。

按用户的完整reference/受控放回定义，9ced94d在GPU2完成64条nominal连续episode，
115.057秒、约140.53MiB；S/P/M几何弱标签为62成功/2失败、4160窗口、256偏好。
62条正例末尾均已离手且在桌面proxy上静止；两条负例在放回阶段失去几何控制/支撑。
这是自动数值核查，未人工确认视频或精确collision pair。独立value schema不放宽旧
H匹配验证合同；只有train seed230且负例不足，labeled `training_allowed=false`。
未开始evaluator fit，采集进程已退出/GPU2释放。下一步为保留clean的阶段定向扰动
Probe，再考虑独立held-out seed组，不能用nominal零计划声称动作收益。
记录：[完整参考value数据Pilot](../src/task/consequence-evaluator/docs/experiments/probes/P-20261008-full-reference-value-data.md)。

后续已完成经验placing、同H策略误差holding/contact及0.5dose采集。0.2contact
三组59/5、64/0、63/1，独立负例不足，原样保留且未fit。新的固定0.5dose（平移仍
0.05，原生实际控制仍[-1,1]）在98ceb76完成192条完整episode：train41/23、val38/26、
test36/28；11904窗口/768偏好、来源组261/262/263互斥，机械数据门槛通过。
官方/自训同H动作投影复现最大误差5.96e-7；所有96计划真实触发，实际命令符合
裁剪后请求，逐episode物理标签重算通过。输出约421MiB raw+69MiB窗口，GPU2释放。
仅生成数据，旧local-event trainer仍不接受新schema。样例出现“中途失抓后恢复放回”
的语义边界，已向用户提问，当前S会判失败；原标签保留，未启动evaluator fit。
历史下一步原为明确恢复成功定义再适配S/P/M训练；当前已被上述ref4_1路线替代。
记录：[value强度校准](../src/task/consequence-evaluator/docs/experiments/probes/P-20261008-value-dose-calibration.md)。

## 2026-10-08 HOCap 外部测试数据获取

用户指定 HOCap 为外部测试候选。官方 Box 的 calibration/models/poses 三包已下载，
SHA256 与第三方 HF 镜像对应 LFS 身份一致，ZIP CRC 校验/解压通过；通过 HTTP
Range/ZIP64 取得全部64条序列元数据及一相机连续32帧标签，不下载整套 RGB-D。
9主体/64序列/72944帧/64物体；shape/mesh引用/手侧/-1缺失值/四元数检查通过。
样本128次物体-帧比较中，相机标签转世界与原生pose平移最大差4.57e-8m，
仅证明发布数据内部坐标一致。新增磁盘约246MiB，全部获取任务已结束，未用GPU。
输出：`outputs/cm-pointflow-effect-pretrain/hocap-test-preflight-20261008-r1/`。
保留test-only：不加入训练、归一化或checkpoint选择；原始FPS/时钟仍未核实，
`test_ready=false`，尚未转换为30Hz H4/K24正式测试或运行模型评价。
[官方合同与待确认项](../src/task/cm-pointflow-effect-pretrain/docs/research/HOCAP_TEST_PREFLIGHT.md)。

按用户要求，8f857ae已在GPU0完成固定latest50000的HOCap帧序列Probe，49.425秒。
全部64序列/9主体，128运动窗口第24帧点EPE17.805mm vs静止63.704mm（降低72.05%）；
64序列均衡near-hand自然窗口20.346 vs54.985mm。但其中17近静止窗口模型22.160mm
vs基线0.356mm，存在明显虚假运动；不能用总体均值掩盖。源码/数据1318项哈希通过，
手21点与32帧发布标签通过10微米门槛；3项合同测试通过。原始FPS仍未核实，
仅以数组帧索引H4/K24和nominal30/900特征尺度进行无拟合评估，UNCLEAR/test_ready=false。
未调参、未换checkpoint；进程已退出/GPU0释放。
[协议与分层误差](../src/task/cm-pointflow-effect-pretrain/docs/experiments/probes/P-20261008-hocap-frame-generalization.md)。

同一HOCap冻结192窗口补测Oak-only latest10000与混合latest50000：
GPU1/a3139fb的r3在40.990秒完成，1323输入哈希通过，混合版复现此前结果，
全部进程退出。128运动窗口h24点EPE22.421→17.805mm（降低20.59%）；
自然64窗口24.675→20.346mm（降低17.55%），但其中17近静止窗口
8.039→22.160mm（2.756倍），虚假运动加重。保留运动收益与近静止退化分别汇报。
两版相同模型/归一化；混合版额外64250更新及loss/batch变化，不能把差异单独
归因于混合数据。原始FPS仍未知，仅帧索引Probe/UNCLEAR，不证明Cm策略收益。
两次工程复现阈值失败原样保留；GPU亚微米数值波动修正为1微米/1微弧度复现检查，
没有改预测、拟合、调参或选择checkpoint；5项合同/回放测试及verify通过。

## 2026-10-08 迁入 main 开发

用户授权将 consequence-evaluator 的已提交历史和当前用户/并行 agent 修改一起
提交并合并到本地 main，后续直接在 main 开发；不推送远程。
合并前 main 是本分支的祖先，可快进保留完整历史；实验卡中的 branch/git_commit
继续表示实际运行身份，不随当前开发分支更名。

## 2026-10-08 用户修改审查与当前 blocker

consequence-evaluator 用户新增的H匹配、最小唯一配对覆盖和重放状态合同保留。
修复repeat验收遗漏未来手/done、生产门槛误封只读审计、native物理枚举/dtype
序列化，以及pickle对象共享导致内容哈希在save/load后变化的问题。
六角色airplane/duck/cup资格36/8/63（各64条），mixed12/train5/balanced5为0/5/4；
生产route仍不可训练。216条旧连续episode经H匹配重新审计，唯一pair=1/1/1，
低于8/4/4；没有启动evaluator fit。独立native twin r4（a7d5146/GPU0）通过：
三fresh进程同前缀状态/RNG精确一致、repeat全部future/action/done/native状态一致，
请求/实际control L2分别0.75865/1.14996。它是engineering_only且不可训练，
不改变ref2 continuous schema或科学结论。用户明确保留连续rollout、不fork；
twin只作工程诊断。对全部可比较窗口穷举后，物理匹配候选4/1/3、加H仍为1/1/1，
配对选择器未漏掉合格episode pair。随后完成不可训练的连续保持阶段定向诊断，
核对合格飞机专家的局部反例和匹配覆盖，生产资格和状态阈值未放宽。
连续hold审计7389d05已完成48条episode（24clean/24hold，23实际触发），
有11403可靠progress帧，但仅1个train唯一pair。93dde07逐帧审计17053事件窗口，
463事件可比较episode pair仍仅1个物理/H匹配；不是8步采样漏配。
12条带hold负事件episode的最近正样本中，11条超过手RMS、10条超过物体旋转阈值。
诊断产物均不可训练，不补采val/test、不启动fit，也不将此解释成Cm无效。
130项Task测试及verify通过；原始失败run、用户其他修改均保留。
GPU1/2三源PointWorld已按原冻结配方完成50000步，group/trainer均COMPLETED，
两rank参数哈希相同，全部自有进程退出，GPU0/1/2释放。
最终固定balanced moving-anchor h24：Oak13.778/GRAB27.795/ARCTIC39.804mm，
macro27.126mm（初始化34.480）；best46000为26.786mm。
latest/final50000及best46000均可加载，含优化器和双rank RNG；未开启新预算。
ref3的独立ContactPose auxiliary ablation另设300步matched短Probe，非主训练续跑。
两臂同latest50000模型/fresh优化器、三源main样本及CP辅助计算；仅loss系数0/0.05不同。
CP前向只允许history几何，future手/effect/label字段均不进入前向；小工程r2通过。
正式短Probe在681340a运行，control第112步后DataLoader worker因Aborted退出，
状态FAILED且进程已退出；无完整对照结果，不能据此判断auxiliary有效性。
原始产物保留，数据加载故障待修复；没有自动重启或扩展预算。
详见[配对覆盖卡](../src/task/consequence-evaluator/docs/experiments/probes/P-20261008-consequence-pair-coverage.md)。

## 2026-10-07 ref2/ref3 运行记录（历史）

新随机自训练s1 parent200资格50/64；s3 bounded220→260固定资格36/64，
恢复Probe PROMISING。十条剩余reference全部恢复；六角色airplane_base已过资格，duck340固定资格8/64，保留旧duck280的0/64记录；cup正训练，其余三角色待训练，
尚无真实evaluator数据/fit。GPU0已自然空闲，duck首轮smoke因缺mesh在PPO前失败；
owned资产补齐；duck真实1024帧geometry smoke通过，533forceproxy帧均有<=1cm gap。
短20epoch不足以复现历史80epoch specialist预算；GPU0重新空闲后，
1386d51从合格s3260完成新duck260→340，端点资格8/64；cup80epoch独立迁移已启动，完成后固定资格已排队；三角色待恢复。ref2改用执行前已知24×18
请求残差计划，三臂E0/object/EI，同容量、几何接触约束、同expert/motion/current
物体手状态配对；89工程测试通过（含motion软链、阶段覆盖、native driver与新路由回归）；GPU1真实8env×128步几何smoke通过，712forceproxy帧均有≤1cm采样gap，
但尚未验证其他物体/桌面反例或完整真实数据，不升级为精确contact GT。

按用户ref3，四源PointWorld正常保存停止step14250，moving-anchor h24验证
Oak9.24/GRAB38.86/ARCTIC51.24/ContactPose27.15mm；best/latest保留。
三源Oak/GRAB/ARCTIC主监督已于23:20启动，commit9019fd4/GPU1/2每卡64，
latest14250模型初始化/fresh优化器，50000新更新，
用户指定截止2026-10-08 10:00。新step0同panel EPE为Oak15.92/GRAB36.13/ARCTIC51.39mm，
不同于旧四源panel，不直接横比。首次250val宏34.48→37.16mm，早期退化，
step1500宏29.88mm曾改善，但step2500宏35.87mm再次高于初始化34.48；
短期波动，不认定稳定改善或平台；
每250步保存latest/best；含val/save实测.744s/update，50000预计09:41，硬截止不变。
ContactPose仅刚性运输辅助；EPIC仍candidate_only，
独立修订坐标约定、双手有效性、gap及OF overlap质量，未开放训练。
旧四源结果不升级为同质监督或Cm收益结论，最终matched trained-policy utility仍OPEN。

## 2026-10-07 用户指定 consequence-evaluator 路线

用户要求将当前PointWorld与数据处理代码合并到本地main，再创建`consequence-evaluator`。
数据扩充提交a620375已合并，实验索引跟进9c555d1；当前新分支继承oracle与PointWorld。
按Task ref1固定K24/Kexec8，先检验连续六专家rollout上的E0(H,A)与Eoracle(H,A,Z_GT)，
暂不推进WM适配或在线proposal。论文原文/源码已核对；独立scalar BT是Robometer改编。
窗口合同、标签mask、两臂模型、有界训练入口与六专家连续采集驱动已实现；CPU测试覆盖真实驱动循环，
但未在真实Isaac Gym上验证，也未采集或拟合真实数据。采集副产物保留measured q/root与扰动审计，标签仍需核验。
独立test评价入口已实现：冻结val选择的权重，报告配对排序与任务/阶段汇总、跨episode未来替换诊断；
31项工程测试通过，不代表真实oracle headroom，也不把重叠窗口当独立样本。
旧oracle outputs、六专家权重未找到，原motion输出缺失。用户已授权重训专家并重新rollout；
外部canonical几何motion与仿真资产可用，13条motion完成CPU合同检查，已复制/软链接到owned输出。
DExplore指向已删除baseline的失效data链接已保留并修复。先新训s1 parent，再检查frame0抓取后迁移s3与其余专家；
新权重不继承旧六专家Validation结论。PointWorld原三卡训练已正常完成10000新更新，源码hash未变；
固定balanced运动anchor h24 EPE14.0672→11.1342mm，best为step9500的11.1026mm，
预训练Probe PROMISING；末尾2500更新仍改善3.49%，仅能判断收益减慢，不能认定完全平台。
best/latest/final均保留，未开启test或新训练预算。专家重建r2的native smoke在首次PPO更新前
因NumPy别名兼容导入顺序失败，原日志保留；调整bootstrap导入顺序后重试原预算内预检。
随后用户授权OakInk2继续：GPU1/2、每卡64/global128，从原latest保留AdamW动量，
沿用末尾lr约1e-5追加最多10000更新，仍受原绝对deadline约束；这是显式两卡迁移，非逐位resume。
GPU0用于consequence专家/rollout。r3重建smoke在首次reset发现CPU/CUDA混用，未发生PPO更新；
task-local入口改用CUDA PhysX tensor pipeline后，按原输入/预算执行r4预检。
启动前用户暂停以上续训安排，优先混合OakInk2/GRAB/ARCTIC/ContactPose，原latest仅作模型初始化，
新优化器/训练从头开始。两卡续训与r4预检均未启动；当前无本会话GPU进程需要停止。
四源全量整合和双卡12步工程检查已完成；GRAB/ARCTIC官方split、原生类别映射，
ContactPose真实世界位姿/30Hz及gap屏蔽已接入，参与者train/val/test隔离。
2026-10-07 19:57:29开始混合训练，运行提交384f860，GPU1/2每卡64/global128，
OakInk2/GRAB/ARCTIC/ContactPose采样50/20/20/10；原latest只加载模型，优化器/调度重置。
训练窗口总4794229，固定64val窗口/source，step500运动anchor h24宏平均52.56→45.74mm；
OakInk2新固定panel9.79→10.49mm，较step250略恢复，其他三源改善，早期遗忘/适配取舍仍UNCLEAR。
沿用原deadline2026-10-08 09:58:37，最多40000新更新，不追加24h。用户随后恢复consequence-evaluator：GPU0启动r4重建，2epoch GPU smoke通过，
64env/200epoch s1 scratch母策略已完成（1114.91s），权重保留；最近接触/保持指标仍0。
固定64条frame0/45帧保持无后续drop资格检查完成（c0b10fb）：0/64合格，
不扩展六专家，先同初始状态参考动作回放诊断；这不构成evaluator/world-model负结论。新增局部物理事件标签、独立专家progress抽样和资格检查共44项工程测试通过；
尚未获得真实evaluator数据或oracle headroom结论。
后续真实物理诊断确认r4存在GPU reset提交/刷新缺陷：首步物体跳1.48m。
task-local修复后同环境降到1.43mm，重复子集reset/FK物理检查通过，51项工程测试通过；
原r4训练和0/64不能用于判断正确实现的学习效果。已找回原s1 corrected tensor，
与r4 canonical在q/坐标/contact上并不等价；旧新均用graspenv+Dexplore_Inspire。
原s1数据GPU物理预检已通过（15.78s，首步位移1.43mm，三次子集reset通过）；
原CmLite权重/转移与s3 corrected输入仍需重建。
当前推进：567c909启动原s1输入/修复reset/Cm-off新重建，GPU0、64env/h32/mb256、
LR2e-5/mini-epochs6、anneal40→80、200epoch；smoke36.18s通过，前80epoch有真实接触/抬升奖励，
训练吞吐约500FPS、显存18251MiB。缺失CmLite奖励未启用，不称原配方精确复现。
CPU已恢复s3 corrected reference（23.17s，几何相对误差<3e-7m，左contact0），迁移仍待母策略资格。
该新母策略200epoch已完成（1123.50s）；独立64条frame0完整episode中50条通过45帧保持/无后续drop，
超过预设8/64数据准备门槛，重建Probe PROMISING（不继承旧Validation或证明Cm收益）。
GPU0下一步从新自训练parent200迁移s3到220，先2epoch工程检查、再20epoch正式Probe部分，
最后固定端点资格检查；权重祖先/RMS/优化器/epoch均核对，LR显式1e-5，保留已结束anneal40→80。
s3首轮200→220已完成，但固定64frame0资格只有2/64通过（未达8），首步位移1.72mm内正常，
有16/64短抬升且训练接触上升。下一步同输入/配方有界续训220→260并再评固定端点；不扩展其他专家。
cup参考输入CPU恢复通过（934帧/相对误差<3e-7m），不是cup抓取证据。
其他专家和连续六专家数据仍未完成；尚未训练E0/Eoracle。
详见[混合预训练卡](../src/task/cm-pointflow-effect-pretrain/docs/experiments/probes/P-20261007-pointworld-multisource.md)。
见[重建Probe](../src/task/consequence-evaluator/docs/experiments/probes/P-20261007-consequence-baseline-rebuild.md)。
见[任务入口](../src/task/consequence-evaluator/README.md)。不改变最终matched Cm-on/off策略utility要求。

### consequence-evaluator 2026-10-08 续跑

六个新 endpoint 已冻结为不同 hash：airplane_base36/64、duck8/64、cup63/64
通过固定资格门；mixed12/train5/balanced5 为0/5/4，保留为 observational
候选。route.json 明确 `all_experts_operationally_qualified=false` 和
`training_allowed=false`，不恢复旧六专家 Validation 结论。queue smoke 的
hard-object oversampling 与 minibatch divisibility 修复已提交
`3b4305b`、`e62dd08`。

首轮三 split 原始采集 144 episodes 后，严格 local-state label 为
train/val/test=0/0/2；独立 proxy/gap 几何审计通过。按 Decision Note 追加
296/297/298 三波次、216 episodes，仍使用 24 步 decision-known residual 和
2 cm hand RMS。追加标签在未改合同下变为 `READY`、pair=2/1/5，审计为
149,841 valid frames、49,388 native proxy、45,984 proxy-near、仅1 proxy-far。
但 2 train/1 val 不足以支撑 32-pair/update 的 evaluator fit；继续训练只会
重复采样并在一个 validation pair 上选模，故本轮不启动 `prepare_windows` 或
`train_matched`，也不放宽阈值。原始、label 和 coverage diagnostic 均保留；
下一步若继续，需设计同一 current state 的 twin residual branches，而不是
再堆 generic waves 或把 hand RMS 放宽后重判。North-star matched Cm-on/off
policy utility 仍 OPEN。

当前实现审查把这项 gate 固化为规则
`geometry-corroborated-H-matched-state-residual-events-v3`：偏好端点必须同时
匹配历史 `H`、当前物体位姿和11点手状态，历史相对RMS上限为0.25；train/val/test
还必须分别拥有至少8/4/4个不同的无序 episode pair。collector 只把该要求冻结进
source manifest；labeler、window 准备、`Windows` 和 `train_matched.py` 均拒绝不满足
的生产 fit 数据。上面的历史
2/1/5 产物因此只保留为 observational audit，不能作为 evaluator fit 输入；当前
route 仍是 `all_experts_operationally_qualified=false`、`training_allowed=false`。

新增的 `consequence_evaluator.twin` 是严格的 Isaac-free v1 合同测试：要求完整
native task/controller buffer、Python/NumPy/Torch RNG、fresh simulator prefix
replay provenance、刚体物体位姿和双分支第0帧锚点。它尚未连接 native collector，
也没有真实 twin branch 产物；这部分仍是下一项实现 blocker，不能把合同测试升级为
科学证据。Task tests 在本轮代码审查后继续作为工程回归，不改变 North-star claim。

2026-10-08 native twin 工程推进：`consequence_evaluator.twin` 新增不依赖 Isaac 导入的
native capture adapter、完整 prefix trace hash/帧数检查和 Python/NumPy/Torch RNG 捕获。
它现在能把初始化后的 native task/controller 边界转成 v1 snapshot，但尚未接入连续
collector 的 fresh-simulator 双分支执行，也没有真实 twin branch 产物；因此 blocker
从“缺少 adapter”收窄为“缺少 native branch runner 与真实覆盖”，不升级为科学证据。

更新：2026-10-06。本摘要整合已交付的主分支与本轮Cm研究事实，不产生正式科研结论，
不纳入其他独立会话尚未交付的结果。完整旧摘要见[状态快照](archive/research/STATE-20260930-before-workflow-simplification.md)。

| North-star | 当前判断 |
| --- | --- |
| Self-trained grasp | PARTIAL：限定 12-motion 的六专家初始观测路由已通过正式 C1 Validation；单一 actor 稳定结果仍未证。 |
| Cm one-step information | PARTIAL：物理效应可学，但依赖表示、分布与目标。 |
| Cm policy utility | OPEN：尚无跨训练 seed 的 matched Cm-on 优于 Cm-off 证据；effect-rank 正式 Validation 的正向主张已 REFUTED。 |
| Generalization | OPEN：未见物体/多轨迹上的 Cm 收益尚未建立，本阶段先聚焦固定任务分布。 |

## 关键事实与边界

- C1 的任务限定 SUPPORTED 不证明单一 actor、未见物体泛化或 Cm utility；冻结已验收的
  六专家 substrate。[正式验证](experiments/validations/VAL-20260926-observation-six-expert-c1.md)。
- HF01–HF05 的已失败局部路线保持冻结，不靠换 seed/门槛重置预算；这不是所有未来 Cm/GPU
  路线的全局禁止。历史路线边界保存在归档实验记录中。
- 用户已授权边界内自主选路线，以及六专家蒸馏与新的 Cm 探索；历史标签不因此升级。
- r6 support 的 teacher label 仅覆盖 source_e260，不能证明六专家蒸馏；其缺失轴字段不能回填。
- r7 轴合约通过，但 contact q10 与 delta 覆盖未过 calibration gate，不生成正式 Cm-on 标签。
  [校准证据](archive/2026-10-04-research-governance/handoffs/CM_SCRATCH_CPU_CALIBRATION_R2_AXIS_20260928.md)。

## 其他已有任务交付边界

主分支已记录的两个旧系统受控任务：一次 fit-only CPU 校准修复，以及独立六专家逐步轨迹蒸馏。
校准任务 2 CPU/15 分钟/1 GiB，蒸馏任务 1 GPU/60 分钟/5 GiB；均只形成 Probe 结论。
旧系统任务的实际终态以各自交付为准，不由新工作流猜测或重新启动。

先验收实际交付，停止失败的局部 calibration tuning；蒸馏独立推进。新 Cm 路线须服务于
真实策略因果增益，保留 matched Cm-off 对照。[实验索引](experiments/INDEX.md)按需检索。
运行细节、失效执行、数值和哈希留在原卡/manifest；资源授权见 [CAMPAIGN](CAMPAIGN.md)。


## 本轮 Cm 策略价值研究

用户授权先取得真实策略 matched Probe，正向则优先正式 Validation；期限为
2026-10-03 23:59（Asia/Shanghai）。完整交互与短期物理预测结合长期价值，
最终要求训练所得actor受益；瞬时物体位移不替代动作价值，跨数据集预训练可选。

HF06 teacher-envelope与HF07 BC物理预测输入实现均为UNPROMISING，保持关闭。
HF08 physical-value已完成：公共池1041599转移、1920完整episode，fit831811行，
开发holdout209788行；同一自训练source_e260、三条canonical airplane motion，
六臂均追加160epoch/327680交互，全部48点actor-only评价完成。原生gate为
UNPROMISING：终点Cm33/384，普通PPO41/384，直接Q40/384。这里只是Probe，
不否定Cm核心假设；North-star policy utility仍OPEN。

R2价值目标审计已由root验收：source主成功15/1920、holdout成功1/384，两个e420
V checkpoint均实际完成7680 optimizer updates。结论为UNCLEAR：V训练充分性和当前
策略校准仍未知；不据此宣称需要重训或已收敛。

在线适配后物体位置误差降至0.78/0.86cm，仍弱于恒定线速度基线；姿态误差
仍约1.34/1.36rad，明显弱于保持状态。模型不能称为已准确预测物理转移。
同source checkpoint的e0重复评价有差异，原生配对只覆盖env/motion/start/时长/
初始高度；初始完整state恢复审计此前已排除为当前阻塞，不以单次realized MC误差
宣称bias。暂不升级Validation，不调参重扫HF08。

collector与V诊断工程已验收；R1 nativecwd资产失败、R2 wrapper在GPU ownership前失败，
两者均无新数据、无V充分性结论。R3/r3b也已FAILED并完成进程清理，均0行；
CM真实输入守卫已验收、集成，main复验11项通过。
当前执行原生Gaussian、frame-0、冻结策略的完整episode诊断，尚无完整采集数据或新训练。
真实环境初始化暴露reward_shaper对象检查错误；修复后首个动作又因采集器误把logstd
当sigma退出。原生model还负责观测归一化，raw网络直调不能替代它。错误属于采集器，
不证明checkpoint标准差无效。root已派固定RL角色做CPU-only原生player合约修复；
原恢复轮截止不延长，未来采集按实际累计成本另记有界预算。单次MC误差不单独证明bias；
HF08 slot不重置。见[修复记录](archive/2026-10-04-research-governance/decisions/D-20261001-native-player-collector-repair.md)。
[HF08实验卡](experiments/probes/P-20260930-cm-physical-value.md)与
[完整结果](experiments/probes/P-20260930-cm-physical-value-results.json)保留边界与数值。
原始数据/checkpoint留在原研究工作树的research/output/P-20260930-cm-physical-value/r7，
未提交Git，不因本次合并移动或删除。

## Ref5：point-flow E → G 与 surface I

当前实现分支 `agent/cm-interaction-oracle`，新 Task 为
`src/task/cm-interaction-oracle/`。新工具/卡已放入 Task；旧工具和历史卡逐步迁移，
根级 Mission、Campaign、seed ledger 与状态保持唯一来源。

- 真正冻结 point-flow K1 E → G 已跑 matched Probe：同容量 H MAE23.3652、
  H+GT E23.2837、GT-trained bridge换预测E23.6477、预测E训练bridge22.8961。
  判定 UNCLEAR：GT仅0.35%改善，预测fit2.01%且CI跨零，收益受单个高RTG episode
  支配。去此episode剩余21个episode预测fit反而差5.82%。
- 同容量GT合同检查（短序列右对齐，避免GRU补零冲淡）：pose K1/K4/K8改善
  0.034%/1.067%/2.748%，full13 K8改善7.42%
  但去高RTG episode后负1.66%；没有合同通过预设门槛。保留K8 pose弱方向性信号，
  不据此扩大K4 predictor或进入policy teacher。
- 8patch×5D GT surface I比同容量零I改善10.83%，CI跨零且约97%收益来自同一
  高RTG episode；pooled I更好，未证明空间topology独立价值，不启动I head/K4 I。
- 工程审计发现fresh shards手root平移最高22.33mm，旧identity-root假设不能迁移。
  用当前measured body pose恢复共同root后，五body最大残差5.35微米；当前/未来root
  cache分开。首轮G失效、no-grad技术失败均留痕；修复后独立review未见其他致命bug。
  CM平移RMSE19.33mm，当前twist persistence4.92mm，不能称准确物理预测。
- 原始episode summary与assembled labels逐条一致：112 episodes仅2次达到45步
  held成功，两次随后都掉落；该source_e260数据的成功无后续掉落数为0。主导held-out
  RTG episode即使最长保持8.87s也随后drop，高RTG不能直接视为最终稳定抓取。
  这些统计不重判已验收六专家baseline，不据此直接训练稀疏success/drop二分类G。

Root选择：保留E物理主路线与修正的几何合同，暂缓局部I/predictor扩展；下一次
最便宜的决策应先核对持续hold/drop任务目标与action contrast。RTG回归不替代抓取、
保持、掉落评价，更不替代matched Cm-on/off训练所得策略的因果收益。
两路使用GPU6/7，均已结束、零新采集；本轮只给Probe判断，North-star Cm utility仍OPEN。

[E→G实验卡](../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261004-pointflow-g.md)，
[surface I实验卡](../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261004-ref5-surface-i-gt-value.md)。

## Task ref1：RECAP-style 相对任务 advantage

同一 Task/branch 已实现并执行 cross-fitted MC V → 32步 advantage 的有界
Decision Probe；task reward 固定 `held/10 + lift_progress/5 + stable`，排除
base/approach，所有模型显式控制 episode-fixed noise。历史90train/22test
episode split、两次V拟合×三fold，gamma0.99，固定16epochs，不改变核心 Mission。

- G0标签计数、episode支持、集中度均达标；held-out V MSE229.58低于zero324.80。
  但独立fit标签一致率 train34.72%/test39.15%，远低于预设75%，结论UNCLEAR。
  按门槛停止，G1 action critic、G2 predicted E/I、G3 selector均未执行。
- 连续advantage test相关性0.978，去最大MC episode仍0.952，保留方向性线索。
  真实32步task reward非零只覆盖16.3% test queries；该区标签一致78.0%，
  零reward区29.6%。后者fit差中位数0.255大于0.05幅度floor，不只是微小量化抖动。
  不能事后用未来reward active筛选测试样本或改一致率分母宣布通过。
- 工程review修复neutral样本污染binary AUC的问题（正式运行前，G0不受影响）；
  后续112 episode target/row/fold/threshold audit精确一致，无进一步致命问题。
  当前暂停原因是标签/value合同可靠性不足，不能写成RECAP或动作信息被否定。

Root Decision：保留相对任务后果方向，下一决策前提是可信标签/value；暂不投入
I/G容量扩展。任何改标签协议须重新明确假设和判定，不放松本轮门槛或追加seed搜索。
本轮GPU6 model执行含smoke<20s，零新采集，任务进程均结束；代码commit9dd503d。
North-star Cm训练策略因果utility仍OPEN，不把离线标签工作算作policy进步。

[相对任务advantage实验卡](../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261004-recap-relative-action.md)。

## Task ref2：冻结连续 advantage 的 action ranking

ref2 明确授权新排序问题，未重跑V或放松旧三分类G0门槛。新 matched pairwise
Probe完整执行 H / HaK1当前动作 / HaK4已记录feedback；同96,705参数、同init、
同pairs/训练设置，既有90/22episode split、全部冻结advantage继续使用。

- 正式测试 macro同motion/phase/noise pair accuracy：H59.712%、HaK1 59.617%、
  K4 60.520%。K1增益−0.095pp；macroSpearman仅+0.0021；打乱当前动作仅降低
  0.133pp，两个独立V target的增益均负；8/19可配对episode改善。UNPROMISING
  针对当前fixed target/data/fit合同，不是物理动作信息或Cm被否定。
- 新审计纠正ref2的排序前提：Pearson0.978并不证明rank稳。两套冻结advantage
  的Spearman train0.363/test0.540，条件pair顺序一致约62.4%/66.8%。去三分类
  boundary并未形成强排序监督；训练93%–95%而测试≈60%，泛化仍弱。
- 配对实际覆盖19/22test episodes、64strata（2,432/2,816anchor queries）；
  macroSpearman使用88strata，包含单episode无cross-pair组，分母明确分开。
  去最大MC episode后K1+1.333pp是弱敏感性线索，未过整体门槛。
- 当前动作进入网络且打乱score RMS0.991；独立review及root SciPy/rawscore
  复算确认pair/input/normalization/target/hash/metrics正确，无进一步致命bug。
  第一smoke仅path-hash metadata失败，修复留痕；正式执行commit1a64eab。

Root按ref2负分支停止这份离线标签上的critic/predicted E/I扩展，未新采集或训练
teacher/policy。下一研究决策应先建立可信物理action/outcome contrast，不能仅
换V seed、width或threshold继续此数据拟合。保留E/I与relative action科学问题，
North-star causal Cm-on/off训练所得策略utility仍OPEN。
本轮GPU6 valid模型执行含smoke<49s，产物<8MB，全部任务结束。

[冻结相对advantage排序实验卡](../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261004-relative-action-ranking.md)。

## Task ref3：真实随机动作干预

固定自训练source_e260、airplane三motion，移除episode噪声，七臂随机分配
四步feedback residual；336完整episode、320实际干预、320完整32步窗口。
每臂37–54次、有效dose100%；全batch reset屏障，pre/post/PD/GT标签经独立review。
全部decision为pre-lift（最高lift8.74mm），early-hold/drop-risk样本为0。

- 按环境留出66train/18test，预测器及scorerOOF也隔离环境。H E/I MSE0.5748，
  Ha0.5841；macro任务排序 H66.78%、direct66.14%、predicted65.17%、GT61.17%。
  七项预设gate全部失败，当前合同UNPROMISING，不启动selector/teacher/policy。
- GT保留contact-retention局部信息，但整体泛化弱；模型容量/OOF噪声差和少量
  history瞬态限制归因。PCA总test能量异常主要由单行支配，典型样本保留率中位
  86.96%，不是所有测试状态丢失93%；不删除outlier重判结果。
- 最便宜的既有数据Decision诊断控制当前状态后，物体短期旋转有弱随机臂响应
  （E/I家族探索性permutation tail0.045），任务窗口/完整episode证据仍不足。
  手实际收到不同扰动不等于已建立任务相关可预测中介链。
- Full summaries105/336达到45步hold、76随后drop；不同采样下的raw arm均值
  不当作策略增益，也不修改旧baseline结论。motion symlink漏预先hash已补明确
  post-run provenance，后续collector修复；原manifest/失败smoke保持可追溯。

Root停止本pre-lift E/I8→Y16/32 PCA/MLP扩展，保留随机干预数据和物理control
问题；不据此终止Cm核心路线。未来研究需要新的阶段/时域决策合同，不换seed/
阈值/encoder继续拟合本合同。单GPU6，正式仿真222s、smoke70s、模型4.86s，
统计1.12s，产物<13MB，进程均结束。North-star训练所得Cm策略utility仍OPEN。

[真实随机动作干预实验卡](../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261004-randomized-action-intervention.md)。

## Task ref4：early-hold interaction retention control

按用户新增ref4，干预移到lift≥3cm且连续6步contact-proxy的early hold。
1,008完整episodes产生494随机干预，全部32步窗口完整、剂量实际执行。
后续214次失败中212有物理高度损失；98次早失败全部保留，不筛掉step8失败者。

- A UNPROMISING：短接触比例调整后臂间范围5.68pp，tail0.1335；I14 family
  tail0.5005，未建立预设action→retention-I可控链。
- B UNCLEAR：GT I令后9..32预测误差改善49.6%，failure AUC0.899→0.949；
  但两主目标均只有2充分支持分层，未达预设3层。原macro排序58.46→77.33%
  受1–2pair小层放大；充分支持层描述性87.76→96.32%，不事后改gate。
  未早失败的92个test trial也有预后改善，但这只是post-treatment描述性子集。
- 独立工程review及root逐项复算通过。未启动C、selector或policy训练。

Root关闭当前early-hold四步feedback residual→I8合同的继续拟合，保留GT I预后
价值。缺口转为可操纵的保持控制变量/动作时域，不能靠更多seed、网络或补B支持
绕过A失败；不否定Cm核心假设。单GPU6仿真约417s、模型2.31s、产物<18MiB，
全部结束。North-star训练所得matched Cm-on/off策略utility仍OPEN。
[early-hold实验卡](../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261004-early-hold-intervention.md)。

## Task ref5：延长 feedback residual 的物理响应

仅做physics Decision，不训练Cm/S/PPO：在同一early-hold区随机分配7方向×K4/8/16，
2,016完整episodes、1,006完整干预窗口。非零cell最少32、两半最少13、pooled zero148；
全部剂量约1.0，native PD和独立标签/OLS/curve复算通过。

结果UNPROMISING：短接触tail0.1645，K16最大绝对效应6.62pp，未达10pp；I16 family
tail0.572，后17..32接触/高度失败tail0.581/0.244，未形成稳定duration→retention-I链。
动作实际改变了手：wristx+在同一step16调整位移随K为4.51/10.47/21.92mm。
部分反向baseline命令与补偿一致，但累计剂量/释放恢复时长同时变化，不能唯一识别
feedback cancellation。secondary all32高度失败有弱tail0.085，不能救援I控制gate。

Root关闭当前K≤16 feedback-residual合同，保留I预后信息及测得的反馈响应；不扩大网络、
时长/幅度/seed扫描或补旧B支持。不否定其他operator、I/Cm核心思想；固定绝对target
仍未测试，若以后推进须有能区分机制的新合同。GPU6仿真787s、CPU统计2.81s，产物
<37MiB，全部结束；25 Task tests与变更检查通过。训练所得matched Cm策略utility仍OPEN。
[duration实验卡](../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261005-early-hold-duration.md)。

## Task ref6：固定 K8 的幅值与局部 interaction authority

alpha1/2/4×七方向联合随机：1,680 episodes、863完整干预窗口，剂量、support、
原生PD、输入哈希及独立/root复算全部通过。任务关联authority gate UNPROMISING：
短接触tail0.8895、alpha4最大效应3.68pp；后9..32接触/高度失败tail0.554/0.1965，
没有合格阈值候选，因此未启动条件性Stage2或神经模型。

局部interaction响应PROMISING：I8、切向投影、ratio family tail均0.0005，主要由
thumb_distal贡献；alpha4 finger−距离+22.95mm且力下降，finger+距离−8.44mm且力上升，
两半同向。wristx+同step8手位移12.95→23.53→47.61mm确认动作authority。
原force norm已捕捉信号，不能将缺少任务差异全部归因方向压缩；全手OR contact-proxy
可掩盖单thumb变化，物理高度结果也未明确分化。只证明此合同的局部proxy可控，
未识别抓持阈值、可用控制区间或Cm收益。

Root保留局部action→I证据，停止本次幅值合同，不扩大幅值/seed/网络以重复authority；
后续需区分可控局部接触是否实际参与承载。GPU6采集player累计671.48s，产物<40MiB，
均已结束；North-star Cm utility仍OPEN。
[幅值实验卡](../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261005-amplitude-authority.md)。

## Task ref7：独立 finger DOF 与实际幅度审计

拆为六独立DOF±（含thumb-yaw）、zero、匹配旧synergy±共15臂，固定K8/20%driver range。
1,680episodes、854完整窗口，support/dose/nativePD/17input hashes及独立复算通过；
各指PD rad/deg、mimic、实际q和真实tip矩阵已写入Task实验卡，ref6旧卡也补PD幅度表。
同20%range并非同毫米：四指局部tip对照范数约15–29mm，thumb约9–14mm，未假定thumb
运动最大或已知冗余/承载身份。

预设own-I+late-task及useful gates均UNPROMISING；但后9..32contact family tail0.0005，
不能概括单指控制无效。middle−保持−10.94pp两半重复；thumb-yaw−保持+10.76pp、高度
失败−11.34pp，两半同向。后者height family tail0.1535及own-I幅度/半包一致性未达标，
保留为下一项针对真实高度安全的窄对照候选，不靠改gate宣称有效或启动Cm训练。
现有净力contact-proxy改善不等于可靠抓持；训练所得Cm-on/off utility仍OPEN。
GPU6累计691.45s（含初始化），均已结束，36tests通过。
[per-finger实验卡](../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261005-per-finger-control.md)。


## Task ref8：GT consequence 对任务信息的保留

复用ref7的854随机窗口，不再采集；step8 E12/I14预测step9..32任务，
688train/166test、125/31环境隔离，六主模型及三预先声明signed-force扩展同容量/init。
GT预后PROMISING：HEI主误差0.73135→0.39311（46.25%改善，环境bootstrap95%
34.02–57.03%），物理高度失败误差改善32.52%（11.24–49.92%）；H+a也有18.99%
主误差改善。跨半包、留一臂PCA3/ridge1桥的EI接触/高度失败误差相对zero改善
21.65/16.14%，相关0.550/0.723；是共享zero的有噪声边际估计，不是因果中介识别。

完整充分性UNCLEAR：HEI再加a主/物理误差点gain−0.33/−0.93%，但one-sided95%
上界7.96/9.64%仍超过预设5%，不能排除有意义的剩余action信息。Signed扩展下a
又有8.68/12.91%增益，末50epochs loss仍降约40%，初始q/dq经PCA与泛化误差均限制判断。
无泄漏/关键工程bug；root和独立审查复算通过，逐指PD/q/真实tip幅度表附新卡。
下一项有决策价值的是独立冻结的conditional predictability（H+a→E/I，再与直接Ha
比较预测consequence的task增益），不以GT预后或点估计等同链闭合，不立即selector/PPO。
本轮未训练Cm；训练所得matched Cm-on/off utility仍OPEN。
GPU6固定拟合5.56s结束，首CUDA初始化前失败0.76s留存；新产物<4MiB、41tests通过。
[GT consequence实验卡](../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261005-gt-consequence-sufficiency.md)。


## Task ref9：conditional consequence prediction 与 OOF 任务价值

冻结ref7数据及ref8的688/166环境隔离划分，E12/I14、intended one-hot保持；
21个固定1500epoch拟合，严格fold-only PCA/尺度/OOF，保存directHa和两个shuffle对照。
总体UNCLEAR，预设A/B/C均未过。Ha I误差比H改善8.50%但区间跨零，E反而恶化20.79%；
冻结Ha置换test动作使I误差增加11.96%(95%4.17–22.24%)，保留局部动作敏感性。
同H候选I差分corr.293、方向62.29%、幅度比.369，未稳定保留动作差异。

OOF后果P_Ha主MSE.6730，弱于directHa.6478；Ha+P_Ha .5995有7.45%改善点估计但
CI跨零。GT同预算oracle .4080仍强；预测后果保留31.18%oracle gain（CI2.90–56.56%），
不能等同超过directHa。跨half预测已做，83行子集GT contrast rank13/9记为NULL/UNCLEAR，
未用伪逆制造逐臂结论。全部14臂预测／GT向量及逐指PD/q/true-tip幅度表在新卡。

按用户要求独立review反常结果，root GPU全权重重放精确一致，无时域/OOF/尺度/shuffle bug。
预测器明显劣于train-mean和persistence，train接近零而test高；末期loss下降不支持追加epoch。
Shuffle较高原sign受共同zero偏移影响，中心化后corr−.076；不救援主结果。
结束当前固定拟合，不selector/PPO；下一Decision应区分受控泛化训练与缺少信息，非更多
memorization。没有关闭Cm核心假设，训练所得matched policy utility仍OPEN。
GPU6主运行59.99s，smoke3.93s，审计仅inference，均结束；新增<12MiB、46tests通过。
[conditional consequence实验卡](../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261005-conditional-consequence.md)。

## Task ref10：几何动作表示与物理残差

按ref10将决策native q/PD指令经explicitroot×FK转成120surface对应点流，
共同提供baseline nominal geometry，对比State/Arm/Joint/Flow和shuffle；
fold-only PCA/尺度、严格environment OOF、固定α32ridge，无新仿真。
原PCA/bilinear残差screen UNPROMISING：Flow I12.440明显崩溃，testshuffle反而
降低误差。按用户要求独立review：无来源/标签/泄漏/单位/求解bug，root GPU全重放0误差。
单窗口/前五窗口贡献57.84%/85.13% Ierror，state×flow乘积超出训练支持并放大输出。

另预先冻结六fit尺度×乘法因子诊断（明确post-result、非独立确认）：移除products
I1.254，固定20mm additive1.138；仍弱于State1.074/Arm1.055/Joint1.041/mean.891。
当前PCA/ridge名义端点路线UNPROMISING，不能否定物理几何动作或OI-CmV2空间网络。
独立NumPy复算六fits/bootstrap一致。GT仍改善task31.50%(CI8.32–51.03)，
P_Flow .7149却输H .6540/P_State .6132，R−.2955；C=true仅胜崩溃directFlow，
root明确不认作uniqueCm收益。不selector/PPO、追加epoch或删outliers。

保留FK/continuous surface合同、OOF工具、14signedarm向量、逐指PD/实测q/true-tip
及名义surface幅度表；后续若继续几何路线，应检验localcontact/sharedspatial归纳
偏置与nominal→realized执行合同。全局matched trained-policy Cm utility仍OPEN。
GPU6两主run最终manifest2.88s/1.19s，首启动hash路径错误在模型计算前终止并留存，
审计只inference；新增<85MiB，50Task tests通过，无剩余本轮GPU进程。
[ref10主实验卡](../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261005-geometric-innovation.md)，
[归因实验卡](../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261005-geometric-support.md)。

## Task ref10 续接：局部 OI-CmV2 空间后果

续读原会话后补上尚未测试的空间路线：复用V13局部交互/token/fusion模块，
固定width32/4tokens/2cmKNN8，名义PD端点点流与persistence残差；全部854窗口，
原688/166环境划分，严格三折OOF，20后果+8同预算直接/GT/预测后果task模型。

本固定合同UNPROMISING：Flow I.8891，State.8837、Joint.8670、mean.8907；
动作shuffle仅增加I误差.565%，I contrast corr.239/sign53.14%/幅度比.066。
P_Flow主误差.5442与directFlow.5468接近且输P_State.5329，五gate均未过；
GT仍改善主误差30.22%(CI11.98–43.15%)。误差稳定，无旧bilinear爆炸。
全部28保存权重、OOF和15Flow候选重放精确一致；独立review与root核验通过。
99.06%窗口有空间边，动作改变邻居/特征且梯度非零；不是动作未接入。

Root停止这个固定fit，下一Decision需区分名义端点与可预测实际执行，不能靠更多
epoch/seed重复弱动作通路。端点/实际tip运动已有明显描述性差距，但不能唯一归因
反馈抵消或断言空间模型不可学。没有selector/PPO或正式科学结论；全局Cm策略
utility仍OPEN。GPU6主162.62s、smoke6.17s、重放3.44s、review GPU.74s；新增约154MiB、53tests通过。
[空间后果实验卡](../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261005-spatial-consequence.md)。


## Task ref10 续接：真实／可预测执行几何

固定8个ridge执行拟合、2个实测oracle与3个可预测空间后果臂；854窗口、原环境
划分、执行训练输入严格source OOF，无新仿真。原全部18q裁剪发现实现错误：URDF
腕部3角为continuous，±pi仅PD尺度fallback。root与独立review确认，保留原run，
修复后重放8ridge、复用2oracle/4control，只重跑3个受影响300update空间臂。
错误投影的预测结果不作为原定连续执行路线负证据。

修正后Ha finger误差比H降低79.81%(CI75.70–83.48)，q18 .4793低于nominal .6090；
但surface XYZ-component RMSE34.09mm vsnominal32.30mm。无拟合FK分解发现真实
腕部+预测手指仅2.13mm，预测腕部+真实手指33.94mm，说明事实端点误差主要来自
腕部；前5窗口贡献83.45%误差，未删样本。名义／当前腕部的因果替代分别32.15／
50.92mm，没有取得oracle精度。不能据此宣称执行不可预测，也不能用oracle部署。

本固定端点空间合同UNPROMISING：PredSurface I.8972差于State.8837/PredJoint.8620，
I contrast corr.152/sign50.86%/幅度.0235、testshuffle penalty.301%，五gate均未过。
实测Surface I.8913未胜nominal；实测Joint.8415比State改善4.77%(CI1.17–8.06)，
仅post-treatment诊断，支持研究表示传递而非断言信息不存在。全部保存权重/候选
重放0、ridge残差≤6.66e−16，独立review重建metrics/bootstrap/OOF一致；动作通路有效
（99.30%窗口有边、test22.81active points），57tests通过，原+smoke+repair约154MiB。
Root停止反复拟合totalendpoint合同，保留可预测finger动作信号；后续区分共同腕部
演化和finger相对几何，或检视spatial瓶颈。没有selector/PPO；全局Cm utility仍OPEN。
[执行几何实验卡](../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261005-execution-geometry.md)。

## Task ref10 续接：动作传递与按手指残差

冻结空间各阶段做24个source-only线性动作decoder：RawHandBase恢复已知输入
nominal≈1.000/forecast.9987，trainedFused仅.3306/.3436；当前有接触支持的
手指子集也只有.4122/.4275。该诊断只说明固定14arm线性恢复弱，不证明物理
泛化或2cm半径丢失：arm均值lookup自己已≈1/.996，LocalFlow均值代理省略
实际边的位置、法向、距离与个别flow。独立复算通过，root收紧结论范围。

据此执行七头匹配factorial：完整按手指mean/RMS相对点流／相同forecast关节，
普通／15候选均值中心化残差；共同H156，复用三环境OOF State，固定300update。
本固定合同UNPROMISING：PredFingerCentered I.9124差于State.8837（gain−3.24%，
CI−5.94..−.62%），也未优于匹配PredJointCentered.9077。测试动作打乱罚2.75%
但CI跨零，I contrast corr.1565/sign56.57%/zeroMSEgain1.58%，三gate均失败。
名义几何中心化相对plain改善3.91%但CI跨零；不把更大响应幅度当作方向正确。

完整H/action/source-norm、四State和七新head/候选重放0；中心化均值约1e−7，
独立review复算metrics/contrast0、bootstrap≤2.39e−7，未见影响负结果的实现错误；
source-OOF/full-test nuisance差及中心化不能修正共同偏差保留为限制，未唯一归因。
62Task tests通过。GPU6 fidelity主11.67s、factorial主9.30s，零新仿真或策略训练。
Root停止这份mean/RMS+中心化fit，不追加epoch/seed；下轮须区分conditional
response、state nuisance或详细contact信息，不能据此关闭全部空间Cm。
两项都不提供strict nested consequence OOF或matched trained-policy utility；
North-star Cm utility仍OPEN。
[动作传递诊断](../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261005-spatial-action-fidelity.md)，
[按手指动作残差](../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261005-relative-finger-innovation.md)。

## 暂停旧执行路线，在原分支推进 ref11

最后一个surface-relative/current-wrist物理basis × Physics/OOFState nuisance
matched ridge32 Probe为UNPROMISING：PhysicsContact I.98857与同配置State.98882
接近，输priorState.88370；OOFStateContact1.03884。动作shuffle罚2.512%
(CI+.279..+4.959%)，raw Icontrast corr.4047/sign62.86%，四gate未过。
Root重放0、独立saved-matrix/statistics复算通过，未见影响负结果的实现错误；
动作scale floor导致不同有效shrinkage，current-wrist/采样normal/有限basis
限制保留，不否定所有contact geometry。GPU6主3.34s，零新仿真或policy训练。
[收尾实验卡](../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261005-contact-innovation.md)。

用户澄清“收尾”仅指暂停旧执行预测路线，继续在原分支
`agent/cm-interaction-oracle` 按Task ref11推进，不另建分支：先以真实hand point-flow为
oracle action，分离 `(H,F_hand)→E/I` 规划问题与以后 `joint→desired flow` 控制。
已完成State、GT endpoint、GT0→4/4→8 chunks及matched shuffled flow比较，
先只看E/I；不继续execution、候选nativearm或任务Y拟合。实际flow是
post-treatment信息，正向只能支持oracle表示Probe，不自动证明前瞻planning或
同状态候选因果contrast。最终matched Cm-on/off训练策略utility仍OPEN。


Ref11首个raw720oracle-flow Probe已完成（原分支，代码452d7e5）。同688/166
windows和125/31environment split，五臂同59066参数/300updates。Chunk E.48968
vsState.52895，改善7.42%（95%CI+.46..+14.04%），E门槛通过；I.55603
vsState.60486，改善8.07%但CI−1.25..+17.62%，预设联合E/I gate为UNPROMISING。
保留E的PROMISING探索信号与I不确定性，不据此关闭oracle flow路线。
trained/frozen shuffle显示action敏感性；Chunk-vsEndpoint CI均跨零，时序优越性
未建立。新H包含清晰current q/dq/geometry，旧State.8837与新State.6049不属
同合同，跨run提升不能归因于flow。test0同状态配对，candidate contrast为UNCLEAR。
完整GPU input/FK/weight/statistics重放误差0，69Task tests通过；GPU6主5.62s，
零新仿真或policy训练。只读检查确认已有restore仅支持coldstate，不能恢复warm
PhysX cache。用户明确暂不做配对数据及相关工程检查/采集，列入延后证据。
独立CPU权重/源统计/bootstrap复算通过（归一化误差≤1.34e-6），报告归档；
当前单seed预测Probe完成。不放松门槛、不追加seed/epoch，也不恢复旧execution
预测支线；尚未完成多seed泛化Validation或真正flow-space planning。
[Oracle flow实验卡](../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261005-oracle-hand-flow.md)。


## Ref12：oracle flow → OOF E/I → Y 链路

按用户ref12在原分支完成现有数据的matched任务预测Probe，配对数据/相关工程
和execution predictor继续暂停。3环境fold的H/PCA/标签stats仅使用各fit环境；
688source每行仅用hold预测，166test用冻结ref11 full-source模型。六新Cm fits
300updates、六同容量Y heads500updates，E12/I14和ref8step9起Y8保持原义。

直接actualflow→Y误差.45788 vsH.63951，改善28.40%（CI12.23..41.48%）；
预测E/I→Y.54712，改善14.45%（CI−1.57..29.47%）；GT E/I.45440，改善
28.95%（CI10.61..42.74%）。R.4991（CI−.0639..1.2790），固定链路gate
UNPROMISING，保留直接flow与GT的PROMISING任务信息，不否定所有flow/Cm。
Hybrid.43694相对直接flow改善4.57%但CI跨零，unique Cm增益UNCLEAR。
OOF source Chunk I.61550 vsfull-source in-sample.39092，后者从未进入Y训练；
OOF/full-test分布与有限优化预算保留为限制，不唯一归因于任何机制。
GPU6主11.72s；全部input/foldnormalizer/FK/权重/OOF/统计重放误差0，零新仿真。
独立CPU核对所有8个Y标签、sourcefold统计/权重/OOF/shuffle/bootstrap/R，
归一化replay≤2.17e-6，未见影响结果的实现缺陷；72Task tests、仓库验证通过。
不追加epoch/seed或进入planner；未来改变模型须另立可判别的Decision实验。
全局matched训练策略Cm utility仍OPEN。
[Oracle flow任务链实验卡](../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261005-oracle-flow-task.md)。

## ref13：Oracle-Y Utility Gate 已完成

用户 ref13 重新启用同前缀配对候选采集，在原分支完成逆向必要性 Probe。
先修复 GPU pipeline/异步前缀污染：最终用GPU PhysX+CPU tensor pipeline、
GPU6 frozen policy，四个同步fork组。32current-only前缀（29s3/3s7）在候选前
冻结；fresh baseline/repeat字节一致，24候选branch前缀差0，32×7结果完整。
原异步46cohort和失败候选保留为工程记录，不作GateA科学证据。

GateA固定合同UNPROMISING：baseline稳定Z23/32，GT-Y选择24/32，gain3.125pp
(anchor95%CI0..9.375)，未达≥5pp/lower95>0；GT-Z候选上限25/32。
仅两处潜在rescue，其中一处由候选顺序平局选中；另一处全部短Y相同，baseline
step89才越过掉落高度阈值，成功候选到step90也仅高于阈值1.838mm。
因此停止当前短Y/固定七候选selector，不把小正点估计当正式控制收益，也不
扩成全部Y、E/I或Cm无效。改变U系数不能拆开相同完整Y标签的候选。
GateB加噪、GateC表示比较和GateD执行/PPO均未启动；旧execution forecast和
进一步MSE调参继续暂停。未来需先冻结具有有意义长期机会的目标/候选合同。

GPU真实point-flow归档14panel FK/live最大tip误差1.714e−5m，Y/Z重建完全一致；
包含样点、代码/mesh和原始panel哈希。独立raw标签/组装/PD/selector/CI复算
全部一致，仓库验证通过。零新fit/policytraining。native累计1679.61s，
产物约420MiB，预算内；78Task/pairedsim+7coldcontract tests通过，1skip。
共享solver四组mosaic、强s3条件群体、forcepair代理及短Z horizon仍限制外推，
不把anchorCI当group数值独立或在线混合选择已验证。最终matched训练策略
Cm utility仍OPEN。[实验卡](../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261005-oracle-y-utility.md)。


## Ref13_1 修正：滚动 GT-Y 可分性

完成纯CPU保存轨迹审计，零新模型或仿真：32×7候选的原Y/Z和tau0标签精确
重放，保持32步窗口、原utility、每8步cadence。124个Z不同的候选pair中50个
初始utility平局；46个在滚动后正确区分，涉及8/32原前缀，提前29–47steps，
记录路径可分性PROMISING。这些pair并非独立实验；8前缀中6个baseline本已
成功，env33已由one-shot救回，不能写成8个新增rescue或rolling策略收益。

env33 middle−/grip+在tau16分开、提前32steps；env36 baseline/thumb−在所有
已保存每8步查询tau0..56仍平局，dense诊断tau57/58才差1、提前32/31steps。
原90步记录的tau56窗口终点88，下一查询tau64需未来到96（缺6steps），不能
补齐或把dense诊断当主cadence成功。后续候选H已不同，无法拼接反事实切换。

因此ref13 one-shot no-go保留，明确不关闭rolling short-Y；后续优先新的共享
当前状态rolling GT-Y干预证据，而非直接改long-Y/扩大authority或继续predictor。
实际rolling收益/可执行上限仍UNCLEAR，原GT-Z25/32只是旧分支机会ceil。
主审计1.43s、产物不足1MiB、14合同回归测试通过；源码/数据哈希和边界冻结。
独立CPU全部rolling标签/risk/U、事件/提前量/统计复算一致；4pair缺后续查询，
不额外声称持续性。去掉这4pair后的42pair仍覆盖全部8信号前缀。
旧execution forecast/PPO和MSE调参继续暂停，最终matched训练策略Cm utility
仍OPEN。[滚动审计卡](../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261005-rolling-gt-y.md)。

## Ref14 主链 / ref14_1 rolling oracle 已完成

用户更新ref14_1，当前只推进真正same-current-state RollingGT-Y Oracle Control。
保持32步Y/U、七个K8候选、每8步重规划与原稳定Z90；新的真实组合执行产生
下一轮状态，禁止候选世界状态拼接。每次fresh冷重放整条实际动作前缀，
PhysX solver历史由重放重建；旧25/32仅历史one-shot机会，不是rolling上限。
原32anchor/四组、已有暴露cohort复用，属于机制Probe，不提供独立Validation。

工程smoke163.16s完成：7anchor新baseline/repeat/旧状态历史动作90步结果
EXACT；4个非零组合动作之后，两次下一状态重放仍EXACT。代码与协议已本地
提交，主运行commit337d3a6，GPU6/7最多4进程（每卡2），<=7200s/4GiB。
固定utility<=1.25，所有baseline达到上界时，baseline优先平局可精确省去
六个候选，不改变选择，不编造其Y；其他情况完整同状态七候选。
主实验外部中断后已由本会话恢复（恢复入口a949b62），原progress保留，当前
进度见同run目录resume_progress.json；原短Y/候选/执行源码与协议哈希未变。
固定32-anchor gate 已完成：baseline 23/32，实际 same-current-state rolling
GT-Y 27/32，救回4、伤害0，净增益12.5pp；paired bootstrap 95%区间
3.125--25.0pp，lower95为3.125pp，达到卡片预先固定的 PROMISING gate。
四组全部完成，baseline repeat/prefix 和 actual mixed path 检查通过；这是暴露
cohort上的机制 Probe，不是独立 Validation，也不能替代最终 trained-policy
Cm-on/off。结果文件为
`outputs/cm-interaction-oracle/rolling-oracle-control-s263-s264/result.json`。
预测器、PPO尚未启动；离线 Y ranking/noise tolerance Probe 已完成：43个完整计划、
339个同状态面板；sigma .10 的 median pairwise accuracy=0.758，sigma .20=0.696，
离线筛选状态 PROMISING。它只提供预测器的排序门槛，不提供 noisy rolling Z
retention；下一步仍需冻结 predictor 输入/输出并做真实 rolling intervention。
原7200s/4GiB主实验预算不重置。
用户授权ref14_2并行工作已完成并本地提交912aa1a：只读完成组数据资产、固定Y
时序诊断、已有flow与tinyFK审计、ranking/noise评估合同；17新测试通过，无
新仿真或训练。[并行交付](../src/task/cm-interaction-oracle/docs/ref/ref14_2_progress.md)。
[固定协议与进度入口](../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261005-rolling-oracle-control.md)。


## Ref14_3：actual-flow learned-Y ranking 已完成

按用户ref14_3进入预测器阶段：已有完整32anchor/fourgroup输出重新打包，
GPU FK重建120点两段actualflow，无新PhysX。四折anchor隔离，source内三折
E/I OOF，matched H/Direct/Bottleneck/Hybrid/GT_EI；保持原Y8和U。
339complete panels/2276strict pairs：Direct57.38%、Bottleneck57.82%、
Hybrid57.95%，均未过冻结70%离线gate，固定合同UNPROMISING。GT_EI67.36%。
三臂flowshuffle退化CI均正，保留动作敏感信号；Bottleneck/Hybrid相对Direct
仅+.44/+.57pp，CI跨零，unique E/I contribution UNCLEAR。
GPU0训练30.79s、savedweight重放3.95s且误差0；独立只读CPU/NumPy核查一致。
H约50%的微小偏差来自跨batch的1.19e−7舍入，仅5strictpairs，原始证据保留。
110Task tests/1skip与仓库changed验证通过。无新增rolling learned-Y、execution
或PPO；不将当前模型、预算、selectedoracle support的负结果升级为全局否定。
Source ranking也仅约.585–.711，下一步最小Decision可另行冻结source-panel
ranking objective比较以区分拟合/泛化问题；不直接扩epoch/seed或消耗仿真。
[Ref14_3协议与结果](../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261006-actual-flow-y-ranking.md)。
最终trainedpolicy matched Cm-on/off utility仍OPEN。


## Ref14 baseline source_e260：冻结 critic 排序已完成

用户指定先比较critic与Y，使用同一source_e260 PPO checkpoint，无新模型训练。
原ref13的32anchor/七候选/fourgroup面板完整冷重放，28候选执行与原H、
动作、physical32以及整个height/pair/valid90 EXACT；独立critic replay误差0。
原PPO奖励含2approach/10held_lift/5lift_progress，已按源公式从保存状态离线
重建；最初遗漏shaping的评分保留为诊断，不作为原PPO评分证据。
同one-shot面板baseline23/32、GT-Y24/32；reward8+gamma^8V8为20/32，
救回1、伤害4；V-only19/32。原冻结gate UNPROMISING，暂不推进这个critic
评分的rolling执行或Cm→V，不改系数寻找正结果。独立组件/排名/统计复算通过，
112Task tests/1skip。重放RNG错误的旧运行保留并排除；修复后总主采集690.254s，
低于原1200s cap。这里只是one-shot潜在结果组合，不能与实际rolling GT-Y27/32
当作同协议对照，也不否定所有critic或全局Cm假设。
[协议、修复和完整结果](../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261006-frozen-critic-ranking.md)。
最终trained-policy matched Cm-on/off utility仍OPEN。


## Ref15：training-only GT auxiliary 有界 Probe 已完成

用户授权以source_e260 PPO为起点，让GT真实h8交互loss直接塑造actor z，
推理仍仅actor；四臂均追加64epoch/131072交互/3072更新。完整96新frame0
episode持续45tick且以后不drop：PPO62、条件GT21、shuffle51、stopgrad62；
source本身3。条件GT较PPO差42.71pp，更多后续掉落。该cohort不同于旧32anchor
GT-Y/critic，不能跨协议比较。A/D整native权重/RMS与逐episode结果EXACT，
独立GT/action/mask/reward/MC/成功统计审计通过，未见实现缺陷导致负结果。

冻结gate为UNCLEAR：条件GT heldout loss仅改善2.24%，未达到5%学习门槛；
同z return readout较PPO改善2.52%，弱于shuffle。原生critic独立于actor z；
readout针对finite source-policy MC与MC-sourceV residual，不是当前策略advantage。
本轮无策略收益，但不能否定已充分学会GT监督的路线。停止固定h8/lambda.05
短配方，不扩epoch/seed/系数、不拟合Cm、不进入Validation。若重开先另冻最小
Decision核实aux可学性/独立覆盖；不把训练reward或loss升级为utility。
训练429.28s/1701.87GPU-worker秒，评价222.44s/552.89GPU-worker秒；117Task
测试通过/1skip及changed验证PASS，原始失败smoke和完整证据保留。
[协议与完整结果](../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261006-gt-interaction-aux.md)。
最终trained-policy matched Cm-on/off utility仍OPEN。


## 新路线：cm-pointflow-effect-pretrain 数据预检

用户明确指定新分支`cm-pointflow-effect-pretrain`，先按Task ref1检查SPIDER
retarget_full是否可用于点流/effect预训练。四来源各一条Inspire，120文件checksum
通过；750帧qpos/qvel/ctrl/time及portable MuJoCo模型加载/FK点流工程检查PASS。
完整Inspire清单1946条，当前只下载4条及全部依赖。尚未开始大规模训练或PPO。

下载SSL EOF已用curl解决。Python3.12/MuJoCo3.7隔离环境可加载原始场景；
原Isaac环境未改。SPIDER18维ctrl不等于现有PPO action，实际轨迹违反当前native
耦合最大约1.017rad，因此native control replay仍需适配/漂移检查，不能假定直接
可用。数据支持样本级离线几何点流提取，不代表Isaac dynamics或策略utility。
[预检及后续数据合同](../src/task/cm-pointflow-effect-pretrain/docs/DATASET_PREFLIGHT.md)。


## 2026-10-06 主工作树删除后的恢复边界

用户确认删除了Ref2Dex-agent-baseline，当前工作树原.git和outputs均指向该目录。
从远程Ref2Dex-agent恢复至1afe075的历史，创建当前目录独立.git并保留
cm-pointflow-effect-pretrain分支；使用index-only read-tree，未checkout或覆盖工作文件。
当前Task未push的提交对象身份未找回，但代码/文档文件还在，保存为恢复快照。
旧.git指针和outputs链接在artifacts/git-recovery-20261006保留，新outputs为本地空目录。
被删除的checkpoint、rollout和结果数组未恢复；此前实验卡所指原始输出现不可访问，
不能把已保留的文档摘要当作原始证据仍可复算。Git恢复不是数据恢复。
本轮OakInk2产物使用Task声明的artifacts/cm-pointflow-effect-pretrain目录。


## 2026-10-06 OakInk2 annotation-only 100-sequence audit

按用户ref2，本Task当前只推进OakInk2，不混入SPIDER/其他数据。100条固定序列及
五个资产包已校验下载，6.29GB；GPU0完成双手11点/物体512点/SE(3)审计877.62s。
646565手帧、477序列物体对，无缺失mesh、无效rigid pose或帧号缺口。
4961958重叠窗口原始静止90.40%；program筛选731579窗口，运动384679，
静止47.42%，支持继续小规模预测Probe。120Hz的h8只有66.67ms，不能沿用30Hz解释。
发现少量相邻物体跳变，影响1344个program窗口；训练前检查并屏蔽跨异常窗口，
按sequence划分、运动/静止平衡，再决定大规模投入。尚未开始预训练/PPO；
数据工程可用不等于可学性、因果effect或策略utility。Git已独立恢复，但历史产物
删除仍影响6个旧测试；新OakInk2流程不依赖这些路径。
[结果与下一步](../src/task/cm-pointflow-effect-pretrain/docs/OAKINK2_DATA_RESULTS.md)。


## 2026-10-07 WM30/K24 架构实现与接口完成

用户指定架构并确认program锚点+0.5m局部物体、三组独立训练及B验证shuffle、
全627条标注/最多4GPU/整组24h。完成1cm稀疏卷积64/128/256、384维8层场景/
4层动作/6层dynamics Transformer、多物体24步SE(3)及解析点损失，37208777参数。
局部选择已核实使用当前mesh几何中心，避免部分标注原点偏离>20cm导致选错。
6项合同测试及12条现有数据三组6步GPU smoke通过，初始化hash一致，checkpoint
保存/恢复/独立推理接口通过。属于工程检查，不作模型质量或policy utility结论。
全量下载（HTTP range续传）与GPU3预处理运行中；627完整校验及split审计后自动
启动GPU0/1/2三组，每组40000更新、有效batch16，共享24h截止时间。不启动PPO。
[实现与运行入口](../src/task/cm-pointflow-effect-pretrain/docs/WM30_INTERFACE.md)；
[固定实验协议](../src/task/cm-pointflow-effect-pretrain/docs/experiments/probes/P-20261007-oakink2-wm30-k24.md)。


## 2026-10-07 WM30 全量训练中间状态

627条全部下载/预处理完成，5060616窗口，501/70/56序列划分。实际运行提交ec4d918，
GPU1/2/3三组训练已运行约5h17m；H33310、H+A27796、shuffle28943/40000，
最慢组估计还需2.3–2.5h，仍在24h预算内。数值/进程正常，但最近运动锚点h24验证
点EPE三组均约28.17mm，与静止基线几乎一致，尚无A收益。不同更新数仅是中间观察。
四个验证窗口的checkpoint核对确认实际近零预测（GT最大90mm、预测最大约0.05mm），
标签非零、A梯度存在，FP32同样近零；不足以确定优化原因或否定方法。保持当前实验
到固定matched更新数，不改活跃配置/不追加预算；原始中间检查记录保留在Task产物。

## 2026-10-07 用户指定 ref3 PointWorld-small 替换

用户指定参考本地PointWorld、按Task ref3换架构并启动训练，随后明确授权停止旧训练。
旧三组均保存checkpoint/正常退出，H40000、H+A33464、shuffle34823，不能形成
matched最终比较；原始产物保留。当前分支不变，Mission/claim不变。

新实现直接使用PointWorld PTv3-small、128维/patch128/1cm，统一场景与24步双手
点输入、object pooling+SE(3)小head，逐时域train-only归一化和官方逐帧运动soft权重。
实测50,495,881参数，完整监督保持每物体512点。重复栅格的SparseConv不稳定问题
在正式运行前修复：唯一voxel聚合+inverse恢复，评估全部层固定序列化顺序。
7合同测试通过/1旧模型可选GPU测试skip；三组6步smoke初始化相同且checkpoint精确
保存恢复，独立val推理/恢复加载通过。80步重复运动batch loss3.016→0.225、
末端anchor EPE44.3→16.4mm，只证明学习链路/小batch拟合，不作泛化结论。

复用627条完整数据及原sequence split，GPU0/1/2三组独立40000更新、有效batch16、
共享24h上限。运行目录outputs/cm-pointflow-effect-pretrain/pointworld-small-wm24-20261007，
启动前统计已冻结、接口已验收；实时进程/进度/运行提交见group_status.json。
实际运行提交f507f18，launcher3942111，worker3942116/3942117/3942118；三组均已
完成正式更新，loss/梯度有限、初始化/data/stats/config身份一致。早期每步约1.2–1.4s，
40000步粗估14–16h，共享24h上限不变。本次改动范围verify通过；全恢复分支仍有
6个依赖已删除baseline产物的历史测试失败（另102项通过），不影响新路线输入。
不启动PPO；最终trained-policy matched Cm-on/off utility仍OPEN。
[协议及证据](../src/task/cm-pointflow-effect-pretrain/docs/experiments/probes/P-20261007-pointworld-small-wm24.md)。

ref4只读诊断已完成：229训练序列/512窗口，528action点平均合为160.55个空间voxel，
同手同关键点跨时间碰撞影响83.70%的点，下一版优先保留时间身份。5mm逐帧selector
原始权重偏低，但归一化后moving-anchor相对均匀监督系数中位1.217，不能据此声称
所有moving监督仅剩2%–3%；部分旋转样本仍被相对降权。审计自身遗漏padding mask
的初次归一化份额已排除并修正，原训练mask正确；三组继续原配方/原预算，结论仍UNCLEAR。
[诊断协议、结果和修正边界](../src/task/cm-pointflow-effect-pretrain/docs/experiments/probes/P-20261007-pointworld-ref4-input-loss-audit.md)。

## 2026-10-08 consequence twin contract hardening

在独立工程复核后，twin v1 继续收紧：native prefix 按 30 Hz control tick 与 60 Hz
physics frame 分离计数；zero-step replay 只接受自然的空列表，显式空二维
action 仍核对 18 维；RNN 必须使用显式 `is_rnn/state` sentinel；Python/NumPy/Torch RNG
格式、非空 physics/history/controller provenance 均在入口拒绝缺失值；native adapter
自动冻结 direct tensor、scalar 和 Enum（包括 `_state_init`）inventory，并提供 CPU/GPU
Torch RNG restore helper。Task tests 为 114 passed，compileall 与 diff-check 通过。当前仍没有 native branch runner
和真实 twin branch 数据，不能把合同测试当作 twin coverage 或 evaluator 科学证据。

## 2026-10-09 native GPU serial reset/replay Probe

为区分“GPU fresh process 行为不稳定”和“同一 simulator 内固定控制能否配对”，新增
`engineering_serial_replay` worker（实现提交 `bf96546`；随后在 `11fd62d` 中强制该
模式只能使用 `gpu_physx_gpu_pipeline`）。在 seed282、原生 GPU PhysX/GPU pipeline、
单环境、同一进程内依次 reset 五个 arms：baseline、zero_repeat_1、zero_repeat_2、
positive、negative；baseline 的执行控制被完整记录，其他 arms 在 tick48--71 重放并
只给候选加固定 residual。实现通过 `tools/verify.py --changed`、py_compile 和
consequence-evaluator 171 tests。

r29 baseline 最高抬升 `0.8241 m`、held484；两个 zero replay 轨迹和 outcome 完全相同，
candidate positive/negative 分别为 `0.7971 m`/485 与 `0.6489 m`/310。每次 reset 前后
absolute frame 保持不变，reset RNG anchor 相同。object pose、hand points、gap、q/dq
和 object velocity 的 zero-pair query-relative p95 均为零；但 contact force p95 仍为
约15.82/17.15，history p95为2.0，首个 task-buffer divergence 是 reset 后 stale
`_curr_obs`。这些 trace 没有覆盖 PhysX warm-start/contact-manifold/island cache，不能
称 full-state twin。冻结 physical-bank/TCC 的只读审计给 baseline 与两个 zero 同一
`Y=0.0411695`，positive/negative 为 `-0.0145127`/`0.0044708`；该结果只说明本次
固定控制下 label 可复现，不是 Gate1 utility 证据。

因此 serial reset/replay 保留为 frozen-control 的工程噪声校准容器，确认它可以保持
GPU baseline 行为，但 strict same-state Gate1、reactive-policy candidate ranking、
evaluator/PointWorld 和正式 utility claim 仍关闭。raw packet、serial noise audit 和
serial GT-value audit 位于
`outputs/consequence-evaluator/gate1-gpu-serial-engineering-20261009-r29/`；不修改
reference bank、Y、policy weights。下一步需要 hidden PhysX state fork/restore，或预先
声明 field-specific noise 与重复数的 statistical paired design，再决定是否重启 Gate1。

为检查 serial arm 顺序，又在提交 `79cdbf3` 下做了 candidate-first fresh launch
（`baseline → positive → negative → zero_repeat_1 → zero_repeat_2`，r30）。与 r29
相同 seed/checkpoint/backend 的 baseline parity 失败：r29 为 `0.8241 m`/484 held，
r30 为 `0.8281 m`/483，action 在 tick47、物理轨迹在 tick48 已出现差异。因此不能
用 r29↔r30 的候选或 noise 差异归因于 warm-cache arm order。r30 内部两个固定控制
zero 仍保持机械字段 exact，而 contact/history divergent；其物理-bank Y 为
baseline/zero `0.0437091`、positive `0.0189845`、negative `-0.0122399`，仍是
engineering-only label diagnostic。该 order Probe 标记 `UNCLEAR`，serial 只保留为
单次 frozen-control field/noise calibration 容器；后续若没有冻结 baseline action stream，
不再追加 order-only launch。

## 2026-10-09 8-env GPU group decision note

当前决定问题：同步 GPU group 是否还能在不改变原生 GPU pipeline 行为的前提下，提供
可用于候选统计配对的短窗口执行容器。

关键证据：4-env group 的 r19 在一次 launch 中达到 `0.8262m/481`，但 r27 同合同
只有 `0.0891m/5`；因此把 env pair 当独立样本或直接进入 Gate1 都不成立。serial
r29/r30 的冻结 executed stream 两个 zero arm 在 object/q/dq 上可重复，但 contact
与 history 仍暴露隐藏 solver 状态。SDK 审查也未发现 GPU PhysX hidden-state clone/
restore API，不能用公开 setter 构造严格 twin。

root 选择先做一次低成本 8-env、72-step engineering Probe：env0 baseline，env2/3
候选，其他 env 为 zero roles；actor 总推理行数固定为 256（每 env 32 copies），
避免把 group size 与 actor GEMM 行数同时改变。先跑两次 fresh seed282 launch；若
两次 env0 都不能达到短窗口原生行为，停止 group 扩展并回到 serial frozen-control
统计设计；若两次通过，再决定是否跑四次完整542-step cluster Probe。所有结果仍为
`engineering_only`，不修改 Y/reference/policy，也不能替代 docs/user/完整链路.md
规定的 strict same-state Gate1。

该 Probe 已在 `01c085a` 下运行一次（r31），使用原生 GPU PhysX/GPU pipeline、seed282、
8 env、32 copies/env（总 actor rows=256）、72 steps。初始 semantic state exact，所有
角色 query 前 controls/done exact，但 env0 最高仅 `0.1439m`、held8；env1 zero 为
`0.1537m`、held8，候选角色几乎不抬升。相同 packet 的 zero-role query 前 object
p95 约 `9.37e-4m`，q velocity p95 约 `0.151`，history p95 `0.0231`；query 后
object displacement p95 约 `0.200m`，candidate effect 未形成稳定 margin。TCC 只读审计
的 progress-start range 为 `1.86e-4`，严格 contract 失败，deadzone 选择 baseline。
因为单次短窗口已低于预设 `0.20m` 筛查线，按停止条件不再跑第二个 8-env launch，关闭
group 扩展；r31 及其 audit 仅保留 engineering evidence，下一步回到 serial
frozen-control/launch-level statistical design 的决策，不进入 formal Gate1/evaluator。

group 关闭后的下一步 Decision Note：为区分“group solver 退化”和“固定控制下候选
标签本身无信号”，再做一个单进程 serial 72-step frozen-control Probe。该 Probe
只记录 baseline executed stream，随后重放同一 prefix 到 tick48 并运行两个 zero
repeat 与 ±candidate 到 tick72；它不评估完整 hold/place，不进入 strict scorer。
若 serial 的 zero Y 仍一致且候选选择稳定为 baseline，近期停止继续仿真，保留
strict Gate1 blocker；若选择不稳定，则只记录为 execution-noise evidence，不把它
升级成 utility 结论。

在 r32 结果出来后，新增一次且仅一次的 serial-cluster Decision Note。r32 的
reactive teacher 在 72 步内抬升约 `0.272m`，超过 `0.20m` 短窗口线；两个 frozen
zero 的 object/hand/q/dq/velocity 几何字段仍 exact，但 contact/history 继续漂移。
因此下一 Probe 使用 native `gpu_physx_gpu_pipeline` 的同进程单 env，先运行一个
`reactive_teacher` 保存 executed action stream，再按固定交错顺序重放
`frozen_zero_1..5`、`positive_1..2`、`negative_1..2` 到 tick72。候选只在
tick48--71 使用 `clip(teacher_action + residual)`，zero 臂全程重放 teacher
action；teacher 不作为 counterfactual zero。审计只报告 paired noise、机械字段与
contact/history 分层结果以及描述性 TCC/Y 对比，不计算置信区间、p 值或选择器结果。
若 teacher lift <`0.20m`、action/reset 合同失败、机械 zero noise 超预设阈值，或
两类 candidate 没有任一重复在主要机械字段达到约 `2x` zero floor，则关闭这条
统计工程路线。无论结果如何，packet 保持 `engineering_only`，不进入 strict
Gate1/evaluator/PointWorld，不修改 reference bank、Y 或 policy。

r33 已按该合同完成（代码 `e1c71b4`，packet 为
`outputs/consequence-evaluator/gate1-gpu-serial-cluster-20261009-r33/serial-cluster.pkl`）。
teacher tick72 最高抬升 `0.2706m`；action、done、reset frame/RNG、schedule 和
executed residual 合同审计全部通过。五个 frozen zero 的 object/hand/gap/q/dq/
object-velocity query-relative p95 均为 `0`；contact/history 仍单独出现隐藏状态
噪声。两次 positive 和两次 negative 的机械轨迹分别重复一致，candidate 相对 zero
的机械 effect 明显超过声明的 field floor；contact-force effect/noise 约为
`1.6--2.5x`。physical-bank/TCC 只读值为 teacher/zero `0.0051948`，positive
`-0.0117869`，negative `-0.0129771`，progress-start range 为 `0`；这只是
固定 executed stream 的影响诊断，不是 candidate utility 选择。

r33 因此通过 cluster 的短窗口继续条件，但不通过 strict Gate1。按预注册上限，
下一步最多再做一个完全相同的 fresh r34 launch，用于检查 launch-level teacher
行为和 paired effect 是否能重复；r34 若 teacher <`0.20m`、机械 zero 不再 exact，
或 candidate effect 不再重复，则关闭 serial-statistical route。无论 r34 结果如何，
不跑 evaluator/PointWorld、不改 Y/reference/policy，也不把十个 arm 当独立样本。

r34 已完成并通过同一 audit（packet 为
`outputs/consequence-evaluator/gate1-gpu-serial-cluster-20261009-r34/serial-cluster.pkl`，
read-only audit 为其 `gate1-gpu-serial-cluster-20261009-r34-audit/audit.json`）。
teacher tick72 最高抬升 `0.3001m`、held13，超过短窗口线但不构成抓取成功；action,
done, reset, schedule, residual 和 hash 合同均通过。五个 frozen zero 的主要机械
字段 query-relative p95 仍全为 `0`，contact hidden noise 继续存在；positive/negative
各两次重复的 mechanical effect 一致。TCC 只读值为 zero `-0.0094461`、positive
`-0.0117899`、negative `-0.0131493`，两类 candidate 都低于 zero，因此没有
candidate utility 信号。

按预注册上限关闭 serial-statistical cluster。它证明了同一 executed action stream
下可见机械 replay 在两个 fresh launch 中可重复，并量化了 contact/history hidden
noise；它没有恢复 PhysX hidden state，也没有把 strict same-state Gate1 变成统计
替代。Gate1、evaluator、PointWorld、Execution Bridge 和 MPC 仍未开始。下一步回到
可验证执行合同设计：优先评估 ACT/open-loop chunk 或原生 PhysX hidden-state
fork/restore 的可行性；在其通过前不改 Y/reference/policy，不fit evaluator。

### ACT deployment-conditioned chunk Decision Note

当前问题是 clean hold-audit 上的 ACT-like chunk 是否覆盖原生 GPU deployment
state。现有 checkpoint 在 clean holdout 上 chunk MSE 约 `1.6e-4`，但直接喂给
r33/r34 teacher query history 时 MSE 为约 `4.2e-4--6.3e-4`，说明存在
deployment-distribution gap。root 先做最便宜的离线判别实验：以 r33/r34 的
`reactive_teacher` history/action 为两个 launch-level episodes，leave-one-launch-
out 训练一个同结构 native 24-step proposal，冻结 train-only standardizer，并在
另一 launch 上评估 full chunk/first-action error。若跨 launch 仍优于现有 clean
checkpoint，下一步才值得收集更多 reactive deployment distribution；若不优于，
关闭该 route-conditioned proposal 路线。该 Probe 不运行仿真、不改变 Y/reference/
policy、不训练 evaluator，结果仍为 engineering-only。

该离线 Probe 已完成，代码提交 `94ef7df`，产物为
`outputs/consequence-evaluator/act-native-chunk-deployment-20261009-r3/`。使用 r33/r34
两个 serial-cluster 的 `reactive_teacher`，每个 launch 13 个 stride-4、非跨 episode
窗口；每折只在另一个 launch 上评估，standardizer 只由训练 launch 拟合，模型选择也只看
训练 launch（修复了初版用 held-out MSE 选 checkpoint 的 validation leakage）。r33→r34
held-out MSE `1.902e-4`、first-action MAE `0.00542`，r34→r33 为
`1.827e-4`/`0.00521`；均优于 clean checkpoint 的 `1.932--2.017e-3` 和
route-s3 checkpoint 的 `3.532--3.963e-4`。两次 launch 的 physics/controller/backend
identity 与 arm schedule 均显式相同，GPU2 运行约21秒、显存约361--369MiB。

因此 deployment-conditioned proposal 在动作空间标为 `PROMISING`，但每折只有一个
launch、13 个重叠窗口，不能形成泛化、行为、utility 或 Gate1 证据。该结果只保留“未来
若获得预算，先收集更多 native reactive-deployment episodes 再拟合 proposal”的工程
方向；不启动新的 ACT candidate/serial simulation，不fit evaluator，不进入 PointWorld。
严格 Gate1 仍等待可验证的 PhysX hidden-state fork/restore 或新的执行合同。

### Clean-only continuous source audit

上一条 offline Probe 暴露了 continuous loader 的来源漏洞：`unlabeled` episode 也可能
带有 intervention。新增的 `clean_only` 合同只保留 manifest 中
`assigned_phase=clean`、`perturbation_tick=-1` 且数组中 residual plan 全零、
`plan_known` 全真的 episode；默认 loader 行为保持不变。用
`continuous-20261008-{train,val,test}-r1` 和 `expert=airplane_base` 审计后，三个 split
各有 8 条 clean episode、1204 个 24-step windows；精确 `s3_airplane_lift` 每个 split
只有 1 条。所有入选数组通过零 residual/finite/known 检查，来源 manifest 中冻结的
`GRAB_00000260.pth` SHA 与当前 history RMS 一致。

这只支持一次 episode-held-out 的 offline chunk fit，不支持 route-specific 泛化或 native
行为结论。下一步最多使用 train split 内的 episode holdout，并在独立 val/test manifest
上报告动作空间误差；在行为 screen 通过前，不启动 GT ranking、evaluator 或 PointWorld。

该 bounded fit 已完成，产物为
`outputs/consequence-evaluator/act-native-chunk-clean-airplane-base-20261009-r1/`。
训练使用 6 条 train episode，2 条 train episode 做 holdout；独立 clean val/test 各 8 条
episode、1204 windows。新 chunk 的 val/test MSE 为 `2.405e-4`/`3.397e-4`，低于只用
train target mean 的 `5.855e-4`/`7.483e-4`，也低于旧 clean checkpoint 的
`7.905e-4`/`8.857e-4`；first-action MAE 为 `0.0118`/`0.0125`。因此该动作空间
Probe 标为 `PROMISING`，但不外推为 native 行为、utility 或泛化结论。下一步只做一次
`open_loop24` native behavior screen；不跑 `receding8`、candidate ranking、evaluator
或 PointWorld。

首次 behavior screen 的 4-env group 没有满足 baseline 前提：reactive teacher 只有
`0.129m/7`，reactive repeat 为 `0.637m/237`；同一提交下独立单环境 baseline 恢复到
`0.8165m/485`，因此该 group packet 标为 `INVALID_IMPLEMENTATION`，不能作为 chunk
负证据。进一步按 motion 分层后，当前 broad fit 在真正的 `s3_airplane_lift` clean
val/test 上 MSE 为 `5.823e-4`/`6.669e-4`，旧 clean checkpoint 为
`2.962e-4`/`2.054e-4`。这说明 aggregate windows 的改善没有覆盖当前抓取路线。

下一步 Decision：利用已有 `continuous-hold-audit-20261008-r1` 中 24 条同一
`airplane_base/s3_airplane_lift` clean episode，在显式 audit-only、冻结 expert RMS 的
合同下做一次 route-specific offline fit；若 episode-held-out 仍不优于 train mean/旧
checkpoint，则关闭该 proposal route；即使优于，也必须先解决 group baseline contract
才可做行为 screen。

route-specific fit 已在 audit-only source 上完成，产物为
`outputs/consequence-evaluator/act-native-chunk-route-s3-engineering-20261009-r1/`。
20 条 episode 训练、4 条 episode holdout；holdout MSE `1.783e-4`，旧 clean
checkpoint 为 `1.985e-4`，train-mean 为 `9.207e-4`，first-action MAE 为
`0.01136`。这是约 10% 的动作空间改善，样本来自同一 hold-audit wave，仍只标为
`PROMISING/UNCLEAR`，不构成行为证据。由于 group baseline contract 已失败，暂不再
启动 native chunk screen；ACT 只保留旧 r2 的 open-loop24 行为工程基线。

### Native GPU CUDA synchronisation probe

旧 ref13 的原始 binary panel 已确认不在当前工作区或历史 baseline worktree；当前只剩
`tmp/ref13` 的日志和 `raw_stat_replay.json`。历史 panel 虽可由源码确认有 32 步
trajectory/native-q，但没有完整 24 步 candidate controls、现行 `object_pose`/11 点手坐标/
timestamps，也属于 GPU PhysX + CPU tensor pipeline，不能重标为当前 Gate1 packet。

为区分 fresh native GPU 行为波动是否主要来自 host/kernel 异步排程，在不改变 backend、
seed、checkpoint、Y 或 reference bank 的情况下做了 `CUDA_LAUNCH_BLOCKING=1` Probe。
两个 fresh 单环境 baseline 都达到约 `0.79m/483 held`，行为 spread 明显比之前的
full-to-zero launch 小；但接触 buffer 仍在 tick44 分叉，完整 state 只在 `44/543` 个
hash 相同，不能声称 same-state replay。相同设置的 4-env group env0 只有
`0.186m/10 held`，zero-pair hand/q/velocity/object-velocity 噪声门也失败。结论是
同步设置可能改善单环境行为稳定性，却没有形成候选执行合同；关闭 synchronisation-only
route，不启动正式 Gate1/evaluator/PointWorld。审计见
`src/task/consequence-evaluator/docs/experiments/probes/P-20261009-gpu-cuda-sync.md`，
摘要产物在 `outputs/consequence-evaluator/gate1-gpu-sync-probe-20261009-r1/` 和
`gate1-gpu-sync-group-20261009-r1/`。

### Native GPU group role/layout probe

进一步审计发现，4/8-env 组把 env0 的世界原点/首行路径与非零 origin env
配对时，固定 controls 下 tick1 就出现约 `3.46 mm` hand、`0.0116` joint-position
漂移；这早于 tick44 的显著 contact-force 分叉。新增的工程入口支持 bounded
`--query-tick`、非默认 `--zero-env-pair`，并把 query 和 scene-layout hash 纳入
replay identity。

一次 tick32 pre-contact query 仍在 tick1 漂移，4-env baseline 只有 `0.245 m`；一次
小 spacing 诊断也没有恢复行为或 noise gate（且会改变 broadphase，只保留为诊断）。
8-env、256 actor rows 下把 zero pair 改为非零 origin 的 `[1,4]` 后，选中 pair 的
geometry p95 通过但 pairwise gate 和行为筛查失败；更稳定的 `[4,6]` pair geometry
接近 exact、72 步 baseline `0.234 m`，candidate effect 与 pairwise gate 仍失败。
预先限定的 full 542-step `[4,6]` confirmation 最终只有 `0.153 m/9 held`，行为、
pairwise 和 candidate-effect gates 全部失败。结论是 group role/layout route 关闭；这些
packet 不能进入 strict Gate1。证据见
`src/task/consequence-evaluator/docs/experiments/probes/P-20261009-gpu-group-contract.md`
及 `outputs/consequence-evaluator/gate1-gpu-precontact-group-20261009-r1/`、
`gate1-gpu-spacing-group-20261009-r1/` 和
`gate1-gpu-nonzero-pair-group-20261009-r1/`、`r2/`、`r3/`。

### Native PhysX binary hidden-state audit

在 Python surface 审计之后又做了归档 Isaac Gym/PhysX native binary 的只读符号检查。
`libcarb.gym.plugin.so` 确实包含 `PxCloneDynamic`、`PxCloneStatic`、`PxCloneShape`
以及 PhysX object/RepX serializer helper，但没有可调用的 `PxSerialization` scene
collection create/serialize entrypoint、`PxCollectionExt` 场景收集器，或 contact-manifold、
warm-start、solver-island、GPU cache 的 fork/restore boundary；`libPhysXGpu_64.so`
也只有内部 contact/solver/cache routines。actor clone 只复制公开 actor 属性，不能复制
scene membership、joints、sleep timer 或 hidden solver state；RepX/object serialization
也不是 live solver snapshot。原始 symbol/hash 清单在
`outputs/consequence-evaluator/hidden-physx-binary-api-audit-20261009-r1/audit.json`，
实验卡为
`src/task/consequence-evaluator/docs/experiments/probes/P-20261009-hidden-physx-binary-api-audit.md`。
因此 strict Gate1 blocker 已从“可能遗漏 Python wrapper”收窄为当前 runtime 没有可调用
的 hidden-state 执行合同；不重复 group/serial/CUDA/CPU/host，也不启动 evaluator 或
PointWorld，等待新 runtime/API 合同或明确的 claim Decision Checkpoint。
