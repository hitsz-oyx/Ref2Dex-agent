# 当前状态

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
