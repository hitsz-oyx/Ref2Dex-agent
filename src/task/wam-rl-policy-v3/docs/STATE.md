# 当前状态

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
