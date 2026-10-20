# 当前状态

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
现有Current421 best仍可作为父策略，不需为迁移重训；下一步可重训正确条件的Scheduled BC。
新训练保留step0基线并记录shape特征漂移；不因ref1_2提出的学习风险修改容量或50k配方。
当前工具只有有界单回合collector，尚不作为多回合长PPO训练入口。

ref1.md 两项新增缺陷已补回归：Scheduled BC 保留所有 task-active 命令，
无效 τ 显式为零、valid=False，使用原时刻真实动作监督；配对 DEV 同样包含启动样本，
另报 valid/invalid 分组。意外掉落不再确认未到达位置的计划 RELEASE，恢复时输出
CONTACT 意图，重抓后保留待执行的 RELEASE；合法中间释放也有正向测试。
v3 共132项测试通过。四进程有序数据回放与串行结果逐数组一致。
r2 在 CPU 数据准备阶段、零次参数更新时停止，保留停止记录。
下一次训练为 r3，所有 active TRAIN54162/DEV5879 行，包含无效 τ TRAIN1502/DEV186 行。
事件修复不改变 task10/local goal 或当前不接收事件的 WAMv4 输入，既有统一目标缓存仍可复用。
这两项修复不等于已获得正向 BC 抓取结果；尚需新训练和配对闭环验证。
额外覆盖“已经到终点但尚未进入 RELEASE 阶段就掉落”：最终释放必须等显式
RELEASE 阶段授权，不能只靠到达终点确认；相关24项定向回归通过。
