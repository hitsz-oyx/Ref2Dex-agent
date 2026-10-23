# τ goal 契约：reset 时已知的任务级目标（brief45 修正）

撤回“把调度器 waypoint 变成 goal 通道”的方向。`ResetWaypointChannel` 的来源校验、
落盘与训练/部署共享实现继续保留，但主条件改为**任务级目标**；局部 waypoint 单列诊断。
此前 v1 的几何阶段合同只认证局部调度机制，不认证“物体最终该去哪”。

## 两个量

| 名称 | 来源与语义 | 用途 |
|---|---|---|
| task-level goal / 任务级目标 | 场景/任务在 reset 前声明的最终物体位姿 + `task_completion` 阶段语义 | 要接回 τ 的任务信息 |
| planner waypoint / 相位机局部 waypoint | `task_scheduler.SchedulerDecision.local_goal_pose`，执行器已有的当前阶段局部目标 | 可选诊断，不能代替任务级目标 |

本 wrapper 的诊断值来自 `OrderedGoalScheduler` 路径子模块；缺完整任务阶段反馈，不能
将其冒充完整 `TaskScheduler.local_goal_pose`。两者都属于局部规划目标，不是最终任务目标。
旧 goal-chain 审计还给路径子模块提供了完整录制路径作为 oracle reference，不能认证成
真实 reset 时已知的任务指令。历史反例与补测门槛见[实验卡](../experiments/probes/P-20261023-reset-bound-goal-channel.md)。

## 可执行合同

实现：[goal_channel.py](../../../wam-rl-policy-v3/src/wam_rl_policy_v3/goal_channel.py)。
v2 明确 `goal_semantics=task_level_terminal_waypoint`，
`time_semantics=task_completion_stage_no_arrival_deadline`，`target_stage=task_completion`。
这是完成任务时应到达的空间 waypoint，不是 t+24 实测终点；没有额外到达 deadline。
24步/30Hz只表示预测时域，局部 scheduler 的 .8s 只是 lookahead 参数。

当前支持场景事前声明的 `fixed_world_translation` 最终目标：绑定 anchor + 明示世界位移，
规则必须显式声明 `target_stage=task_completion`。不按物体名称、身份或执行结果选规则。
一般任务/完整路径/释放时刻不在这个生产器的认证范围，不凭局部 waypoint 补造最终意图。

| 判据 | 执行方式 |
|---|---|
| 任务级最终位姿完全由 reset 来源生成 | `task_goal_pose` 从绑定 anchor/rule 重建并核对；后续观测不改变最终目标 |
| 指令与观测分开 | `task_instruction` 不能是 `reset_measurement`；当前物体 pose 合法作 anchor，但不能独自声明任务 |
| reset 时可得 | 来源记录 role/kind/id、observed/available 时间和 execution/reset/clock；时间均 ≤ reset |
| 未来真实 pose 是违规标签 | 拒绝 future role、晚于 reset 的时间、本执行冒充独立录制、重算 hash 后替换最终 pose/路径 |
| 每 tick 因果消费 | observed ≤ available ≤ decision；当前 pose 只用于坐标编码，当前手/物观测推进单列诊断 scheduler |
| 训推同实现 | 都调用 `ResetWaypointChannel.update`，返回任务级目标的32个同ID query delta、points、bool valid |
| 局部 waypoint 隔离 | 仅写 `planner_waypoint_diagnostic`，不进入主条件；mode/index 是局部规划诊断 |

v6 `online()` / `predict()` 可显式指定任务级语义编码；默认 legacy planner 语义保留，
禁止静默把旧语义或旧 checkpoint policy 当成已接通任务级条件。合同测试在非零平移/旋转下
验证共享 producer 与真实 online/collate 三字段逐位相同；这不认证模型实际利用了该信息。
生产训练 reader、缓存身份和真实 collector 接入仍须通过独立准入门，当前保持 HOLD。

## 可认证落盘

每个新执行目录保存 `instruction.json` 和 `waypoints.jsonl`。前者保存 reset 规则、最终
目标、阶段语义和来源；后者保存 observed_at_ns / available_at_ns / decision_at_ns、tick、
instruction/reset 绑定、任务级 goal pose/delta，局部 waypoint 放在独立命名字段。
`available_before_reset` 只证明指令/reset 来源；当前观测与局部诊断的可用时刻单独记录。
同 tick 完全相同重试返回相同结果且不追加，冲突重试报错；已有目录拒绝覆盖，重启需内存
重放或新目录，不自动续写。写入预算拒绝不推进 scheduler/tick。每执行指令≤1MiB、
总记录≤16MiB，可用磁盘低于20GiB停止。

来源必须在可信 collector 边界真实记录；hash/自报 metadata 不能证明伪造来源真实。
合成写盘/重放只认证工程机制，不追认历史 reset，也不计入旧6,000窗口合法覆盖。
覆盖审计复用 brief35 与冻结 r7 的逐窗口有序 join：当前0合法、6,000缺证、0确证未来违规。
Shadow3母序列/30窗口仅有原生任务参数候选，缺完整原执行/reset/来源绑定，不升级为合法。

## 对照与结论

主比较固定为 **B合法任务级目标 vs A unknown goal**；同数据/预算/模型/面板，event均unknown。
C局部 waypoint 为诊断格，沿用历史反例；补测前必须写出机制理由及旧反例不适用的依据，
当前无该理由，不分配新预算。不得把 oracle 未来端点变成 B，亦不得把旧 C 算作干净 A/B。

来源覆盖为零时不运行科学比较，不启动 v6 重训。即便未来合法任务级目标接回后成功率
不升，也只能写“该通道未通过执行落点”；拇指限位、approach 未完成、约82mm残差等执行
瓶颈，与 τ 是否知道目标独立。离线改善不等于执行成功；负结果不能推出“任务信息无用”。
