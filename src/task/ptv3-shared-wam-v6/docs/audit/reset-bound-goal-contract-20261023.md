# τ goal 契约第二期：reset-bound waypoint

本合同将 goal 定义为 **空间 waypoint + 有序几何阶段语义**：
`spatial_path_waypoint / ordered_geometric_stage_no_arrival_deadline`。
它是当前阶段希望到达的位姿，不是实测 t+24，也不保证在 t+24 到达。
`reference_endpoint_time_s=null`；24 步 @30Hz 是 WAM 预测时域，scheduler 的
`horizon_s=.8` 是 lookahead 尺度参数；reference_times 是计划参数化时钟，均不是到达承诺。

## 第一步：可执行判据

实现：[goal_channel.py](../../../wam-rl-policy-v3/src/wam_rl_policy_v3/goal_channel.py)。
两端都用 `ResetWaypointChannel` → **同一个** `OrderedGoalScheduler / update_schedulers`。
本版本支持事前声明的 fixed_world_translation，不按物体名称、身份或结果选规则。
未声明的任务、一般路径以及 TaskScheduler 完整阶段反馈不在此生产器的认证范围内；
未来若使用 `SchedulerDecision.local_goal_pose`，必须连同 TaskSpec、反馈来源和同一
TaskScheduler 状态绑定，不能拿 terminal_goal_pose 或未来实测状态替代。

| 判据 | 执行方式 |
|---|---|
| reset 时可得的任务规则、anchor、query 和 interaction surface | 每项显式 source_role/source_kind/source_id、内容 hash、observed/available 时间、execution/reset/clock 身份；时间均 ≤ reset |
| 任务指令与观测分开 | task_instruction 不能是 reset_measurement；当前 object pose 可作 reset_observation，不能独自变成任务意图 |
| 禁止本次执行的未来真值 | future_realized_pose 角色拒绝；时间晚于 reset 拒绝；独立录制引用的 source_execution_id 必须不同于目标执行 |
| 禁止改名后塞未来路径 | 消费时从绑定的 anchor + 明示世界位移重建路径并核对，不能只检查字段名或 hash |
| 同一 scheduler 状态 | 从该 reset 连续逐 tick 重放；不在每个窗口重置 pointer/history；手 slot、物 slot、goal_type、SchedulerConfig 明示绑定 |
| 在线因果性 | update 只收一帧 pose/hand11；observed ≤ available ≤ decision，且匹配 reset 的30Hz tick；不接受未来数组 |
| 统一模型输入 | 两端都把 scheduler 位姿变成当前物体坐标系的32个同ID query delta；float32 delta/points + bool valid；与 v6 online/collate 比较 |

“只用 reset 时可得信息”限定**任务规则、参考及其初始化**；之后 scheduler 允许消费
截至当前 tick 的真实观测推进阶段，不能消费观测未来。如果要求之后连当前观测都不可
使用，则不是现有闭环 scheduler 的定义，本合同不做这种隐藏改变。

负控制与正控制见
[test_reset_goal_channel.py](../../../wam-rl-policy-v3/tests/test_reset_goal_channel.py)：
未来真实 pose 必须拒绝，当前物体 pose 必须合法；另测伪造合法 role 但未来 timestamp、
自身执行冒充独立录制、重算 hash 后替换路径、跨 reset、跳历史与未来 live 观测。
这些是通用来源判据，不依赖物体种类或指标。

来源记录必须由可信 collector 在真实边界写出。时间戳和 hash 不能证明伪造 metadata
真实，也不能替旧数据补造原 reset 历史。合成正控制只证明机制可执行，不计入6,000覆盖。

## 第二步：来源与绑定

新生成链在 reset 处保存完整 instruction（四项来源及内容）、每个 update 保存
waypoint receipt（instruction hash、reset、tick、观测 hash/时间、mode、confirmed_index、
target_reference_index、goal_type、goal pose/delta）。语义是几何阶段，不认证抓取/释放。
instruction 的 available_before_reset=True 只指事前规则与 reset 来源；动态 waypoint
receipt 显式保留 live_observation_available_before_reset 和 availability_scope，
绝不声称 t>0 的测量在 reset 前已知。

Shadow 先例可以用 nominal_constructor anchor + 原声明 displacement + position_only
生成；不拿 post-reset 观测替换 nominal anchor，不发明其余5,970窗口的任务规则。
新生成链绑定新执行，不能倒贴给旧轨迹。既有 collector 尚未接入此 wrapper，必须在真实
collector 的 reset/观测边界调用后，才有真实新执行证据。

覆盖率审计直接复用 brief35 逐窗口证据与 r7 有序元组 join；不重新调查已核实事实。
本轮结果及成本见同目录实验卡，缺绑定仍须判 unproven，不能填 True 或当作确证非法。

## 第三步：干净对照边界

当前不重训 v6、不换模型、不以历史 oracle/planner 替换作对照。合法-only 覆盖为零时
不启动比较；先冻结 paired preregistration，待同一批训练和评价行都具备来源绑定才可运行。
接回 goal 的任何后续训练须服从“不重训 v6”的本轮限制，不能以预注册自动获得授权。
