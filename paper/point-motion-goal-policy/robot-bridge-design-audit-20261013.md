# 物体点 goal 如何接现有 WAM → Chunk-R 机器人链

日期：2026-10-13。只读审查当前源码；未改模型、启动训练或重新执行机器人。本文描述接口与必要桥接，不声称新 encoder 已有效。

## 明确可以沿用的链

`当前场景/手历史 + 成对物体 goal → Actor 的手位移 tau → 当前机器人状态 + tau → Chunk-R → 18D native command（12个active）→ 实际执行`

**只换 EgoDex 的 goal encoder，不会自动让既有机器人 adapter/R 获益。** 有两道必须显式处理的接口：人体/机器人手的实际几何差异，以及新版 goal tokens 与旧 decoder/checkpoint/cache 的兼容。目标条件通过 Actor 到达机器人；R 没有单独读取 object goal 的入口。

## 源码证据：Actor 输出是什么

- `src/task/goal-point-wam/src/goal_point_wam/model.py:258–272` 输出 `[B,24,2,11,3]`；`data.py:139–146` 将训练目标定义为 `future_hand - current_hand`。它是以米表示的手点位移，不是机器人关节 action。
- human 数据的槽位是 wrist、五个近端关节点、五个指尖。robot 数据通过 `src/task/trajectory-policy/src/trajectory_policy/wam_shadow.py:10–17,57–79` 把实际 Inspire link 点重排进入同样的 2×11 槽。该文件 `:1–5` 明确提醒：语义配对不等于解剖点位置相同，尚没有自动拟合 embodiment transform。
- `wam_shadow.py:35–59` 用**当前机器人物体逆变换**，将所有已测 H4 手/物体历史和已知 goal 放入同一当前物体坐标。Actor 的 tau 因而在当前物体参考系；`:21–32` 的 `active_hand_tau` 只选右手并恢复 native link 顺序，不再施加新旋转。
- EgoDex 的同型输出来自人体关键点，现有视频点目标在当前相机坐标。形状一致只能保证张量接口；仍需共同参考系设计、人体到机器人适配监督和物理执行验证，不能等同能力直接迁移。

## R 具体读取和输出什么

- `src/task/trajectory-policy/src/trajectory_policy/retargeter_data.py:18–28`：87D current state = q18、dq18、当前手11×3（物体局部）、当前物体3×4变换、velocity6；tau 为 future hand 相对当前 hand、再旋到当前物体参考系，`[24,33]`。
- `learned_retargeter.py:30–47`：24 步 tau + current state 进入 decoder，输出 `[8,12]` active native action；嵌入 18D native action 并 clamp。
- `chunk_retargeter.py:12–27` 继承该合同。`execute_wam_native_bridge.py:364–375` 用当前真实机器人状态与 Actor 预测的 tau 调用 R，执行 action chunk 的第一步，下一 tick 重规划。
- R 的训练使用实际未来 robot FK hand 和**实际 applied commands**：`retargeter_data.py:31–41`、`chunk_retargeter.py:30–40`。这属于离线监督标签，不是部署时查询未来真实手。

## 当前的物体 goal 已经从哪里进入

- 主 goal：`wam_shadow.py:80–94` 取固定 canonical geometry 的同32点，在当前 pose 和 `planned_object_goal` 下形成 start/end 点。`execute_wam_native_bridge.py:339–343` 当前传入的是允许的参考示范 `reference['obj'][t+24]`。
- 三时刻 robot goal：同 rollout `:344–352` 从**预先已知参考物体轨迹**取 t+8/+16/+24；不是从当前自主 rollout 偷读未来物体测量。
- `src/task/trajectory-policy/src/trajectory_policy/robot_context.py:7–20` 将这三姿态相对当前物体表示为 translation3+rotation6D，共27D；proprio 为 q18+dq18+previous applied active12，共48D。
- `src/task/goal-point-wam/src/goal_point_wam/control_context.py:8–37` 用两个零初始化门控 MLP 将48D/27D转成128D residual，加到所有现有手/时间 query；这和物体点 goal memory 是两条条件路径。
- `src/task/trajectory-policy/src/trajectory_policy/wam_control_data.py:46–56` 训练时构造同样的参考目标/proprio；human 样本 `robot_mask=false`，robot adapter 不对其施加条件。

## 为什么不能换 encoder 后直接复用旧结果

- `effect_consistency.py:35–41` 的 contexts 在 `no_grad` 中编码 scene、goal 和 observed hand。`control_context.py:47–56` stage1 只训练 control adapter，stage2 再允许 trajectory decoder；goal/scene/action/World 保持冻结。
- `train_control_context.py:157–180` 缓存 contexts，并冻结旧 R；`:289–314` 用 trajectory loss、固定 R 的 control-aware loss、World effect loss共同更新。`train_visited_realign.py:43–45,67–83` 同样保持已经选择的 R 与非 actor/adapter 模块冻结。
- 新 paired encoder 产生新 token 分布，需要在 human 预训练阶段真正训练它，再在机器人阶段重新编码缓存、显式加载新 goal 权重并做 matched robot adaptation。旧 adapter checkpoint 是在旧 token 分布上学到的，不能只改 class 后严格加载旧权重就视为完成。
- `model.py:231–242` 的 query 注入假设 `2*N+1` start/end token 排列，并分别池化两块；`:244–257` 的 memory 注入才对 token 数量比较通用。Dex4D 的 `[N,6]` pair encoder 不能直接输出 N tokens 塞进旧 query pooling：即使N为偶数使长度校验通过，两半也不再表示start/end。最小方案选择现有memory路径；若以后需要query，应显式改pooling并记录新checkpoint合同。扩展时序goal还需要time/offset与mask，不能只加更多无时序点。

## 是否能完全避开 future-hand oracle

**执行 Actor→R 的自主链可以。当前整套对照 rollout 仍计算 oracle 分支，不能声称整个脚本从未访问 future hand。**

- `wam_native_server.py:202–212` 严格允许的请求字段只有 tick、已测 object/hand H4、planned object goal；有 adapter 时再加 proprio/robot_goal。没有 future hand 字段。
- 同文件 `:222–229` Actor 只读取 scene、goal、observed hand 条件；返回预测 tau。
- `execute_wam_native_bridge.py:360–367` 的 future oracle 用在**GT 对照 row**。WAM row 的 R state 由当前值与重复当前手的 dummy 构造，dummy tau 随即丢弃；R 实际收到的是 server 的 `wamtau`。
- 若需要完全不读取 future-hand 文件的部署入口，应关闭/移出 GT row，仅保留当前状态→预测 tau 的分支。这是工程入口调整；不需要将未来手作为用户 goal。
- 离线训练仍可用录制未来手作为 trajectory target、用执行过的 hand action训练 World；应区分监督标签与推理输入。`effect_consistency.py:47–60` 的 World 只接 scene 和 hand-motion execution condition，没有物体愿望 goal；该9D hand-motion proxy也不等于原生关节控制动力学（`wam_consequence.py:1–4`）。

## 负结果不能只归因于 goal

目前没有“只干预配对 goal 编码”的 matched robot 实验，因此不能说旧 actor 负结果由 goal 编码导致。已知还有 independent R 的状态覆盖问题：`P-20261013-goal-point-wam-visited-realign.md:161–172` 中，GT tau 输入的 held-out exact 状态仍有约280mm误差，意味着有正确目标手轨迹也不保证当前 R 正确映射；其余状态与当前 actor 失配也须分开看。稀疏 wrist 参考 Probe 的失败只关闭该低LR、小样本 residual recipe（`P-20261013-goal-point-wam-reference-wrist-goal.md:164–175`），没有证明 object goal 或 reference conditioning 无效。

## 最小可执行适配顺序（尚未执行）

1. **human 决策实验**：保留现有手位移输出与 World 合同，仅比较旧 endpoint encoder 和显式配对 encoder；数据、decoder容量和训练预算匹配。目标物体分割/时间标签由 root 的 EgoDex audit 准入，不能拿全 scene 点当目标。
2. **robot 同合同适配**：从当前 canonical 点与已知参考 SE(3) 生成同构 pair goal；保持当前 measured H4、48D proprio和Chunk-R，重新训练/校准 actor+robot adapter。先固定同一个合格 R 比较，避免同时换 R 后无法归因。
3. **分开报告**：手轨迹 first8/24误差、预测 tau→R 的 native PD误差、执行 clipping/物体跟踪/接触保持分别验收。人体 held-out 变好只支持预训练阶段；robot rollout 的正向信号才支持执行迁移。
4. 如果 GT tau→R 也失败，修 R 的当前状态覆盖；如果 GT通过而 predicted tau失败，才集中修 actor/goal。joint R fine-tuning 是后续单独变量，不因引入 paired goal 自动启用。

上述方案复用作者的**配对物体目标表示**，继续使用本项目的 Actor、机器人数据和R，不调用 Dex4D 的 XArm/LEAP policy 或 SimToolReal 的 IIWA/SHARPA action 输出。
