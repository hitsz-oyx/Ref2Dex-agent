# REGRIND 源码审计：哪些部分能迁移到 trajectory-policy

日期：2026-10-11。审计对象是只读副本
`third_party/regrind` 的 commit `38347a9e30184620df04e19c63c7c72378cae103`
（初始 release）。本记录核对源码实现，不把 REGRIND 的论文结果或其示例 checkpoint
当作本项目证据；没有运行 Isaac Sim、Drake 或 REGRIND 训练。

## 结论

我们现在可以准确地称为 **REGRIND-inspired closed-loop executor**，但不能称为
REGRIND 的完整实现。REGRIND 的实际训练问题是：给定一条已经离线 retarget 的
hand-object demonstration，每个 control step 读取当前状态，并让策略输出围绕当前
reference target 的 residual。它没有一个从测量历史 `H` 生成候选 `τ` 的高层 actor，
也没有本项目显式的“预测 24 步、执行 8 步、再重规划”合同。

当前最值得保留的共同接口是：

```text
nominal hand goal + bounded residual -> measured-state feedback -> native command
```

本仓库的 `GoalConditionedExecutor` 已经把这个接口实现为
`τ -> q_des/velocity_des -> wrist feedforward + bounded residual -> native command`，
并且每次调用都重新读取 measured `q`（`src/task/trajectory-policy/src/trajectory_policy/goal_executor.py:L1-L11,L75-L153`）。
但是 REGRIND 的 root wrench PD、目标尺度、residual 维度、reset 语义和 reference reward
都不能直接复制到 DExplore。当前 route 仍应把 GT `τ` 作为执行上界，把生成 `τ` 作为
部署候选，在同一个 executor 中比较。

## REGRIND 源码实际实现

### 1. 离线 retargeting 是 hand-object interaction-mesh 优化，不是在线 policy

README 将项目描述为 retargeting-guided RL，并把 retargeting、replay 和 RL training
分成三个入口（`third_party/regrind/README.md:L7-L9,L28-L42,L62-L79,L81-L121`）。
`scripts/retarget_hand_object.py` 读取 MANO hand 与 object pose/joint，先把 30 Hz
demo 插值到 120 Hz（`.../scripts/retarget_hand_object.py:L46-L135,L195-L203`），然后
调用 `HandInteractionMeshOneStageRetargeter`，最后保存
`robot_pos/robot_quat/robot_joints/object_pos/object_quat/object_joint/robot_keypoints/
mano_joint_coords`（`.../scripts/retarget_hand_object.py:L212-L280`）。

retargeter 对每个时间帧建立由 MANO keypoints 和物体 keypoints 组成的 Delaunay
interaction mesh，在 Drake 中迭代求解。其明确写出的目标/约束是 object frame 中的
Laplacian deformation、非穿透、joint limits、trust region 和相邻帧平滑/nominal
tracking（`.../retargeting/retargeter.py:L85-L115,L543-L680`）；整个序列逐帧求解并把
上一帧解作为下一帧初值（`.../retargeting/retargeter.py:L339-L541`）。这比当前
D288 的 FK/坐标恢复更强地利用了物体几何，但它依赖完整的离线 MANO/object 轨迹，
不能作为部署时的 `H` 输入。

源码还有一个需要保留的实现审查结论：`activate_obj_non_penetration` 在
`HandInteractionMeshOneStageRetargeter.__init__` 中只被保存
（`.../retargeting/retargeter.py:L61-L84,L125-L133`），`solve_single_iteration` 没有
读取它，非穿透约束仍在 `L635-L655` 无条件加入（只跳过 object-ground pair）。脚本
虽然在 `.../scripts/retarget_hand_object.py:L220-L255` 先传 `False`、随后赋值 `True`，
这个开关当前并不构成可靠的行为分支。借鉴时要以实际约束路径为准，不能照抄配置名。

### 2. Action 是 reference target 上的 bounded residual

Leap/Wuji 的手指 action term 先把原始 action 限制到 `[-1,1]`，再加到当前
`motion_target`（也可选 `current_obs` 或 `zero`）的 hand joint target，最后按 joint
limits clamp 并写入 position target（`.../mdp/actions.py:L19-L81`）。浮动 wrist 的
`SE3ImpedanceActionTerm` 接收 6 维 position/axis-angle residual，把它加到当前
motion target pose，再以当前 measured pose/velocity 计算 world-frame wrench
`Kp(target-current)-Kd*velocity`（`.../mdp/actions.py:L96-L239`）。因此其控制模式是

```text
reference pose + policy delta -> low-level position/impedance control
```

这是 REGRIND 最直接可迁移的思想。实现细节不能直接搬：Leap 的 root action 是 6 维、
手指是 16 维，Wuji 是 20 维；root/finger scales 和 gains 在
`.../config/leaphand/leap_base_env_cfg.py:L58-L75` 与
`.../config/wujihand/wuji_base_env_cfg.py:L89-L103` 固定为其 floating-hand 资产调过的
值。两种 free-hand 资产还显式 `disable_gravity=True`
（`.../robots/free_leap_right_hand.py:L19-L48`，Wuji 对应
`.../robots/free_wuji_right_hand.py:L53-L83`）。

REGRIND 的 residual 也不是当前 executor 的 exact implementation：它通过 action
manager raw clip 限制 Gaussian output；当前 executor 对 12 个 active native coordinates
使用 `tanh` 和显式 `residual_limits`，再走 coupling 和 native clamp
（`goal_executor.py:L75-L141`）。当前设计因此更适合保留为 DExplore 的有界接口，而
不是把 REGRIND 的 action scale/gain 当成可迁移超参数。

### 3. Reference 每个环境 step 前进，没有显式 24/8 receding horizon

`MotionCommand` 把整段 retargeted trajectory 载入 GPU，保存 object/hand qpos、qvel、
keypoints 以及目标 object/hand 的 pose、velocity 属性
（`.../mdp/commands.py:L49-L315,L584-L655`）。`_update_command` 每个环境 step 将
`time_steps` 增加一并夹在当前 demo 末端；`phi` 是当前 demo 内的归一化相位，
`command` 只返回这个 phase（`.../mdp/commands.py:L459-L474,L981-L998`）。所以
REGRIND 是逐 reference-frame 跟踪，不能据此说它已经实现了本项目的 24/8 chunk。

它的 reset 语义尤其不同：`_resample_command` 按采样的 demo timestep 直接把 object 和
robot root/joint state 写回 reference pose，还可加小扰动
（`.../mdp/commands.py:L901-L979`）。训练配置允许 RSI 从失败 episode 的时间 bin
重新采样（`.../mdp/commands.py:L773-L833,L1221-L1226`），而各 `*-Play-*` 配置关闭
RSI 并关闭若干 early terminations（例如
`.../config/leaphand/leap_scissors_env_cfg.py:L61-L86`）。这使其训练/评估从 reference
state 开始，不能替换当前从真实 frame-0 state 开始的部署合同。

### 4. Policy observation 有当前 reference target 的间接入口

REGRIND policy group 包含 noisy/delayed current object pose、current wrist pose、
current finger joint、last action、phase，以及 action term 当前使用的 wrist/finger
base target；critic 另外加入 fingertip positions 和 hand joint velocities
（`.../dexterous_env_cfg.py:L141-L215`）。具体 action base observation 直接从
`SE3ImpedanceActionTerm.get_base_pose()` 和 finger action term 取 motion target
（`.../mdp/observations.py:L208-L247`），而非只给历史测量。观测噪声、0--2 step
continuous delay 和一致性分组在 `.../dexterous_env_cfg.py:L149-L185,L342-L358` 及
`.../envs/events.py:L201-L270` 实现。

因此 REGRIND 的 policy 不是“纯当前历史、不含 phase/reference”的 policy：它虽然没有
把完整 future trajectory 拼到 actor，但 actor 可以看到当前 desired hand target 和
phase。当前 trajectory-policy 的 H 合同明确不读 future q/object、phase/clock、
tactile；不能因为 REGRIND 这样做就把 `target_*` 或 `phi` 加入当前 actor/executor。

### 5. Reward/termination 是 reference tracking objective

默认 reward 直接比较当前 object keypoints、object linear/angular velocity、wrist pose，
再加 action-rate、action-bound 和 early-termination penalty
（`.../dexterous_env_cfg.py:L430-L477`；实现见 `.../mdp/rewards.py:L25-L113`）。默认
termination 包括 demo end、object deviation、object bounds、hand-object distance、
table contact 和 timeout（`.../dexterous_env_cfg.py:L480-L505`；实现见
`.../mdp/terminations.py:L32-L86`）。这套 reward 的主要目标是跟随 demonstration
object/hand reference，而不是独立的“抓住、抬升并保持后检查掉落”。当前 Mission 的
真实 lift/hold reward 不能被 REGRIND 的 tracking reward 替代。

### 6. PPO、随机化和资源规模是其专用训练配置

RSL-RL 配置使用 24 steps/env、20,000 iterations、1024/512/256/128 MLP、5 epochs、
4 minibatches、`gamma=.998`、`lam=.95`、adaptive KL 和 `learning_rate=.001`
（`.../config/leaphand/agents/rsl_rl_ppo_cfg.py:L14-L41`；Wuji 对应配置相同）。训练
入口默认由 4096 个并行环境运行，保存 env/agent YAML 并调用 `runner.learn`，且以
`init_at_random_ep_len=True` 启动（`.../dexterous_env_cfg.py:L520-L579`；
`.../scripts/rsl_rl/train.py:L113-L216`）。场景还配置了 friction/mass/gain/default
joint/observation-lag randomization、gravity curriculum 和后期 random push
（`.../dexterous_env_cfg.py:L218-L426`）。这些是可参考的鲁棒性训练工具，不是当前
小规模 H→τ Probe 的可比预算或成功证据。

## 与当前 trajectory-policy 的关键差异

| 维度 | REGRIND 源码 | 当前 route 的含义 |
| --- | --- | --- |
| 高层输出 | 没有 H→τ actor；offline retargeted demo 每 step 给出 target | `H → independent actor → c → D → 24-step τ`；高层是否能提出可执行 τ 仍是主要 blocker（`src/task/trajectory-policy/README.md:L3-L5,L28-L38`） |
| 执行节奏 | demo index 每 env step +1；无 24/8 chunk | 每 24 步计划执行 8 步，之后重读 H/q 并重规划（`.../tools/run/probe_history_actor_execution.py:L216-L264`） |
| 反馈 | root wrench PD 用 measured pose/velocity；finger target 是 reference + delta | `GoalConditionedExecutor` 每次用 measured q 构造 velocity-FF target、bounded residual、coupling 和 native clamp（`goal_executor.py:L108-L153`） |
| reference 输入 | actor 通过 action bases 看到当前 hand target，并看到 phase；reward/termination 使用 future demo target | actor/executor 不读 future q/object、phase/clock/tactile；只有计划 `τ` 和实时状态进入共享 executor |
| reward | reference object/hand tracking | 真实 object lift/held/drop 语义；不能直接使用 `object_keypoints_pos_error_exp` 作为 Mission 结论 |
| reset | 直接写回 demo timestep 的 object/robot 状态，可 RSI | 真实 frame-0 初始化；不能用 reference reset 掩盖 startup 误差 |
| retargeting | Drake interaction mesh + collision/joint constraints | 当前 D288 主要是 URDF/FK 与轨迹坐标恢复；learned/local retargeter 仍需 measured-state 验证 |
| 资产/单位 | Leap/Wuji free-hand，root wrench，16/20 finger DOF，gravity disabled | DExplore 的 18-D native command、PD offset/scale、coupling 和动力学；只能迁移抽象，不迁移数值 |

这解释了为什么当前接口审计可以是正向的，却不能把 REGRIND 的 reference tracking
结果当成 H→τ 或 Cm utility 证据：两者的 reference 来源、reset、reward 和输入权限
不同。

## 可直接迁移的组件

1. **把 nominal reference 与 residual 分层。** 保留一个纯函数接口
   `goal -> bounded residual -> native command`，每控制步传入 live state；当前
   `GoalConditionedExecutor` 已满足这点。GT、generated τ、将来的 residual policy
   都必须调用同一个 adapter，避免各 runner 重复实现 conversion。

2. **保留 requested/applied/clipping 三层日志。** REGRIND 的 action term 明确区分
   raw action、processed action、position target 和最终物理写入；当前 executor 也返回
   `target/intended/command/clipped`（`goal_executor.py:L128-L141`）。后续物理 Probe
   应至少记录首 8 步 wrist/finger target error、native clipping 和 object held/drop，
   不能只报告离线 τ 或 policy loss。

3. **明确 goal velocity 的语义，但不要复制其 gains。** REGRIND 能从 demo 求出 target
   linear/angular velocity（`.../mdp/commands.py:L564-L606,L633-L639`），但其
   `SE3ImpedanceActionTerm` 实际 wrench 只使用当前 velocity 的 damping
   （`.../mdp/actions.py:L213-L239`）。当前路线额外使用
   `wrist_feedforward(reference_velocity)`；这是一项本项目自己的 damping compensation，
   应作为独立对照保留，不能称为 REGRIND 原样实现。

4. **训练期可借鉴 failure-aware sampling。** `MotionCommand._adaptive_sampling`
   按失败时间 bin 与 uniform mixture 重新采样，且记录 entropy/top-1 bin
   （`.../mdp/commands.py:L773-L833`）。若当前 PPO 继续推进，可只在训练 sampler 中
   使用真实 rollout 的失败 bin；不把 bin/phase 暴露给部署 actor，不把 reference
   target 写回 state，不改变 H 输入合同。

5. **离线 interaction-mesh retargeting 可作为标签/coverage Probe。** 若当前
   DExplore hand/object URDF 与 mesh 可在 Drake 中建立，可用 REGRIND 的输出 schema
   生成更高质量的 11-point hand τ，再比较 startup/contact/velocity/FF 误差。它只能
   作为离线数据预处理；生成后的 τ 仍需经过当前 executor 的实时 q 反馈，并不能把
   future object 或 retargeter 的 q 送入 H actor。

## 不能直接迁移的假设

- **reference 可用性。** REGRIND 每步拥有 demonstration target，当前 H→τ 部署没有
  future q/object/reference；加入 `target_hand_*` 或 `phi` 会改变当前任务合同。
- **reset 是 reference state。** `_resample_command` 的 state teleport 与 RSI 适合
  imitation/reference tracking，不能证明从真实 startup 能抓住。
- **reward 是跟踪 reward。** Object keypoint/velocity tracking 不是当前 autonomous
  lift/hold/drop 目标；用它会把“跟上 demonstration”误当作抓持成功。
- **控制器数值。** `Kp/Kd`、action scales、gravity disabled、16/20 finger DOF、世界
  wrench 施加位置都绑定 REGRIND 资产，不能套到 DExplore 18-D native action。
- **相位作为输入。** REGRIND `command=phi` 是显式 phase；当前 route 要检验纯测量历史
  是否足够，不能偷偷恢复 phase/clock shortcut。
- **离线 full-trajectory 依赖。** Interaction mesh 的 object-frame human/object
  points 与逐帧 SQP 需要完整 demo 和 object mesh；不能作为在线 learned retargeter
  的可用信息。
- **训练规模与评估语义。** 4096 env、20k iteration、RSI、gravity curriculum 和
  `*-Play-*` 放宽终止条件，不能与当前 16/64 env 单 motion Probe 直接比较。

## 最小下一步实验

### Probe A：同一 executor 的 GT/generated τ 物理对照

固定当前 DExplore native offset/scale、`GoalConditionedExecutor`、24/8 chunk 和真实
frame-0 reset；只把 `reference_q/reference_velocity` 分成 GT τ、dense τ、当前冻结
generated τ 三组。报告首 8 步 wrist/finger tracking p95、clipping、hold45、最长 held
和 terminal-held。判别规则是：GT 失败则先修 executor/控制语义；GT 成功而 generated
失败则把主要 blocker 留在 H→τ/coverage，而不是继续改 REGRIND residual 数值。

### Probe B：小幅 τ 扰动的执行 basin

以已通过的 GT τ 为中心，只做预先固定的 wrist translation/rotation 与 finger-joint
小扰动，仍使用同一个 executor 和真实 measured q；不训练模型、不加入 future object、
phase 或触觉。比较扰动幅度与 startup tracking、clipping、held/drop 的关系，判断
“bounded residual + velocity FF”是否扩大了可执行 basin。若 GT 很窄，下一步应修低层
目标/预载；若 basin 足够宽而纯 H 仍失败，预算应转回高层 τ proposal。

### Probe C：只有在 A/B 通过后再做 failure-aware H→τ PPO

按 REGRIND 的 failure-bin sampler 做**训练期**起点重采样，保持当前 H、D、executor、
真实 lift/hold reward 和无 phase/future-reference 输入不变；与相同交互预算的 direct
native 或现有 trajectory PPO 做 matched 对照。只做一个短 Probe，记录 sampler 的
失败 bin/entropy、chunk return、startup tracking、held/drop 和 clipping。若 A/B 没有
低层正向信号，不启动这一步，也不把 REGRIND 的 reference tracking reward 接进来。

## 对当前决策的影响

源码核对支持继续使用“共享 goal-conditioned closed-loop executor”，但不支持把当前
路线改写成“完整 REGRIND”。优先顺序应是：先用相同 executor 完成 GT/generated τ
物理分解，再决定高层 H→τ 是否值得补训练；interaction-mesh retargeting 和
failure-aware sampling 都是有边界的候选工具，不能替代真实自训练策略和 matched
Cm-on/off 验证。
