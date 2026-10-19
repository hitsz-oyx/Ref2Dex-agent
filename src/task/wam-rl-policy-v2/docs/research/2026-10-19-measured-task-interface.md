# 统一任务语义与实测执行接口

依据用户提供共享会话末尾讨论实施。当前交付是可在归档/原生状态上运行的公共接口及
小规模工程验证；不将这些新模块描述成已切换的 BC/PPO 训练入口。历史冻结运行保持原合同。

## 任务与阶段

`TaskSpec` 保存有序物体 SE(3) 路径、原始 source frame/fps、计划 CONTACT/RELEASE
事件、终态 HOLD/RELEASE/UNKNOWN 和目标约束。只简化连续近似重复姿态，保留事件帧和
非连续回到原点的路径。DExplore 二值接触注释按任一已选接触部位生成计划区间，合并
不超过2帧的空隙、过滤短于3帧的区间；多个区间全部保留，不把最终断触自动解释为放置成功。
裁剪只去首接触前和末释放后的参考路径，源帧身份/原始序列长度不变。

`TaskController` 是归档重放与原生状态共用入口，内部由实测力和手物相对运动生成
`InteractionFeedback`，再交给 `TaskScheduler`。ACQUIRE/TRACK/RELEASE 是当前执行阶段；
计划事件仅提供 WAM 的意图上下文，时间到达不触发当前阶段。抓取候选持续0.12秒可请求
相对固定探测起点最多3mm的试探目标；物体与手实际运动耦合才确认 TRACK。此观测仍为
启发式，不能当力闭合证书。恢复抓取保留已确认路径进度，禁止跨恢复间隙做穿越插值。
事件按新的实测接触/释放边沿确认，不因时间或持续同一接触跳过后续多个事件。

终态 HOLD 要求实际终点、稳定及抓持证据持续1秒；RELEASE 还要求实际释放与支撑
证据，缺少支撑观测时保持 UNKNOWN，不伪报放置成功。完整路径完成与终态保持分开。
Shadow 使用 position_only，不继承初始姿态作为硬朝向要求。

## 执行器观测

统一的是字段语义与构造函数；不同机器人的 native state/action 维度及控制频率独立。
actor 的 task10 是当前物体系局部目标平移3、rotvec3、confirmed progress1、当前阶段
onehot3。计划事件类型和时间不混入 actor task10。

| 机器人 | robot | task | native world tau + valid | current / scheduled |
| --- | ---: | ---: | ---: | ---: |
| Shadow | 原生300 | 10 | 264+1，60Hz八点 | 310 / 575 |
| Inspire | 312+DOF力18+接触力15+shape64+力有效位2=411 | 10 | 264+1，30Hz八点 | 421 / 686 |

Inspire 的312保留原实测布局及任务终点XYZ。新力字段不做原始[-5,5]裁剪，由训练集
归一化处理。reset 时 DOF/接触力清零并无效，下一物理步后才有效；valid 位不等于零力。
ObjectShapeEncoder 使用物体系点云，保留物理尺寸，经 PointNet/max pooling 得到64维；
当前还没有训练/绑定这个编码器，不能用随机特征开训并宣称已完成形状适配。

`ExecutorLayout` 使用命名字段，`expand_named_model` 按列迁移归一化与首层权重，新增列
权重零初始化，保留1024/1024/512/512 ELU网络和原裁剪前缀。两机器人τ统一为
palm、五bases、五tips；旧Inspire交错base/tip的τ必须显式使用列置换，同时迁移
首层权重与归一化参数。共用转换函数按30/60Hz取样，不能直接复用旧native顺序τ。
兼容迁移有测试；新增输入
不是旧PPO的普通resume，后续训练必须保存新的合同与已训练形状编码器身份。

`wam_goal` 始终显式返回空间 waypoint 的 SE(3) goal_delta。只有声明训练过
`next-contact-release-plan.frame30.v1` 合同的模型才接收计划事件，类型数值为
NONE=0/CONTACT=1/RELEASE=2，时间单位为30Hz步，已知NONE与缺失mask分开。
该编码与WAM-v5新增可选事件分支源码对齐；既有v4/v5检查点尚未训练此输入，
require_events=True时拒绝调用，默认兼容路径明确记录event_consumed=False。
没有修改正在训练的 WAM-v5，也没有宣称旧模型已理解新事件。

## 奖励与数据

TaskReward 在增广状态(实测状态,当前阶段)上计算 γΦ(s',phase')−Φ(s,phase)，
终止潜势强制为0，事件奖励只支付一次。静止时该项未必为0；测试验证跨阶段循环与终止
的折扣回报望远镜关系，避免把相位改变制造的局部正奖励误认为额外任务进展。
此奖励模块尚未替换冻结 PPO 入口的旧奖励。

完整原始660条需要在新 TaskSpec 下重放；旧 task-active 筛掉的命令不是永久坏数据。
旧归档有 body/object 接触力但没有 DOF 力，不能拼接别的执行所得力值。
`collect_inspire_expert.py` 复用既有官方 teacher 数据生成器，动作、PD、状态与新增力
必须在同一轮执行共同保存。teacher 只生成数据，不作为最终策略权重。
`inspire_physics_host.py --extended-state` 为可选字段，默认旧入口不变。

工程入口：`tools/audit/replay_task_contract.py`（完整归档接口重放）、
`tools/audit/check_inspire_force_state.py`（双环境真实 step/reset）、
`tools/run/collect_inspire_expert.py`（新增力的原生采集）。新模块通过后仍需更新全量数据、
训练 shape/current BC/条件BC、评估抓取，再决定是否恢复六卡 PPO。
