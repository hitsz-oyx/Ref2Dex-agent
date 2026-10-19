# Teacher rollout任务与实测执行接口

以用户最新纠正为准：DExplore checkpoint只用于生成teacher rollout。
TaskSpec/goal来自同轮实测物体轨迹与接触，不使用DExplore训练reference_obj/reference_current。
此前将训练参考当任务目标、进而得到0/660的判断已撤销；不是本路线的失败证据。
历史reference_task_spec只保留供显式历史审计，新的task_from_teacher_archive不读取这些字段；
回归测试把训练参考改为NaN/极端值，新的任务packet逐位不变。

## 完整数据与审计入口

最终原始数据：`outputs/wam-rl-policy-v2/inspire-teacher-rollout-20261019-r5/manifest.json`。
最终准入：`outputs/wam-rl-policy-v2/inspire-teacher-rollout-audit-20261019-r2/manifest.json`。
使用GPU0–5、固定官方teacher、原完整660序列与589 TRAIN/71 DEV划分、30Hz。
原16环境并行中部分PhysX批次重复崩溃；reset与buffer修复未独立消除所有故障，aggregate计数
未超限。实测两个8环境半批均通过后，仅补采缺失64条。新run以只读软链接复用r4中有效596条，
没有再次复制或重采；r4仍是r5的必要数据依赖，不能因其整体FAILED状态而删除。

同轮捕获q/dq、DOF力18及有效位、25个body状态/接触力、object位姿/速度/接触力、hand11、
canonical/applied动作、PD目标和tick，以及512点CAD/法线。不同执行的力和状态不拼接。
reset首帧力为invalid/零，step后有效；大于5的原生力值保留。

审计复用原audit_current_rollout的独立PD/接触/hand/时钟检查，关闭训练reference诊断条件，
新增force18与validity检查。660条工程全PASS。质量mask依据同轮实测耦合、unsupported loss、
几何穿透、过大位移/旋转步及可对齐的旧人工标注；不使用训练reference误差。
接触/耦合/支撑仍是启发式，不作为抓取成功证书。跳变阈值沿用原crossing保护，表示保守排除
不可确认的片段，不宣称这些运动必然违反物理。桌面位姿来自原生静态环境初始化，不是目标轨迹。
旧人工unusable/uncertain只在q/obj/hand同帧对应时迁移，纯reference tracking_error不用于新准入。
失败前50帧缓冲，H4/future24支撑窗口不得跨异常；训练用连续安全前缀，不拼接掉落后的片段。
TaskSpec同样截到该前缀终点，保留完整source_length/原帧身份，禁止把失败尾段作为前缀的未来goal。

筛选结果：563条前缀、60529动作；TRAIN501/54544覆盖50物体，DEV62/5985覆盖42物体。
TRAIN的362条前缀含23043帧耦合，覆盖46物体；其余4物体保留前缀仅含接近阶段。
原始660条保留供追溯，训练准入剔除不等于删除原始数据，也不等于563条成功抓取。

## 状态、网络与任务语义

| 输入 | Shadow | Inspire |
| --- | ---: | ---: |
| robot | 原生300 | 原312+DOF18+contact15+shape64+valid2=411 |
| task10 | 局部目标平移3、rotvec3、confirmed progress1、当前phase onehot3 | 相同语义 |
| current / scheduled | 310 / 575 | 421 / 686 |
| 世界系τ | 60Hz八个canonical hand11位移 | 30Hz八个canonical hand11位移 |

TeacherRolloutCorpus只读准入前缀并按时间重建TaskController，返回measured347、task10、CAD512、
原生独立action12及active mask。measured347不含任何虚构的shape64；MeasuredInspirePolicy
在GPU用可训练PointNet编码CAD得到shape64，再接共享NormalizedCommandMLP。
隐藏层保持1024/1024/512/512、ELU；首312维保留原裁剪，新力不裁到[-5,5]。
形状编码器在BC中联合训练，之后可冻结供PPO；目前只有接口/梯度检查，尚无训练后shape权重。

TaskSpec为teacher实际有序SE(3)轨迹、原帧/fps、实测接触导出的离线计划事件及HOLD/RELEASE。
已知计划可用于下一次执行，在线反馈始终使用下一次执行的当前/过去实测值。
连续近重复位姿简化，保留初始anchor、事件帧、非连续回到起点和实际终点。
ACQUIRE/TRACK/RELEASE只由实时反馈转换。ACQUIRE确认cursor冻结，临时记录接触候选运动，
确认耦合后提交；候选丢失撤销未确认后缀。静态接触候选可发最多3mm试探目标。
计划事件时间到达不触发抓取；几何路径完成不代表最终稳定抓持/释放成功。
Shadow position_only保持无硬朝向要求。终止潜势为0，跨phase采用增广状态的γΦ'−Φ，事件只支付一次。

τ统一palm/五bases/五tips，旧Inspire交错顺序迁移时同时置换首层权重和归一化。
WAM事件接口使用next-contact-release-plan.frame30.v1，与独立WAM-v6事件分支对齐；
旧未训练事件的模型明确event_consumed=False，要求事件时拒绝调用，不影响外部WAM训练。

## 验证与后续接线

114项Task测试通过；真实GPU0八个episode的current421/scheduled686前向通过，新增力进入
首层前的保真误差0。该检查使用未训练模型、invalid零τ格式fixture，0优化，不是闭环成功率。
旧BC权重的按列迁移检查另保留，不能把它理解为新观测已学会使用。

新数据/旧数据在目标定义上不同，旧reference条件的物理/τ缓存禁止静默复用。
旧319 BC入口也拒绝extended-force corpus，避免忽略新增力、形状、phase及质量准入。
后续需将TeacherRolloutCorpus/MeasuredInspirePolicy接到两阶段完整BC训练与官方PPO入口；
本轮未启动新BC或PPO，也未修改旧冻结checkpoint。
