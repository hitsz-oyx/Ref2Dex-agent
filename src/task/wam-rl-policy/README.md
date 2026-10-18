# WAM-assisted robot RL

2026-10-18 [手物历史条件的调度与3D审阅](docs/experiments/probes/P-20261018-interaction-goal-scheduler.md)：
修复慢速误超时、低频错过里程碑及goal整段跳变，新增因果hand11/物体随动代理。
弱proxy只影响短目标，不作为几何进度硬门；显式外部准备/失败信号仍可暂停确认。
17项单测、19片段独立复算及Firefox真实交互检查通过；Clock10Hz边界仍保留失败。
新展示构建入口`tools/audit/build_goal_scheduler_viewer.py`，没有修改WAM。

2026-10-18 [独立goal调度器](docs/experiments/probes/P-20261018-goal-scheduler.md)：
`OrderedGoalScheduler(reference_poses, reference_times_s, object_points)`逐环境保存有序进度，
`update(object_pose, time_s, execution_ready=None, lost_control=False)`返回短目标SE(3)、
当前物体坐标系的配对点delta、模式与选择原因。只做goal选择，不调用WAM或输出机器人动作。
回放入口为`tools/audit/replay_goal_scheduler.py`，包含离线交互HTML；真实执行收益尚未验证。

2026-10-18 [完整物体轨迹到WAM短目标的调度器调研](docs/research/2026-10-18-goal-scheduler-literature.md)：
区分已知未来参考与真实未来oracle，核查既有短目标/进度接口，提出按状态推进的有界前视方案。
该调研记录设计依据；独立模块的实现与回放见上方Probe，尚未进入WAM或策略执行对照。

2026-10-16 [共同抓取起点上的协调奖励Probe](docs/experiments/probes/P-20261016-wam-coordination-bonus.md)
已完成：固定DEV同初态BC/none/geometry/WAM保持28/26/23/25，当前抬起后辅助奖励接法
UNPROMISING，停止增加其权重或训练预算。实现提供`--coordination-kind none/geometry/wam`、
真实next-state相对构型评分、逐步跌落禁奖及独立审查；下一步先定位关键转折阶段的目标条件
与预测信息，不扩大BC或把一般预测提升等同于策略收益。

2026-10-16 [独立PPO与运动参考Probe](docs/experiments/probes/P-20261016-independent-motion-ppo.md)：
`--bc-kind none --action-kind independent-gaussian`可不传BC checkpoint，跳过官方actor加载与BC正则；
`--observation-kind motion-none/motion-geometry/motion-wam`支持固定时间窗的因果运动参考。
三臂原生资格已通过，任务到达/保持尚未改善，下一步研究接触学习信号。

2026-10-16 [学习短期终点Probe](docs/experiments/probes/P-20261016-wam-learned-short-goal.md)已完成：
终点回归未过预设门槛，WAM条件运动预测保留局部改善。下一步先建立独立动作与运动参考接口，
再检验原生执行收益；不将BC或复现旧策略未来作为最终研究目标。

2026-10-16新增[任务读出与目标语义Probe](docs/experiments/probes/P-20261016-wam-task-readout.md)：
`task_readout.py`支持状态/目标查询、真实短期物理辅助监督与冻结预测输入，不依赖BC动作
监督。新读出未超过原mean，进一步定位到预训练短期终点条件与整任务goal的差异；
下一步优先短期目标桥接，BC不再作为研究主线前置关卡。具体结果见[当前状态](docs/STATE.md)。

实现冻结WAM/BC reference上的小型在线控制器。路线输入见
[用户参考](docs/user/ref/ref1.md)；当前先做Shadow Stage-0 residual PPO，
之后才根据结果决定RL-token readout与独立动作支路。

`src/wam_rl_policy/ppo.py`定义默认327D观测及可选539D冻结表征观测、24D canonical残差与PPO；
`shadow_loop.py`连接既有原生Shadow BC，`tools/run/shadow_ppo.py`复用固定collector
的环境初始化和物理循环。代码不修改第三方物理实现。

当前已按用户授权启动128个TRAIN物体的四卡同步PPO，200轮目标、50/100/200轮固定DEV
读出；另完整训练fresh单拍BC MLP并完成388DEV/1552起点native基线，held396/1552。
同起点WAM9k held511/1552；远端MLP PPO200轮和探索噪声对照尚无明显改善，
正在检查等容量raw归一化输入与BC隐藏特征；此前两物体Stage-0仍UNCLEAR。详见[当前状态](docs/STATE.md)。

使用`/home2/wyy/miniconda3/envs/graspenv/bin/python`，入口先加载IsaacGym再导入Torch。
`--mode bc/zero/train/evaluate`分别进行BC、零残差、PPO和确定性PPO评估。
TRAIN-only在线探索，DEV只用于固定终点评估；固定原官方reward。
研究协议与输出路径见[首轮Probe](docs/experiments/probes/P-20261016-shadow-bc-ppo.md)。

规模入口为`tools/run/launch_scale_ppo.py`；`prepare_scale_panels.py`固定对象和资产，
`evaluate_scale.py`执行完整DEV/阶段面板，`run_scale_queue.py`监督有界读出队列。
`train_current_bc.py`整理同拍标签并训练MLP；`--bc-kind mlp`连接相同native PPO。
参见[规模Probe](docs/experiments/probes/P-20261016-shadow-scale-continuous.md)。
