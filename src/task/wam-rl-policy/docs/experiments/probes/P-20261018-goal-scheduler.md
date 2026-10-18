---
schema: ref2dex.probe.v2
probe_id: P-20261018-goal-scheduler
experiment_id: P-20261018-goal-scheduler
date: 2026-10-18
task: wam-rl-policy
branch: main
git_commit: recorded-in-run-manifest
claim_id: C1
hypothesis_family: HF-ordered-goal-scheduler
probe_index_in_family: 1
seed_pool: debug
seeds: [0]
decision_changed_if_positive: keep ordered state-based scheduling as a candidate for later frozen-WAM and physical execution checks
decision_changed_if_negative: fix progress, rotation, timeout or recovery semantics before any policy experiment
status: COMPLETED
run_id: goal-scheduler-replay-20261018-r3
---

# 独立有序物体轨迹调度器：几何工程 Probe

Result: 10项单测通过；r3的13片段/4197帧通过独立几何复算，12个运动片段完成有序参考位姿，1个持续未准备片段预期TIMEOUT。仅工程资格，不代表机器人任务成功。
Decision: 保留有序进度与按状态前视作为候选，先审阅可视化和真实任务参考；不修改WAM，不启动策略训练。

Class: Blocker / engineering smoke。用户最新明确先只实现 goal scheduler，不修改 WAM。
当前只回答：静止准备、纯旋转、回环、落后和偏离时，调度输出是否遵守有序与有界合同？
这会决定是否保留当前调度逻辑；最便宜方法是合成合同测试和已有真实记录的受控重放。
不训练网络、不运行策略或新物理仿真，不将几何回放升级为任务成功证据。

## 协议

- 输入完整已知 SE(3) 参考、同身份物体局部点、当前实测位姿/时间；可选外部
  `execution_ready`、`lost_control`，未知 readiness 明确为 unknown，不能冒充接触认证。
- 按物体点路径累计距离建立有序里程碑，允许用户指定不可跳过的索引；仅达成下一
  里程碑才确认进度，局部匹配不能跨过该里程碑。准备阶段进度冻结，仍保留小幅前视意图。
- 正常目标在参考上选择，偏离过大时标记恢复连接段；平移/旋转/对应点位移各自有界。
  末端需持续达标，停滞和总时限明确返回 TIMEOUT，不能跳过难点或永久等待。
- `horizon_s=.8` 只声明下游希望实现短目标的物理时域，不认证可达性；参考速度允许变化。
  此模块不支持必须保持原时间戳、必须等待/释放或冲击速度等额外任务合同。
- 合成：静止准备后抬升、往返重复姿态、纯旋转、外部注入侧向偏离、持续未准备的超时。
- 真实记录：既有 `shadow-value-scale1-32obj-20261016-r1/rank00/last-episode.npz`。
  按终点位移确定性选择4个不同TRAIN物体（非代表性/随机面板，不按success筛选），
  每个比较原速度和0.6秒起始延迟+半速重放。物体和手使用同一记录帧，参考在开始时已知。
- 对照为按墙钟取 `t+.8s` 的参考目标；只比较目标距离/超前量和几何边界，
  边界来自我们算法的设计，不得把更短距离本身称为执行收益。

## 环境、预算、停止

Python3 / NumPy / SciPy / Matplotlib，实际版本写入 manifest。纯几何数组计算使用CPU，
不调用 WAM 或批量神经推理，不占GPU。最多4个物体、180秒、100MiB新增产物；
可用空间低于20GiB不启动。固定代码提交后运行，输出独立新目录，不覆盖已有产物。
非有限、不合法SE(3)、跨reset记录、目标越界、进度倒退或超时中止并保留证据。

入口：[replay_goal_scheduler.py](../../../tools/audit/replay_goal_scheduler.py)。
实现：[goal_scheduler.py](../../../src/wam_rl_policy/goal_scheduler.py)。
依据与设计范围见[调研](../../research/2026-10-18-goal-scheduler-literature.md)。

## 当前结果

首轮r1完成12片段/3656帧，7.62秒，所有有界/单调进度检查通过。逐状态复查发现
纯旋转因短目标角度上限而频繁误标RECOVER；已修正为沿原参考缩短前视，只有真正需要
离开参考作连接时才标恢复。r1保留，不以它的模式统计作为最终结果。

r2固定代码05616c34重放12片段通过，6.46秒；纯旋转不再误标恢复。随后将TIMEOUT
返回的live-pose hold明确标为非参考目标，并增加可视化超时片段。r3为最终重放，
10项针对性单测已通过；尚不代表真实执行或完整参考任务成功。

### 最终r3

- 实际代码 `ee9a915d`，新目录
  `outputs/wam-rl-policy/goal-scheduler-replay-20261018-r3/`。
  13片段/4197帧，CPU几何重放6.93秒；没有神经推理、训练或新仿真。
- Python `/usr/bin/python3` 3.8.10、NumPy1.24.4、SciPy1.10.1、Matplotlib3.7.5；
  `CONDA_PREFIX`继承机器base环境，但实际运行解释器为系统Python，未使用graspenv。
- 5个合成片段包括静止准备、往返、纯旋转、侧向偏离、停滞超时；8个已有记录重放为
  Camera/Clock/Sheep/Pillow各原速与延迟半速。12个运动片段完成有序参考位姿，
  停滞片段8秒后TIMEOUT且返回当前位姿hold，没有宣称任务完成。
- `audit.json`不调用scheduler，独立重算全部4197帧的SE(3)/对应点边界、当前锚点delta、
  目标是否位于声明的参考插值上；检查每次进度增加都有实测里程碑达标证据，
  readiness=false时不增加进度、TIMEOUT不冒充参考目标。全部PASS。
- 四个真实延迟半速片段中，时钟目标对应点距离p95为282–297mm，状态目标为25.5–27.0mm。
  后者更近是显式限幅的预期结果，只说明目标跨度得到控制，不是物理可达性或收益证据。
- `review.html`为兼容本机旧Node语法的最终独立可视化，`replay.html`保留运行时原版。
  13片段×起点/中点/终点的39次DOM/canvas-stub渲染及播放交互通过，12051次绘点坐标有限；
  没有真实浏览器截图测试。已人工查看`overview.png`统计图。
- 核心产物：`manifest.json`、`replay.json`、`audit.json`、`review.html`、`overview.png`。
  连同HTML约40MiB；r1/r2中间运行保留。WAM模型、训练、数据接口均未修改。

## Limitations / future evidence

对象里程碑不能确认抓稳、释放、任务功能或真实可达性。瞬间穿过里程碑但未在采样帧
进入容差时，初始实现可能过于保守；本轮真实回放专门检查这个问题。
时钟与状态目标不同，只能说明几何调度差异；是否帮助 WAM τ 和 RL 需要之后单独实验。
WAM模型、训练、数据接口保持不变。

里程碑切换仍可能产生短目标跳变；接入下游tau之前还应决定刷新/保持间隔，并检查是否
需要沿参考平滑前视。当前没有把未知readiness自动替换为几何抓稳判据。
本轮Python/Markdown检查通过；仓库全局verify仍被trajectory-policy已有6张实验卡
缺少v2元数据阻塞，详细原始输出保存在`tmp/goal-scheduler-research/verify.log`，未修改这些卡。
