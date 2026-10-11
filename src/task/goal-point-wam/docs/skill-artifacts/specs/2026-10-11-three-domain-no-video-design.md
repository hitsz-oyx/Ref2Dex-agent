# Goal-Point-WAM 三域无视频训练设计

## 目标与已确认约束

本路线把 `ref1.md` 的 PointWAM-inspired WAM 结构落到现有三域 3D 数据上，先验证
scene/action/world 三个分支在真实 3D 手物轨迹上的接口和学习能力。三域固定为
OakInk2、GRAB、ARCTIC；视频、RGB、tactile、robot policy、evaluator 和 PD loss
不进入本阶段训练。

本阶段的成功标准是：

1. 三域样本在同一数据契约下可以被稳定采样，源级 split、mask、单位和 30 Hz 时间
   轴不漂移；
2. scene encoder 不接收 goal，world decoder 不接收 goal；
3. trajectory 分支能够在显式 start/end hand goal 条件下输出未来双手轨迹；
4. world 分支能够用已执行的未来手轨迹预测三域刚体 effect；
5. 单批 smoke、短 GPU Probe、固定三域验证面板和 checkpoint replay 都通过；结果只
   标记为工程通过或 Probe 的 PROMISING/UNPROMISING/UNCLEAR，不升级为机器人策略结论。

现有 `cm-pointflow-effect-pretrain` 的 `MixedWindows`、三域 manifest、PointWorld
刚体 effect 合同和 PTv3-small 配置是输入依赖。现有 `PointWorldWM` 中“把未来动作
直接拼入 PTv3 scene token”的完整 forward 不作为新 WAM 的整体实现复用，因为它
违反 ref1 要求的 goal-blind scene/world 隔离。

## 数据契约

### 三域样本

通过现有 `MixedWindows` 读取已冻结的 main-three manifest：

- 30 Hz，4 帧历史，24 帧未来，当前 anchor frame；
- 当前 scene/object 点和历史运动特征仍使用现有 18D scene feature；
- 每只手 11 个语义点，保留 right/left 双手布局，总形状为 `2 x 11 x 3`；
- future hand action 作为已执行轨迹 `tau_executed`，形状为 `24 x 2 x 11 x 3`，并
  保留逐点 validity mask；
- object effect 继续使用现有每个局部物体的 24 步刚体 `SE(3)` target、object mask
  和 canonical 512 点；
- OakInk2、GRAB、ARCTIC 的 source-aware sampling、split、normalization hash 和
  fixed balanced validation panel 全部沿用，不复制或重写原始数据。

### Goal 合同

三域没有独立的 robot goal 标签，因此第一阶段使用显式的 endpoint-conditioned
teacher contract：

- `P_s`：当前 anchor frame 的双手 11 点；
- `P_g`：同一窗口未来第 24 步双手 11 点；
- `goal_valid`：分别由当前手 mask 和末端手 mask 组成；不能用零坐标伪造缺失点；
- trajectory 训练 target：`tau_target = tau_executed - P_s`，按逐点 mask 计算；
- world 训练输入：真实 `tau_executed`，而不是 trajectory decoder 的预测结果。

因此本阶段验证的是“给定三维起点和终点 hand goal 的轨迹拟合，以及已执行 hand
trajectory 的物理 effect 预测”。它不声称已经解决从语言、视觉或机器人状态推断
goal 的问题，也不把 future-derived `P_g` 当成部署时可用的观测。

### Robot state 边界

ref1 中的 `H328` robot proprioception 保留为后续 adapter 接口，但三域数据没有真实
机器人 `H328` 合同。本阶段不拼接伪造的 H328，不训练 `L_PD`，trajectory decoder
使用 scene/goal context；后续 DExplore/robot data 接入时再启用 `RobotStateAdapter`。

## 模型结构

### SceneHistoryEncoder

新建 goal-point-wam 的 scene encoder，复用 PointWorld PTv3-small 的 vendor、hidden
dimension 128、固定 1 cm workspace 和 patch-size 配置，但只输入当前 scene 和历史
压缩特征。future hand trajectory、`P_g` 和任何未来 object effect 都不能进入该
encoder 的 sparse coordinates 或 features。

encoder 输出两种 token：

- scene spatial tokens `Z_H`，供 trajectory/world decoder cross-attention；
- object pooled tokens，保留现有 object identity、canonical geometry 和 target mask。

voxel origin 使用固定 workspace contract，不能从 batch、future goal 或 action 动态
计算。有效点在进入 sparse backbone 前必须先 mask 非有限值；同一样本单独推理和换
batch 组成应保持输出一致。

### GoalEncoder

使用轻量 shared point MLP 加 start/goal type、hand id、keypoint id embedding，再用
两层 set/cross attention 编码 `P_s` 与 `P_g`，输出 `Z_G`。它只服务 trajectory
branch，不能作为 world decoder 的输入。

### TrajectoryDecoder

使用 `24 x 2 x 11` 个时间、hand、keypoint query。每个 query 对
`[Z_H, Z_G]` 做 cross-attention，输出相对于 `P_s` 的三维位移，恢复为
`tau_hat: (B, 24, 2, 11, 3)`。invalid goal/target 点由 mask 排除；不通过复制、零填充
或 future scene 信息补齐监督。

### ExecutedActionAdapter 与 WorldDecoder

`tau_executed` 经过 action projection、time/hand/keypoint embedding 后形成 action
tokens；它在 scene encoding 之后进入 world decoder。world decoder 只接
`[Z_H, Z_tau_executed, object tokens]`，不接 `Z_G`，输出现有三域的 per-object
24-step rigid `SE(3)`。第一版保留现有 analytical canonical-point effect 与物理
loss，以便把新结构的变化和数据/标签变化分开。

训练图固定为：

```text
history scene ──> SceneHistoryEncoder ──> Z_H ───────┬──> Goal/Trajectory branch ──> tau_hat
                                                     │
executed tau ──> ExecutedActionAdapter ──────────────┴──> WorldDecoder ──> E_hat
start/end goal ──> GoalEncoder ──────────────────────> trajectory branch only
```

禁止 `Z_G -> WorldDecoder`，禁止用 `tau_hat` 替代第一阶段的 `tau_executed`，从而先
隔离 goal shortcut 和 rollout error。

## 优化与训练入口

第一阶段总损失为：

\[
L = L_{traj} + \lambda_{world} L_{world},
\]

其中 `L_traj` 是逐点 masked Huber，`L_world` 复用三域刚体 effect 的现有 physical
loss 与 source-aware normalization。`lambda_world`、学习率、microbatch、source
draw 和 validation seeds 写入新配置，不修改历史配置或 checkpoint。

训练入口分三层：

1. **contract smoke**：CPU 读取每个 source 的 train/val 样本，检查形状、单位、mask、
   endpoint goal 和有限值；检查 batch 组成不改变固定 voxel/output contract；
2. **GPU engineering smoke**：单 GPU 或用户授权的空闲 GPU，少量 update，检查 finite
   loss/grad、显存、参数 hash、checkpoint save/reload 和 world/trajectory 两个 head
   都收到梯度；
3. **三域 Probe**：使用固定 balanced validation panel，报告 OakInk2/GRAB/ARCTIC
   分源 trajectory EPE、world point EPE、moving/static 分层和 goal/world ablation。
   训练输出使用 `outputs/goal-point-wam/<run_id>/`，不覆盖现有 PointWorld 输出。

GPU 运行必须记录 GPU utilization、allocated/reserved memory、update throughput 和
   预计完成时间；遇到 OOM、非有限值、manifest/hash 漂移、foreign process 或超出
   Campaign 预算时停止并保留部分 checkpoint。

## 测试与证据边界

新增测试覆盖：

- 三域 adapter 的统一 `24 x 2 x 11 x 3` goal/action/mask 合同；
- 缺失手点不会被零坐标当作有效 target；
- goal 只影响 trajectory head，world 输出对 goal 替换保持不变；
- scene encoder 不读取 future effect/goal，future label mutation 不改变 world 输入；
- 固定 voxel contract 对单样本、不同 batch composition 和 action/goal 替换保持稳定；
- 两个 head 的 loss、梯度、checkpoint replay 和 source-aware validation 指标有限。

工程 smoke 只能形成运行/接口证据。单 seed 或短 Probe 只能标记为 PROMISING、
UNPROMISING 或 UNCLEAR；只有后续预先注册的 matched multi-seed Validation 才能
形成正式架构或跨域泛化结论。

## 分阶段实施与停止条件

1. 先实现 data adapter、模型模块和合同测试；未通过前不启动正式训练。
2. GPU engineering smoke 通过后，再运行小规模三域 Probe；若任一 source 无法稳定
   采样或 world/trajectory head 没有梯度，先修实现，不解释为方法负结果。
3. 若 Probe 只显示 fit improvement、dev 或 static baseline 失败，保留 checkpoint 和
   诊断，标记 UNCLEAR，停止无界追加 epoch；下一步由 goal contract、表示能力或
   数据覆盖的证据决定。
4. 视频、RGB、PointWAM human-video pipeline、机器人 policy、PD、evaluator 和
   full-archive 数据扩展均留到本阶段三域合同稳定之后。

## 明确不在本规格内

- 不接视频或 RGB，不下载或处理新的视频训练样本；
- 不改变 `cm-pointflow-effect-pretrain` 的历史 model、manifest、stats 或 checkpoint；
- 不把 endpoint goal teacher contract 当作部署时可获得的 causal goal；
- 不声称当前结果等同于官方 PointWAM、Motus 或机器人控制收益；
- 不创建新 Git 分支，不覆盖用户未跟踪文件。
