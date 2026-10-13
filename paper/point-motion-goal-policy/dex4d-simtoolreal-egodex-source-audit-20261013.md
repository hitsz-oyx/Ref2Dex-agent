# Dex4D / SimToolReal 对现有 EgoDex 的源码适配审计

日期：2026-10-13。范围：只读源码审查、现有 EgoDex 窗口上的小型数据合同 smoke；没有安装依赖、下载权重、运行作者训练或机器人执行。

## 结论边界

**不能说这些完整策略可以直接在我们的 EgoDex 上运行。** Dex4D-Simulation 和 SimToolReal 的训练入口是机器人仿真 RL；EgoDex 的人体手关键点不是它们的机器人关节位置、速度和控制 action。即便补齐软件依赖，数据语义仍不匹配。

Dex4D 的成对点编码可独立复用，Vision 的 UV+深度反投影函数已在我们的真实窗口上执行；它们不等于完整策略适配成功。Dex4D-Vision 的 2D 跟踪模块可接普通视频，但完整 EgoDex 3D 目标构造还需要真实内参、深度尺度、逐帧相机位姿和物体点身份。

| 源码 | 直接读取我们现有 EgoDex 训练 | 可复用的边界 | 仍缺的关键内容 |
|---|---|---|---|
| Dex4D-Simulation | 否 | `[current_xyz, goal_xyz]` 配对点编码、部分点观测训练思路 | XArm6+LEAP 状态/action、仿真环境、物体资产、PPO expert/DAgger rollout，或改写为人体关键点预测器并重新训练 |
| Dex4D-Vision | 可适配视频预处理，不能原样作为完整米制 3D goal 管线 | 目标 mask 内采点、CoTracker 包装、单帧反投影 | 权重与依赖；EgoDex 时变外参、尺度与目标分割适配；质量验证 |
| SimToolReal | 否 | SE(3) 目标推进、抓取 bbox 点构造和误差度量 | IIWA+SHARPA 的 29DOF 状态/action、对象 SE(3)/bbox、机器人与物体资产；不是任意点轨迹策略 |

## 审计版本

本次用户授权克隆到 `third_party/` 的以下检出，仅含源代码/文本，未检出大型 mesh/权重。检出中没有这些大文件是本次审计的选择，**不能据此声称作者没有公开它们**。

- `third_party/dex4d-simulation-egodex-audit-20261013`：`0dfccd82797e929cda6ab665003bc27a6b8cd31a`，上游 `https://github.com/Dex4D/Dex4D-Simulation`。
- `third_party/dex4d-vision-egodex-audit-20261013`：`13c2f02902dc9b7f87d3db68eda69df417d851da`，上游 `https://github.com/Dex4D/Dex4D-Vision`。
- `third_party/simtoolreal-egodex-audit-20261013`：`313d5aea1f507c6cfe097b672b62945d7b0bbff5`，上游 `https://github.com/tylerlum/simtoolreal`。

真实数据使用 `outputs/goal-point-wam/video-object-goal-audit-20261013-r1/inputs/r03/{clip,projection}.npz`。这是我们已有预处理产物，不是原始 EgoDex 的完整 schema；本文只对现有产物和已读上游源码作兼容性判断。

## Dex4D-Simulation：输入不是人体手，训练不是视频 BC

以下路径均相对于 Dex4D-Simulation 检出根目录。

1. `dex4d_policy/dex4d/tasks/xarm6_leap_hand_ap2ap.py:85–106` 设置 student 观测维度 `66 + 6*N` 和 `numActions=22`，注释明确 XArm6 6DOF + LEAP 16DOF。
2. `dex4d_policy/dex4d/algorithms/rl/dagger/module.py:183–200` 将这 66D 拆为 `robot_qpos:22`、`robot_qvel:22`、`action:22`，剩余为配对对象点；不能拿双手 XYZ 展平替代。
3. 同文件 `:131–138` 把对应索引的当前位置和目标位置拼成 `[B,N,6]`。Transformer 版同样使用 `Simple6DPointNetBackbone`。这部分是可移植的表示方法。
4. `tasks/xarm6_leap_hand_ap2ap.py:992–1028` 从**同一对象局部 mesh keypoints**经当前/目标物体变换生成点对，后续再施加遮挡。EgoDex scene points 尚不等于物体 mesh 点，需先确定物体/部件身份和对应关系。
5. `algorithms/rl/dagger/dagger.py:238–275` 加载 PPO expert，分别根据 student obs 和 full state 求 action，再 `env.step` 并缓存 expert action。`:442–451` 做 student action 对 expert action 的 L1 BC。它没有接受人体视频轨迹直接代替 expert action 的现成路径。
6. `tasks/xarm6_leap_hand_ap2ap.py:331–370` 读取 UniDexGrasp `datasetv4.1` 的 `qpos/scale/plane` 抓取先验；`:377–398` 加载 XArm6+LEAP URDF；README 的 Dataset 节要求另备 UniDexGrasp 物体资产。只提供 EgoDex mp4/HDF5 不足以生成这些内容。
7. `tasks/xarm6_leap_hand_ap2ap.py:1237–1270` 将 action 解释为实际 arm/hand 控制目标；输出并非人体关键点预测。

**采用边界：**如果我们借用其 6D 点编码，然后改手观测、预测头、loss 和数据 loader，得到的是本项目的适配模型。不能称作“Dex4D 原策略直接在 EgoDex 训练”，也不能将我们的 smoke 当作作者性能复现。

## Dex4D-Vision：视频入口可接，3D 坐标需要修改

以下路径相对于 Dex4D-Vision 检出根目录。

- `cotracker3.py:141–148` 离线入口读取 RGB video，调用 `cotracker3_offline`，可以提供当前帧 segmentation mask。`:225–232` 支持任意视频路径与已有 mask 路径。因此原始 EgoDex 视频不是 2D tracker 的格式障碍；现有环境缺 `cotracker`/`sam2` 与相应权重，尚未跑其模型推理。
- `run.py:57–64` 需要作者预设 checkpoint，然后估深度。`:69–92` 从 `outputs/realsense/<exp>/intrinsics.npz` 和首帧 RealSense 深度读取标定；没有首帧深度时，**metric depth 仍乘 0.75，relative depth 倒数后乘 550**。这些数值不是 EgoDex 的已验证米制校准。
- `utils/util.py:77–83` 无内参文件时默认 `(fx,fy,cx,cy)=(385,385,320,240)`；我们导出 r03 的 640×360 内参是 `(245.54465,245.54465,320,180)`。直接使用默认 K 会改变空间几何。
- `visualize_track_4d.py:47–85` 将每帧 UV 与同帧 depth 反投影到**各帧自己的相机坐标**。`:87–102` 只接受一组 rotation/translation；`:235–258` 从一个 calibration JSON 读取后对全部帧应用同一外参。这适用于作者固定相机标定，并没有接受 EgoDex `transforms/camera[t]` 的现成时变外参路径。
- 同文件 `:130–132` 还硬编码作者机器人工作空间门槛 `z<0.602`、`x<-0.5`、`|y|>0.5` 来删除点。在 EgoDex 坐标系直接运行会误删，必须换成数据自身的判定。

EgoDex 当前窗口已有相机位姿。共同参考坐标应明确构造为 `X_ref(t)=T_ref<-origin * T_origin<-camera(t) * X_camera(t)`，而不是用首帧外参处理所有帧；手与物体点需要同一坐标系。本项目 `point_target` 已使用当前相机作为共同参考，移植时不能再次把它误认为每帧相机坐标。

## SimToolReal：SE(3)+抓取 bbox，固定机器人接口

以下路径相对于 SimToolReal 检出根目录。

- README 的 Train a Policy 调用 Isaac Sim/Gym RL 训练。不是下载 EgoDex 后启动监督学习的入口。
- `isaacgymenvs/utils/observation_action_utils_sharpa.py:51–89` 明确 29 个关节：IIWA 7 + SHARPA 22。`:187–211` 组成 `N_OBS=140`，含关节位置、速度、前次 action target、FK palm/fingertip、物体旋转、对象/目标相对点及对象 scale。
- 同文件 `compute_observation:266–334` 断言 `q/qd/prev_action_targets` 为 `[N,29]`，object/goal pose 为 `[N,7]`（xyz+xyzw）；调用实际机器人 URDF FK，并核对关节名顺序。EgoDex 的人体手 `[2,11,3]` 不满足接口含义，不能伪造零 qpos 来声称适配。
- 同文件 `:640–670` 从对象 SE(3) 和 grasp bbox scale 构造四个固定偏移 keypoints。这些不是视频中任意 N 个 surface tracks；未知刚体 pose、抓取 bbox 或可形变对象均没有直接输入映射。
- `dextoolbench/objects.py:11–25` 对象有 URDF 与 **policy-normalized grasp-bbox scale**，明确 scale 不等于米制物体尺寸。把当前 scene 点云外包框直接当该 scale 未经过定义转换。
- `observation_action_utils_sharpa.py:445–499` 输出 29D joint targets：手 action 先映射到关节限位、EMA；arm action 以速度比例乘 dt 后累计。不同于我们的机器人 18D state/12D active control 或人体 XYZ target。
- 同文件 `:213–216` 和 `deployment/rl_policy_node.py:38–39` 使用作者固定世界到机器人关系；`deployment/goal_pose_node.py:297–308` 还把 world goal 的 y 减 0.8，并设一个固定对象 scale。EgoDex 原点不自动等于其 world/robot frame。
- `docs/deployment.md:9–11,147–149` 要求 RGB-D、SAM+FoundationPose 和米制对象 `.obj` 来产生 pose。仿真运行还需要对象/机器人资产。EgoDex RGB、人体变换、场景轨迹不能单独替代对象 mesh 与可执行机器人状态。

可借鉴的是 `goal_pose_node.py:44–61,148–180` 的 bbox-keypoint 误差与按完成进度切换 goal。要在我们场景应用，须使用我们确认的对象 pose/点对与坐标合同，独立接入本项目 Actor/控制器；不能直接部署作者 policy 到本项目机器人。

## 实际 smoke 与失败记录

环境：`/home2/wyy/miniconda3/envs/graspenv/bin/python`。仅几十点单次 forward/几何计算，CPU 启动成本低，未占 GPU。结果保存于 `outputs/goal-point-wam/dex4d-simtoolreal-source-audit-20261013/smoke.json`。

| 检查 | 实际结果 | 能证明什么 |
|---|---|---|
| r03 真实现有窗口 | current `[1792,3]`、future `[24,1792,3]`、human hand `[28,2,11,3]`；endpoint object valid 原样全空 | 已有场景数据，没有机器人 qpos/actuation 或现成目标物体标签 |
| `visualize_track_4d.py` 正常导入 | 成功 | 几何可视化模块在当前环境可导入 |
| 原函数 `tracks_to_pc`，用 r03 当前真实 UV、K、Z | 32 点输出有限，反投影最大误差 `0.000076 mm` | UV/Z/K 的几何合同可对接；**不涉及深度或跟踪模型推理** |
| 原 `Simple6DPointNetBackbone` 类隔离执行 | AST 提取原类，随机权重，`[1,32,6]→[1,128]` 有限 | 编码器算子可用；不是完整 module import，不是 policy checkpoint 成功，点取自 scene、未声称都是目标物体 |
| 完整 `algorithms.rl.dagger.module.ActorPointNetTransformer` 导入 | `ModuleNotFoundError: No module named 'h5py'` | 当前环境依赖未满足；未逐项补装，也未训练 |
| SimToolReal 原观测模块导入 | `ModuleNotFoundError: No module named 'yourdfpy'` | 当前环境缺 FK 依赖；不代表装好后人体输入就兼容 |
| 当前环境依赖检查 | 有 torch/numpy/open3d/viser/isaacgym/pytorch3d；缺 cotracker/sam2/rospy/pointnet2_ops/yourdfpy | 包发现仅表示可定位，不代表所有二进制/模型运行可用 |

额外做了真实 r03 窗口的坐标反例：对未来参考点逆变换到 t+24 相机，再错误地仅用 t=0 相机外参映回世界。1586 个当前与末端有效场景点，坐标误差 median **4.12 mm**、p95 **5.42 mm**。这只衡量“固定外参代替逐帧外参”的误差，不是 tracker 精度或物体运动误差；即使此窗相机运动小，也已经接近我们此前 5mm 几何审核尺度。

## 对下一步的约束

1. 本轮能支持“源码中的配对点编码/反投影可拆用”；不能支持任何完整作者模型在 EgoDex 上训练或执行的结论。
2. 若本项目近期仍是 EgoDex 预训练人体手与场景预测，最便宜适配是使用现有 loader 和监督目标，仅移植配对 goal 编码；必须保持物体身份、对应关系、mask 和共同坐标正确。这是**改造工作量较小的候选方案**，尚没有训练收益证据。
3. Dex4D/SimToolReal 完整机器人策略复用属于另外的仿真/机器人训练路线，需要目标硬件状态、控制合同与资产投入。不能把“源码公开”当作“直接适配现有视频训练”。
4. 如果要用 Vision 改善小物体采点，可先只接 mask 内更密的 2D query，不必替换已经有的整个 metric geometry；随后用逐帧 EgoDex camera transforms 做小样本标签质量审核。

本审计没有证明哪条方法效果好；它排除了“直接拿上述完整代码喂现有 EgoDex 就能训练 goal policy”的实现假设。
