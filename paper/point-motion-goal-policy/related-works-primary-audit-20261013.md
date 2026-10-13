# 非语言的物体点运动 goal：相关开源工作一手核查

后续本地源码检查：完整作者策略尚不能直接读取现有EgoDex训练；部分goal/几何组件已通过真实样本工程检查。详见[兼容性结论](egodex-compatibility-summary-20261013.md)。

核查日期：2026-10-13。问题是：把「物体上哪些点应如何运动」直接给策略，而不是让策略从语言猜测任务。仅核对作者论文、项目页、官方仓库 README 和源码；没有下载模型、训练或复现结果。仓库公开与方法已复现是两件事。

## 判断

**存在直接匹配的开源方法。COIL 最贴合可变数量、可变轨迹长度的 3D 物体点 goal；Dex4D 最贴合灵巧手的当前–目标对应点控制；Im2Flow2Act 提供完整度较高的 2D flow → 闭环机器人动作实现。**语言可以保留为可选上游规划器，也可以用手工指定、参考演示或其他规划器产生同格式 goal。上述工作不能直接证明本项目的人类视频预训练或 Cm 有效，需要独立对照。

| 工作 | 实际下游控制条件 | 对我们的匹配 | 公开范围 |
|---|---|---|---|
| [COIL](https://coil-manip.github.io/) | 3D 物体点的有序目标序列、当前跟踪点、点云、本体状态 | 最直接的 object-motion goal Actor | [仿真数据采集、策略训练、评估源码](https://github.com/realquantumcookie/COIL)；README 没有承诺完整真机部署包 |
| [Dex4D](https://dex4d.github.io/) | 当前与下一目标的 3D 对应点对、本体状态、上一动作 | 灵巧手最直接匹配 | [Simulation](https://github.com/Dex4D/Dex4D-Simulation)、[Vision](https://github.com/Dex4D/Dex4D-Vision) 有源码；[Hardware](https://github.com/Dex4D/Dex4D-Hardware) 当前仅 Coming Soon |
| [Im2Flow2Act](https://im-flow-act.github.io/) | 对应物体点的 2D flow plan、当前跟踪、视觉/几何、本体状态 | flow 接口与训练链路参考 | [flow 生成、flow-conditioned policy 训练/评估源码](https://github.com/real-stanford/im2flow2act) |
| [ATM](https://xingyu-lin.github.io/atm/) | 当前图像与预测的任意点 2D 轨迹 | 点轨迹引导可以借鉴，但不限物体点 | [轨迹预训练、下游策略及 LIBERO 评估源码](https://github.com/Large-Trajectory-Model/ATM) |
| [Track2Act](https://homangab.github.io/track2act/) | 物体 2D tracks → 刚体变换/末端计划 → 残差修正 | 物体 motion 转控制的替代方案 | [官方库](https://github.com/homangab/Track-2-Act)已核实轨迹预测训练/推理；未核实完整机器人残差执行链路公开 |
| [Point Policy](https://point-policy.github.io/) | 物体及机器人点的历史 → 未来机器人点 | 表征/人到机器人转换参考；不等同于用户输入未来物体 goal | [官方策略源码与数据入口](https://github.com/siddhanthaldar/Point-Policy) |

## COIL：最贴合我们想定义的接口

论文的 task representation 是 `H×K×3`：同一组物体点在多个目标步骤的位置。可在多个物体上选点，也可给静止物体约束。策略融合 scene point cloud、当前 tracked keypoints、proprioception 与整段目标，输出动作序列；不要求手或末端 goal。平台是 Franka Research 3 + Robotiq 平行夹爪，非灵巧手。[论文 §3–4](https://arxiv.org/html/2512.05953v1)

**时间语义是顺序，不是固定秒数 deadline。**目标可以稀疏、不等间隔；只需按顺序达到，执行速度在线适应。不要把其归一化位置编码误读为严格物理时间条件。[论文 §3.1–3.2](https://arxiv.org/html/2512.05953v1)

源码已核查：`ctx['flows']` 为 `B×T×M×4`（xyz + valid），`obs['tracked_flow']` 保留相同点索引的当前/历史位置及遮挡信息。`policy_feature_extractors.py` 通过时空注意力把这些与点云融合；`flow_encoder.py` 根据有效步骤构造顺序编码及 attention mask，没有语言字段。[数据构造](https://github.com/realquantumcookie/COIL/blob/main/flow_manipulation/train_utils.py)、[特征提取器](https://github.com/realquantumcookie/COIL/blob/main/flow_manipulation/networks/policy_feature_extractors.py)、[编码器](https://github.com/realquantumcookie/COIL/blob/main/flow_manipulation/networks/flow_encoder.py)

开源范围是 ManiSkill/UniEnv 仿真侧采集、训练、评估，MIT；存在真实训练入口 `training/flow/train.py`，不是仅项目网站。标签来自仿真成功演示的 hindsight flow，随机化点数、步骤数、dropout、轨迹噪声；我们的训练可借鉴，部署时仍需独立指定目标对象和点身份。[README](https://github.com/realquantumcookie/COIL)、[训练入口](https://github.com/realquantumcookie/COIL/blob/main/training/flow/train.py)、[随机化配置/实现](https://github.com/realquantumcookie/COIL/blob/main/flow_manipulation/pointcloud/flow.py)

## Dex4D：灵巧手和对应关系最值得优先看

AP2AP 下游不用语言。其政策读当前与目标物体点、本体状态和上一动作；目标轨迹可以来自生成视频或人类示范。真机是 xArm6 + 16 DoF LEAP hand，另展示 XHand 转移。序列由外部推进：对应可见点平均误差小于阈值时换下一目标，政策本身主要处理一个当前目标点集。[论文 §III–V](https://arxiv.org/html/2602.15828v1)

**对应点必须成对编码。**每个 `i` 组成 `[current_xyz_i, target_xyz_i]` 六维，再用 PointNet 和池化；两边整体同步重排不影响输出，但独立重排会改变匹配含义。仿真 task 用同一 object-local 点集分别变换到当前/目标位姿；student 的 `ActorPointNetTransformer` 实际 `cat([object_kp,goal_kp],dim=-1)`，不是把两个无序点集独立池化。[task 源码](https://github.com/Dex4D/Dex4D-Simulation/blob/master/dex4d_policy/dex4d/tasks/xarm6_leap_hand_ap2ap.py)、[student 模型源码](https://github.com/Dex4D/Dex4D-Simulation/blob/master/dex4d_policy/dex4d/algorithms/rl/dagger/module.py)

对遮挡，student 用部分点蒸馏：一侧平面裁剪，再随机高度遮掉大量点；当前和目标同步 mask，保留对应关系。`utils/util.py` 的高度 mask 对两侧同索引置零。项目页展示少于 10 可见点仍可操控的例子；这不是我们数据上的可靠性保证，却说明上轮「至少 16 点」只是我们审核刚体标签的保守门槛，不是 point-goal 政策的普遍必要条件。[mask 实现](https://github.com/Dex4D/Dex4D-Simulation/blob/master/dex4d_policy/dex4d/utils/util.py)、[项目页](https://dex4d.github.io/)

其 RL teacher 不只依赖 object-goal distance，还含手/指到物体距离、手指弯曲、成功奖励等，先学抓起再学附近位姿转换。**没有手 goal 不代表不需要接触/抓取学习信号。**student 所谓 action world model 联合预测动作和下一步机器人关节状态，输入含 goal；它不等同于我们评估候选动作后果的 goal-blind object World/Cm。[论文 §III-C](https://arxiv.org/html/2602.15828v1)

公开状态已实际检查：Simulation 有 PPO、DAgger、task/model 源码与训练命令；Vision 有 `cotracker3.py`、`real_time_tracking.py`、depth/segmentation 工具；Hardware 目前只有 README，写着 Coming Soon。[仿真库](https://github.com/Dex4D/Dex4D-Simulation)、[视觉库](https://github.com/Dex4D/Dex4D-Vision)、[硬件库](https://github.com/Dex4D/Dex4D-Hardware)

## Im2Flow2Act：语言仅在上游，动作政策直接读 flow

完整系统用初始图像和 task description 生成物体 flow，再由 flow-conditioned policy 执行；上游可以被外部指定轨迹替代。物体 flow 是有身份的 2D 点随时间的位置，非手/末端轨迹；当前点在线跟踪，辅助几何包括 depth-lifted point cloud。训练控制政策使用仿真 play data，论文覆盖刚体、关节物体和可变形物体。[论文](https://arxiv.org/abs/2407.15208)、[作者仓库](https://github.com/real-stanford/im2flow2act)

训练代码实际读取 `episode_flow_plan`、`initial_flow`、`current_flow`、`point_cloud`、图像和 proprioception，送入 diffusion action model；这条训练调用没有语言输入。`target_flow` 参与辅助 alignment，不能混同为部署时实测未来。README 提供独立 flow/policy 训练和评估、数据及 checkpoints 入口，MIT；完整在线 tracker+policy 提示至少 24 GB 显存。我们可直接借其物体内采点、同点跟踪、flow 与当前状态分离的合同。[控制训练源码](https://github.com/real-stanford/im2flow2act/blob/main/scripts/controll/train_flow_conditioned_diffusion_policy.py)、[README](https://github.com/real-stanford/im2flow2act)

## ATM / Track2Act / Point Policy：为什么不是同一件事

- **ATM** 轨迹预测器用语言 + 当前图像生成任意场景点的 2D tracks；论文 §IV-B 明确下游动作政策不再需要语言。官方 `libero_vilt.yaml` 与 `train_libero_policy_atm.py` 的两个 `use_language_token` 均为 false；`vilt.py` 仍收到 `task_emb` 是因为内部 frozen track predictor 需要它。若我们直接提供轨迹，需绕过生成器改接输入，不能把现成 `act()` 当成外部 object-goal API。默认点也不只物体，可能包含机器人。[论文](https://arxiv.org/html/2401.00025v3)、[配置](https://github.com/Large-Trajectory-Model/ATM/blob/main/conf/train_bc/libero_vilt.yaml)、[训练脚本](https://github.com/Large-Trajectory-Model/ATM/blob/main/scripts/train_libero_policy_atm.py)、[策略源码](https://github.com/Large-Trajectory-Model/ATM/blob/main/atm/policy/vilt.py)
- **Track2Act** 不以语言驱动：初始/goal image → 2D tracks → 物体刚体变换 → 末端计划，另用机器人残差政策闭环修正，主要在 Spot arm 上实验。可直接提供 tracks 来替代图像到tracks，但刚体变换和抓取到末端映射仍限制适用性。官方 README 与顶层文件目前明确公开的是 track prediction 的 train/inference，不能推断整个论文机器人系统已开源；许可 CC-BY-NC。[论文](https://arxiv.org/abs/2405.01527)、[官方库](https://github.com/homangab/Track-2-Act)
- **Point Policy** 没有语言，但当前 object/robot 点历史是 observation，预测未来机器人点是 action representation。`agent/point_policy.py` 的 `act()` 读 `point_tracks_*`、输出 robot point future；未见用户提供的 desired object future 输入。可借鉴语义点标注、跨对象对应、手→机器人几何转换，不能直接说它证明 object-goal Actor。平台 Franka 平行夹爪。[论文](https://arxiv.org/html/2502.20391v1)、[模型源码](https://github.com/siddhanthaldar/Point-Policy/blob/main/point_policy/agent/point_policy.py)

## 补充：位姿目标执行和3D flow优化

[SimToolReal](https://github.com/tylerlum/simtoolreal) 提供对象位姿目标驱动灵巧手的RL路线；[Dream2Flow](https://github.com/KDharmarajanDev/Dream2Flow) 提供物体3D flow到轨迹优化控制的路线。两者的公开范围、静止pregrasp处理及本地goal编码检查见[本地设计影响](local-design-implications-20261013.md)。

## 给本项目的可执行建议（研究者推断）

1. **最便宜下一 Probe：保留明确点配对，先做 paired encoder 对照。**以 `[current_xyz_i,target_xyz_i]` 或 `[current_xyz_i,delta_i]` 同 token 表示，先检查同点整体重排等价、仅 target 重排可区分，再做小训练。多 waypoint 时再增加 step/order 与 valid mask。借鉴依据是 Dex4D 实际实现，不必先复制完整 RL 框架。
2. 将「轨迹顺序」和「必须在某秒到达」拆成两个任务合同。当前手远时物体短期不动并不异常，但仅给很近的静止 goal 不足以告诉策略接下来要抓什么。先用更晚 object waypoint、有序稀疏目标与闭环进度推进，比把 0.8 秒静止段当最终意图更清楚。COIL 与 Dex4D 都提供无需手 goal 的控制先例。
3. 数据仍需目标对象/部件 mask 与正确 point identity。少量好点优于大量背景点；点数不足与身份错误是不同问题。先为 3–8 个明确物体点保留遮挡掩码、做 motion label 可视化，再决定是否需要补点。不能因为上述论文成功就放过我们已观察到的瓶子未来点全部失效。
4. Actor 条件化期望，World/Cm 预测实际候选动作后果。可借这些工作建 Actor 的 goal 接口；不能直接把 Dex4D/COIL goal-conditioned 辅助预测头当作本项目 Cm 的因果证据。仍需正确 goal / null goal / shuffled goal 的动作与成功率对照，以及 Cm-on/off 对照。

本报告只形成源码与方法匹配判断，未形成正式性能结论。建议优先阅读顺序：COIL goal contract → Dex4D paired encoding/masking → Im2Flow2Act 视频到 flow 的数据处理。
