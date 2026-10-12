# EgoDex 物体监督：可以从视频生成，不能只靠截窗

核查：2026-10-12。只读官方材料和已有代码；未启动模型、下载数据或修改运行中的训练源码。

## 已核实的边界

EgoDex 公开 HDF5 主要提供相机、人体/手部 SE(3)、可选关节置信度与语言属性，没有发布逐物体 pose、mesh 或深度标注。因此“没有原生物体标签”成立；“不能得到物体监督”不成立。已有 RGB、内外参和米制手标签能够用于额外重建。[Apple 官方 schema](https://github.com/apple-aiml-research/ml-egodex#dataset-structure)

PointWAM 正是这样利用 EgoDex：VGGT-Ω 估计深度，以原生手深度比值的稳健中位数恢复米制尺度；CoTracker3 离线追踪首帧查询点，再用深度、内参和相机位姿提升到共同世界坐标。它得到**场景点轨迹伪标签**，不要求预先指定物体类别或 mesh。窗口内保持点对应，跨窗口不保证。论文给出的窗口是 1 秒、11 个 10 Hz 帧，再插值为 31 帧；这不等于我们当前 H4/K24、30 Hz 原生窗口。[PointWAM §3.4 / B.1](https://arxiv.org/html/2610.02840v1#A2.SS1)

截至本次核查，作者项目仍显示 Code soon，不能声称已复用作者发布的正式训练代码；仓库已有相关 pipeline 是自身实现或其他模型组件。[作者项目](https://chrockey.github.io/PointWAM/)

## 与当前代码的差别

[pointwam_data.py](../../../cm-pointflow-effect-pretrain/src/oakink_wm/pointwam_data.py) 已包含手尺度校准、轨迹 lift、首帧点选择和窗口封装，说明这条路线已有工程基础。不过该产物声明 `training_allowed=false`，且全片重建不能自动认证为在线可用的当前观测。[既有远端审计](2026-10-12-remote-data-access.md) 记录的 48 个工程 clip 同样未获训练准入；数量不是合格物体监督的证明。

当前 [GoalWindows](../../src/goal_point_wam/data.py) 的 goal/world 依赖目标对象及其未来刚体 effect，而 PointWAM 人类视频监督是每个场景点的未来位移。**场景轨迹不自动等于每物体刚体标签**。现有 [track_object.py](../../../cm-pointflow-effect-pretrain/tools/run/ref5_data_expansion/track_object.py) 尝试 FoundationPose 路线，需要深度、mask 和 mesh；这是我们选择的物体 pose 路线要求，不是 PointWAM 方法本身要求。

## 能改变下一步决策的最小路线

先选已有官方 TRAIN 的一两段明显刚体搬运视频，不扩大下载规模：

1. 在当前帧/已观测历史中标出一个目标物体，首帧可先人工 mask；沿视频追踪该物体表面点，用原生相机与手尺度恢复米制 3D 轨迹，保存逐点可见性与深度质量。不要按未来移动大小挑选输入点或物体。
2. 对已选对象保持稳定 point IDs，直接以当前/终点对应表面点构造最多 32 点的 object goal，并保留有效性 mask。目标编码器使用点集，因此不必先得到 6-DoF pose 或完整 mesh；此源可先提供 object-goal 条件下的原生手轨迹监督。当前 loader 用 R/t 生成 goal，需新增显式 point-goal 数据适配，不能直接把新产物冒充原合同。
3. 对满足非共线几何和覆盖率的同物体点，再稳健拟合各时刻相对 SE(3)，检查刚体残差、投影与相机运动补偿；通过后接入当前 world 标签。不通过则只屏蔽 world loss，仍可使用可信 point goal 和手标签；非刚体 scene trajectories 可留作独立 point-flow 辅助监督。以上是**本仓库拟议适配**，不是论文声称做过的物体拟合步骤。

最便宜的决策 gate 是视频上的点轨迹/物体轮廓叠加、静态背景漂移、手深度尺度一致性与 SE(3) 残差；先证明少量片段标签可信，再训练和扩大处理。目标点轨迹可信即可试 actor+goal；刚体拟合进一步通过才接 world。频繁遮挡或深度漂移失败意味着先换片段或改善伪标签，而非否定 EgoDex。

离线使用未来帧生成**监督标签**是合理的；我们当前部署式合同还要求 scene/history **输入**来自当前及过去。若深度教师对全片联合推理，初始深度也可能利用未来，因此须分别记录 teacher labels 与 causal observation。用户指定的目标终态作为条件是模型接口的一部分；它不授权把其他未来视频特征塞入当前 scene。当前混合训练保持不变，后续伪标签应在独立产物与准入记录中验证。
