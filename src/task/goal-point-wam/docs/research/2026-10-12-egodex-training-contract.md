# EgoDex 接入 Goal-Point-WAM：标签合同核对

核查日期：2026-10-12。范围为官方论文/仓库、当前 loader 源码和已有远端库存记录；没有下载大数据、运行训练或更改运行中的源码。本文区分官方规范、已读本仓库代码和仍需实际数据验证的事项。

## 官方标注与能直接学习的目标

官方 MP4 与同名 HDF5 一一配对；视频 30 Hz，HDF5 每行对应同索引视频帧。`camera/intrinsic` 为 3×3，`transforms/<joint>` 和 `transforms/camera` 为 N×4×4；全部在录制 session 的 stationary ARKit origin 中，episode 之间 origin 不保证相同。置信度可缺失；存在时范围为 [0,1]，零代表未检测/完全遮挡。[官方 schema](https://github.com/apple-aiml-research/ml-egodex#dataset-structure)

论文每手 25 关节，包括 wrist 和 24 finger joints。原生动作监督是双手腕/指尖运动；论文误差以米表示，因此原生位移采用米尺度，不做额外任意缩放。论文没有发布可直接用于当前刚体 world 分支的 object R/t、metric depth、object mesh 或 point tracks；这是根据其五种公开模态得出的接入判断，不代表任何私有采集链路都没有深度。论文允许从官方 TRAIN 再划分 fit/dev；官方 TEST 保留评估，不能因为已有 test 小样本就混入训练。[论文 §3.3、§4.1–4.3](https://arxiv.org/html/2505.11709v1)

官方骨架命名为 `<side>Hand`、`<side>ThumbKnuckle`、`<side>{Index,Middle,Ring,Little}FingerKnuckle` 和相应 Tip；四个非拇指另有 Metacarpal 与 IntermediateBase/Tip。Knuckle 与 Metacarpal 是不同字段，不应只按整数下标猜测映射。当前每手 11 点最接近 wrist + 五 Knuckle + 五 Tip；与 MANO/OakInk 的 MCP 解剖等价仍应保留“近似”说明。[官方 skeleton_tfs.py](https://github.com/apple-aiml-research/ml-egodex/blob/main/utils/skeleton_tfs.py)

## 坐标、时间与有效性

官方转换为 `T_camera_joint = inverse(T_origin_camera) @ T_origin_joint`，可用原生相机直接把原生手标注投影到视频；内参示例 `fx=fy=736.6339, cx=960, cy=540`。以这一实际发布转换为准，勿因一般 OpenGL/ARKit 约定自行再加 y/z 反射。[官方 data_utils.py](https://github.com/apple-aiml-research/ml-egodex/blob/main/utils/data_utils.py)

对 H4/K24 轨迹窗口，应在当前 t 选择一个固定 anchor，例如当前相机坐标，并把 t−3…t+24 全部变到该坐标；不能每帧换相机再把相机运动误算成手运动。变换依据只用 t 及之前观测；未来手位置仅作为监督。原生 frame_id/30 为源时间，不直接沿用旧 pilot 的 11 原帧插值成 31 帧时钟。米尺度、SE(3) 刚性、帧长度与手臂/掌宽合理性须在实际输入上复核。

官方投影可能有 Vision Pro 多相机合成视差；小幅 RGB 重投影误差不能单独否定 3D 标签。[官方 visualize_2d.py](https://github.com/apple-aiml-research/ml-egodex/blob/main/visualize_2d.py)、[官方说明](https://github.com/apple-aiml-research/ml-egodex#code-structure-and-installation)

有效性必须逐点、逐时刻保存：有限的 SE(3) 与已知 confidence>0 才为已知有效；缺失 confidence 不可假称 confidence=1 或完全不可用，可另记 `confidence_known`。零坐标只是张量填充，不能在 action/trajectory loss 中成为观测或标签。计算当前速度/加速度也必须核对对应历史点有效，不能对零填充计算运动特征。

## 已观察的本仓库合同

- [当前 GoalWindows](../../src/goal_point_wam/data.py) 为 H4/K24、30 Hz、2×11 点；从 `action[..., :3]` 读取未来手位置，预测相对当前的位移。`_object_goal` 要求一个当前目标对象与未来刚体 effect；`_cap_scene` 拒绝零对象。
- [基础 Windows](../../../cm-pointflow-effect-pretrain/src/oakink_wm/data.py) 将整个窗口变到**同一当前目标对象 frame**；历史产生速度、加速度、三步位移，未来不能进入 scene encoder。
- [原生 EgoDex 手转换器](../../../cm-pointflow-effect-pretrain/tools/run/ref5_data_expansion/prepare_egodex_hands.py) 输出 right/left、每手 11 点、原生 camera/intrinsics、逐点 confidence/mask。该旧产物合同明确 `HAND_ONLY_OBJECT_PENDING` 与 `training_allowed=false`，没有发明 object poses。
- [旧 PointWAM 数据工具](../../../cm-pointflow-effect-pretrain/src/oakink_wm/pointwam_data.py) 使用 10 点（六 palm offset+四 fingertip）、离线场景教师以及 y-up→z-up proper rotation；其输出不能冒充当前 11 点原生手合同。用户 untracked `pointwam_core.py` 另有 10 点 named map，默认手序也不同；本次只读、不修改。
- [既有远端库存核对](2026-10-12-remote-data-access.md) 记录 48 个工程 qualified clip，样例手 `[31,2,10,3]`、场景轨迹、`strictly_causal_observation=false`；它们不是已经满足当前 object-goal+rigid-world 的新增源。该记录是既有审计，本次未重新读取每个远端原始 HDF5。

## 对本次训练的决策建议与未决检查

可以加入 EgoDex 的原生手轨迹监督，但应明确为**缺少 object goal/world 标签的额外人类动作数据**。最低成本方案是为新增源显式屏蔽缺失 object-goal tokens 和 world loss，保留有标签的 observed hand/history→future-hand 学习；三原生源仍提供 object-goal 与 world 监督。另一种方案是独立的 hand-endpoint-goal 预训练阶段，但它改变条件合同，应在实验卡写明，不能无声替换当前 object-goal 模型。

Root 本轮选择隔离的新 auxiliary path：EgoDex 原生 11 点、30 Hz H4/K24，在固定**当前相机** frame 中构造观测/未来轨迹；共享 scene/goal/trajectory 的 actor forward，但将 object-goal 全部置无效/null，**直接绕过 world forward**，避免全 masked world decoder 的 NaN。与 main3 完整 WAM batch 以 3:1 交替，EgoDex 比例 0.25；这是训练配比，不是按数据集规模自然采样。当前仅选既有 26 条官方 TRAIN raw HDF5（13 fit/13 dev）以及本地 6 条（3 fit/3 dev）；是否真实满足这些数量、CRC/SHA 与实际字段，由 root 的数据审计确认，本次不冒充实测完成。

若采用 confidence≥0.5，0.5 是**本次设计选择**，官方只说明 [0,1] 与零的意义，没有找到官方推荐的 0.5 阈值。须记录筛选后的每手/每关节/每 record 覆盖率；confidence 缺失另标 known=False，不通过把缺失值补成 1 来伪装高质量。

这一方案的科学范围是额外人类手运动监督，不能说明 EgoDex 教会模型 object goals、刚体 world 或物体因果效应。无 goal 的辅助学习可能降低主模型对真实 object goal 的敏感性，需同时监视现有 main3 固定面板/goal sensitivity 与 EgoDex held-record actor EPE。record 0/1 分离只证明 record 分离；没有可靠主体/session 元数据时，不能声称 subject/session/scene 分离。原生相机 frame 与 main3 目标物 frame 的输入分布差异，以及近似关节解剖映射，都属于外部数据域差异；本次 Probe 不将其等价于匹配对照的收益证据。

全假 world/goal masks 尚非当前 loader 支持的合同；空 scene/object 会触发校验或 decoder 全 masked 风险。允许有限的无效占位 slot 仅为形状兼容，必须在输入池化、goal attention、loss、metrics 与归一化中一致排除，不能当作一个静止真实物体。world 仍应在三原生源学习，不能用 EgoDex 的缺标签数据报告 world 改善。

本地最小 gate：验证官方 TRAIN provenance/CRC/SHA、record 分离 fit/dev；逐点映射/手序/30 Hz 时钟；固定当前 anchor；confidence/history masks；null goal 与无 world label 的有限前向/反向；改变伪占位对象不改变任何 EgoDex 监督或分数；有限 checkpoint replay。小规模 GPU Probe 再检查 held-record actor EPE 是否优于静止及训练能否稳定，随后才扩展远端数据与 DDP。

仍待实际输入核对：源 HDF5 字段集合/完整性、原生 camera 概念在选定 episode 的正深度投影、掌宽/位移范围、每手有效覆盖、分组/subject/session 元数据是否足以防泄漏、拟加入 TRAIN 子集确切规模与磁盘预算。完整 `.zip.part` 的大小不是已完成归档的证据；现有 admission flags 必须保留，由新运行 manifest 声明新的可审计准入范围，不修改原始工程记录。
