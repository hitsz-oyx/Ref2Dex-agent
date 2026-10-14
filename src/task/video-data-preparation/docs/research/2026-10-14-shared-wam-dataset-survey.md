# Shared-WAM 数据源接入调研

日期：2026-10-14。范围：为 `video-data-preparation` 的 EgoDex 接入和 `ptv3-shared-wam` 训练寻找补充数据。本次只核验公开的第一方论文、官方格式说明和官方读取代码；没有下载数据、申请凭证或跑接入实验。下面的“适配建议”和成本排序是基于字段合同的工程判断，不是已验证的训练收益。

当前建议：**优先 H2O / HO-Cap 的原生 RGB-D 小集，DexYCB 做单手抓取对照，随后 HOT3D / TACO；HoloAssist / EgoVerse 用于后续扩量。** 先减少目标误选和 metric 伪标签误差，再扩大视频规模。HOI4D 的 15Hz 与 SHOW3D 的移动 world frame 是需要明确处理的合同差异；详细依据、下载边界与最小检查见下文。

## 判断标准

当前合同要求：观察仅使用当前及过去 H4，30 Hz；场景为非手 2048 点和 18 维特征；双手各 11 个语义点；固定当前选择的 32 个目标表面 query，输出 future24 的同点轨迹与 endpoint；相机、手、物体共用米制坐标。原始数据不必长成此 tensor，但必须能说明几何、时间和有效性来源。

不能把另一个数据集的“GT pose”强塞进 EgoDex 的 `confidences` 字段：有效性、可见性、置信度是不同含义。新来源应由独立 provider 生成同一 Shared-WAM 合同，保留 `source_id`、标注来源和质量 mask；缺失的手和关节保持无效。当前 EgoDex 11 点语义为 wrist、五指 knuckle、五指 tip；21 点骨架需要按名字映射，尤其 thumb 的关节命名/解剖位置不能仅凭数组序号视为完全等价。现有实现依据：[native_hand.py](../../src/video_data_preparation/native_hand.py)。

## 有手物几何标注的候选

下表来源与具体限制在后面的各数据集小节逐条给出。“绕过”指新 provider 可使用原生标注解析标签，并不表示当前 EgoDex CLI 可直接读取这些数据。

| 数据集 | 真实观测与时间 | 手标注 | 目标身份与轨迹 | 能否绕过 Omega / CoTracker | 对我们最有价值的用途 / 主要缺口 |
|---|---|---|---|---|---|
| H2O | ego + 四固定视角 RGB-D，30 fps | 左右各 21 点、MANO、每手是否标注 | 物体类 ID、mesh、逐帧 6D、camera-to-world | 原则上可同时绕过；scene 用当前 depth，query 用 mesh/pose | 最接近 ego 双手全合同；需下载凭证，核对深度单位/对齐和可见 mask；先限刚体 grab/place |
| HO-Cap | 固定 RealSense RGB-D + HoloLens ego RGB；帧时钟首样本核验 | 左右手 keypoints/MANO，依序列 hand sides | 类别、mesh、seg、逐帧6D，固定桌面世界系 | 固定RGB-D路径原则上可同时绕过；ego scene另核验 | 可按subject小包接入；pick/place/交接贴合任务，多对象需当前目标规则 |
| DexYCB | 八固定视角 RGB-D，30 fps | 每序列操作手 21 点 / MANO，左右均有案例 | `ycb_grasp_ind` 指定目标，像素 `seg`、全部物体 6D | 可同时绕过；有直接当前 depth/seg 与物体 pose | 最简单的原生几何对照；非 ego，单手，不补齐另手 |
| TACO | ego RGB-D、相机逐帧外参；官方按30fps解码 | 双手 MANO | tool/target分别有mesh与逐帧6D，自动分割 | 原则上可同时绕过，仍须对齐/可见性检查 | 后续双手工具操作扩展；target角色不等于手中操作物 |
| HOT3D / HOT3D-Clips | ego 30 fps；Aria 有 RGB，Quest3 仅双目灰度；没有发布的原生 dense RGB-D | 双手 UmeTrack / MANO，缺手允许；QA/可见性信息 | 多刚体 identity、扫描 CAD、世界系 6D；Clips 有渲染和 SAM2 mask | future query 不需 CoTracker；完整当前 scene 仍需解决深度 | 最适合 ego 双手物体轨迹与 selector 审计；不能把 SLAM 全段 map 当因果当前 scene |
| HOI4D | ego RGB-D，**15 fps**，20 秒序列 | MANO 单手操作，官方不覆盖双手操作 | 实例 CAD、物体/部件 pose、手物 2D seg、camera pose | 刚体/部件轨迹原则上不需 CoTracker；depth 不需 Omega | 场景/实例多样性强；15→30 Hz 和单手是实质适配成本，标注齐全率未实测 |

### H2O：优先解决 ego 双手的原生标签对照

**已确认事实。** 官方论文报告五台同步 Azure Kinect、30 fps、RGB 与 depth 为 1280×720；四个参与者、八种物体、36 类动作。571,645 是包含多个视角的帧数，不能视为相同数量的独立 ego 时间点。标注经过多视角拟合、人工检查和时间平滑，并非无误差传感器真值。[原论文 §3–4](https://arxiv.org/html/2104.11181v1)

官方格式有 `cam4` ego、`rgb/`、`depth/`、`cam_pose/`、`hand_pose/`、`hand_pose_MANO/`、`obj_pose_RT/` 与 `cam_intrinsics.txt`。`cam_pose` 为 camera-to-world 4×4；每手由是否标注加 21×3 坐标组成，物体提供 class ID 和 4×4 RT。intrinsics 包括 fx/fy/cx/cy/width/height。支持仅下载 ego 或仅下载 pose；后者不足以建立我们真实观察 scene。[官方数据说明](https://github.com/taeinkwon/h2odataset)

**适配判断。** 当前 depth 回投影得到 scene，当前物体 ID+mesh/pose 构造候选表面点，再由原生 future object pose 解析 32 条刚体轨迹。先只用 grab/place 等物体主体可近似刚体的窗口；open/squeeze、可动盖子和变形包装不能由一个主体 RT 解释全部 query。发布目录没有在上述说明中明确承诺逐像素手物 mask，需以当前 mesh 渲染结合 depth 检查遮挡；渲染完整轮廓不等于真实可见轮廓。官方 README 的 `cam_to_hand` / `cam_to_obj` 文字不足以独立证明每个矩阵的方向与单位，须用一段重投影确认。

**接入前最便宜检查。** 获得合法访问后取 2 个训练 ego 片段和 1 个已有 validation 片段，只核查连续 28 时间点的 RGB-D/左右手/object RT，测量投影重合、物体 mesh 与 depth 的尺度一致性，并渲染当前/endpoint 的固定 query。先判断无需两路 teacher 是否能产出少量合格窗口，再决定下载规模。

**获取限制。** 官方下载要填写身份/机构并接受条款，凭证邮件有效七天；仅学术、非商业用途，禁止向第三方转交数据。本次没有提交身份或验证凭证获取成功。[官方注册与条款](https://h2odataset.ethz.ch/)

### DexYCB：最低复杂度的 depth + 目标 ID + 刚体标签样例

**已确认事实。** 八台同步且外参标定的 RealSense D415 固定相机，640×480、30 fps；10 位参与者、20 个抓取对象、1,000 序列、约 582K 多视角 RGB-D 帧。每次单手从桌面拿起指定目标，周围有其他物体；左右操作手分不同试次。其多视角帧数不是独立动作数量。[官方论文 §2](https://dex-ycb.github.io/assets/chao_cvpr2021.pdf)

reader 给出彩色图、对齐到彩色图的深度、label NPZ、intrinsics、`ycb_ids`、`ycb_grasp_ind`、`mano_side`。标签包含像素 `seg`（background=0，YCB=1…21，hand=255）、物体 `pose_y[num_obj,3,4]`、手 `joint_3d[1,21,3]`（camera 坐标）/MANO；未标注或不可见手的 `joint_3d` 为 -1，不得当真实位置。[官方字段说明](https://github.com/NVlabs/dex-ycb-toolkit#loading-dataset-and-visualizing-samples)

**适配判断。** 这是当前“选错旁边积木”问题最便宜的对照：用 `ycb_ids[ycb_grasp_ind]` 锁定目标，用当前 seg+原生 depth 取 query，从物体 pose 生成 future trajectory。单固定相机可以直接采用它的 camera frame 作为窗口 anchor；多视角要共用 calibration，并将同一次动作的所有视角放入同一 split。非操作手无源标注，另一侧必须全 mask，不复制操作手。原生 21 点可按 wrist/MCP/tip 映射；官方定义可读到具体关节名称。[官方 reader 与 split 实现](https://github.com/NVlabs/dex-ycb-toolkit/blob/master/dex_ycb_toolkit/dex_ycb.py)

MANO renderer 有 mm→m 与图形相机轴翻转，不应将它的渲染坐标混入我们的 OpenCV camera 坐标；depth 的单位也应独立核对。[官方 renderer](https://github.com/NVlabs/dex-ycb-toolkit/blob/master/examples/visualize_pose.py)

**最小检查。** 先一个主体压缩包+calibration+models，选少量 train/val 序列与一个视角，检查固定 query 的物体 pose 传播及 depth/mesh 一致性。如果只需要验证 adapter，可以进一步从已合法持有的数据挑 3 个序列。此来源适合作为处理线路的可解释控制，但不能由它独自支撑 ego 或双手泛化结论。

**获取限制。** 官方提供 Google Drive 全包或按主体拆包；网站标注全压缩包 119G、单主体约 12G、models 1.4G，仅作为官方压缩下载量，不是解压后磁盘预算。数据 CC BY-NC 4.0；toolkit GPL-3.0，MANO 另有条款。本次只确认下载入口存在，没有验证传输成功。[官方下载页](https://dex-ycb.github.io/)

### HOT3D：ego 双手原生物体标签最有价值，但完整 scene 不免费

**已确认事实。** 833 分钟、19 人、33 个刚体、30 fps；Aria 包括一个 RGB 和两个灰度 stream，Quest3 两个灰度 stream。手和物体由扫描模型及外部 mocap 标注；Aria 还有 SLAM scene points。没有将该数据宣传为原生逐像素 dense depth 的依据。[官方项目页](https://facebookresearch.github.io/hot3d/)

Clips 共 3,832 个，每段 150 帧/5 秒，2,804 个来自 train，1,028 个来自 test；test 的目标/手 GT 被移除，不能直接给我们训练/验证标签。train clip 提供 cameras/hands/objects JSON、手 shape 与 object GLB；camera calibration 支持 fisheye 模型，object pose 为 `T_world_from_object`。其 `masks_amodal` 来自 CAD 渲染，`masks_modal` 和部分 visibility 为 SAM2 生成，不能当成人工像素 GT。[官方 Clips 格式](https://github.com/facebookresearch/hot3d/blob/main/hot3d/clips/README.md)

部分格式文字对 pose 方向描述有歧义；官方 viewer 的实际计算是 `T_world_from_object @ object_vertex → world`，随后用 camera pose 逆变换投影。以此与重投影检查作为适配依据。[官方 viewer](https://github.com/facebookresearch/hot3d/blob/main/hot3d/clips/vis_clips.py)

**适配判断。** 在 current-time native object pose 下选择可见 CAD 表面 32 点，使用 future object pose 得到米制同点轨迹，能彻底省掉 object CoTracker。手用 UmeTrack 或 MANO 解码后映射 11 点，缺失侧保持 invalid。多物体 ID 仍不自动告诉我们任务目标：应使用明确指定的 current object ID，或仅 H4 选择规则；不得按哪个物体未来运动最大来选。官方 hand decoder 提供 keypoint/mesh 输出，可复用而无需自己猜骨架。[官方 Clips 解码工具](https://github.com/facebookresearch/hot3d/blob/main/hot3d/clips/clip_util.py)

有两种 scene 方案应分别声明：一是当前 CAD 渲染/可见表面重建，仅覆盖已建模对象和手；二是仅 H4 Omega 深度补真实背景，native pose 只用于物体标签与尺度/重投影核查。前者不能冒充与 EgoDex 同分布的真实 depth scene，后者仍有推理成本，但省去 future CoTracker。鱼眼图像应先一致地去畸变或用官方 camera model 回投影，不能直接给当前 pinhole reader 一个普通 3×3 K。[官方相机 provider](https://github.com/facebookresearch/hot3d/blob/main/hot3d/data_loaders/AriaDataProvider.py)

**因果边界。** 该 provider 的 `get_point_cloud()` 返回 MPS 的 semidense scene map，不以 current timestamp 限定。因此没有额外逐点观测时序过滤证据时，不能直接把整段 map 裁到当前视野就认作 H4 内获取的 scene；“投影在当前视野内”不证明“只用了过去”。全片段重建地图属于额外 oracle-map profile。

**最小检查。** 2 个 `train_aria` Clips + 对应简化 object GLB，按源 sequence 分组，核对双手映射、鱼眼重投影、可见 query 与 pose 传播，再运行与当前 EgoDex selector 的当前帧对照；不下载全 VRS，不先处理 Quest3 灰度域。训练 split 内划小 dev，不能使用标签隐藏的官方 test 充 dev。

**获取限制。** Clips 有官方指向的 Hugging Face 入口；全 VRS 数据需要在 Project Aria 注册取得有期限的下载 URL。数据/手标注/模型分别受 HOT3D 与相关第三方条款约束；Apache-2.0 是 toolkit 许可，不能替代数据许可。本次未成功读取 Project Aria 的具体许可正文，不能据此宣布可商业使用，也没有申请身份。[官方获取及许可说明](https://github.com/facebookresearch/hot3d#step-2-sign-up-and-get-the-download-links-file)、[官方 Clips 发布入口](https://huggingface.co/datasets/bop-benchmark/hot3d)

### HOI4D：扩场景适合，直接沿用 30 Hz 双手接口不适合

**已确认事实。** 2.4M RGB-D 帧、4,000 序列、9 人、800 个实例、16 类、610 个室内场景；公开有 camera 参数、手 pose、物体 CAD、标注下载入口，数据 CC BY-NC 4.0。[官方项目页](https://hoi4d.github.io/)

原论文说明 RealSense D455 和 KinectV2 同步采集，每视频 **15 fps、20 秒**；限制节明确不覆盖双手操作。物体包含刚体和关节物体，原始标注之间做 world-space 插值再优化，不能因为写着 frame-wise 就假设所有发布序列的全部28帧合同齐全。[官方论文 §3–4、§7](https://hoi4d.github.io/HOI4D_cvpr2022.pdf)

原生目录包含 RGB/depth 视频、object pose JSON、hand/target segmentation、scene点云和 `output.log` camera trajectory。MANO pickle 包含 `poseCoeff`、`beta`、camera-frame `trans`、21 点 `kps2D`；官方 right-hand 解码示例把 MANO mm→m。rigid mesh 是其 pose canonical frame，关节物体有单独部件 mesh 和部件 pose。[官方数据组织与字段](https://github.com/leolyliu/HOI4D-Instructions)

官方 decoder 强制以 15fps 导出 RGB 和 depth；4D 示例代码还硬编码300帧、Kinect 默认内参，不宜直接照搬作为我们的多设备 adapter。应使用下载的真实 camera calibration，检查 depth 编码与RGB对齐，不以脚本默认配置替代。[官方 decoder](https://github.com/leolyliu/HOI4D-Instructions/blob/main/utils/decode.py)、[官方 RGB-D 回投影示例](https://github.com/leolyliu/HOI4D-Instructions/blob/main/prepare_4Dseg/scene2frame.py)

**适配判断。** 对刚体或选定刚体部件，可从 current mask/depth + CAD alignment 得到 query identity，由 per-part pose 生成标签。需要确认 `dimensions` 与 mesh 物理尺度的组合方式，不得逐帧对目标形状施加变化 scale 导致假变形。另一手缺失应 mask，未知 confidence 保留“不可用”。更大的阻力是时钟：直接把15Hz的24步当30Hz会把1.6秒运动误当0.8秒。若未来要使用，建议独立15Hz profile，或明确记录重采样/插值来源；H4观察插值不得读取current之后的帧，且不能把复制图像宣传为新的30Hz真实观测。

**最小检查。** 先 3 个 pick/place 刚体训练片段和 1 个可用验证片段，审计每种标注的frame ID、时间间隔、齐全率、mesh/pose/seg对齐；仅当物理时钟和原生label可靠，才投入新profile。官方下载入口存在，不代表已经验证链接或百度提取成功，也未确认全量解压大小。

ARCTIC、OakInk2、GRAB 已属于当前 Shared-WAM 原有三域；可用现有产物作为 adapter/训练基准和控制组，本次不把它们计为新增数据来源，也不因存在 geometry 就重新下载。

## 新增候选：HO-Cap 与 TACO

### HO-Cap：优先做原生 RGB-D 的小规模接入对照

官方任务覆盖 pick-and-place、双手交接和按物体功能操作，提供物体模型及逐帧手物标注，数据许可为 CC BY 4.0。下载可以按 subject 进行；项目页列出的最小 subject_5 压缩包为 3.2 GB，另需 calibration、models、poses、labels 等共享包。这些是发布包大小，不能当作解压后总占用。[项目与分包清单](https://irvlutd.github.io/HOCap/)

实际可接入的字段比宣传页更明确：

- 固定 RealSense 相机的 RGB、depth、K、外参进入固定桌面世界系；官方 loader 读取深度时除以 1000。
- 图像 NPZ 提供 `obj_poses`、`hand_joints_3d`、`hand_joints_2d`、`seg_mask` 和物体类别；缺 NPZ 返回空字典，应按缺标注处理。
- 序列 `meta.yaml` 提供 `num_frames/object_ids/mano_sides`，`poses_o.npy/poses_m.npy/poses_pv.npy` 区分物体、手和头戴相机位姿。[SequenceLoader](https://github.com/IRVLUTD/HO-Cap/blob/main/hocap_toolkit/loaders/sequence_loader.py)、[逐帧标签使用示例](https://github.com/IRVLUTD/HO-Cap/blob/main/examples/image_label_viewer.py)

**适配判断：**先选单个固定 RealSense 视角，用当前真实深度构造非手 scene，用原生物体 SE(3) 生成固定点未来标签。可以避开语言 grounding、单目尺度标定和 CoTracker 三个环节；这是基于发布字段的工程推断，尚未完成实际样本读取。头戴 HoloLens 视角另作下一步：当前公开 loader 直接反投影的是 RealSense 深度，不能因为数据含 HoloLens 就宣称 ego RGB 已有同像素密集深度。第一份样本还必须核查帧时间、NPZ/pose 坐标方向、手侧顺序及无效值；官方元数据读取代码没有明确 FPS 字段，本调研不假定所有序列严格 30 Hz。

原生 annotations 来自多视角模型估计和优化，需抽查重投影，不能视作无误差测量。固定 RGB-D 视角的对照能判断 adapter 是否正确，但不单独证明 ego 场景泛化。[官方论文 §4、§9](https://arxiv.org/html/2406.06843v4)

### TACO：后续扩展双手、工具和被作用物体

发布 V1 有 2317 个序列，官方完整 ego 模态列表列出 2212 个；包含 ego RGB-D、双手 MANO、tool/target 姿态和模型、相机参数，2D 分割为自动标注。数据许可为 CC BY 4.0，预发布小集另有作者提供的百度网盘入口。不要把预发布版和 V1 混为一个快照。[官方数据说明](https://github.com/leolyliu/TACO-Instructions)

官方投影代码实际读取逐帧 `egocentric_frame_extrinsic.npy`，约定为 world-to-camera；物体 mesh 文件以 cm 存储并乘 0.01，depth 解码说明的 scale 为 4000。示例按 30 fps 解码和回放，但接入仍应核验视频时间戳与 pose 长度，不能依靠 ffmpeg 的 `fps=30` 自动补帧证明原始同步。[投影代码](https://github.com/leolyliu/TACO-Instructions/blob/master/dataset_utils/project_pose_to_egocentric_view.py)、[深度与序列列表说明](https://github.com/leolyliu/TACO-Instructions#data-instruction)

**适配判断：**能够走真实深度 + 原生 pose 标签线路，模型无需照搬 TACO 的工具操作网络。第一阶段选单个刚性对象作为 goal target，其他对象作当前 scene context。特别注意其 `target` 指“被工具作用的对象”，不一定是我们要预测的“手里移动的对象”；例如搅拌时勺子是 tool、碗是 target。应根据当前任务角色预定义选哪个 slot，禁止根据未来位移选对象。待单目标合同跑通后，再考虑 Shared-WAM 已有的多物体输出能力，不在初次接入时扩大 EgoDex 的 O=1 profile。

## 扩规模候选与暂缓项

| 数据集 | 确认提供的关键内容 | 对本流程的判断 |
| --- | --- | --- |
| HoloAssist | RGB、AHAT depth、双手和头/相机位姿、内参、语言与动作区间 | 原生手与近场深度有价值；仍需目标选择与物体轨迹生成 |
| EgoVerse | episode Zarr、RGB、头部位姿、内参；部分 episode 有完整 21 点手标注 | 接近 EgoDex 扩量路线，必须按字段筛子集；发布合同未要求物体轨迹或 depth |
| SHOW3D | 双手模型、物体 6D、ego/exo 图像、相机标定、文本 | 当前发布 world 随背部 rig 移动；未补 rig 世界运动前，暂缓原合同接入 |
| GigaHands | 双手、物体 mesh/pose、RGB 多视角和文字 | 可扩张几何运动库；必须选择与物体同坐标的手数据，scene 仍需补充 |
| ADT | 场景/物体、相机 pose、GT 系统生成的 depth/instance segmentation | 很适合几何和目标审计；逐指手标注覆盖未确认，不优先做完整双头训练 |
| Ego-Exo4D | 部分时间点双手 3D、相机标定、MPS 位姿及场景点云 | 不能假定完整连续 H4+24 标注，物体点轨迹仍须生成 |
| FPHA、HO-3D/H₂O-3D | RGB-D、手标注、部分或全部物体 mesh/pose | 可作为规模较小的补充；与 H2O 区分，优先级低于上述原生 RGB-D 路线 |
| EPIC-KITCHENS/VISOR、普通 Ego4D FHO | 手/活动物体 mask 或 box、动作及交互描述 | 可帮助审核/学习目标选择，完整米制手物监督需额外估计，当前不优先 |
| DROID 等机器人演示 | RGB、机器人关节/夹爪状态与动作 | 当前 2×11 人手合同无法直接套用；末端位置不能伪装成逐指标签 |

上表是字段适配判断，并非训练效果排序。对“未要求/未确认”的字段，只说明当前读取的发布合同不足以保证，不声称所有原始传感器数据都不存在。

### HoloAssist：近场原生深度有帮助，但要处理 AHAT 特性

论文报告 166 小时，当前项目页写 169 小时；记录二者版本口径，不将其相加。发布目录区分 depth/RGB 相机 calibration、camera pose 和左右手同步文件。[项目页](https://holoassist.github.io/)、[官方格式说明](https://holoassist.github.io/data_links/README.html)

补充材料列出 RGB 29.5 Hz、depth 32.4 Hz，AHAT 超过约 1 m 存在 phase wrap。因此不能直接把深度 PNG 当作普通、已对齐 RGB 的 Z-depth：先处理有效近场范围、相机坐标及时间同步，再判断目标是否有可信 metric depth。[官方补充材料 Table 2](https://openaccess.thecvf.com/content/ICCV2023/supplemental/Wang_HoloAssist_an_Egocentric_ICCV_2023_supplemental.pdf)

**适配判断：**手与相机可替换原生 reader；当前 mask、稀疏 2D tracking 和 future validity 仍要保留。观察和标签优先使用有效原生 depth，缺深度时直接 mask/拒绝，不能默默回填未来估计。论文/发布说明未承诺逐帧目标 mesh+6D，所以暂不把它列为消除目标误选的首选。depth 全包已超过本项目当前 400 GB 产物预算，只考虑有界 episode 下载。

### EgoVerse：值得跟进的新扩量来源，但先筛完整手点

论文版本报告 1362 小时、约 80k episodes；网站快照已写 4003 小时，属于持续增长的口径。只选定一个 release/index 再计数。[论文 arXiv:2604.07607](https://arxiv.org/abs/2604.07607)、[项目快照](https://egoverse.ai/)

Zarr 合同要求头部位姿与相机内参；完整 `left/right.obs_keypoints` 是可选字段，单个 `obs_ee_pose` 不能提供我们的 11 点手。完整点以 MANO 顺序 21×3 存储在 SLAM 世界系；帧数必须按 `total_frames` 截断，不能把 chunk padding 当作数据。合同未要求密集 depth、object instance mask 或物体 6D。[官方 contribution schema §6](https://github.com/GaTech-RL2/EgoVerse/blob/main/CONTRIBUTING_DATA.md)

**适配判断：**先筛完整手点、可信头部到 RGB 相机外参、实际时间戳和有效内参的 episode，再复用我们的 depth/mask/track teacher；RGB-only 子集不会自动消除当前问题。缺手点、异常/缺失值不伪造。公开处理代码可读，实际下载还依赖其数据服务配置，本轮没有完成数据获取或使用任何凭据。[官方工具入口](https://github.com/GaTech-RL2/EgoVerse)

### SHOW3D：姿态很丰富，但固定世界系是当前接入 blocker

这是 CVPR 2026 数据；当前发布为 60 fps，适配 30 Hz 可用相隔 2 帧的采样。hand v3 与 camera pose 修订已发布，应固定版本。**物体 pose 不覆盖所有视频**：其 v1 专项说明列出 467 个 recordings，需和 hand、video、calibration 的 index 取交集，而不能按完整视频数量估算可训练规模。[项目](https://show3d-dataset.github.io/)、[hand v3 格式](https://huggingface.co/datasets/facebook/show3d-dataset/blob/main/hand_pose/README.md)、[object pose v1 格式](https://huggingface.co/datasets/facebook/show3d-dataset/blob/main/object_pose/README.md)

最关键的是，官方 `scenes/README` 明确“world”为背部 rig frame，并非固定物理世界；跨帧桌面点坐标会因人走动/转身而变化，估计 rig 在物理世界中的运动仍属未来工作。[相机与 world 合同](https://huggingface.co/datasets/facebook/show3d-dataset/blob/main/scenes/README.md)

**适配判断：**直接做 `p[k]-p[t]` 不符合我们的目标位移语义。短片里运动小不等于固定世界系已成立；不以更宽刚体门掩盖它。可用于当前 mask、手点投影和对象身份审计；作为完整 Shared-WAM 样本时，先有 rig→固定世界的可信轨迹，或另立明确的 body-relative 预测 profile。后者改变监督含义，需要独立设计，不能与当前数据混用。

### 其他可用来源的边界

- **GigaHands**：已公开 object pose/mesh 和手物对应工具；原先 `keypoints_3d_mano` 曾被归一化并重新居中，新发布有与物体坐标对齐的平滑手数据。必须取后者或统一恢复坐标，不能把 motion-generation 格式直接用于共同 anchor。数据许可为 CC BY-NC 4.0，下载走 Globus；目前公开物体 motion 集约 3.3k 序列，勿把全体手动作都当成完整手物监督。[官方 README 与更新](https://github.com/brown-ivl/GigaHands)
- **ADT**：原生相机/物体轨迹、instance ID、深度均由 GT 系统生成；深度明确为相机 Z、mm，部分 skeleton/带人体遮挡版本可选。它有真实视频，也有合成渲染，不能把 GT 系统深度称为传感器实测深度；人体 skeleton 不等于双手逐指标注。可先作 depth lifting、instance selection 和刚体标签的诊断集。[官方 ADT 格式](https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/aria_digital_twin_dataset/data_format)
- **Ego-Exo4D**：EgoPose 在有效标注时间点提供双手共 42 点和相机矩阵；MPS 场景点云关联多个时间点，并非当前帧密集深度。原生手 JSON 的连续窗口覆盖要单独统计，不能将稀疏手点插值成真实 30Hz 标签。[EgoPose](https://docs.ego-exo4d-data.org/annotations/ego_pose/)、[MPS](https://docs.ego-exo4d-data.org/data/mps/)
- **FPHA**：真实 RGB-D 和 21 手关节，但 object pose/mesh 只覆盖物体子集；只用手物字段齐全的动态序列。[作者项目](https://kcvl-kaist.github.io/FPHA/)
- **HO-3D/H₂O-3D**：前者单右手，后者双手；均有对应 depth 和手物姿态，是 **不同于 H2O** 的数据集。旧可视化代码需要适配当前 Python，手部模型另遵循 MANO 获取流程；连续时钟及缺标注须核验。[作者仓库](https://github.com/shreyashampali/ho3d)
- **EPIC-KITCHENS/VISOR 与 Ego4D FHO**：VISOR 提供手/活动物体分割和交互关系，部分 dense mask 为插值；FHO 主要通过 box 标注交互。更适合 selector 的训练/审核集；两者这些标注本身不足以提供我们需要的连续米制逐指与物体点标签。[VISOR](https://epic-kitchens.github.io/VISOR/site)、[FHO 格式](https://ego4d-data.org/docs/tutorials/FHO_Overview/)
- **EgoPAT3D**：有 RGB-D，但 hand 是 MediaPipe 推断，动作 target 主要来自片段末帧手中心经点云 registration 回投，不是物体表面 32 点的轨迹真值。与我们的双头监督缺口不同，不优先增加适配工作。[作者数据说明与标签生成](https://github.com/ai4ce/EgoPAT3D)
- **DROID**：官方 RLDS schema 提供夹爪位置、7维关节位置、机器人笛卡尔位置和动作；原始数据另有 stereo 信息，但这些字段不能直接还原我们的逐指手标签。适合未来明确的机器人 provider，与本轮人手视频接入区分。[官方 schema](https://droid-dataset.github.io/droid/the-droid-dataset)

## 接入方式：复用样本合同，按数据来源替换 teacher

对有真实 depth + 对象 mesh/SE(3) 的候选，建议在 reader 之后加入新 provider，继续复用 anchor、scene sampler、手点映射、shard/snapshot、collate 和训练接口。无需把它们强行送进 EgoDex 的 GroundingDINO→SAM2→Omega→CoTracker 全流程。

给定在当前帧选择的对象表面点 `p_t^W`，将其固定在物体局部坐标：

```text
q_i      = inverse(T_WO(t)) · p_t,i^W
p_k,i^W  = T_WO(k) · q_i
U_k,i    = T_AW(t) · p_k,i^W − T_AW(t) · p_t,i^W
V_k,j    = T_AW(t) · h_k,j^W − T_AW(t) · h_t,j^W
G_i      = U_24,i
```

观察点可来自当前可见、去手后的 depth，并通过当前 CAD/mask 核验归属。这保留同一批点的身份，标签计算通常只需矩阵运算，避开每窗深度网络和点跟踪推理。对已获得 metric depth 的来源，不再使用 EgoDex 手/深度尺度比门；替换为传感器/标注单位、重投影、CAD-depth 残差及位姿连续性检查，不能解释为放宽 EgoDex 的现有门。

先维持 visible-observation profile：只对当前可见表面采样，未来 visibility/occlusion 单独保存；如果原生 pose 能监督遮挡中的点，明确另列 pose-complete label profile，不把“物体 pose 有值”与“点在 RGB 中可见”混淆。首次不把全 CAD 后表面填进 scene，也不把其他未标注背景标成未来静止。

所有来源仍须满足：

1. right/left 与 wrist、五个 MCP、五个 fingertip 的语义映射；缺手/关节显式 mask。原生数据没有置信度时，采用其标注有效性协议并记录来源，不能直接套 EgoDex 的 0.5 阈值或假造 confidence。
2. 按真实时间构造 H4 与 0.8 秒 future24；速度和加速度以真实 dt 或明确的 30Hz重采样计算。60→30可选帧，15→30涉及插值，二者不等价。
3. 固定物理世界系和当前 anchor；RGB/depth/resolution/intrinsics 一致。整段 SLAM 点云与离线估计 depth 的因果边界须审计。
4. 当前对象身份和 query 不按未来运动选择；未来只能生成 U/V/G 和标签 mask。
5. 按完整 recording/session/subject 分组；同一演示多个相机视角不能跨 TRAIN/dev。官方 TEST 保持独立，多数据集共享 mesh 不等于同一 episode。
6. 独立 source/profile/版本清单与冻结 snapshot，不把几何真值、CAD渲染和视频伪标签合成无法追溯的一个 profile。

## 下一步决策与最便宜的检查

本轮形成**接入优先级建议**，尚无新增数据下载、adapter 实现或训练结果，不构成任何数据集能改善最终机器人策略的证据。

建议先保留 EgoDex 线路，用 **H2O 或 HO-Cap 小份原生 RGB-D 数据建立对照**，再扩真实视频；已存在的 OakInk2/GRAB/ARCTIC 继续作为训练基线。选择逻辑：

- H2O 若下载权限可用，优先验证 ego 双手 RGB-D；若获取不便，HO-Cap 可按一个小 subject 包启动固定 RGB-D 对照，随后才处理头戴视角。
- DexYCB 用于最省实现的单手抓取对照；同源多视角只保留一个观察视角起步，避免无用下载和重复推理。
- HOT3D 用于明确对象身份、ego 双手的标签与 selector 审计；TACO 用于更复杂的双手/工具目标域。两者都先做少量片段。
- HOI4D 在我们决定接受 15Hz来源的显式重采样 profile 后再接入；HoloAssist/EgoVerse 在 selector 改善后用于扩量。
- SHOW3D 等当前缺固定世界运动的数据不立即进入原合同。大规模视频本身不能解决目标身份和 metric 几何缺口。

建议的下一次 **Decision Probe**（尚未执行）：一个来源、2个独立 recording或subject组、每组至少10个真实连续窗口；先读取原始少量样本，验证时钟、单位、投影和手物对应，再形成20–50窗的可审查 TRAIN/dev。它检验“原生几何/对象标注是否能以更低处理成本产生完整双头窗口”，结果决定接入原生标签路线还是先修 adapter。停止条件是字段不齐、时间/坐标无法确定或当前目标身份不可信；不通过插值、复制或放宽门补齐。具体下载量与运行预算在执行前按包粒度核实，本调研不预报未经实测的吞吐/GPU小时。

下载时只取必要视角、标注、模型和 calibration；若使用 Hugging Face，按仓库规定先查询 ModelScope，再尝试国内镜像，之后才按需代理。本轮链接可读性不等于所有下载包在远端已获取或下载权限已获批。
