# 新增数据集接入优先级：结合近期仓库更改

日期：2026-10-14。所属 Task：video-data-preparation。读取时 HEAD 为 `8e680e49`，收尾补读同日新增的 `25f2cefd` 现场报告。本轮完成一手来源与读取代码调研，未下载新数据、申请访问、安装环境、实现 provider 或启动训练。下面是接入建议，不是训练收益或科研结论。

## 近期更改怎样影响选择

- `da694f1f` 引入 PTv3-Shared-WAM v2：共享几何/手部编码器、双预测头、按 horizon 标准化；当前 v2 长训仍为 OakInk2/GRAB/ARCTIC。旧 Task 承担 RL660 的机器人联合微调，v2 还没有 robot/Chunk-R/native 执行入口。[v2 README](../../../ptv3-shared-wam-v2/README.md)、[v2 STATE](../../../ptv3-shared-wam-v2/docs/STATE.md)
- `0576812f`、`4e4b2b00`、`836dc509` 把 UniDexGrasp++ 回放变成按需动态样本：20 对象、320 条执行、TRAIN2560/dev640 窗，mesh 分组隔离；H4 场景运动来自观察历史。当前 source4/完整 CAD/ShadowHand profile 仍只允许工程读取，不能直接作为 Inspire 控制标签。[UniDex 实验卡](../experiments/probes/P-20261014-unidexgrasp-local-admission.md)、[provider](../../src/video_data_preparation/unidexgrasp_data.py)
- `25f2cefd` 的 16:25 现场快照更新了视频瓶颈：新 EgoDex r4/r5 两批完成，922 次尝试、196 数值候选，已批准集仍 TRAIN5/VAL0；下载与旧重建继续，新的 consumer 没有持续生产。Open-AoE 两段仅过手/相机检查，扩展按用户要求暂停。[现场报告](../user/ref/ref1_1.md)
- RL660 已补录完整机器人手点、状态和实际命令，但近期控制目标/选点与 native 数据入口仍有待修项。因此新增人手或异构机器人数据首先服务 U/V 手物轨迹预训练；是否改善 Inspire 的 R/PPO 另行验证。[RL660 实验卡](../../../ptv3-shared-wam/docs/experiments/probes/P-20261014-ptv3-shared-rl660-joint.md)

**选择依据：**优先取得新的连续交互、物体多样性和可信标签，同时衡量制作完整窗口的成本。大库小时数、静态抓取数和多视角图像数，不能代替可用的独立动态轨迹数。

## 对当前模型“可用”的具体含义

模型样本是米制同一 anchor 下的观察 H4、30Hz future24、双手各11点及有效 mask、当前场景18D特征、当前固定32个物体 query 及其未来位移/endpoint goal。缺手保持 invalid，腕/MCP/tip 按语义映射，不用夹爪末端冒充逐指骨架。[现有样本合同](../../src/video_data_preparation/sample_contract.py)、[Shared-WAM 数据实现](../../../ptv3-shared-wam/src/ptv3_shared_wam/data.py)

当前场景有两种可区分的来源：既有 CAD/pose 几何路线，以及 RGB-D/视频可见点路线。只做明确声明的完整 CAD profile 时，有物体 mesh 和当前 pose 就可建立对象几何；要做相机可见场景则还需当前 depth/遮挡依据。不能把两种输入的完整性、normal 或未来可见性混为一谈。

现有 `SharedWindows` 与 `MixedWindows` 明确限制三域/既有来源；v2 训练还绑定原 TRAIN 统计和抽样权重。新增文件不会自动进入长训，需独立 provider、source/profile 准入、来源抽样与新 TRAIN 统计后才接新 run。沿用已有工程 provider 不代表通过 source admission。[来源限制](../../../cm-pointflow-effect-pretrain/src/oakink_wm/multisource.py)、[v2 训练入口](../../../ptv3-shared-wam-v2/src/ptv3_shared_wam_v2/training.py)

## 当前建议

| 目的 | 优先候选 | 为什么 / 尚需核验 |
| --- | --- | --- |
| 最低增量成本增加物体覆盖 | 扩现有 UniDexGrasp++ | 已有采集与按需读取代码；先完成现有20对象 source admission，再决定跨mesh/scale扩量，不重复研究静态标签 |
| 新增机器人动态来源 | DexMimicGen | 已发完整动态演示与 state playback；先取真正 dexterous 任务，核验 control clock、FK与物体标签 |
| 新增真实手物几何与物体覆盖 | GigaHands 的刚体成功tracking子集 | 有双手/mesh/物体pose；发布物体序列约3.3k，需使用同坐标手数据、保留缺帧与估计标签来源 |
| ego刚体轨迹与对象身份 | HOT3D | 手物camera/CAD标签较齐；完整 CAD profile 可先接，dense observed scene 另处理 |
| 双手工具/多对象交互 | TACO | ego RGB-D与双手、tool/target pose；选操作对象角色、核对深度/尺度和齐全序列 |
| 新机器人多任务候选 | DexVerse | 已发1,900条baseline演示；IsaacLab依赖、记录手型和接触语义的审计成本较高，先检查元数据 |
| 千小时以上视频扩量 | Open-AoE；有访问条件再考虑 EgoSuite | 有手/相机字段，但物体轨迹仍须制作；Open-AoE继续列备选，不自动恢复已暂停的扩展 |

上表是基于字段与已有工具的工程排序，来源证据见下文和[已有大规模语料调研](2026-10-14-shared-wam-dataset-survey.md)。若只选择**一个全新机器人来源**，建议先查 DexMimicGen 的 LiftTray/BoxCleanup；若优先增加**真实物体与双手几何**，建议先查 GigaHands 的刚体子集。小规模原生 RGB-D 对照则用 HO-Cap/DexYCB，避免把扩量目标局限在小基准。

## 动态机器人来源

| 来源 | 已发布内容与优先级判断 | 关键缺口 |
| --- | --- | --- |
| DexMimicGen | 完整动态 HDF5 演示和 MuJoCo 模型/状态；优先核验六个灵巧手任务 | 部分任务只有夹爪；时钟需读 env_args；官方许可文字冲突 |
| DexVerse | 1,900 条 baseline trajectories，38 个 task/version 集、19 task families；优先小份核验 | manifest 未标每条 hand；simulation replay 未验证；历史时钟/资产需匹配 |
| DexWild | 动态真人和真实 LEAP 机器人演示；作为更晚的真实机器人扩量备选 | 未见原生对象 mesh/逐帧 SE3；33+h，官方库约 2.14 TB |
| DexTrack | 已发布 GRAB/TACO retargeted references、资产和部分 checkpoint | reference 不等于真实物理执行；生成 rollout 需额外仿真预算 |
| GraspM3 | 百万级手部 grasp motion sequences，不能简单归为静态 | 未核实完整逐帧物体动态；邮件申请，限制修改/再分发 |

### DexMimicGen：现成动态仿真演示，先区分灵巧手与夹爪

官方仓库实际为 [NVlabs/dexmimicgen](https://github.com/NVlabs/dexmimicgen)，
数据在 [MimicGen/dexmimicgen_datasets](https://huggingface.co/datasets/MimicGen/dexmimicgen_datasets)，
当前 HF 总文件量 59.9 GB。它不是只发布末帧 grasp qpos：官方
[playback_datasets.py](https://raw.githubusercontent.com/NVlabs/dexmimicgen/main/scripts/playback_datasets.py)
读取每个 `data/demo_*/states`、`actions`、`model_file` 和 `ep_meta`，逐状态加载并 `sim.forward()`。
因此可在匹配模型上恢复手 FK、物体 body SE3 和当前相机渲染；是否已带可用 depth 要核验具体文件。

官方 [environments.md](https://raw.githubusercontent.com/NVlabs/dexmimicgen/main/environments.md)
明确：DrawerCleanup、BoxCleanup、LiftTray 用双 Panda 加灵巧手；Coffee、Pouring、CanSortRandom
用 humanoid 灵巧手；Threading、ThreePieceAssembly、Transport 是双 Panda 平行夹爪。
不能把九个任务全部计成五指手样本，也不能把其他 MimicGen 夹爪库混入本合同。
先取 BoxCleanup/LiftTray 或 CanSortRandom 的少量轨迹，避开关节物体与倒液体的非刚体标签范围。

时钟来自 HDF5 `data.attrs.env_args` 中的真实控制配置；viewer 输出 `fps=20`
只是视频写出设置，不能证明数据为 20/30 Hz。需要确定 states 与 actions 的 pre/post 对齐，
再用真实时间间隔生成 30 Hz 窗口。完整模型点云应明确为 CAD profile；可见 scene 则单独渲染当前 RGB-D。

许可存在明确冲突：[GitHub README](https://github.com/NVlabs/dexmimicgen#license)
写数据 CC BY 4.0，[官方 HF metadata](https://huggingface.co/datasets/MimicGen/dexmimicgen_datasets/blob/main/README.md)
写 CC BY-NC-SA 4.0。记录待澄清并保守采用较严格条件，不能声称许可已统一。

### DexVerse：小体量现成轨迹，但公开资产不等于多手演示

正式源码是 [ycyao216/DexVerse](https://github.com/ycyao216/DexVerse)，依赖 Isaac Sim 5.1 / Isaac Lab 2.3.2。
[HF 数据卡](https://huggingface.co/datasets/dexverse/DexVerse_release)目前说明 baseline demos 公开，
1,900 条、每个已发布 task/version 50 条，库总量约 4.09 GB；它与 GitHub README 的
“demonstrations coming soon”文字存在发布快照差异，应以实际清单核验。
CC BY 4.0 仅覆盖 curated manifest 中的文件；机器人、物体资产和源码保留各自原许可。

[release_manifest.json](https://huggingface.co/datasets/dexverse/DexVerse_release/raw/main/demonstrations/release_manifest.json)
列 `episodes`、任务、文件、`benchmark_revision` 等，但没有每 episode 的 hand identity。
`simulation_replay_verified=false`，设备字段为未知；OpenDoor v0/v1 没有演示。
HF 列出 Inspire/Allegro/LEAP/Shadow 等 robot asset bundles，**不能据此宣布这些手均有动态演示**。
默认任务使用 Shadow tabletop 配置；实际 hand 必须读取真实 pickle 的 `robot_type` 并核对实体。

[record_demos.py](https://raw.githubusercontent.com/ycyao216/DexVerse/main/scripts/record_demos.py)
保存 `robot_type`、任务、USD 等 metadata，`episodes[].states` 是 T+1 个 scene snapshots。
Isaac Lab [get_state](https://raw.githubusercontent.com/isaac-sim/IsaacLab/v2.3.2/source/isaaclab/isaaclab/scene/interactive_scene.py)
给出 `states[t]['rigid_object'][entity]['root_pose']` 与手的
`states[t]['articulation'][entity]['root_pose'/'joint_position']`。
`is_relative=True` 只减去 env_origin，不是手腕坐标；需统一米制锚点和四元数约定。

[create_demo_files_sequential.py](https://raw.githubusercontent.com/ycyao216/DexVerse/main/scripts/demo_tools/create_demo_files_sequential.py)
的 `--set-state` 分支要求 T+1 状态，通过 `scene.reset_to(..., is_relative=True)` 恢复，并刷新 scene/sensors。
匹配任务和资产后，FK、逐帧对象位姿与当前 RGB-D 可以由此恢复；不能把只导出的 policy obs
视作完整几何，也不能把 goal/target marker 混入真实场景输入。

当前 [dexverse_base_env_cfg.py](https://raw.githubusercontent.com/ycyao216/DexVerse/main/source/dexverse/dexverse/tasks/dexverse_base_env_cfg.py)
为 `decimation=2`、`sim.dt=1/120`，控制频率 60 Hz；30 Hz 可隔帧取样。
该值不自动证明历史 v0 记录频率，要与 `benchmark_revision` 和记录 metadata 匹配。
首选 GraspCup/GraspPan/BimanualLiftTray 的小份只读状态检查，而非关节工具任务。

局部物理风险：当前
[grasp_bleach_cfg.py](https://raw.githubusercontent.com/ycyao216/DexVerse/main/source/dexverse/dexverse/tasks/config/functional/grasp_bleach_cfg.py)
显式 `object_collision_enabled=False`。这要求核查该任务/version 的记录语义，不能推广到所有
DexVerse 任务，也不能按今天源码倒推历史演示有实现错误。状态回放可提供几何轨迹，
不能单独证明 action-driven replay 或接触物理成功。

### DexWild：真实机器人连续轨迹，缺对象侧原生标签

[官方项目](https://dexwild.github.io/)链接
[boardd/dexwild-dataset](https://huggingface.co/datasets/boardd/dexwild-dataset)，数据卡报告
9,505 episodes / 33+h / 3.5M+ transitions，其中机器人分项合计 1,588 条；项目页的
9,290 human demos 与论文机器人计数是不同快照，不能相加当当前发布量。
HF 列 MIT；全库约 2.14 TB，超过本仓库总产物预算，不能全量下载。

发布格式有 MANUS/LEAP 状态、wrist tracker、手掌相机、ZED 图像和 timesteps。
这些足以确认动态演示，不足以确认已有物体 CAD/逐帧 SE3；当前官方 schema 未列对象侧字段。
要接完整 WAM 仍需对象重建/跟踪 teacher、相机/固定世界校准和 hand FK。
喷瓶/倒水可以筛刚体容器；花材/衣物应留在当前刚体范围之外。先不为扩量购买全库处理成本。

## 其他来源与排除理由

- [DexTrack](https://github.com/Meowuu7/DexTrack)发布的是 retargeted GRAB/TACO references，
  支持 floating Allegro 与 Franka+LEAP；输入参考和物理 tracking 输出要分开。
  checkpoint 可生成机器人 rollout，但不是已核实的现成全量 rollout 库；与已有 GRAB/TACO 分组去重。
- [GraspM3](https://github.com/lihaoming45/GraspM3)报告 1,364,360 手部 grasp trajectories / 8,152 meshes，
  v2增加 simulation success 与对象 rotation；没有据此确认逐帧对象 SE3和30Hz。
  邮件申请需真实身份/导师资料，非商业且禁止未经授权修改/再分发，本轮未提交申请。
- [DexGraspNet](https://github.com/PKU-EPIC/DexGraspNet)与
  [DexGraspNet2](https://github.com/PKU-EPIC/DexGraspNet2)主要发布静态 grasp/scene/graspness，
  能提供形状/末姿态先验，但没有直接满足 future24 的动态交互标签；两者数据均 CC BY-NC 4.0。
- [DexData/Dexbotic](https://github.com/dexmal/dexbotic/blob/main/docs/ModelZoo.md)是通用机器人数据格式/模型生态，
  文档中的 arm+gripper 字段不能证明已有逐指五指手和对象 SE3；仅因名称含 Dex 不优先接入。
- [DexWM](https://github.com/facebookresearch/dexwm)公开 RoboCasa random 数据与代码，
  官方目录明确包含 `gripper_open_and_close`，不把方法名 dexterous 当作五指手数据来源证据。
- UniDexGrasp++ 已有本地动态采集与 provider；它比新增静态抓取库更接近训练准入，
  真实状态和当前限制见[已有接入笔记](2026-10-14-unidexgrasp-admission.md)，此处不重复增加“新数据集”数量。

## 真实手物几何来源

### GigaHands：优先补真实物体覆盖，但只取标注交集

官方全库口径为34小时、56人、417对象、约14k手动作片段；实际发布物体motion约3.3k。
手、物体pose/mesh、相机与对应表有官方入口，还提供1段/5段demo，适合先核验reader。
原 `keypoints_3d_mano` 被归一化和重新居中；应取 `keypoints_3d_mano_align` 或核验原MANO
与物体坐标的一致性。下载走Globus，CC BY-NC 4.0。[官方发布说明](https://github.com/brown-ivl/GigaHands)

对象meta有 presence/tracking success；官方报3356个成功tracking序列。非刚体/关节物的
success可能只按mask IoU>60%计，不能据此认定SE3准确解释每个表面点。先取明确刚体，
并用原始pose索引统计连续H4+24覆盖，不按全库手片段数推算双头窗口。
[对象标注合同](https://github.com/brown-ivl/GigaHands/blob/main/README_object.md)

官方可视化默认对object pose插值、居中9帧平滑，会读取anchor后的信息；不要直接用该
处理结果制作观察H4/速度/加速度。缺帧与离线估计来源须保留，观察和标签独立生成。
可先沿现有完整CAD路线解析物体点位移；真实背景scene/遮挡另行制作。
[可视化源码](https://github.com/brown-ivl/GigaHands/blob/main/render_mesh_video.py)

### HOT3D：ego刚体与对象身份较齐，先做CAD/pose provider

官方833分钟、19参与者、33个刚体，30fps；手、物体与相机pose、扫描CAD均有标注，
适合补ego交互和对象身份。多视角370万以上图像不能作为370万个独立时间点。
Clips为3832个5秒片段，有官方toolkit；下载要按所选Sequence、Hand、Object组件许可办理。
[官方论文](https://arxiv.org/abs/2411.19167)、[项目及下载](https://facebookresearch.github.io/hot3d/)

物体未来query可由刚体pose解析，省去CoTracker；当前CAD profile也能先独立接入。
如果要求真实可见背景scene，不能假定存在原生dense RGB-D，也不能使用全段SLAM地图
冒充当前观察。Aria RGB与Quest灰度、鱼眼相机模型、手标注有效性分开核验。
[官方格式/toolkit](https://github.com/facebookresearch/hot3d)

### TACO：双手多对象与工具交互，数据角色需明确

V1共2317序列，其中2212列为完整ego模态；有RGB-D、双手MANO、tool/target模型和pose、
camera及自动2D分割。先取train的明确刚体操作，保留官方泛化test。
[官方发布清单](https://github.com/leolyliu/TACO-Instructions)

`target`是被工具作用物，不必然是当前要预测的手持对象；例如勺子/碗应按预定任务角色
选择slot，不能根据未来位移挑对象。官方投影代码将mesh乘0.01、读取world→camera外参；
深度解码、实际时钟和可见性须由样本复核。原生物体pose可替换tracking teacher，当前depth
可替换单目teacher，这是字段适配推断，还没有本仓库实测。
[投影代码](https://github.com/leolyliu/TACO-Instructions/blob/master/dataset_utils/project_pose_to_egocentric_view.py)

### 小规模对照与暂缓项

HO-Cap有RGB-D、手物pose和模型，CC BY 4.0，可按subject分包；最小subject包3.2GB，
另需共享labels/models/poses。固定RealSense路径能减少depth/目标tracking重建；头戴视角
不自动有同像素dense depth，实际时钟仍需检查。[官方分包](https://irvlutd.github.io/HOCap/)、
[reader](https://github.com/IRVLUTD/HO-Cap/blob/main/hocap_toolkit/loaders/sequence_loader.py)

H2O/DexYCB保留作为已有调查中的原生几何工程对照。HOI4D对象/房间覆盖丰富，但15Hz→30Hz
会改变标签来源，须显式重采样profile，不能把插值叫真实30Hz。SHOW3D的发布world是随人
移动的背部rig frame，固定世界运动仍缺；目前只推荐投影/身份审计，不直接混入现有位移合同。
[HOI4D](https://github.com/leolyliu/HOI4D-Instructions)、[SHOW3D坐标合同](https://github.com/facebookresearch/show3d)、
[已有对照集详情](2026-10-14-shared-wam-dataset-survey.md)

## 大规模视频：保留扩量方向，先补物体侧

Open-AoE官方nano约3h、tiny约100h已发，full 2000h仍上传；有30fps双手MANO和米制相机，
camera_traj不含depth，未列原生物体6D。当前本地手/相机检查能复用，但物体teacher、当前
目标解析与观察边界仍是缺口。保留备选，尊重已暂停的扩展。
[官方数据卡](https://huggingface.co/datasets/inclusionAI/OpenAoE-2000h)、
[已有本地Probe](../experiments/probes/P-20261014-openaoe-local-admission.md)

EgoSuite作者公告首批10000h，100000h是总体计划；当前EgoStandard卡仍标planned duration，
完整库分发于需审批Bucket，实际manifest才是发布量依据。手21点/相机有格式，depth可选；
“raw MCAP”不代表手pose为无误差采集真值。已有访问条件后优先核验hand+camera+depth子集，
本轮未代填身份或申请。[作者公告](https://huggingface.co/blog/LightwheelAI/egosuite-open100k)、
[数据卡](https://huggingface.co/datasets/LightwheelAI/EgoStandard)、
[手格式](https://docs.lightwheel.net/egocentric_data/mcap_data/pose/hand)

EgoVerse/WIYH/EgoSteer继续作为已有调研中的千小时备选或派生手标注；不把可选/占位手字段
当完整骨架，不将派生集与EgoDex/OakInk2原视频重复计数。纯视频大库增加teacher成本，当前
已有EgoDex目标误选及VAL准入未解决，暂不优先接更多RGB-only库。
[已有大规模来源核验](2026-10-14-shared-wam-dataset-survey.md)

## 下一步最小可判别工作（未执行）

先选一个新来源，做**Blocker/Decision接入Probe**。它检验“原生状态/几何标签能否低成本
生成独立TRAIN/dev的完整手物窗口”，结果决定接入该来源还是先修reader/换候选。
最便宜的阶段是只读release索引与少量完整episode，核查时间、单位、手型、mesh、pose，
再从至少两个独立录制/示范组各取10–25个窗口，展示当前/+8/+16/+24同点轨迹。

DexMimicGen先按单个真实灵巧手任务核验；GigaHands先按明确刚体/成功tracking/手物字段
交集核验。DexVerse先做小文件元数据检查，确定手型与记录版本，再决定是否值得部署其
IsaacSim环境。首份工程窗口不要启动全库下载、重建或训练；数据读取和统计使用CPU，模型
计算/批量FK和可加速仿真按实际收益优先GPU。

source通过时再接双头工程smoke，然后建立独立matched短训Probe，比较已有三域底座与
固定比例新来源混合。先看held-record/object的first8/middle手物误差与原域退化；endpoint
goal已作为输入，target h24低误差不能独立证明动力学学习。控制收益仍需RL660/Inspire上的
实际策略比较，异构手action不直接进现有R。

停止条件：记录时钟无法确定、H4/标签错位、刚体query身份不可解释、有效字段交集过小、
目标物或手型语义不可信。缺帧保持invalid，不能靠平滑/复制或放宽门制造合格率。
正式采集/训练前在Task-local实验卡写预算并核对全局六卡/400GB限制；现场报告的资源是
历史快照，不能替代启动时检查。HF获取遵循ModelScope→国内镜像→代理的顺序。
