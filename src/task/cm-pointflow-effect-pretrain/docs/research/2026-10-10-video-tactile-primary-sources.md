# 大规模视频与触觉：PointWorld 数据接入主源核查

日期：2026-10-10。所属 Task：`cm-pointflow-effect-pretrain`。
范围：官方论文、项目、源码、数据卡与公开文件清单；本次没有下载视频/训练数据、启动训练或修改 ref9。
已阅读 Task README、ref9、REF6_OBJECTFORESIGHT_EPIC、HOCAP_TEST_PREFLIGHT 和新合并 EPIC 实验卡。
`paper/README.md` 是现有论文产物入口，本次没有发现对应数据源的独立文献笔记。
小文件证据保存在工作树 `tmp/video-tactile-primary-sources/`，不提交该运行目录。

## 影响当前决策的判断

**建议：保留 OakInk2/GRAB/ARCTIC 的物体运动主监督；先把已有 EPIC 候选转成可审计的弱点轨迹辅助源；EgoTouch 先做独立触觉 schema 审计。**
这不是“把所有视频都塞进现有 loader”：当前模型没有 RGB 输入，预测每个对象的 SE(3)，再解析传播固定 canonical points。
大规模视频、可靠的 metric 3D object supervision、真实触觉是三种不同资产，数量不能互相替代。
尤其不能把弱数据的重建自一致性残差解释为真实毫米级 GT 精度。

| 来源 | 真正可用的监督 | 可以服务的近期目标 | 当前不能默认成立的事 |
| --- | --- | --- | --- |
| OakInk2 | 双手 MANO、物体模型、逐帧 `obj_transf`、多视角图像/相机、任务 program | 当前 action/hand-conditioned object motion 主源 | 发布 preview 及历史缺上传问题使版本固定仍必要；没有真实 pressure |
| GRAB | marker-based 拟合手/人体与刚体物体轨迹、扫描 mesh、几何计算 contact | 当前固定点/SE(3) 主源 | contact 是解析几何量，不是实测压力；公开展示视频不是大规模自然 RGB 视频监督 |
| ARCTIC | 同步双手与关节物体 mesh/pose、动态几何 contact，2.1M video frames | 当前刚体部件运动及 articulation | 整个关节对象不能强行按一个刚体 SE(3)；contact 不等于 force |
| HOCap | 多 RGB-D、重建 mesh、优化/半自动手物 pose | 现有冻结外部测试 | 不是 MoCap；本 Task 已约定 TEST_ONLY，不能为了规模纳入训练或选 checkpoint |
| EPIC-Contact | 人工核验中心帧 contact/拟合 geometry，其余帧随 WiLoR 传播 pseudo-GT | 接触位置先验、弱手动作/几何辅助 | 非中心帧不是独立测量 object dynamics；不能证明手动作因果效应 |
| ObjectForesight-EPIC | 视频提取 masks、估计 depth/camera、重建 mesh、估计 6DoF | 大规模 object-only 或点轨迹弱监督候选 | 原生 3D hand action 缺失，尺度/相机方向/时间 gaps 仍需审计 |
| EgoDex | 829h/30Hz 视频，ARKit 手/上体/head/camera transforms 与部分 confidence | 手动作表示、视觉观测到手轨迹先验 | 官方字段没有 object mesh/逐帧 object 6DoF，也没有真实 tactile |
| EgoTouch | RGB、双手手姿、归一化手套传感值、head/wrist tracker | 独立 hand/tactile 表示及未来触觉预测 | 不提供当前完整的 object geometry+motion 标签；pressure grid 不是441个独立物理 taxels |

上述合同分别来自 [OakInk2 toolkit](https://github.com/oakink/OakInk2#dataset-format)、[GRAB 项目](https://grab.is.tue.mpg.de/)、[ARCTIC 项目](https://arctic.is.tue.mpg.de/)、[HOCap 项目](https://irvlutd.github.io/HOCap/)、[EPIC-Contact 官方卡](https://huggingface.co/datasets/Sid2697/epic-contact)、[ObjectForesight 官方卡](https://huggingface.co/datasets/raivn/ObjectForesight-EPIC)、[EgoDex 官方 README](https://github.com/apple-aiml-research/ml-egodex)、[EgoTouch README](https://github.com/Jianyi2004/TouchAnything)。

## EPIC 为什么值得先做，但不能直接加进当前主 loader

**事实：** EPIC-Contact 官方区分一张中心帧的人工核验标注和其他帧的 propagated pseudo-GT；后者绝对 camera placement 近似，随附 quality scores。
项目说明 clip pose 通过 WiLoR 手运动传播。[官方卡的 two label tiers](https://huggingface.co/datasets/Sid2697/epic-contact#two-label-tiers)、[官方 annotation pipeline](https://sid2697.github.io/epic-contact/index.html#dataset)

**推论：** 用传播后的手动作预测传播后的物体轨迹，可能学到标签生成器的绑定规则。
这仍可作为抓持 transport 先验，但不能用于宣称真实接触动力学或未来动作的干预效果。
即使未来手打乱后误差恶化，也只能说明预测模型利用了该输入及其与标签的相关性。

**事实：** ObjectForesight 将大量视频提取成 pseudo object trajectories；论文的“2M+”是生成的短窗口，不是2M独立对象或采集 episode。
官方提取管线使用 EgoHOS/SAM2/SpaTrackerV2/TRELLIS/FoundationPose，原生发布并没有本 Task 所需双手 semantic future action。
源帧和重初始化段必须留在 manifest；尺度、camera direction 不能只照 metadata。
[论文](https://arxiv.org/abs/2601.05237)、[官方卡](https://huggingface.co/datasets/raivn/ObjectForesight-EPIC)

**本仓库已有证据：** 新合并的 `P-20261008-epic-scene-flow-p03-03` 是数据处理 Probe；P03_03 在转换器现有 side-valid 政策下报一个 H4+K24 window，P03_13 因手帧稀疏没有该窗口；下文的独立 review 发现前者也未满足逐语义点质量合同，不能解释为训练就绪。
卡片明确 scene label 是 RGB/depth pseudo-label，手估计还含插值；静态 reprojection residual 是自一致性检查。
其 `scene_flow_candidate.npz` 包含 `scene_points_world`，由背景重投影和物体 mask 内光流/depth 构造，不是当前对象实例的 canonical 512 points+SE(3)。
[本地实验卡](../experiments/probes/P-20261008-epic-scene-flow-p03-03.md)、[converter](../../tools/audit/convert_epic_scene_flow.py)

**实现决策：** 保留 source ID、帧号/实际时钟、有效 mask、point track identity 和 label provenance；先支持显式 `weak_scene_point_flow` loss/adapter。
背景静态点与运动前景应分别采样/计分，避免全场景低误差被8192个静态点支配。
若坚持现有 object-SE(3) decoder，则先按实例/刚体 segment 拟合并审计残差，不能将非刚体轨迹硬标成刚体 GT。
直接点轨迹 decoder 是另一个实现选择，需要单独 Probe；本报告没有决定改架构或把已有候选升级为训练就绪。

## EgoTouch 的实际 schema 与 ref9 修正

**官方规模声明：** 1,891 episodes、208 tasks、约20h/2.1M frames、30Hz、3路 RGB。
这些数字描述采集/论文规模；下一节的公开 release 清单并不等于这个声明的完整性证明。
[论文 §2](https://arxiv.org/html/2605.13083v1#S2)

### 传感值不是统一物理 pressure image

论文附录说明每手原始256路8-bit传感值来自异步串口；物理布局不是规则16×16图片。
映射后是稀疏21×21手形图，无效位置为 NaN，右手水平镜像以统一 canonical layout。
预处理包括可选首帧 baseline subtraction、右手坏列插值，以及 tactile/bending sensors 分别归一化。
因此其中包含 bending-related signal，归一化值没有校准为 Pa 或 N。[论文 Appendix 8.1–8.2](https://arxiv.org/html/2605.13083v1#A1)

本次固定 TouchAnything 源码到 `d74f9ef5c189a957b7ff72781a0c998e41b45a56`，实际取得的左右 grid mapping 均为217项。
它们把 `"row,col"` 映射到 raw sensor index；**217项不证明217个独立、等面积、同类型压力传感器**。
[left mapping](https://github.com/Jianyi2004/TouchAnything/blob/d74f9ef5c189a957b7ff72781a0c998e41b45a56/configs/pressure_position_mapping_left.json)、[right mapping](https://github.com/Jianyi2004/TouchAnything/blob/d74f9ef5c189a957b7ff72781a0c998e41b45a56/configs/pressure_position_mapping_right.json)

converter 从 `pressure_grids.npz` 读取 `left_pressure_grid/right_pressure_grid`，要求长度等于 video-derived T。
`pressure/attrs` 写入 `grid_size`、`separate_normalization`，以及 `tactile_max_left/right`、`bend_max_left/right`。
只看到最大值字段，还没有确认 normalization maxima 的计算范围、全部 taxel 的 bend/contact 分类、面积、力响应曲线。
必须保存并查明这些信息，才能决定哪些 cell 可以当 contact target。
[固定 converter](https://github.com/Jianyi2004/TouchAnything/blob/d74f9ef5c189a957b7ff72781a0c998e41b45a56/scripts/core/convert_to_hdf5.py#L591)

官方另有可视化 `ta_to_mano_mapping_left_visual.json`，`coordinate_space` 明示 display grid/right flipped，`positions["r,c"].mano_vid` 是 vertex list。
列出的 vertex IDs 大于标准 MANO 778 vertices，说明不能直接拿标准 MANO 索引使用；其细分 mesh身份/映射方法仍未核验。
它不是面积/法向/pressure-to-force标定。[官方可视化映射](https://github.com/Jianyi2004/TouchAnything/blob/d74f9ef5c189a957b7ff72781a0c998e41b45a56/scripts/tools/mano_visualization/ta_to_mano_mapping_left_visual.json)

### 手姿、世界系与时钟不能仅按字段名认定

HDF5 主字段为 `wilor_left/right_joint_xyz (T,21,3)`、`wilor_left/right_valid (T,)`。
它们是 WiLoR 图像估计，不能称为 MoCap GT；converter 的 valid 只做形状、有限性和很宽的异常值检查，并非准确度置信度。
legacy `left/right_joint_xyz` 在有 Rokoko JSON 时复制其值，无数据时填零；README 的“placeholder”不是每个文件都无数据的证明。
`poses/chest_pose,left_pose,right_pose` 来自 Vive tracker trans/rot；tracker pose不自动等于光学相机 pose，需要外参和轴/quat约定。
本次源码不能确认 WiLoR joints 已与 Vive 处于同一个 metric world frame。
[converter 的手姿与 tracker 写入](https://github.com/Jianyi2004/TouchAnything/blob/d74f9ef5c189a957b7ff72781a0c998e41b45a56/scripts/core/convert_to_hdf5.py#L778)

论文 acquisition 是软件同步：异步 readers 缓存最新 valid sample，再按30Hz collection tick保存 snapshot。
共享 row/frame_index 说明保存索引对齐，不能保证相机、pressure、hand 测量瞬间无延迟或无重复。
本次取得的 converter 取三视频最短长度 T；`timestamps = arange(T)*(1000000//30)` 是重新生成的名义微秒时钟。
其 `load_wilor_jsonl` 按行号 i 写数组，未按原 `frame_index/ts` join；因此审计要用原始 JSONL 检查 gaps/重复/坏行。
[论文同步策略](https://arxiv.org/html/2605.13083v1#A1.SS1)、[converter 时钟](https://github.com/Jianyi2004/TouchAnything/blob/d74f9ef5c189a957b7ff72781a0c998e41b45a56/scripts/core/convert_to_hdf5.py#L690)

### `joints_worldspace` 整理版是独立派生 release

MIT-Media-Lab 的卡描述 `joints_worldspace`、world-to-camera extrinsics、MANO及逐帧 tactile/video索引；自行切分、单手 episode 和VLM instruction不是官方录制单位。
卡报告 pressure/video 长度在其1,566 source scenes一致，这是发布者报告，本次没有独立复算其数据内容。
当前 v1 卡的2026-09-08 revision 修复旧版无效手帧被吞入 episode，改为连续 valid run；现为107,364 episodes/3,123,675 samples。
[派生 release 主源卡](https://huggingface.co/datasets/MIT-Media-Lab/egotouch-annotations-v1#revision--2026-09-08-re-cut)

`leftfix` 修复左手 rotation convention，称 joints/translations/tactile 不变，但仍列旧版111,159 episodes/3,687,389 frames。
**不能默认 leftfix 已包含后续 validity re-cut 修复**；使用前要核验 immutable revision、parent hashes、episode IDs及连续 valid frames。
这些派生 sample counts 含左右手切分/重叠，不能与官方约2M frames相加解释为新采集量。
[leftfix 卡及 audit hashes](https://huggingface.co/datasets/MIT-Media-Lab/egotouch-annotations-v1-leftfix)

## 公开下载完整性：本次实际检查

固定官方 raw Hub 到 `cfdbb0ac31cc2af4247943820aa250575e7e6637`；API为public/ungated，无 `cardData`。
分页读取26页文件元数据，共25,048 entries、22,897 files、88,508,555,957 bytes（88.51GB，约82.43GiB）。
其中1,933个录制目录均列3路视频、pressure grids、pressure/Rokoko/Vive JSON；1,930列 WiLoR，3个缺 WiLoR。
全录制目录未列 `camera_matrix.txt`；它是否另存在未公开校准包尚未知。
目录存在/非零大小不能证明正确、同步、decoded frame完整或 hand validity。
[固定文件树 API](https://huggingface.co/api/datasets/zhouzhoujy/EgoTouch/tree/cfdbb0ac31cc2af4247943820aa250575e7e6637?recursive=true&expand=false)

同 revision 的 `split.json` 为作者本机绝对 HDF5 paths；按末三层 `category/task/recording` 规范化后有2,228互斥记录。
train1,665 / val208 / test_seen208 / test_unseen147，其中207/34/29/62在当前 raw recording清单未匹配（共332）；另37个raw不在split。
这个匹配只能证明 manifest与当前raw命名清单不一致，不能确定是缺上传、改名、旧split或预处理剔除。
不要用split绝对路径直接运行，更不能将所有raw重新随机划窗造成录制泄漏。
[固定 split.json](https://huggingface.co/datasets/zhouzhoujy/EgoTouch/resolve/cfdbb0ac31cc2af4247943820aa250575e7e6637/split.json)

官网仍留 upload-in-progress 提示；因此 ref9 的“先5–10 episode”合理，但不应假定公开包恰好完整覆盖论文1,891 episodes。
TouchAnything代码MIT许可不自动代表录制数据的许可；官方raw Hub没有dataset card/明确data license字段，本次尚未确认数据许可。
派生MIT-Media-Lab卡写CC BY-NC 4.0，但派生许可不能自行扩大原始录制权利。
EgoDex官方数据写CC BY-NC-ND；EPIC-Contact/ObjectForesight为CC BY-NC且有EPIC/MANO附加条件；不同来源不要写成统一MIT数据集。
[EgoTouch代码/上传状态](https://github.com/Jianyi2004/TouchAnything)、[raw Hub](https://huggingface.co/datasets/zhouzhoujy/EgoTouch)、[EgoDex license说明](https://github.com/apple-aiml-research/ml-egodex#license)

## 触觉到 Isaac 与部署缺失模态

NVIDIA源码把 `acquire_net_contact_force_tensor` 组织为 `num_bodies×3`，每刚体的3D net contact force。
这与归一化、稀疏、部分bend混合的手套grid不是同一随机变量。
[NVIDIA Factory 源码](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs/blob/main/isaacgymenvs/tasks/factory/factory_base.py#L151)

**物理推论：** 即使有Pa值，压力积分到force也需要surface area、normal、接触body归属及shear信息；net合力还会丢失分布和抵消方向。
因此不能把217/441 grid cells reshape成5×3 force，或用统一标量归一化宣称物理对齐。
可先做finger/palm region tokens，共享contact event/relative activation表征，保留source type与validity；pressure、bend、sim-force各有专属head。
不能将视觉估计的 tactile 或几何distance-generated contact称为实测 tactile。

**部署决策：** 先确定policy实际能观测哪些量；缺 wrist RGB/real tactile 的部署，训练输入也要提供显式missing mask与相同缺模态评估。
最便宜路径是将真实 tactile用于辅助输出/teacher，再让部署模型仅用已有history+候选手轨迹；不要让效应预测依赖部署根本拿不到的 C_t。
TouchAnything view dropout只支持其ego始终保留、随机缺wrist的设定，不能据此宣称支持无RGB、无pose或无pressure的当前PointWorld。
没有随机动作干预和robot执行，未来手轨迹+触觉预测优于shuffle也不证明改善抓取RL；最终仍回到MISSION的matched Cm-on/off。

## 最小下一步与停止条件（建议，未执行）

1. **EPIC Decision：** 已有34 clips中先按原视频/subject划分，检查source identity、clock/gaps、hand连续valid、实例与point身份；统计真实运动前景和完整H4+K24窗口数。
   能稳定产出多个独立有效窗口才做小GPU matched辅助loss Probe；否则保留candidate，不用大训练掩盖数据缺口。
2. **EgoTouch Blocker：** 先固定许可/公开版本、核对split/raw，选择train中具备必需文件的5–10个录制做schema审计；不要先全量HDF5展开。
   核验JSON帧索引/传感快照重复、video长度、WiLoR world变换/validity、pressure NaN/坏列、bend mask、左右镜像/region映射及normalization范围。
3. **Tactile Decision：** schema通过才比较history-only、history+future-hand、matched shuffle；未来24步target，切分按source recording，单独检查onset/offset和活跃区域。
   先问手trajectory是否提供可重复的预测增量；输入不能看future pressure/RGB，loss不能把NaN/bend当contact或被大量零格主导。
4. **Evidence 延后：** 多seed、真实Pa/N标定、完整EgoTouch object reconstruction、robot tactile硬件与causal策略收益；它们当前不应阻塞EPIC的最低成本数据审计。

本次结果是数据可用性与方法适配调研，不是新训练Probe或正式科研结论；没有将触觉路线或弱视频预训练标为SUPPORTED。

## 本地数据与独立实现审查（root 核验）

本节以当前源码和实际文件为依据，不把历史卡片的工程通过当成训练资格。
合并提交为 `a039077b2f5ec4c4349b0708353c6755db19bfde`，保留完整 EPIC 分支
历史；旧工作树32个产物文件共25,391,996bytes已哈希核对后迁入本工作树。
旧路径保留兼容软链接，分支未删除。

### 可以立即沿用的三源数据

root 核对 sharedstats 的现行 mixed manifest：
`outputs/cm-pointflow-effect-pretrain/pointworld-main3-sharedstats-20261009/data/processed/manifest.json`。
其SHA256为 `babaf17fcb3e414472406247adb9d7abbf95bdbea86158372df4f053ab479c58`；
三个 source manifest 的哈希匹配，原始 source 根目录和所有split索引均存在。

| 来源 | train anchors | val anchors | test anchors | 当前定位 |
| --- | ---: | ---: | ---: | --- |
| OakInk2 | 3,860,393 | 656,368 | 543,855 | 现行主监督 |
| GRAB | 207,179 | 18,291 | 41,916 | 现行主监督 |
| ARCTIC | 310,906 | 41,030 | 0 | 现行主监督；公开test在本产物中为空，不补造 |

合计4,378,478个train anchors，数量包括重叠窗口，不能视为同等数量独立episode。
OakInk source manifest含627条序列；GRAB/ARCTIC共享native manifest共1636条，
不能把1636分别作为两来源的序列数。采样权重由manifest给定的5/9、2/9、2/9
决定；配置中的.6/.2/.2是source内部类别采样。
本次仅核对文件和manifest，不重新审计所有原始几何或宣称跨域泛化已验证。
本地核对摘要：`tmp/video-tactile-data-audit-20261010/local-main3-sources.json`。

### EPIC 目前仍被哪些具体合同阻塞

1. **来源标识错误已定位：** converter将所有`audit.source_clip`硬编码为
   `P03_03_23`，P03_13也被误标。实际输入目录basename分别为`P03_03_23`与
   `P03_13_12`。旧输入SHA不同，不能推断旧tensors用了同一片段；旧audit保留，
   来源应按文件SHA/原始文件回溯。最小修复记录实际scene目录、source起始帧及
   sampled FPS，仍不生成training manifest。
2. **逐点手validity缺失：** `load_contact_hand`只用side-valid，没有使用
   `joints_valid_*`或quality CSV。root在两份原始NPZ独立复算semantic11：
   plate有37个无效语义点，仅20/30 source rows全11点有效；bottle有29个，
   仅15/30全有效。plate最初10行存在无效点，所以原28/28手帧和1窗口只适用于
   宽松side-valid政策；目前不能作为严格训练窗口。
3. **来源split冲突：** 两份对象都列在ObjectForesight train，但plate属于
   Contact train、bottle属于Contact test。不能只继承ObjectForesight split；
   当前bottle不得用于Contact train接入，原始视频/参与者与重叠帧一起隔离。
4. **时钟未精确对齐：** 当前每2源帧采样实际为29.97002997Hz，timestamps
   保留真实source_fps，audit30Hz仅名义值。P03_13 `action.meta.json`的
   start_frame是1929，而旧窗口表现与传入1927一致；元数据还含pad=5，
   起点/padding语义尚未恢复，不能猜一个数字后称已解决同步。
5. **point identity不完整：** LK失败后保留旧坐标却推进previous图像，后续
   可重新valid；没有验证重获物理点身份。两份候选的采样后moved masks分别出现
   12/19次invalid→valid转换。应永久失效或经验证后开启新track身份。
6. **背景误差会掩盖运动：** static分支每帧重投影后再采depth，后续帧未排除
   hand/object mask，不能等同于精确静止点。两份候选static点相邻步长p95约
   8.23/9.12mm，已有非零伪运动；8192背景点相对187/86运动点更会支配平均loss。
7. **伪动作与标签共享来源：** scene branch未读FoundationPose/TRELLIS的
   object运动标签，但hand canonicalization用了Contact的object姿态和shape，
   并依赖每个未来帧的物体depth fit放置。未来hand与未来object flow共享标签
   生成来源；shuffle退化也不能单独证明真实动作因果信息。
8. **训练接口不兼容：** 当前`MixedWindows`限定已注册三/四来源与严格30Hz
   的poses/canonical/index，decoder为逐对象24步SE(3)。EPIC的逐point future
   validity/occlusion不能通过改manifest名字接入。

对应源码：[converter](../../tools/audit/convert_epic_scene_flow.py)、
[Contact audit](../../tools/audit/audit_epic_contact_pair.py)、
[当前多源loader](../../src/oakink_wm/multisource.py)。
对应旧运行：`outputs/cm-pointflow-effect-pretrain/P-20261009-epic-scene-flow-p03-03-auto/`
和`P-20261009-epic-scene-flow-p03-13-auto/`。
以上是当前两片段和当前converter的实现/证据限制，不是“EPIC方法无效”的科研结论。

### 当前路线选择

本地计算沿用已有三源与sharedstats checkpoint；先保留严格measured-object评价，
不要通过加入不合格pseudo-label追求训练规模。若要投入视频，最低成本下一步是
让已有片段的source-split、手逐点质量、采样时钟和持续point身份达到明确合同，
再决定是刚体适配还是独立point-flow辅助head。
触觉可并行做5–10个train录制的schema小审计，先分清pressure/bend与时间/空间
语义；仅下载审计需要的label文件及必要视频，不先完整88.5GB下载或解压。
未来任何数据获取先检查ModelScope，再HF国内镜像，最后用户代理；模型计算
优先空闲GPU，纯文件/schema统计使用CPU。

触觉路线建议先用auxiliary target/teacher保留部署时无需真实手套的输入合同。
单独EgoTouch head通过，不代表把它拼到物体head会有效；共享encoder和robot
contact融合仍要matched小Probe检验，不能用跨数据集非配对样本伪造联合标签。
当前尚未实现新head、改变模型架构、启动新训练或进行大下载。

### 同日后续：修复与真实标签抽样已完成

上述问题清单描述修复前状态；最新可执行证据见
[EPIC readiness卡](../experiments/probes/P-20261010-video-data-readiness.md)和
[EgoTouch schema卡](../experiments/probes/P-20261010-egotouch-label-schema.md)。

EPIC converter现为v2：使用semantic11逐关节/finite/quality有效性、两端插值
支持域、原始train/test划分、metadata时钟，以及永久失效的LK身份；背景每帧
排除hand/object遮挡。上游
[step1_split.py](https://github.com/RustinS/ObjectForesight-Data/blob/main/step1_split.py)
先算`ts_start-pad`再写入metadata.start_frame，故padding已经包含在1929中，
不能再减5或用Contact首帧1927代替。原始源帧号和local索引分别保存，实际
stride-two时钟为29.97002997Hz；仍未进行严格30Hz重采样。

34个本地clip中14个P01_03 clip缺少action.meta时钟，另20个有时钟。两份
Contact形成4个时间重叠候选，95个H4+K24起点中0个通过严格手标签检查。
已验证的plate/bottle首窗口手有效帧17/28、0/28，完整运动track176/74。
仅有运动track不能补足动作条件手输入，也不能直接接入当前刚体decoder。
这阻塞当前本地EPIC子集的联合训练，不反驳视频自监督动力学路线。

EgoTouch从固定raw revision下载两个TRAIN任务的10个标签文件，共485754B，
全部Git/LFS校验通过；ModelScope检索未发现对应release，国内镜像超时后
通过项目代理下载官方源。压力/Wilor/Vive原始frame_index在样本中完全一致，
但Wilor时间戳是相对30Hz合成时钟，而sensor/Vive使用绝对采集时间戳，
mouse实际步长33–34ms。网格每手217/441位置有效，其余是NaN；压力/bend
各自归一化，两个episode的max分别20/30与50/45，不能当跨episode的统一
物理力单位。两个样本均`aligned_to_vive=false`，无手关节confidence/validity，
无camera_matrix，且clip-level左右contact均false。因此样本只能用于数据
接口检查，不能形成world-point监督或证明触觉有效。

下一步选择：视频可先探索不依赖未来手标签的point-track预测，然后单独检查
与native手动作条件表示的迁移；触觉先采contact-positive TRAIN样本、保留
frame_id/未测量mask/pressure-bend区别，并核对RGB与相机关系。训练主来源
仍为OakInk2/GRAB/ARCTIC，现有sharedstats checkpoint和全部旧实验输出保留。

### 同日后续：可运行视频 Probe 与 RGB/传感时钟

上述“未启动训练”和下一步选择属于初始审计阶段。后续
[视频点动力学卡](../experiments/probes/P-20261010-video-point-dynamics.md)
记录 history-only PTv3 点轨迹训练：h24 与独立 h8 各500步，均未通过收益门槛。
窗口起点建轨迹改善训练长期支持，但开发 h24 仍只有一个片段；h8 两片段宏平均
24.808mm 对静止9.464mm。完整源码/目标mask/分片指标经独立最小复算未见
推翻结果的缺陷。此设置 UNPROMISING，视频路线仍 UNCLEAR，不据此扩训。

触觉的两个原始 TRAIN 录制现在各有43/47帧胸前 RGB，与标签原始帧号一致，
相对时间误差<1微秒/0.333ms；仅属接口一致性，不证明物理同步。40任务原始
contact flag全false，不能解释为没有物理接触。217有效格对应左138/右147个
unique raw index，包含重复映射；样本NPZ存global max而固定converter期望
per-hand max，会产生元数据零默认值。后续历史传感输入不能依赖未来整段峰值。
细节与原始文件SHA见[触觉卡](../experiments/probes/P-20261010-egotouch-label-schema.md)。
