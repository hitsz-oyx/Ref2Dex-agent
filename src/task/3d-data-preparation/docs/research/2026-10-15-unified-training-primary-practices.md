# 多源联合训练的实际处理：一手论文与官方代码核验

日期：2026-10-15。定位：帮助决定当前多源 3D 预训练的最小清洗与统一合同；不是复现论文收益的实验结果。
本笔记区分「论文声明」「公开代码明确实现」「未查到或不确定」，避免把统一文件格式等同于物理语义统一。

## 1. UniDex：直接联合 H2O / HOI4D / HOT3D / TACO 的例子

对象是 [UniDex: A Robot Foundation Suite for Universal Dexterous Hand Control from Egocentric Human Videos](https://arxiv.org/abs/2603.22264)，
官方代码 [unidex-ai/UniDex](https://github.com/unidex-ai/UniDex)。注意不同于 UniDexGrasp。
核验版本：`97d869e0f2d1ec0372cd3cdf28dde66b4e3f216d`，官方仓库的稀疏只读快照在
`third_party/UniDex-source-audit/`；仅获取数据 loader、少量脚本/配置和辅助代码，未下载原始数据、模型或大 URDF mesh。
此前 `paper/` 未有 UniDex schema；`paper/dextrack/schema.json` 已有 DexTrack 的公开实现边界记录，本轮不重复建立。

### 1.1 从论文中能确认什么

论文 §3 声明：四源视频按语言指令切成轨迹、过滤无效片段，再生成机器人手的监督；§3.2 使用指尖 IK，
先对「每数据集 × 每机器人手」做 GUI 基础校准，再调整少量接触丰富帧，让接触更合理。视觉侧移除人手点并插入机器人手几何，
按深度顺序重投影。它面向 retargeted robot actions，而我们需要保留 human/object 运动真值；手动改变手基座的位置不能自动视为我们的标签清洗。
[论文 §3.1–3.2](https://arxiv.org/html/2603.22264v1#S3)

**没有在论文中查到**：通用的原始 timestamp 重采样合同、缺帧最大插值跨度、跨源自动 SDF 穿透阈值、统一法线清洗阈值、
逐来源采样权重。论文给出的 30 fps 数据描述不能替代这些未说明的步骤。

### 1.2 代码中的具体统一与清洗行为

| 关注点 | 公开代码明确实现 | 对我们的价值 / 边界 |
|---|---|---|
| 单位与 MANO | HOI4D 用左右独立 MANO layer，`use_pca=False, flat_hand_mean=True`，输出 joints 除 1000 后加 translation；TACO 同样左右独立但 `center_idx=0`，没有相同的 `flat_hand_mean=True` 显式设置。 | 值得学习：每源单独 decode 到米制 joints；不能把每源 MANO profile 设成同一个默认值。[HOI4D adapter](https://github.com/unidex-ai/UniDex/blob/97d869e0f2d1ec0372cd3cdf28dde66b4e3f216d/HandAdapter/dataset/HOI4D.py#L17-L38)、[TACO adapter](https://github.com/unidex-ai/UniDex/blob/97d869e0f2d1ec0372cd3cdf28dde66b4e3f216d/HandAdapter/dataset/Taco.py#L16-L37) |
| 坐标系 | HOT3D 把 world hand points 乘 camera inverse 后，再用 `diag(1,-1,-1)` 改轴；左右手分别读取，缺文件保持 `None`。 | 值得学习显式坐标链和保留 missing。变换本身必须按我们来源的原始定义验证，不能照抄旋转矩阵。[HOT3D adapter](https://github.com/unidex-ai/UniDex/blob/97d869e0f2d1ec0372cd3cdf28dde66b4e3f216d/HandAdapter/dataset/hot3d.py#L124-L145) |
| 关节点语义 | `MANO_TIP_INDEX_MAP` 对 H2O/HOI4D/TACO 用 `[4,8,12,16,20]`，HOT3D 用 `[16,17,18,19,20]`。 | 同样 21 点的 index 也不统一。我们 21→11 必须逐源 semantic map。[映射常量](https://github.com/unidex-ai/UniDex/blob/97d869e0f2d1ec0372cd3cdf28dde66b4e3f216d/HandAdapter/hand_processor.py#L57-L62) |
| 时间降采样 | H2O/HOI4D/TACO 配置 `sample_stride=2`，注释均写 30→15 fps；各 `_build_window` 在 frame indices 上取 stride。 | 值得学习把 stride 放配置，仍不足以证明 native fps。[H2O config](https://github.com/unidex-ai/UniDex/blob/97d869e0f2d1ec0372cd3cdf28dde66b4e3f216d/config/dataset/H2o.yaml#L13)、[HOI4D config](https://github.com/unidex-ai/UniDex/blob/97d869e0f2d1ec0372cd3cdf28dde66b4e3f216d/config/dataset/HOI4D.yaml#L13)、[TACO config](https://github.com/unidex-ai/UniDex/blob/97d869e0f2d1ec0372cd3cdf28dde66b4e3f216d/config/dataset/Taco.yaml#L13) |
| HOT3D 升采样 | 配置 stride=2，注释 10→5 fps；再 `interpolation_factor=3`，注释 5→15 fps。base loader 将相邻 action 向量逐元素线性插值；不是新增观测点云，也不是 timestamp-aware 对齐。 | 可以统一 action 输出步长；不可声称恢复了原本缺失的观测/接触细节。我们的旋转应按旋转插值处理、保留观测/插值 mask，避免直接套向量插值。[HOT3D config](https://github.com/unidex-ai/UniDex/blob/97d869e0f2d1ec0372cd3cdf28dde66b4e3f216d/config/dataset/Hot3D.yaml#L13-L14)、[插值实现](https://github.com/unidex-ai/UniDex/blob/97d869e0f2d1ec0372cd3cdf28dde66b4e3f216d/src/dataset/base.py#L317-L348) |

### 1.3 不能照搬的时间证据冲突

同一版本 `scripts/process_HOI4D.py` 两次调用 ffmpeg `-vf fps=fps=15` 解码 RGB/depth，
但 `config/dataset/HOI4D.yaml` 的 stride=2 注释写 30→15 fps。
因此当前公开代码存在 **处理脚本与配置注释不能同时解释的时钟缺口**；不能断言论文实际训练一定用了错误 fps，
因为作者可能用了其他预处理产物，但也不能仅引用他们的 30 fps 描述来解决我们的 HOI4D 时钟问题。
[解码脚本](https://github.com/unidex-ai/UniDex/blob/97d869e0f2d1ec0372cd3cdf28dde66b4e3f216d/scripts/process_HOI4D.py#L10-L24)

### 1.4 缺手、接触与穿透：要分清观测缺失和合法目标

- **代码明确实现**：retarget HDF5 为 joint/pose arrays 预分配 `NaN`，只有该侧存在时才写值，保留左右槽位和 frame tag。
  [HDF5 writer](https://github.com/unidex-ai/UniDex/blob/97d869e0f2d1ec0372cd3cdf28dde66b4e3f216d/HandAdapter/hand_processor.py#L746-L805)
- **代码明确实现**：训练 reader 遇到缺手 wrist 会换成 `UNSEEN_POSE`（base 为 identity），NaN joints 归零；action 同样会构造占位目标。
  这是它的 robot action 缺失编码，不等于真实静止手，也不适合我们的 metric displacement 目标。我们应保留有效性 mask 并屏蔽相应 loss。
  [reader state](https://github.com/unidex-ai/UniDex/blob/97d869e0f2d1ec0372cd3cdf28dde66b4e3f216d/src/dataset/base_retarget.py#L350-L397)、[reader action](https://github.com/unidex-ai/UniDex/blob/97d869e0f2d1ec0372cd3cdf28dde66b4e3f216d/src/dataset/base_retarget.py#L430-L495)
- **代码明确实现**：IK 限制 joint ranges、damping，执行 mimic constraints 和 clipping，用 fingertip residual 挑最优结果。
  在本次核对的公开主要路径中**未看到**基于物体 SDF/接触动力学的自动碰撞去穿透与片段剔除；不能把 kinematic IK 等同于 physical validation。
  [IK 实现](https://github.com/unidex-ai/UniDex/blob/97d869e0f2d1ec0372cd3cdf28dde66b4e3f216d/HandAdapter/hand_processor.py#L306-L408)
- **论文声明**：少量 contact-rich frames 的 GUI 修正提高 plausibility。这支持我们为每源做少量代表序列 overlay/视频复核；不支持一开始就给全量数据运行昂贵物理优化。
  [论文 §3.2.1](https://arxiv.org/html/2603.22264v1#S3.SS2.SSS1)

### 1.5 混合、归一化与划分：别人的实现也有我们不该学习的部分

`MixtureDataset` 将四源窗口按长度拼接，共用合并 normalizer；`train.py` 直接 `shuffle=True`。
公开实现没有来源平衡权重，因此抽到某来源的概率由它的窗口数决定。
[mixture class](https://github.com/unidex-ai/UniDex/blob/97d869e0f2d1ec0372cd3cdf28dde66b4e3f216d/src/dataset/mixture.py#L7-L45)

同一训练入口在滑窗生成后做 `random_split`，默认 val_ratio=0.05。TACO 等滑窗的间隔是 `chunk_size//6`，
存在邻近、重叠窗口跨 train/val 的风险；这是从代码顺序推断的风险，没有实测作者 split 产物。
我们的 episode/group-disjoint split 应继续在切窗之前固定。
[训练划分](https://github.com/unidex-ai/UniDex/blob/97d869e0f2d1ec0372cd3cdf28dde66b4e3f216d/train.py#L160-L187)、
[TACO window](https://github.com/unidex-ai/UniDex/blob/97d869e0f2d1ec0372cd3cdf28dde66b4e3f216d/src/dataset/Taco.py#L58-L105)

它的 normalizer 合并 min/max 使用各输入 stats 的 min/min 和 max/max，配置是 action/state 的 minmax、pointcloud identity。
这不是针对我们 24-horizon 米制位移目标拟合的 TRAIN-only moments，不能移植其数值或声称它已解决我们新增源统计的问题。
[normalizer](https://github.com/unidex-ai/UniDex/blob/97d869e0f2d1ec0372cd3cdf28dde66b4e3f216d/src/utils/normalizers.py#L38-L75)、
[base normalization config](https://github.com/unidex-ai/UniDex/blob/97d869e0f2d1ec0372cd3cdf28dde66b4e3f216d/config/dataset/normalizer/base.yaml)

## 2. UniDex 证据改变的工程决策

以下是本项目建议，不是 UniDex 作者结论：

1. 每源 decode 到共同米制几何，再统一时间网格；保留 native timestamps/frame ids 和真实观测 mask。时间统一、MANO decode、窗口过滤分开做。
2. 首先修复确定的单位/法线/坐标错误和官方 QA 失败窗口。小 gap 插值是可选操作，须带 gap 上限与插值 mask；不把大 gap 压缩成连续轨迹。
3. 用每源少量接触丰富片段的 overlay 检查 side、joint order、CAD 与 pose chain；先判定系统性问题，再选择局部过滤。不要默认全量 retarget/物理去穿透。
4. 继续按来源和 episode/object/subject lineage 划分，再构建窗口；新增源比例显式配置，避免大源按窗口数淹没其他来源。
5. 不要求所有来源都通过才开始混合 Probe；通过同一合同的清洁子集可先做最小实验，其余源继续保持隔离。

## 3. 本轮未覆盖与证据边界

- UniDex 不是我们的 object-dynamics/world-model 目标。它证明四源可以经过 source-specific adapters 联合使用，不证明我们每个原始 object pose 都可靠。
- 没有下载作者 80 TB 处理产物，未执行其 pipeline，不能把源码配置等同于论文运行时完整协议。
- UniHand/Uni-Hand 存在重名：Being-H0 的 UniHand corpus、IRMVLab 的 Uni-Hand forecasting、以及 arXiv:2602.21631 的 4D motion modeling 不是同一对象；引用时必须用题名/作者/URL 指定。
- DexTrack 的已有索引记录公开代码不包含完整 homotopy/generalist 迭代；本轮不把其方法描述当作开箱即用的四源清洗工具。
  [现有 DexTrack schema](../../../../../paper/dextrack/schema.json)、[官方仓库](https://github.com/Meowuu7/DexTrack)

## 4. EgoVLA：有效窗口过滤、共同当前坐标系和噪声来源降采样

对象为 [EgoVLA: Learning Vision-Language-Action Models from Egocentric Human Videos](https://arxiv.org/abs/2507.12440)，
官方代码 [RchalYang/EgoVLA_Release](https://github.com/RchalYang/EgoVLA_Release)，不是 Biscue5 的另一个同名项目。
核验源码版本 `09645b9d1729f266f0e532d506a5e57d4a8c09e3`；只获取十个处理文件，保存在
`third_party/egovla-data-audit-20261015/`，清单为 `SOURCE_MANIFEST.json`。以下没有运行其数据处理。

**论文声明**：混合 HOI4D/HOT3D/TACO/HoloAssist；将未来 world-frame wrist 转换到当前 camera frame，
避免相机运动混进手运动监督；HoloAssist 标签较噪，只均匀取其十分之一。RGB 取样 3fps 与预测输出
30Hz 是不同频率，不能据后者推断各源 native fps。手统一到 MANO，使用平均 shape。
[论文 §3.1–3.2 与附录1](https://arxiv.org/html/2507.12440v3#S3.SS1)

**代码明确实现**：

- HOT3D `parse_single_seq_hand` 要求至少一侧满足当前可见性条件，且未来该手标注有效率与相关投影
  在画面内比例均超过90%；两侧都失败则跳过，保留逐时刻 hand flags。
  [HOT3D utils.py L545–597](https://github.com/RchalYang/EgoVLA_Release/blob/09645b9d1729f266f0e532d506a5e57d4a8c09e3/human_plan/dataset_preprocessing/hot3d/utils.py#L545-L597)
- HOI4D 也检查当前和未来投影覆盖，拒绝不足的窗口。`frame_repeat` 可选择把数组插值扩长，
  但已发布启动脚本未设该参数，默认是1；不能宣称作者实际执行了15→30升采样。
  [HOI4D utils.py L337–343、L416–425](https://github.com/RchalYang/EgoVLA_Release/blob/09645b9d1729f266f0e532d506a5e57d4a8c09e3/human_plan/dataset_preprocessing/hoi4d/utils.py#L337-L343)、
  [启动脚本](https://github.com/RchalYang/EgoVLA_Release/blob/09645b9d1729f266f0e532d506a5e57d4a8c09e3/human_plan/dataset_preprocessing/hoi4d/hf_dataset/generate_dataset_hands.sh)
- `expand_first_dim` 将包括 camera pose 在内的数组展平成标量，再以 `interp1d` 插值；这是可选路径，
  不能直接用于我们需要保持 SO(3) 的物体/相机旋转。我们的候选方案是平移线性、旋转SLERP，并核验
  正交性/行列式。[数组插值实现](https://github.com/RchalYang/EgoVLA_Release/blob/09645b9d1729f266f0e532d506a5e57d4a8c09e3/human_plan/dataset_preprocessing/hoi4d/utils.py#L51-L82)、
  [SciPy SLERP 官方说明](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Slerp.html)
- `split_train_eval` 按完整 sequence 列表每十条选一条eval，再在各集合内生成窗口。
  值得借鉴切窗前划分；序列隔离仍不自动等于subject/object隔离。
  [split helper](https://github.com/RchalYang/EgoVLA_Release/blob/09645b9d1729f266f0e532d506a5e57d4a8c09e3/human_plan/dataset_preprocessing/utils/funcs.py)

**用于我们时的调整**：借鉴“窗口有效率门、共同当前frame、限制噪声源占比”。90%的画面内比例是RGB任务
特有条件，不能作为我们的原生3D数据通用门；缺失的一侧手不应导致整个单手交互样本被拒。
平均MANO shape会改变接触位置，不建议用它覆盖已有真实shape/关键点。原始3D仍用当前物体anchor和米制尺度。

## 5. Being-H0 / JALA：语义标准化、伪标签清洗与质量分层

**Being-H0论文声明**：已有mocap/SLAM标注转成MANO；只有3D关节时优化拟合MANO；无3D标注时用HaMeR，
检测左右手身份不连续、补小缺口，拟合加关节约束和时间平滑。motion tokenizer使用15fps、1秒窗口，
运动放在同一个参考帧下；另做视角/分布平衡。[§4.1.4、§4.2、§5.2.1](https://arxiv.org/html/2507.15597v1)

**对我们的选择**：学习“统一语义、查身份跳变、控制来源分布”。我们已用米制3D点，不必先把所有真实关节
重新拟合MANO；弱透视图像对齐也不替代我们的3D坐标链。小gap插值和平滑主要用于估计标签，不应默认
改写mocap/仿真真值，尤其不能抹掉接触瞬间。该版本明确没有交互物体6D建模，不能作为物体标签清洗的直接验证。

**JALA论文声明**：继承MANO标准化，对RGB估计做时间平滑和左右纠正；野外视频先检测手，再验证是否为
手部操作，排除无手、闲置和干扰手片段；可选HaWoR伪标签只留confidence>0.65的片段。
有姿态标签和无姿态标签的数据使用不同监督路径，而不是给无标签数据填零目标。
[§4.3、§5.1–5.2](https://arxiv.org/html/2602.21736v1)

**对我们的选择**：工程上保留“真实观测 / 插值 / 伪标签 / 无效”的类别与loss mask；当前新增native3D无需
重新跑WiLoR/HaWoR，也不能移植某个估计器的0.65阈值。不能把所有静止片段删除：抓持、接触保持以及无动作
状态仍可能是我们需要学习的转移；筛掉的是无关/错误片段，静止与运动比例由sampler控制。

以上Being-H0/JALA是论文层证据；本轮未核验其完整数据生产代码或作者最终产物。

## 6. 对当前两个问题的可执行判断

以下为root基于本地代码与上述一手来源的工程建议，不是论文已经证明的效果。本轮只做调查，没有修复数据或开放训练。

### 6.1 时钟可以分来源解决，不要求所有来源都有纳秒时间戳

- 固定采样率来源：官方采集频率证据 + 原始frame id + 缺帧/重置检查，可建立 `t=(frame_id-frame0)/fps`；
  不因缺少硬件timestamp而永久禁止。DexYCB论文明确采集30fps，仍须确认我们的文件未做额外抽帧。
  [DexYCB原论文](https://dex-ycb.github.io/assets/chao_cvpr2021.pdf)
- timestamp来源：HOT3D按选定stream做单调时间网格，保留nearest residual、重复观测和官方QA/availability；
  不机械沿用旧audit的2ms门。当前bulk converter实际使用16ms，是否收紧应由源cadence与时间误差对应的
  运动误差决定，而非把两者不一致自动写成数据错误。
  [本地bulk入口](../../tools/run/convert_hot3d_bulk.py)、[本地对齐实现](../../tools/run/convert_hot3d_pilot.py)
- 低频来源：先确认native clock，再按需要转换；插值点标明synthetic，保留原观测，不能把带宽增加当作新数据。
  旋转采用合法旋转插值，长gap/reset切段。也可先让该源留在工程集合，不阻止其他来源先训练。
- HOI4D现存冲突需要具体文件证据裁决，别人的30Hz训练配置不能代替证明。当前不盲选15或30。

### 6.2 法线、QA与几何异常主要是可逆工程处理

- 已有DexYCB只读probe支持法线方向与独立mesh重算大致一致，因此非零法线的候选修复是单位化；它不会
  自动修正翻面。零/极小或相反面抵消法线按CAD对象隔离或从可信面法线重建，不随意选正负号。
  [本地probe](../../../../../tmp/goal-supervision-20261015/normal-repair-probe.json)、
  [法线估计与方向歧义官方说明](https://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html#Vertex-normal-estimation)
- 修复写新geometry版本，不改原CAD/pose；保持query点身份及相同metric scale，或者显式重建其对应标签。
- QA无效和缺失帧用mask/窗口门处理，避免删除帧后压缩时间轴；单手来源允许另一手全invalid。
- 可先隔离少量坏CAD，按来源记录清洗前后序列、有效帧、窗口、对象计数和拒绝原因，不必等全量修复。

### 6.3 最小落地顺序

1. 固定来源profile（fps/clock、米制坐标、side/joint map、目标对象），生成版本化法线修复与隔离清单。
2. 根据QA、reset、时间gap和目标有效性构造窗口；按来源抽少量接触/高速/静止片段做几何overlay，独立复算位移。
3. 先确定episode/group划分和source采样比例，再从准入TRAIN拟合24-horizon统计。明确哪些DEV衡量新episode，
   哪些另衡量新subject/object，不把所有DEV强制object-disjoint而清空整源。
4. 选择通过合同的清洁来源子集接入production adapter，做真实GPU有界mixed Probe；按来源报告hand/scene物理EPE，
   不被总体loss掩盖。未通过的源继续隔离，数据量增长不自动升级科学结论。

不把全量SDF去穿透、全量重新MANO拟合、语言重标注或RGB-D重建作为当前native3D开训前置条件。

补充核验：此前ledger把H2O的220个TEST NPZ称为orphans。2026-10-15本机对账为TRAIN/DEV524、TEST220、
磁盘744，`disk - train_dev == heldout_test`且交集/缺失/未归属均为0；这些文件是预期隔离测试数据，不是坏数据。
它们不应进入TRAIN，审计器应识别独立TEST manifest。

## 7. 追问：来源配比、抽样权重和loss权重（2026-10-15）

三件事分别记录：语料构成比例、训练sampler概率、loss系数。图中的数据比例不能证明每个batch固定按此配额抽样，
更不能推断为loss乘数。来源对梯度的影响还受有效标签数、目标归一化尺度和reduction方式影响。

### 7.1 UniDex

公开代码按训练窗口数拼接并shuffle，`p_s=N_s/sum(N)`，没有独立source weights，四源数值百分比未公开。
模型是归一化82D FAAS、30步的全元素flow MSE，没有source/hand/actuator/pad监督mask；补零维仍被加噪并参与loss。
共享minmax具有量纲平衡作用，不能解释成每源独立加权。完整固定commit与代码链路见
[专项审计](2026-10-15-unidex-mixture-loss-audit.md)。公开入口还有缺失配置/命名空间，不能直接当作论文最终运行manifest。

### 7.2 EgoVLA

Figure 3来源构成：**HOI4D39%、HOT3D23%、HoloAssist25%、TACO13%**。HoloAssist先均匀取原始数据十分之一，
降低噪声来源占比；“取十分之一”不等于最终占10%。百分比通过官方HTML附带SVG的扇区、数字位置和颜色图例核对，
原SVG与审查PNG在 `tmp/multisource-mixture-loss-20261015/egovla-source-split.{svg,png}`。
[论文 §3.1 / Figure 3](https://arxiv.org/html/2507.12440v3#S3.F3)

附录给固定任务项权重：

```text
L = 20 * L_wrist_translation + 5 * L_wrist_rotation + 5 * L_hand_joint
```

平移与MANO PCA joint是L2平方误差；rotation从rot6D转换矩阵后算平方误差。三项权重在预训练/后训练均相同。
这是不同预测量的权重，没有查到按HOI4D/TACO等来源额外乘loss系数的声明。
[论文附录2.1、Table 3](https://arxiv.org/html/2507.12440v3#A2.SS1)

### 7.3 Being-H0 / UniHand-2.5M

训练子集不是原始166.5M指令数据的自然比例。论文先做source/task平衡后抽约2.5M，图中列出以下近似K计数：

| source | sampled instructions, K |
|---|---:|
| EgoDex | 414 |
| OakInk2 | 391 |
| ARCTIC | 296 |
| TACO | 293 |
| HOI4D | 290 |
| H2O | 240 |
| HOT3D | 210 |
| DexYCB | 174 |
| Taste-Rob | 146 |
| FPHA | 73 |

图中取整计数合计2527K，故不声称这些是精确的2,500,000条manifest计数。任务构成约为motion generation1.6M、
contextual prediction0.5M、motion-to-text0.4M，即约64:20:16；它也是任务样本构成，不是三项loss系数。
已直接查看作者发布图，保存在 `tmp/multisource-mixture-loss-20261015/being-h0-data-recipe.png`。
[官方图](https://arxiv.org/html/2507.15597v1/04_unihand.png)、[论文 §5.1](https://arxiv.org/html/2507.15597v1#S5.SS1)

特殊loss处理有两个阶段，不能混在一起：

- **Motion tokenizer**：`L_recon + 0.02*L_commit + 1.0*L_wrist`，额外照顾wrist重建。
- **VLA预训练**：next-token CE；对motion位置以50%概率屏蔽非motion词表logits；只平均token loss处于
  第15–95百分位的项，排除最低15%和最高5%。这是token-level过滤，不是“给噪声来源统一乘0.05”。

[公式6、§4.1.4、§6.1.1](https://arxiv.org/html/2507.15597v1)
本轮没有核验其最终训练loader，不能把图中比例进一步说成每batch配额或精确重采样算法。

### 7.4 JALA / UniHand-Mix

论文Figure 3给来源构成：

| source | share |
|---|---:|
| Ego4D | 32.9% |
| EgoDex | 14.7% |
| OakInk2 | 12.4% |
| HOI4D | 10.0% |
| ARCTIC | 9.7% |
| TACO | 9.7% |
| H2O | 6.5% |
| DexYCB | 4.1% |

对应约5M lab + 2.5M wild，约2:1，正文说完整7.5M训练一epoch。Figure3任务构成为motion generation42.5%、
video without motion29.1%、motion continuation15.5%、motion description12.9%。没有据此假定固定batch来源配额。
[论文Figure 3、§6.1](https://arxiv.org/pdf/2602.21736#page=8)

明确的预训练loss为 `1_labeled * L_MCP + 0.5 * L_Align`，MCP是motion-token CE，Align是embedding的L1距离。
有标签时二者共同训练；对论文描述的wild/video-only路径仅用Align。正文同时提到少量wild高置信伪标签，
没有最终manifest/code就不能推出每个伪标签样本实际是否激活MCP；最保守的确认是不能给无标签视频伪造motion目标。
未查到按每个来源再设专属lambda。[公式2/3/6与§6.1](https://arxiv.org/html/2602.21736v1)

### 7.5 与我们现有实现的关系

当前main3 source sampler是OakInk2:GRAB:ARCTIC=5:2:2（约55.6:22.2:22.2），来源内部再按存在的运动类别
分层抽样；它已经不是按437.8万窗口自然比例训练。
[sampler](../../../cm-pointflow-effect-pretrain/src/oakink_wm/multisource.py)

v2为`L_hand + L_scene`，手/物体各自使用TRAIN-only的24×XYZ标准化和validity mask；scene loss还给运动点
较高相对权重（tau=2mm，floor=0.1，sigmoid连续过渡）。没有source-specific loss multiplier。
[v2 loss](../../../ptv3-shared-wam-v2/src/ptv3_shared_wam_v2/objectives.py)、
[配置](../../../ptv3-shared-wam-v2/configs/pretrain_remote_80k.json)

本项目建议：先显式确定source sampler并按最终TRAIN重新拟合共同统计，保持手/物体有效性mask；不一开始叠加
另一套source loss weights。归一化平方误差本身对物理误差隐含`1/std^2`权重；当前masked mean按全batch有效点
汇总，因此各源有效手点、对象数也影响实际贡献。若目标是每个窗口/每个来源严格等权，需要先在各窗口/来源内
归一化再聚合，不能仅把sampler设为均匀就宣称loss贡献均匀。新增源的确切比例留待准入后按有效窗口和目标域确定。
