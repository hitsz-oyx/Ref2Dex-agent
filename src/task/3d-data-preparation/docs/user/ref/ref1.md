可以，而且现在有一条比 EgoDex 简单得多的扩量路线。

**我建议我们暂时把数据原则改成：只收“已有 3D 手 + 物体 pose/mesh + 连续时序”的数据。完全不做 RGB→3D 重建。** 对我们当前 Shared-WAM 来说，这其实很合适，因为我们可以直接用 CAD/mesh + 逐帧 SE(3) 构造 `scene/U/G`，用手标注构造 `H/V`，根本不需要图像。

更关键的是，最新 **UniDex 官方仓库本身就在用 H2O、HOI4D、HOT3D、TACO 做预训练数据，而且已经提供了 H2O/HOI4D/HOT3D/TACO → Allegro/Ability/**Inspire**/LEAP/Shadow 等多种手的 retargeting 入口。**这和我们现在的需求几乎重合。[GitHub](https://github.com/unidex-ai/UniDex/blob/main/doc/DATASET.md)

### 我会这样排优先级

| 优先级 | 数据 | 规模 | 对我们直接可用程度 | 我的判断 |
|---|---|---:|---|---|
| **S+** | **UniDexGrasp++ 全量扩展** | 5355 物体基础；我们当前已打通策略回放 | ★★★★★ | **先扩这个** |
| **S** | **TACO** | 2317 条 motion，206 个高分辨率物体模型 | ★★★★★ | 非常适合 |
| **S** | **HOT3D** | 833 min，1.5M multi-view frames，19人、33刚体 | ★★★★★ | 很适合 |
| **S-** | **HOI4D** | 2.4M frames，4000 sequences，800实例 | ★★★★☆ | 规模最大的人手候选之一 |
| **A** | **H2O** | 4人、8物体、36动作，约57万多视角帧 | ★★★★★ | 很干净，但物体少 |
| **A** | **DexYCB** | 1000 sequences，20 YCB物体，582K RGB-D帧 | ★★★★★ | 非常容易接 |
| **A-** | **HO-Cap** | 9 subjects | ★★★★★ | 标签完整，规模中等 |
| **B+** | **DexTrack retargeted data** | GRAB/TACO → Allegro/LEAP | ★★★★☆ | 很直接，但和已有数据有重复 |
| B | DexCap / DexterHand | 人手→LEAP，直接3D参数 | ★★★★☆ | 数据规模暂时不够明确 |
| 不优先 | DexGraspNet / ContactPose | 1.32M static grasps / 2.9M images | ★★☆☆☆ | **主要是静态抓取，不适合 K24 dynamics** |

下面几个我认为最值得马上做。

---

## 1. 第一优先：把 UniDexGrasp++ 直接扩到全对象

这个其实已经不是“候选”，而是我们**已经证明工程可行**的来源。

当前仓库已经做到：

\[
20\ objects
\rightarrow320\ trajectories
\rightarrow3200\ H4/K24\ windows
\]

而且真实 30 Hz：

- hand trajectory；
- object SE(3)；
- object CAD；
- fixed 32 query；
- H4 scene motion；
- future24；
- 成功/失败真实 rollout；

全部都已经通过 provider audit。

UniDexGrasp/DexGraspNet 本身覆盖 **5355 个物体、133+ categories、1.32M 静态稳定抓取姿态**；UniDexGrasp++ 的策略则可以把这些静态目标变成真实连续执行轨迹。[GitHub](https://github.com/PKU-EPIC/DexGraspNet?utm_source=chatgpt.com)

我们当前仓库已经估算过，如果仍按：

\[
16\ trajectories/mesh
\]

扩：

\[
2066\ TRAIN\ meshes
\rightarrow \sim33056\ trajectories
\]

如果使用全部约 3084 mesh-scale：

\[
\sim49344\ trajectories.
\]

这已经是一个非常不错的机器人世界模型预训练集了。

而且它最大的优势就是：

\[
\boxed{\text{没有任何视觉重建噪声}}
\]

物体 pose、hand state、CAD 都直接来自 simulator。

**所以这一项我认为可以直接排第一。**

---

# 2. TACO：我认为是最好的人手新数据源

TACO V1 有：

\[
2317\ motion\ sequences
\]

覆盖：

\[
151\ \langle tool,action,object\rangle
\]

组合，还有：

\[
206
\]

个高分辨率 object models，并提供 hand-object pose / mesh annotation；其中 2212 个 sequence 的完整 egocentric modalities 可用。[GitHub](https://github.com/leolyliu/TACO-Instructions?utm_source=chatgpt.com)

但对我们来说最重要的不是视频。

我们可以直接丢掉 RGB/D：

```text
MANO hand
object pose
object mesh
camera / world transform
       ↓
Shared-WAM provider
```

然后：

\[
H_{t-3:t}
\]

直接从 MANO FK 得到；

固定 CAD query：

\[
p_i
\]

利用逐帧 object pose：

\[
T_t^{-1}T_{t+k}p_i
\]

就直接得到：

\[
U_{1:24}.
\]

根本不用 CoTracker。

更有意思的是，DexTrack **已经提供了 TACO → robot hand 的 retargeted data**。[GitHub](https://github.com/Meowuu7/DexTrack?utm_source=chatgpt.com)

所以 TACO 可以同时给我们：

\[
\text{human hand version}
\]

和：

\[
\text{robot-retargeted version}.
\]

**我会把 TACO 放在人手数据第一优先。**

---

# 3. HOT3D：非常适合我们的“pose+CAD”路线

HOT3D 有：

- 833 min；
- 1.5M multi-view frames；
- 30 FPS；
- 19 subjects；
- 每段最多约 6 个物体；
- 33 个 rigid objects。[Facebook Research](https://facebookresearch.github.io/hot3d/?utm_source=chatgpt.com)

而且官方数据有：

- hand pose（MANO/UmeTrack）；
- object identity；
- object CAD；
- world-frame object 6D pose；
- 时间戳。[GitHub](https://github.com/facebookresearch/hot3d?utm_source=chatgpt.com)

这意味着我们甚至**不需要它没有 dense depth 这件事**。

因为对于当前 WAM，可以定义一个统一的：

\[
\boxed{\text{full-CAD geometry profile}}
\]

也就是和我们 UniDexGrasp++ 一样：

```text
当前 object CAD
+ 当前 hand
+ object future SE(3)
       ↓
U/V/G
```

这样反而避免：

> Human dataset 用 visible point cloud，simulation 用 full CAD

这种观测合同不一致。

对于我们现在想“先把大规模数据弄起来”，**HOT3D 比 EgoDex 合适得多。**

---

# 4. HOI4D：规模很大，值得第二批接

HOI4D 是：

- **2.4M frames**
- **4000 sequences**
- 9 participants
- **800 object instances**
- 16 categories
- 610 rooms

而且有 frame-wise：

- 3D hand pose；
- object pose；
- object CAD/mesh；
- hand action；
- 4D scene annotations。[GitHub](https://github.com/leolyliu/HOI4D-Instructions?utm_source=chatgpt.com)

所以从数据内容上讲，它几乎就是：

\[
\boxed{\text{现成的 WAM supervision}}
\]

最大问题只有一个：

**我们仓库已经核过它的时间频率和当前 30Hz 合同不完全一致。**

但这和 EgoDex 的困难不是一个量级。

EgoDex 是：

\[
RGB
\rightarrow depth
\rightarrow mask
\rightarrow track
\rightarrow metric3D
\]

HOI4D 只是：

\[
15Hz\ pose
\rightarrow30Hz\ pose
\]

如果我们真要统一 30Hz，可以对：

- translation：linear interpolation
- rotation：SLERP
- MANO/joint：pose interpolation

做一次离线 resample。

我认为这个代价完全能接受。

因此 HOI4D 应该是**第二波最大的人手数据来源**。

---

# 5. H2O：不算最大，但应该非常早接

H2O 有完整的：

- 左右手 3D pose；
- MANO；
- object 6D；
- object meshes；
- camera pose；
- scene point cloud；
- RGB-D。

而且是同步 30 FPS。[H2O Dataset](https://h2odataset.ethz.ch/?utm_source=chatgpt.com)

我们甚至只需要下载：

```text
pose annotations
+ object meshes
```

不需要 RGB-D。

NVIDIA 的 H2O processing 文档甚至指出，pose archive 本身就带 MANO，object meshes 也可以单独下载，所以如果只需要几何数据，数据量很小。[GitHub](https://github.com/nvidia-isaac/video_to_data/blob/main/robotic_grounding/docs/H2O_SETUP.md?utm_source=chatgpt.com)

问题只是数据 diversity 比较有限：

\[
4\ subjects,\quad8\ objects.
\]

所以：

> **很适合作为 clean human-hand source，但不能承担主要扩量。**

---

# 6. DexYCB：最容易做的 human adapter

DexYCB：

- 1000 sequences；
- 10 subjects；
- 20 YCB objects；
- 30 FPS；
- 21 个 3D hand joints；
- MANO；
- 6D object pose；
- segmentation；
- object models。[Kinetic Blocks](https://kineticblocks.com/open-data/dexycb?utm_source=chatgpt.com)

基本可以：

```text
joint_3d
  ↓
11 semantic points

pose_y + YCB mesh
  ↓
32 fixed query trajectories
```

直接出：

\[
H4/K24.
\]

不足是 unique motion 只有约一千段，规模不算特别大，而且通常单手。

但它非常适合测试：

> 新的统一 human provider 有没有写对。

所以我会把它作为**低成本 sanity dataset**，不是主扩量。

---

# 7. HO-Cap：也几乎能直接接

HO-Cap 已发布：

- 3D hand keypoints；
- MANO；
- object 6DoF；
- segmentation；
- object model；
- 9 subjects。[GitHub](https://github.com/IRVLUTD/HO-Cap?utm_source=chatgpt.com)

它跟 DexYCB 类似：

\[
\boxed{\text{pose + mesh 直接生成我们要的 tensor}}
\]

所以也非常容易。

但规模不如 HOI4D/HOT3D/TACO，因此放第二梯队。

---

# 8. DexTrack 对我们其实很特别

DexTrack 官方直接提供：

- GRAB retargeted data；
- TACO retargeted data；
- Allegro；
- LEAP+Franka；
- object assets。[GitHub](https://github.com/Meowuu7/DexTrack?utm_source=chatgpt.com)

所以我们可以直接拿：

\[
\text{robot hand trajectory}
+
\text{object trajectory}
\]

而不用自己做人→机器人的 retarget。

这个很适合我们。

但是问题是：

\[
\text{GRAB}
\]

我们本来就在用；

TACO 如果又用 human version，又用 DexTrack retargeted version，它们不能被当成两份独立任务数据。

所以 DexTrack 的作用我会定位为：

\[
\boxed{\text{embodiment augmentation}}
\]

而不是数据量翻倍。

---

# 9. DexMimicGen 我不会排很前

它确实很大：

\[
>20,000\ demonstrations
\]

9 tasks，而且官方直接发布 HDF5 数据集。[DexMimicGen](https://dexmimicgen.github.io/)

但这里“dexterous manipulation”主要是任务层面的双臂 manipulation，很多环境的数据接口强调 EEF / robot states，不一定给我们需要的**灵巧手指骨架轨迹**。

所以它适合 world/object/action 模型，但不一定最适合我们现在：

\[
22\ semantic\ hand\ points
\]

的手轨迹 WAM。

我会先核具体 embodiment，再决定要不要接。

---

# 两个“大但现在不要”的数据

### DexGraspNet

很诱人：

\[
1.32M\ grasps,\quad5355\ objects
\]

但本质是：

\[
\boxed{\text{static grasp poses}}
\]

没有：

\[
H4\rightarrow future24.
\] :chatgpt-content-reference{index="13"}


所以除非像 UniDexGrasp++ 一样**启动 policy rollout 把它变成动态轨迹**，否则不能直接当我们的 WAM sequence。

### ContactPose

也有：

\[
2.9M\ RGB-D
\]

和 hand/object pose/contact。[GitHub](https://github.com/facebookresearch/ContactPose?utm_source=chatgpt.com)

但核心也是 functional grasp capture，更偏接触/静态抓取，而不是长时间 manipulation dynamics。

暂时不用。

---

# 我认为我们现在最合理的数据扩展路线

其实不用再设计很多套 provider。

我们应该定义一个非常简单的 **`PoseMeshWindows`**：

输入统一成：

\[
\{t,\ hand_t,\ T^{obj}_t,\ mesh\}
\]

然后所有这些数据集都走同一条转换：

\[
T_{A}=T_{obj,t}^{-1}
\]

手：

\[
h_k^{A}=T_A h_k
\]

物体固定 CAD query：

\[
p_i
\]

未来：

\[
p_{k,i}^{A}
=
T_A T_{obj,t+k}p_i
\]

最后生成：

\[
H4
\]

\[
V_{1:24}=h_{t+k}^{A}-h_t^{A}
\]

\[
U_{1:24}=p_{t+k}^{A}-p_t^{A}
\]

和：

\[
G=U_{24}.
\]

这和我们现在 **UniDexGrasp provider 的数学合同几乎完全一致**。

数据集之间只需要写 reader：

```text
H2O reader
TACO reader
HOT3D reader
HOI4D reader
DexYCB reader
HO-Cap reader
          ↓
      PoseMeshWindows
          ↓
 Shared-WAM collate
```

这样我们不是每个数据集重新写一套 pipeline。

---

## 如果现在让我只选 4 个开始

我会直接选：

\[
\boxed{1.\ UniDexGrasp++\ 全量}
\]

机器人侧最大、最快。

\[
\boxed{2.\ TACO}
\]

人手 + tool/object，数据形态非常适合。

\[
\boxed{3.\ HOT3D}
\]

30Hz、手/物体/CAD完整，14小时左右连续交互。

\[
\boxed{4.\ HOI4D}
\]

真正的大规模人手扩量，4000 sequence / 800实例。

然后 H2O、DexYCB、HO-Cap 用作 clean auxiliary / validation。

而且还有一个非常重要的事实：**UniDex 2026 自己现在公开的预训练数据栈就是 H2O + HOI4D + HOT3D + TACO，并且它的 HandAdapter 已经支持 Inspire。** [GitHub](https://github.com/unidex-ai/UniDex/blob/main/doc/DATASET.md)

所以这不是我们自己随便拼的组合。

它基本就是：

\[
\boxed{\text{目前已经有人验证过的一套大规模 human→dexterous 数据路线}}
\]

只是我们不需要它的 RGB 模型部分，只取它们已经存在的 **3D hand/object annotations** 来训练 WAM。

**因此我建议接下来先停 EgoDex，先做一个统一 `PoseMeshWindows`，第一批同时接 TACO + HOT3D；UniDexGrasp++ 在另一边继续扩对象。** 这样最快能把数据规模真正拉起来。