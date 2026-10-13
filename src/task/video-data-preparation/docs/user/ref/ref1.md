可以，而且我现在的判断是：**我们这版 Shared-WAM 比之前 Goal-Point-WAM 更适合往 EgoDex 靠。**真正需要改的主要是数据适配层，不是模型主体。

但要说清楚：**当前仓库的 `SharedWindows` 还不能直接吃 EgoDex。**它目前硬编码 OakInk2/GRAB/ARCTIC，并且 `point_labels()` 假设物体未来由 SE(3) `effect` 生成。EgoDex 没有这个刚体 `effect` 合同，所以要新增一个 `EgoDexSharedWindows`，直接提供 point-track supervision。

### EgoDex 和我们现在的模型其实很匹配

EgoDex 官方数据原生就是 30 Hz，并给了相机内外参和大量手部关节 SE(3)，包括手腕、指尖、指关节等；这些 pose 都在同一个 episode 内固定的 ARKit 世界坐标系中。也就是说，**手和相机这两部分完全不用我们重建**。[github.com](https://github.com/apple-aiml-research/ml-egodex?utm_source=chatgpt.com)

我们模型需要：

\[
(S_{\le t},H_{\le t},G)\rightarrow(U_{1:24},V_{1:24})
\]

EgoDex 可以直接提供其中很大一部分：

- \(H\)：直接从 HDF5 的 3D hand joints 取；
- \(V\)：直接取未来 24 帧的 3D hand joints；
- 相机：官方已有；
- 频率：正好 30 Hz；
- 真正缺的只剩 **目标物体的 3D 点轨迹 \(U\)**，以及对应的 endpoint goal \(G\)。

所以现在最费时间的那套：

```text
11帧@10Hz
→ 全场 dense CoTracker
→ 全场 VGGT depth
→ 31帧重采样
→ 12000点全未来轨迹
```

对我们现在的 Shared-WAM 来说，**做得太重了。**

---

## 我会把 EgoDex pipeline 改成两层

第一层只是给 PTv3 当前 observation，用相对便宜的 **H4 scene context**：

\[
t-3,t-2,t-1,t
\]

我们只需要用它得到当前场景点及历史运动特征：

\[
[n,\ v,\ a,\ \Delta p_{H4}]
\]

也就是现在 PTv3 的 18D feature 前 12 维。

不需要把这些场景点一路 track 到 \(t+24\)。

第二层只针对**目标物体**做 future tracking：

\[
\boxed{32\sim128\text{ 个 object points}}
\]

从 \(t\) 跟到 \(t+24\)。

这样直接构造：

\[
U_{i,h}=p_{i,t+h}-p_{i,t}
\]

以及：

\[
G_i=
[p_{i,t},p_{i,t+24}-p_{i,t}].
\]

这恰好就是我们现在的 goal contract。

### 整个 pipeline 可以变成

```text
EgoDex HDF5
├── native camera pose
├── native 3D hand joints
│        ↓
│   H4 + V1:24
│
RGB / video
│
├── H4 scene sparse tracks
│        ↓
│   current PTv3 scene
│
└── target-object mask
         ↓
    32~128 query points
         ↓
       CoTracker
         ↓
   sparse depth lifting
         ↓
 P_t ... P_t+24
    ├── U
    └── endpoint-delta G
```

这样才真正适合大规模 EgoDex。

---

## 还有几个接口要改，但都不是架构问题

第一处是 hand。当前 Shared-WAM 要：

\[
2\times11
\]

即每只手：

\[
\text{wrist}+5\text{ MCP/knuckle}+5\text{ fingertips}.
\]

你们之前 PointWAM-style EgoDex pipeline 生成的是每手 10 点，而且其中有 6 个近似 palm points。**新 Shared-WAM 不应该继续用那套近似点。**

EgoDex HDF5 本身就有大量手部 joint transforms，所以直接映射成我们的 11 semantic points。这样反而更干净。[GitHub](https://github.com/apple-aiml-research/ml-egodex?utm_source=chatgpt.com)

第二处是时间。当前 Shared-WAM 是：

\[
H=4,\qquad horizon=24,\qquad30Hz.
\]

EgoDex 原生正好 30 Hz，所以直接取：

\[
[t-3,\ldots,t+24]
\]

即可。

**不要再先降到 10 Hz、再插值回 31 帧。**

那套是为了复刻 PointWAM 数据合同，不是我们现在模型必须的。

第三处是 scene label。当前 `point_labels()` 是：

\[
T_{obj,t+h}P_i-T_{obj,t}P_i
\]

适合 OakInk2/GRAB/ARCTIC 的刚体 pose。

EgoDex 改成直接：

\[
\boxed{
scene\_target[h,i]
=
P_{t+h,i}-P_{t,i}
}
\]

即可。

模型 `SceneFutureHead` 根本不在乎 label 是 SE(3) 算出来的还是 tracker 得到的，它最终看的就是：

\[
[B,O,24,32,3].
\]

所以这一层完全兼容。

---

# 最大的问题其实只有一个：目标物体怎么找

EgoDex 原生没有像我们 main-three 那样现成的：

\[
\text{target object slot + object pose}.
\]

所以真正要解决的是：

\[
\boxed{\text{当前画面里，哪一片是被操作的目标物体？}}
\]

但注意我们现在也不需要把**所有对象**都检测出来。

对 EgoDex 第一版完全可以设：

\[
O=1
\]

只监督 manipulated object。

于是数据只需要：

1. 当前目标物体 mask；
2. 从 mask 内选 32–128 个 query points；
3. 对这些 query 做 24 帧 tracking；
4. lift 到 3D。

整个背景场景仍然可以进 PTv3，当 context 使用，但**不要求 background 有 future label**。

这会大幅降低处理难度。

---

## 当前 PTv3 的 18D feature 也能构造

现在主三域是：

\[
[\text{normal}_3,
v_3,
a_3,
\Delta p_{H4,3},
metadata_6].
\]

EgoDex 可以构造前 12 维：

- normal：当前 depth point cloud 局部估计；
- velocity：H4 track；
- acceleration：H4 track；
- history displacement：H4 track。

metadata 里的 target/static/type 等可以由 EgoDex adapter 显式填；hand 已经单独成为 22 个 token，不需要再塞进 PTv3 voxel。

所以也不需要修改 PTv3 网络结构。

---

# 为什么这样会比现在快很多

你们当前 pipeline 里写的是：

> 640×360 首帧每 2 pixel 一个 query。

数量大约是：

\[
320\times180\approx57,600
\]

个 tracking queries。

然后还要处理整个 1 秒 clip。

但我们真正作为 future supervision 需要的可能只有：

\[
32\sim128
\]

个目标物体点。

即使 observation H4 再选：

\[
1k\sim4k
\]

个 scene points，也只需要追踪 **4 帧历史**，而不是 24 帧未来。

计算结构从：

\[
\boxed{
\text{几万点}\times31\text{帧}
}
\]

变成：

\[
\boxed{
\text{少量scene点}\times4
+
32\sim128\text{ object points}\times24
}
\]

这是数量级上的区别。

真正比较重的就只剩 depth backend。

---

# 所以我会把目前架构和 EgoDex 的对应定成这样

| Shared-WAM 字段 | EgoDex 来源 |
|---|---|
| `start_hand` | 原生 HDF5 3D joints |
| `start_hand_history` | 原生连续 4 帧 joints |
| `trajectory_target` | 原生未来 24 帧 joints |
| `xyz` | 当前 RGB + depth lifting |
| `features` | H4 sparse scene motion |
| `scene_points` | 当前目标物体 32 points |
| `scene_target` | 同 32 points 的未来 24 帧 3D tracks |
| `start_object_points` | 同一批 32 当前 points |
| `goal_object_points` | 同一批 points 在 \(t+24\) 的位置 |
| `goal_object_delta` | 自动相减 |
| `object_goal_valid` | track/depth validity |

这一套和模型完全对得上。

---

## 我甚至建议我们把数据层正式改成两类 provider

现在：

\[
\texttt{SharedWindows}
\]

其实应该抽象成：

\[
\boxed{\texttt{SharedSampleContract}}
\]

底下两个 provider：

```text
RigidPoseProvider
  ├ OakInk2
  ├ GRAB
  └ ARCTIC

TrackedPointProvider
  └ EgoDex
```

前者：

\[
SE(3)\rightarrow U
\]

后者：

\[
3D\ track\rightarrow U.
\]

但最终都产出完全一样的：

```text
scene_points
scene_target
scene_valid

start_hand
start_hand_history
trajectory_target

start_object_points
goal_object_points
object_goal_valid
```

模型层完全不知道样本来自哪一种。

这才是后面大规模混训最合理的工程结构。

---

# 一个需要特别注意的地方

我**不建议现在为了 EgoDex 去解除 current-goal point correspondence**。

恰恰相反，CoTracker 天然给我们 point ID：

\[
i_t\leftrightarrow i_{t+24}.
\]

这正好满足：

\[
p_{i,t}\leftrightarrow p_{i,t+24}.
\]

所以 EgoDex 虽然没有 object SE(3)，反而非常适合我们现在这种 **tracked point goal** 表达。

---

## 最终判断

\[
\boxed{\textbf{架构能适配，而且适配程度很高。}}
\]

但我不会继续沿用当前 EgoDex full-reconstruction pipeline 原样扩规模。

我会改成：

\[
\boxed{
\text{EgoDex native hand}
+
\text{H4 sparse scene context}
+
\text{32–128 sparse target-object 3D tracks}
}
\]

而不是：

\[
\text{dense whole-scene 31-frame reconstruction}.
\]

这样既更贴合我们现在的 Shared-WAM，也更有希望把 EgoDex 从现在这种“几万条处理很慢、failure 很多”的工程状态真正扩到几十万甚至百万级 windows。

我们的目标物体能不能直接从language里面获得？


有，而且我觉得**场景点应该和“目标物体监督点”彻底分开设计**。这是现在往 EgoDex 扩规模时最值得简化的一步。

我建议把输入拆成两类：

\[
\boxed{\text{Scene Context Points}}
\qquad+\qquad
\boxed{\text{Target Object Track Points}}
\]

后者就是我们前面说的 32–128 个有 correspondence 的点；前者只是给 PTv3 看当前环境，**不要求未来 correspondence，也不要求每个点都有 future label**。

### 1. 场景点只负责“看懂当前局面”

它们回答的是：

> 手在哪、目标物体周围有什么、桌面在哪、附近有没有障碍物、局部几何是什么样。

所以当前帧只要有：

\[
S_t=\{x_j,f_j\}_{j=1}^{N}
\]

就够了。

不需要：

\[
x_{j,t}\leftrightarrow x_{j,t+24}.
\]

换句话说，EgoDex 完全没必要再给几千上万个 scene points 全部做 24 帧 CoTracker。

---

## 2. 我会把场景点控制在 1k–4k 左右

我们当前 PTv3 后面本来还会压成最多：

\[
256\text{ scene tokens}.
\]

所以前面搞 1 万甚至几万轨迹点的收益很有限。

我会从当前 RGB/depth 点云中做**分层采样**，而不是纯随机：

- 一部分来自 target object mask；
- 一部分来自 hand 周围，例如手附近 15–30 cm；
- 一部分来自 target 周围局部场景；
- 一部分保留全局桌面/其他物体 context。

比如一个 2048 点版本，可以大致让 target + hand-near 区域占主要比例，剩下保留全局背景。具体比例不需要一开始卡死，核心是**不要让大面积桌面把真正 interaction 区域淹没**。

---

## 3. Hand 不应该再作为 scene points

这个和我们新架构正好一致。

现在：

\[
22\text{ hand tokens}
\]

已经是独立的一等 token，所以 EgoDex 的 scene cloud 里应该把 hand mask 掉：

\[
S_t=\text{non-hand scene}.
\]

这样不会出现：

\[
\text{hand point + object surface}
\]

在 1 cm voxel 里混合的问题。

---

# 4. H4 动力学怎么办？

这是唯一稍微麻烦的部分。

我们现在 scene feature 里有：

\[
[\text{normal},v,a,\Delta p_{H4},metadata].
\]

我不建议为了这些特征去做 24 帧 dense tracking。

只需要对**最终选中的 1k–4k scene points**做短历史：

\[
t-3,t-2,t-1,t.
\]

于是：

\[
v_t\approx30(p_t-p_{t-1}),
\]

\[
a_t\approx900(p_t-2p_{t-1}+p_{t-2}),
\]

\[
\Delta p_{H4}=p_t-p_{t-3}.
\]

这只是 4 帧短 track，成本比未来 24 帧低很多。

所以整个负担应该是：

\[
\boxed{
2k\text{ scene points}\times4\text{ frames}
}
\]

加：

\[
\boxed{
64\text{ target points}\times24\text{ frames}.
}
\]

而不是：

\[
10k+\text{ points}\times24\text{ frames}.
\]

---

# 5. 甚至第一版可以更激进

我们可以先问一个问题：

> PTv3 真的必须拥有 scene velocity / acceleration 才能训练好吗？

因为 hand 本身已经有 H4，而且 target object 又有 goal。

所以可以做一个 EgoDex V0：

\[
f_j=
[\text{normal},0,0,0,\text{metadata}]
\]

也就是 scene 只提供当前几何。

然后手仍然有完整：

\[
H4.
\]

如果这版能训练，再增加短 scene tracking。

这会把 EgoDex pipeline 进一步简化成：

```text
当前帧 RGB
    ↓
depth
    ↓
1k~4k current scene points
```

完全不需要给背景做 temporal tracking。

我觉得这是一个非常值得做的 ablation：

\[
\text{current-scene-only}
\quad vs\quad
\text{H4-scene-motion}.
\]

因为如果二者差距很小，我们能省掉大量预处理成本。

---

# 6. Scene future supervision 只监督 target object 就可以

对 EgoDex 我第一版甚至不会强求周围每个物体都有：

\[
U.
\]

直接：

\[
O=1.
\]

即：

\[
\boxed{\text{只有 manipulated object 有 scene future label}}
\]

然后：

\[
\hat U\in\mathbb R^{1\times24\times32\times3}.
\]

PTv3 场景里仍然可以看见其他对象、桌面和背景，但这些只是 context，不进 \(L_S\)。

这其实很符合我们的目标：

> 学会手和目标物体的 interaction。

没必要为了“World 要完整”强迫 EgoDex 重建整个世界。

---

# 7. 我会把 EgoDex 的 sample 定义成这样

```text
Observation
├── 1~4k current non-hand scene points
├── 22 native hand semantic points + H4
└── 32~128 target-object current points

Goal
└── target-object endpoint delta @ t+24

Supervision
├── hand future 24 steps
└── target-object tracked future 24 steps
```

数学上：

\[
S_t
\]

只是当前 context；

\[
H_{t-3:t}
\]

是手历史；

\[
G=
\{[p_i,\Delta p_{i,24}]\}
\]

是任务；

然后：

\[
(\hat U,\hat V)
=
F(S_t,H_{t-3:t},G).
\]

这已经足够。

---

# 8. 对当前代码需要改一点数据合同

现在 `goal_collate/_cap_scene` 还有比较强的“scene point 必须属于某个监督 object slot”的历史假设。

但对 EgoDex，我们最好允许：

\[
\boxed{\text{context-only scene point}}
\]

即：

- 有 `xyz/features`；
- 可以进 PTv3；
- 但没有 `scene_target`；
- 不属于任何需要预测 future 的 object。

我会给它一个明确的 background/context id，例如：

\[
scene\_object=-2
\]

而：

- `-1` 继续保留 hand；
- `0...O-1` 是有 future supervision 的 object；
- `-2` 是纯 context。

然后 PTv3照常吃它，但 SceneFutureHead 不给它 loss。

这对以后 VITRA、真实相机数据都很重要。

---

## 所以我的最终想法

不要把“场景点”理解成：

> 世界模型必须预测这些点未来全部怎么走。

更合适的是：

\[
\boxed{
\text{场景点 = perception context}
}
\]

而：

\[
\boxed{
\text{目标物体 tracked points = dynamics supervision}
}
\]

这两个集合可以重叠，但**职责不同**。

如果按这个思路改，我们的 EgoDex pipeline 会轻很多：

\[
\boxed{
\text{current sparse scene}
+
\text{native hand}
+
\text{sparse target tracks}
}
\]

我认为这是目前最适合大规模 EgoDex 的版本。