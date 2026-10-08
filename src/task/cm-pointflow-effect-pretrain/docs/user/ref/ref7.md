可以解决，但**不能跨数据集配对**。正确做法是：让手轨迹和物体轨迹都从 **同一个 EPIC clip** 里产生。

我现在看下来，有一条比我们自己从头做更合适的路线：

### 1. 最干净的现成桥梁：EPIC-Contact

2026 年刚公开的 **EPIC-Contact** 已经在 EPIC-Kitchens 上做了我们缺的这一步：它提供约 **2.3K clips / 62.3K frames**，每帧有相机坐标下的 MANO 手 mesh，同时有 object pose/mesh；中央帧是高质量拟合，其他帧提供传播得到的 pseudo-GT，并带逐帧质量分数。[Hugging Face](https://huggingface.co/datasets/Sid2697/epic-contact/blob/0df7796dba1acdc4d0260b69662524916e9f7079/DATASET.md?utm_source=chatgpt.com)

也就是说，对这部分数据，我们根本不用再做：

\[
\text{ObjectForesight object}+\text{另外估手}
\]

直接：

\[
\boxed{
\text{EPIC-Contact RGB}
\rightarrow
\text{MANO hand}_{1:T}
+
\text{object}_{1:T}
}
\]

再从 MANO 取我们需要的 11 个 semantic keypoints，就能生成：

\[
(H,A,Z)
\]

完整样本。

这个数据集甚至就是专门解决 in-the-wild egocentric hand-object 3D pose 的。HOPformer 官方代码和 checkpoint 也已经放出来了。[arXiv](https://arxiv.org/abs/2606.30598?utm_source=chatgpt.com)

---

### 2. 但 EPIC-Contact 只有 2.3K clips，ObjectForesight 有 29K clips

所以如果我们想吃完整的 ObjectForesight-EPIC：

\[
29,006\ clips
\]

最自然的办法就是：

\[
\boxed{
\text{同一个EPIC视频}
\rightarrow
\begin{cases}
\text{ObjectForesight}&\rightarrow object\ 6DoF\\
\text{3D hand estimator}&\rightarrow hand\ trajectory
\end{cases}
}
\]

这样不会有“不同视频拼接”的问题。

ObjectForesight 已经给我们：

- RGB clip；
- camera extrinsics；
- intrinsics；
- metric-ish depth；
- object mask；
- hand mask；
- object mesh；
- object 6DoF。[Hugging Face](https://huggingface.co/datasets/raivn/ObjectForesight-EPIC?utm_source=chatgpt.com)

所以只需要补：

\[
RGB_t\rightarrow hand_t^{3D}.
\]

---

## 3. 手用什么估？

现在有三个现实选择。

**WiLoR** 最省事，已经是很成熟的 in-the-wild MANO hand reconstruction，而且 EPIC-Contact 自己生成 hand prior 就用了 WiLoR。[GitHub](https://github.com/rolpotamias/WiLoR?utm_source=chatgpt.com)

**WildHands** 更专门针对 egocentric 场景，而且论文专门在 EPIC-Kitchens 上评估过，对严重遮挡、近距离透视这些问题就是针对性设计的。[arXiv](https://arxiv.org/abs/2312.06583?utm_source=chatgpt.com)

**HOPformer** 最强的一点是直接 joint hand-object estimation，而且刚刚在 EPIC-Contact 上发布，但它的 EPIC 数据主要是有限的 rigid object classes，所以我不会用它直接扫完整29K ObjectForesight。[GitHub](https://github.com/Sid2697/HOPformer?utm_source=chatgpt.com)

因此我会选：

\[
\boxed{\text{WiLoR/WildHands 做全量}}
\]

EPIC-Contact/HOPformer 用来做**质量验证和校准**。

---

# 4. 真正困难的不是手指姿态，而是“绝对3D位置”

这点一定要注意。

WiLoR 给出的 MANO hand pose 很好，但直接预测的 hand placement 未必就是我们需要的精确 metric world position。

而 ObjectForesight 已经有：

\[
D_t,\quad K,\quad T^w_{c,t}.
\]

所以我们可以借 ObjectForesight 自己的 metric geometry 对手做定位。

大致是：

\[
\text{WiLoR}
\rightarrow
\text{MANO shape + articulation + 2D placement}
\]

然后利用：

- EgoHOS/VISOR hand mask；
- ObjectForesight depth；
- 相机内参；

把 hand root/depth 对齐到真实 scene depth，再得到：

\[
T^c_{\text{hand}}.
\]

最后：

\[
T^w_{\text{hand}}
=
T^w_cT^c_{\text{hand}}.
\]

于是 hand 和 object 最后都在**同一个 ObjectForesight world frame**：

\[
\boxed{
P^{hand,w}_t,\quad
P^{object,w}_t
}
\]

这样才能真正用于我们的模型。

---

# 5. 我不建议一上来自己做完整29K

这里有一个很好的验证办法。

EPIC-Contact 已经有 hand pseudo-GT。

我们可以先找：

\[
\text{EPIC-Contact}
\cap
\text{ObjectForesight-EPIC}
\]

也就是两个数据源中相同的 EPIC `video_id + frame_id`。

对于这些重叠片段：

ObjectForesight 给：

\[
\text{object pose + depth + camera}
\]

EPIC-Contact 给：

\[
\text{MANO hand trajectory}.
\]

这样首先能直接构造一批完整：

\[
(H,A,Z).
\]

同时还可以拿它验证我们的：

\[
\text{WiLoR/WildHands + depth alignment}
\]

到底误差有多大。

如果这个 pipeline 在 EPIC-Contact 上表现可靠，再扩到29K。

这个顺序会比现在 EgoDex→ObjectForesight 的方式稳很多。

---

## 6. 这样我们的数据体系就会非常自然

### 高质量3D

OakInk2 / ARCTIC / GRAB / HOT3D：

\[
\text{GT hand}+\text{GT object}.
\]

### 中等质量 in-the-wild

EPIC-Contact：

\[
\text{pseudo/fit hand}+\text{pseudo/fit object}.
\]

### 超大规模

ObjectForesight-EPIC：

\[
\text{estimated hand}
+
\text{existing ObjectForesight object}.
\]

### EgoDex

\[
\text{GT hand}
+
\text{PointWAM-style scene track}.
\]

这样我们不再要求每个数据集都具有完全相同的原始 annotation，而是最终全部转换到统一：

\[
\boxed{
H_t,\quad A_{1:24},\quad Z_{1:24}
}
\]

合同。

---

### 我目前最推荐的实际顺序

不要先解决“29K EPIC 全部怎么补手”。

先做：

\[
\boxed{
\text{EPIC-Contact}\rightarrow
\text{我们的 PointWorld 格式}
}
\]

因为它已经同时有手和物体。

然后检查它和 ObjectForesight 的重叠片段，再做：

\[
\boxed{
\text{WiLoR/WildHands}
+
\text{ObjectForesight depth/camera}
}
\]

的 hand reconstruction 对照。

**如果这一步可靠，ObjectForesight-EPIC 就可以真正变成我们的大规模 \((H,A,Z)\) 数据源。**

这比继续让 EgoDex 自己恢复 object 6DoF 简单得多，因为现在只需要补“手”这一侧，而且目前已经有 EPIC-Contact 作为直接验证集。