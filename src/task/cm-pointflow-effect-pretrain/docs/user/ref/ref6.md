是的，这两个点其实把问题说明白了：

## 1. 为什么 PointWAM 用 EgoDex 没有我们这么多问题？

因为 **PointWAM 根本没有把 EgoDex 转成“物体 mesh + 物体 6DoF”**。

它绕开了我们现在最困难的部分。

PointWAM 对 EgoDex 的处理是：

\[
\text{RGB视频}
\rightarrow
\text{2D scene tracks}
\rightarrow
\text{单目depth}
\rightarrow
\text{3D scene point trajectories}
\]

具体来说：

- 手：直接用 EgoDex 自带的 3D hand joints；
- scene：在每个 clip 第一帧铺大量 2D query points；
- 用 **CoTracker3** 跟踪这些点；
- 用 **VGGT-\(\Omega\)** 预测每帧 monocular depth；
- 用 EgoDex 的 metric hand depth 给 monocular depth 做尺度校准；
- 再结合 camera pose，把 2D tracks lift 到统一 world frame；
- 最多保留约 12k scene points。

它**不关心“这个点属于杯子还是桌子”**，也不要求：
- 分割目标物体；
- 重建物体 mesh；
- FoundationPose；
- 物体 canonical frame；
- 6DoF object pose；
- 每个物体的刚体对应关系。

论文明确说，他们这么做就是为了 **“不需要 task-specific object or keypoint selection”**。[Anonymous View](https://rt.http3.lol/index.php?q=aHR0cHM6Ly9hcnhpdi5vcmcvaHRtbC8yNjEwLjAyODQwdjE\&utm_source=chatgpt.com)

所以 PointWAM 的监督本质是：

\[
p_i^t \rightarrow p_i^{t+\tau}
\]

即：

> **这个场景点未来跑到哪里。**

而我们目前 OakInk2/PointWorld 路线是：

\[
T_{object}^{t}
\rightarrow
T_{object}^{t+\tau}
\]

再由 rigid transform 推整个 canonical object cloud。

这两条路线的要求完全不同。

---

### 这就是为什么 PointWAM 看起来“没问题”

它接受的 pseudo-GT 更弱，但更容易大规模生成：

\[
\boxed{\text{scene point trajectory}}
\]

而我们在 EgoDex 上试图生成的是：

\[
\boxed{\text{metric object mesh + persistent identity + 6DoF trajectory}}
\]

后者当然困难一个数量级。

而且 PointWAM 的 scene trajectory **也不是 GT**。它同样依赖：

- CoTracker3；
- VGGT-\(\Omega\) depth；
- hand-based metric scaling。

所以不能理解成“别人把 EgoDex 完美转成真3D了”。

更准确地说是：

> **别人选择了一个对伪标签更友好的世界模型表示。**

这个区别非常关键。

---

# 2. 那我们是不是应该考虑 PointWAM 那种 Pipeline？

**我认为应该。**

而且这可能比继续 EgoDex→ObjectForesight 更合理。

目前我们自己其实已经有：

\[
\text{手点}
+
\text{object canonical points}
+
\text{24步 object SE(3)}
\]

这对 OakInk2/HOT3D/ARCTIC 这种真3D数据很好。

但对于 EgoDex，没有必要非把它也塞进这个 schema。

完全可以允许两种 supervision：

### 强监督数据

OakInk2 / ARCTIC / GRAB / HOT3D：

\[
\text{persistent object points}
\rightarrow
\text{24-step exact-ish rigid trajectory}
\]

### 弱监督大数据

EgoDex：

\[
\text{scene points}
\rightarrow
\text{tracked 3D point trajectories}
\]

然后共享 PointWorld encoder / dynamics backbone。

这其实就是 PointWAM 的思路。

---

# 3. ObjectForesight 已经处理好的 EPIC 数据——**能用，而且很值得用**

我重新核对了它现在公开的数据。

不是只有论文里说“我们处理了 EPIC”。

作者已经把**处理后的原始提取结果直接放出来了**：

\[
29,006\ clips
\]

\[
34,286\ object\ trajectories
\]

其中：

- train：29,499 object trajectories
- val：4,787

总大小约：

\[
\boxed{0.84\ \text{TiB}}
\]

而且每个 object trajectory 已经包含：[Hugging Face](https://huggingface.co/datasets/raivn/ObjectForesight-EPIC?utm_source=chatgpt.com)

- RGB action clip；
- SpaTrackerV2 intrinsics；
- camera extrinsics；
- metric-ish depth；
- sparse 3D tracks；
- EgoHOS hand/object masks；
- SAM2 / amodal masks；
- **TRELLIS object mesh**；
- **FoundationPose 每帧 6DoF**；
- tracking quality；
- hand-contact flag。

最关键的是：

\[
\boxed{\texttt{foundationpose10/poses.npz}}
\]

里面直接有：

\[
T_c^o(t)
\]

即每帧 object-in-camera 6DoF。

换句话说：

> **它已经替我们做完了现在 EgoDex pipeline 最痛苦的部分。**

---

# 4. 所以我前面说“ObjectForesight-EPIC只能辅助”有点保守了

重新看完整字段之后，我会提高它的优先级。

对于我们现在 PointWorld 的 **object future** 部分，它几乎已经可以直接用：

\[
T_c^o(t)
\]

加：

\[
T_w^c(t)
\]

得到：

\[
T_w^o(t)
\]

再从 TRELLIS canonical mesh 固定采样：

\[
P_o
\]

得到：

\[
P_t=T_w^o(t)P_o
\]

于是直接生成：

\[
P_t\rightarrow P_{t+1:t+24}.
\]

也就是我们现在 OakInk2 的 object supervision。

**不需要重新跑 ObjectForesight。**

---

# 5. 但有一个重要缺口：它没有 EgoDex 那样的 3D hand GT

这是 ObjectForesight-EPIC 不能无脑直接拼进当前 OakInk2 loader 的原因。

目前它有：

- hand mask；
- hand-object detection；
- moved-by-hand flag；

但公开数据的核心 contract 里没有：

\[
\text{3D semantic hand keypoints}_{1:24}.
\]

而我们当前 PointWorld 输入里，action 是：

\[
\text{left/right hand 11 semantic points}_{1:24}.
\]

所以如果完全保持当前 architecture：

\[
(H,A_{\rm hand})\rightarrow object\ future
\]

ObjectForesight-EPIC **还缺 action 这一边**。

---

# 6. 但这个缺口比 EgoDex 当前问题容易得多

现在我们做 EgoDex 是：

> 手已经有 GT，缺物体 → 自己重建整个物体系统。

这是最难的。

如果换 ObjectForesight-EPIC：

> 物体已经处理好了，只缺 hand trajectory。

我们只需要补：

\[
RGB\rightarrow 3D\ hand.
\]

这个问题明显成熟很多。

例如可以用 PointWAM 自己使用的思路：

- VITRA-style 3D hand reconstruction；
- MANO hand reconstruction；
- 或其它现成 ego-hand 3D estimator。

而且**即使 hand 有些误差，也只是 action conditioning 噪声**。

相比之下，物体 pose 有误差会直接污染：

\[
\text{world model target}.
\]

这是很大的区别。

---

# 7. 还有一个更激进、我觉得更合理的方案

实际上 ObjectForesight-EPIC 里已经有：

\[
\text{SpaTracker 2D tracks}
+
\text{depth}
+
\text{camera}.
\]

所以我们甚至可以同时得到两种 supervision。

### A. Object-centric

\[
\text{mesh + 6DoF}
\rightarrow object\ point\ trajectory
\]

### B. PointWAM-style scene-centric

\[
\text{SpaTracker tracks + depth + camera}
\rightarrow scene\ point\ trajectories
\]

也就是说一份数据可以同时给：

\[
\boxed{
\text{object rigid dynamics}
+
\text{generic scene dynamics}
}
\]

这反而比 EgoDex 更方便。

ObjectForesight 官方模型本身就是从这批数据训练：

\[
\text{scene point cloud}
+
\text{past object poses}
\rightarrow
\text{future 6DoF trajectory}.
\]

公开 repo 给出的 EPIC 结果已经做到 translation ADE 大约 **1.9 cm**、FDE 约 **3.5 cm**。[GitHub](https://github.com/RustinS/ObjectForesight?utm_source=chatgpt.com)

所以这批数据至少已经被作者实际证明能训练 3D dynamics model，而不是只发布了“看起来能用”的中间产物。

---

# 8. 那现在应该怎么选择？

我现在会把数据路线改成：

### 第一层：高质量真3D

\[
\boxed{
OakInk2 + ARCTIC + GRAB + HOT3D
}
\]

负责可靠 supervision。

### 第二层：大规模 pseudo-3D

\[
\boxed{
ObjectForesight\text{-}EPIC
}
\]

负责大量 object/scene dynamics。

### 第三层：EgoDex

这里有两个选择。

**方案A：继续 PointWAM-style 转换**

不再恢复 object 6DoF，只做：

\[
\text{CoTracker3}
+
\text{VGGT depth}
+
\text{native hand GT}
\]

生成 generic scene flow。

**方案B：干脆 EgoDex 只用于 hand/action side。**

---

# 9. 我现在反而更推荐 A

因为 PointWAM 已经证明这种预处理在 **1.15M human demonstration episodes** 上可以真正产生策略收益：

> human-video pretraining 给 DexJoCo 平均成功率带来 +56.9pp；去掉 scene-trajectory supervision，又掉约 10.9pp。[Anonymous View](https://rt.http3.lol/index.php?q=aHR0cHM6Ly9hcnhpdi5vcmcvaHRtbC8yNjEwLjAyODQwdjE\&utm_source=chatgpt.com)

所以它不是“一个可能的简化方案”，而是目前已经有实验证据支持的 pipeline。

并且论文附录把转换写得非常明确：

\[
RGB
\rightarrow CoTracker3
\rightarrow VGGT\text{-}\Omega\ depth
\rightarrow hand\ scale
\rightarrow world\ 3D\ tracks.
\]

---

# 10. 所以我对当前 EgoDex 支线的建议会变

**不要再扩大 EgoDex→ObjectForesight。**

蓝色方块 pilot 可以结束，用于说明这条 pipeline 工程上可行。

但大规模 EgoDex 建议直接改成：

\[
\boxed{
\text{PointWAM-style dense scene trajectory extraction}
}
\]

而 ObjectForesight 的角色改成：

\[
\boxed{
\text{直接下载官方处理好的 EPIC pseudo-3D}
}
\]

而不是：

\[
\text{拿它的代码再处理 EgoDex}.
\]

这能省掉非常多工程。

---

## 最终我会这样排优先级

\[
\boxed{\text{OakInk2 full}}
\]

当前已经在跑。

然后：

\[
\boxed{\text{ObjectForesight-EPIC}}
\]

直接接作者现成数据。

再：

\[
\boxed{\text{EgoDex PointWAM-style scene tracks}}
\]

不再 object reconstruction。

最后：

\[
\boxed{\text{HOT3D/ARCTIC/GRAB}}
\]

作为高质量3D锚点和 validation。

---

这里最核心的认识变化是：

> **我们之前默认“world model = object 6DoF world model”，所以 EgoDex 很难。PointWAM 证明 world model 完全可以是 persistent scene-point trajectories。**

这可能正是我们现在 pipeline 应该做的简化。

而 **ObjectForesight-EPIC 是可以直接用的**；真正要决定的只剩一个工程问题：我们是把它适配成当前 `object SE(3) → 512 points` 的 OakInk2 contract，还是趁这个机会把数据 contract 扩展成 **PointWAM-style scene flow + object rigid flow 双监督**。我现在更倾向后者，因为这样 EgoDex 也自然接进来了。