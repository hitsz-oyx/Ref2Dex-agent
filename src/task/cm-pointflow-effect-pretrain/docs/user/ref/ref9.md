我查完以后，结论很明确：

\[
\boxed{\text{EgoTouch 不能和我们当前整条 PointWorld 主线“完全匹配”}}
\]

但它对我们想加入的 **触觉分支非常匹配**。更准确地说，它适合训练

\[
\boxed{
(\tau^{hand}, C_t)\rightarrow C_{future}
}
\]

却**不能直接训练**

\[
\boxed{
(\tau^{hand}, C_t)\rightarrow(E^{object}_{future},C_{future})
}
\]

因为它缺我们最关键的 **物体 3D/6DoF future supervision**。

### 先看它真正有什么

官方 EgoTouch 有 1,891 个 episode、约 200 万帧、208 个任务，30 Hz，同步提供三路 RGB、双手 21×3 的 3D hand joints，以及左右手连续 pressure grid。官方 HDF5 格式里 pressure 是 `T×21×21`，hand pose 是左右各 `T×21×3`，同时还有 chest/left/right camera pose 和 timestamps。[GitHub](https://github.com/Jianyi2004/TouchAnything/blob/main/README.md?utm_source=chatgpt.com)

所以对于我们：

\[
\tau^{hand}_{t:t+24}
\]

它是非常自然的，因为：

- 30 Hz：和我们现在 chunk=24/30 Hz 完全对齐；
- 3D hand trajectory：现成；
- tactile：和 hand pose frame-level 同步；
- 双手都有；
- 数据量也够大。

而且一个独立整理版甚至验证过 tactile 和 `joints_worldspace` 是**逐帧一一对应**的，没有重采样，pressure length 和视频帧数一致。[Hugging Face](https://huggingface.co/datasets/MIT-Media-Lab/egotouch-annotations-v1?utm_source=chatgpt.com)

---

## 真正不匹配的地方：没有 object trajectory

这是最大问题。

官方公开的数据结构里能看到：

- RGB；
- hand joints；
- pressure；
- camera poses；
- glove/object masks；
- object IDs；

但**没有**

\[
T^{object}_{world}(t)
\]

这种逐帧 object 6DoF pose，也没有像 GRAB/OakInk2 那种已经对齐好的：

\[
\text{object mesh}
+
SE(3)_{1:T}.
\]

公开 HDF5 的 `poses/` 只有 chest/left/right camera pose；`masks/` 有 object segmentation，但不是 object 6DoF。[GitHub](https://github.com/Jianyi2004/TouchAnything/blob/main/README.md?utm_source=chatgpt.com)

所以我们最核心的：

\[
\tau^{hand}
\rightarrow
E^{object}
\]

在 EgoTouch 上**没法直接监督**。

这一点和 OakInk2/GRAB 正好相反：

- OakInk2 / GRAB：object geometry / pose 强，真实 tactile 没有；
- EgoTouch：真实 tactile 强，object 6DoF 弱。

因此不能指望一个 EgoTouch 数据集把我们所有东西都包掉。

---

## 第二个不完全匹配：它的 tactile representation 和我们现在不是一种东西

EgoTouch 是：

\[
C_t\in\mathbb R^{21\times21}
\]

pressure map。

而我们 Isaac 里目前拿的是类似：

\[
f^{finger}_t\in\mathbb R^3
\]

几个 rigid body 的 net contact force。

这不是同一个空间。

EgoTouch tactile 是**手套表面的二维 pressure field**；我们现在的是**刚体上的3D合力**。

所以如果想把 EgoTouch pretrain 直接迁到 Isaac：

\[
C^{EgoTouch}
\rightarrow
C^{Isaac}
\]

需要一个共同 representation。

比如我更建议统一成：

\[
\boxed{
C_t=
\{c_j,p_j\}_{j=1}^{N}
}
\]

其中：

- \(c_j\)：hand surface 上的语义位置/token；
- \(p_j\)：pressure/contact magnitude。

也就是说不要让 EgoTouch 保持一张 21×21 图、Isaac 保持 5×3 force，最后硬拼。

最好统一成 **hand-surface tactile tokens**。

---

## 第三个问题：3D hand 并不是我们想象中的完美 MoCap GT

这里官方描述有一点容易误读。

README 概述写的是 42-joint bimanual 3D pose / motion capture；但它实际发布的 HDF5 字段明确叫：

```text
wilor_left_joint_xyz
wilor_right_joint_xyz
```

也就是 WiLoR 得到的 hand pose，并带 valid mask。[GitHub](https://github.com/Jianyi2004/TouchAnything/blob/main/README.md?utm_source=chatgpt.com)

另外一个整理后的 annotation release提供了：

- MANO hand pose；
- `joints_worldspace`；
- world/camera transforms；

但这个整理版后来甚至专门修复过**左手 MANO rotation convention**，说明 hand coordinate/rotation pipeline 也不是完全无坑。好在它说明 `joints_worldspace` 和 tactile 本身不受这个 left-rotation fix 影响。[Hugging Face](https://huggingface.co/datasets/MIT-Media-Lab/egotouch-annotations-v1-leftfix?utm_source=chatgpt.com)

所以如果我们只用：

\[
21\times3\text{ joints}
\]

做 \(\tau\)，问题不大。

但如果想进一步用：

\[
MANO\ pose\rightarrow robot\ retarget
\]

就必须认真处理 coordinate convention。

---

## 第四个问题：没有 robot action

EgoTouch 是人类数据。

所以没有：

\[
A^{robot}
\]

也没有：

\[
q^{robot},\dot q^{robot}.
\]

因此它不能拿来训练我们现在的 execution layer：

\[
\tau\rightarrow A_{\text{native}}.
\]

这一层仍然必须来自：

- Isaac Gym；
- robot demonstration；
- 或 retarget 后的机器人数据。

这和 PointWAM 一样：human data 更适合 world/trajectory pretraining，robot data 才负责 retarget/control。

---

# 但它非常适合我们哪一块？

我认为非常适合单独预训练：

\[
\boxed{
(\tau^{hand}_{past/future},C_{past})
\rightarrow
C_{future}
}
\]

甚至先做最干净的三组：

\[
C_t\rightarrow C_{future}
\]

\[
C_t+\tau^{hand}_{future}\rightarrow C_{future}
\]

\[
C_t+\text{shuffle}(\tau^{hand}_{future})
\rightarrow C_{future}.
\]

如果第二组明显优于第一组，而且 shuffle 后掉下来，就证明：

\[
\boxed{
\text{future hand trajectory 对 future tactile/contact 有预测价值}
}
\]

这个和我们 PointWorld 现在：

\[
H+\tau\rightarrow E^{object}
\]

的实验逻辑完全一致。

---

# 如果硬要把 EgoTouch 接入完整 PointWorld，有两种办法

第一种，也是我更推荐的：

\[
\boxed{
\text{分开预训练，再在 Gym/机器人数据上融合}
}
\]

即：

EgoTouch：

\[
\tau^{hand}\rightarrow C_{future}
\]

OakInk2/GRAB：

\[
\tau^{hand}\rightarrow E^{object}_{future}
\]

Gym：

\[
\tau^{robot}
\rightarrow
(E^{object}_{future},C^{sim}_{future})
\]

最后在 Gym 上把两个 head 对齐：

\[
\boxed{
PW(H,\tau)
\rightarrow
(E^{object},C)
}
\]

这样最现实。

第二种是给 EgoTouch 自己补 object tracking。

因为它有：

- 三路 RGB；
- camera poses；
- object masks / IDs；

理论上可以做：

\[
RGB+mask+camera
\rightarrow
object\ 3D/6DoF
\]

然后构造：

\[
\tau^{hand}
\rightarrow
(E^{object},C)
\]

但这就相当于我们又新建一个 **EgoTouch object reconstruction/tracking pipeline**。工作量不会小，而且得到的 object 6DoF 是估计量，不是 GT。

我不会把它作为第一选择。

---

## 对我们当前接口逐项打分

| 我们需要的东西 | EgoTouch | 匹配程度 |
|---|---|---:|
| 30 Hz temporal data | 有 | ★★★★★ |
| future hand trajectory | 双手21 joints | ★★★★★ |
| real tactile | dense pressure | ★★★★★ |
| tactile-hand 同步 | frame-level | ★★★★★ |
| world/camera coordinate | 有 camera pose/world joints | ★★★★☆ |
| object segmentation | 有部分 masks/IDs | ★★★☆☆ |
| object 6DoF trajectory | **没有直接提供** | ★☆☆☆☆ |
| object point cloud/mesh | **没有标准监督接口** | ★☆☆☆☆ |
| robot q/action | 没有 | ☆☆☆☆☆ |
| task success labels | 不是我们的机器人任务定义 | ★☆☆☆☆ |

所以如果你问：

> “它是不是能完全替掉我们 OakInk2/GRAB/Gym 数据？”

答案是：

\[
\boxed{\textbf{不能。}}
\]

如果问：

> “它是不是目前非常适合给我们的 WM 补 tactile pretraining？”

答案是：

\[
\boxed{\textbf{是，而且非常合适。}}
\]

我现在会把它定位成我们数据体系里的**第四类数据源**：

\[
\text{OakInk2/GRAB}
\Rightarrow
3D\ geometry/effect
\]

\[
\text{EgoTouch}
\Rightarrow
\boxed{tactile/contact dynamics}
\]

\[
\text{Gym}
\Rightarrow
\boxed{robot action + object effect + simulated contact}
\]

最后 Gym/机器人数据负责把三者粘起来。

另外，官方 EgoTouch Hugging Face 当前约 **88.5 GB**，但项目 README 仍保留“dataset upload in progress / accessible files may be incomplete”的提示，因此如果真准备接入，最好先做一次文件级 completeness audit，而不是默认所有 1,891 episode 都已经完整可用。[Hugging Face](https://huggingface.co/datasets/zhouzhoujy/EgoTouch/tree/main?utm_source=chatgpt.com)

**所以我不建议现在直接大规模下载训练。先拿 5–10 个 episode 做一个 `hand trajectory ↔ tactile` schema audit，确认 pressure 的空间语义、world-joint 坐标和我们想要的 tactile-token 映射。**这一步通过以后再决定是否把 EgoTouch 正式纳入 PointWorld。