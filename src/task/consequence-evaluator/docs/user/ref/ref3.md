对，我把 `consequence-evaluator` 分支里**数据集相关提交也重新完整看了一遍**。你的感觉是对的：目前不只是 evaluator 还有问题，**数据集整合这条线也有几个实质性问题，其中两个我认为必须先修，否则继续训练会把结论搞混。**

目前分支里的数据工作主要有两块：

- `d6fb20f → 7c85d5d`：OakInk2 + GRAB + ARCTIC + ContactPose 四源混合训练；
- 最新 `3666bbe`：EPIC-Contact 与 ObjectForesight overlap 的 candidate audit。

我先给结论：

> **四源混合现在“工程上能跑”，但 supervision 并没有真正同质化；最新 EPIC-Contact 则连 training-ready 都还没到。**

---

## 一、最新 EPIC-Contact 这部分确实还有明显问题

最新提交：

`3666bbe audit EPIC-Contact paired candidates`

其实很谨慎，它自己就写了：

```text
status='CANDIDATE_ONLY'
training_allowed=False
```

所以它**还不是一个已经解决的训练数据 pipeline**。

而且我看到至少 4 个具体问题。

### 1. ObjectForesight 相机外参 convention 处理有 bug 风险

现在代码直接：

```python
camera_extrinsics = extrinsics[local]

poses = camera_extrinsics @ poses
hand = R_camera @ hand + t_camera
```

并且直接记录：

```python
convention='c2w'
```

也就是说代码假设 ObjectForesight 的 `extrinsics` 都是：

\[
T^w_c.
\]

但 ObjectForesight 官方数据明确说：

> 每个 trajectory 都有自己的 `extrinsics convention`，大部分是 `c2w`，但有 **782 条是 `w2c`**。[Hugging Face](https://huggingface.co/datasets/raivn/ObjectForesight-EPIC?utm_source=chatgpt.com)

所以现在不能直接：

\[
T^w_c = extrinsics
\]

必须读官方 metadata：

\[
\texttt{extrinsics\_conv}
\]

如果是：

\[
w2c
\]

就要先求逆：

\[
T^w_c=(T^c_w)^{-1}.
\]

这个我认为属于**真正的坐标系 bug**，不是小优化。

---

### 2. hand validity 在 resample 后被直接抹掉了

源数据本来有：

```python
right_valid
high_confidence
hand_valid_ok
```

前面确实做了：

```python
hand_valid &= quality_hand_valid
```

但是重采样以后直接：

```python
candidate_hand_valid[:, 0] = True
```

也就是说：

> 只要生成了30Hz插值点，最后全部当成有效手。

这个不合理。

特别是 EPIC-Contact 官方明确区分：

- manually verified central frame；
- propagated per-frame pseudo-GT；
- 每帧有质量信息用于过滤。[Hugging Face](https://huggingface.co/datasets/Sid2697/epic-contact/blob/0df7796dba1acdc4d0260b69662524916e9f7079/DATASET.md?utm_source=chatgpt.com)

现在代码虽然把：

```python
source_high_confidence
source_hand_valid
interpolated_frame_mask
```

保存下来了，但**并没有真正用它们决定训练有效性**。

更严重的是：

```python
gap_target = ...
```

发现 source frame 有 gap 后，仍然照样插值，然后：

```python
pose_valid = np.ones(...)
candidate_hand_valid = True
```

这意味着未来一旦把 `training_allowed` 放开，gap 中间人为插出来的数据也会进入训练。

正确做法应该是：

\[
\boxed{
\text{gap / low-confidence}
\rightarrow mask
}
\]

而不是：

\[
\text{记录一下但照样valid}.
\]

---

### 3. 左手被直接丢了

代码现在只读取：

```python
mano.j3d.cam.r
right_valid
```

然后：

```python
annotated_side='right'
left_hand_present=False
```

但 EPIC-Contact 官方 per-frame 数据实际上同时有：

- `mano.j3d.cam.l`
- `mano.j3d.cam.r`
- `left_valid`
- `right_valid`

。[Hugging Face](https://huggingface.co/datasets/Sid2697/epic-contact/blob/0df7796dba1acdc4d0260b69662524916e9f7079/DATASET.md?utm_source=chatgpt.com)

所以当前 converter 本质上是：

\[
\text{EPIC-Contact}
\rightarrow
\text{right-hand-only subset}
\]

而不是我们现有：

\[
[\text{right},\text{left}]
\]

统一双手 contract。

如果 clip 真的是左手操作，它现在基本没法用。

这个不是致命 bug，但会直接损失数据，而且会形成很强的 hand-side bias。

---

### 4. 当前并没有真正完成“ObjectForesight + EPIC-Contact 配对”

这点容易被最新 commit 名字误导。

现在 `--objectforesight-poses` 的作用基本只是：

- 检查 frame range；
- 检查 frame id 是否对应。

代码明确写了：

```python
used_for_training=False
reason='independent overlap audit only'
```

而最终 object pose 实际还是：

\[
\text{EPIC-Contact object.rot / object.cam\_t}
\]

并没有使用 ObjectForesight 的：

\[
T^c_o.
\]

因此现在做的是：

> “EPIC-Contact clip 能否借 ObjectForesight camera trajectory 转到 world frame”

而不是：

> “ObjectForesight 的29K轨迹已经成功补上 EPIC-Contact hand”。

这两个差别很大。

所以之前我们讨论的目标：

\[
\text{ObjectForesight object}
+
\text{EPIC hand}
\]

实际上还没完成。

---

# 二、更加严重的是四源混合里的 ContactPose

这个问题我认为比 EPIC-Contact 更值得马上处理。

现在训练是：

\[
OakInk2:50\%
\]

\[
GRAB:20\%
\]

\[
ARCTIC:20\%
\]

\[
ContactPose:10\%.
\]

但是 ContactPose 的“手轨迹”跟另外三个源不是同一种东西。

代码里自己写得很清楚：

```text
Fixed hand articulation is an annotation assumption,
not a measurement of finger motion
```

ContactPose 的做法是：

先有一套固定：

\[
J^{hand}_{canonical}
\]

然后根据每一帧：

\[
T^w_o(t)
\]

和：

\[
T^o_h(t)
\]

算：

\[
P^w_{hand}(t)
=
T^w_o(t)T^o_h(t)P_{hand}.
\]

尤其对于固定抓取，它本质上很接近：

\[
P_{hand}(t)
\approx
T_o(t)\cdot const.
\]

问题来了。

我们训练的是：

\[
(H,A_{hand}^{future})
\rightarrow
Z_{object}^{future}.
\]

那么 ContactPose 里：

\[
A_{hand}^{future}
\]

本身就是通过 object motion 构造出来的。

换句话说：

\[
\boxed{
future\ object
\rightarrow future\ hand
}
\]

然后训练模型：

\[
future\ hand
\rightarrow future\ object.
\]

这个很容易形成**近似标签泄漏**。

它不是严格意义上的 leakage，因为 hand/object 是共同运动，但它会让模型学到一个非常简单的规律：

> “手整体移动多少，物体就移动多少。”

对于 grasped-object transport 数据这个规律完全成立。

但对我们真正关心的：

> 手即将怎么动 → 会不会推动/抓住/滑落物体

就不是一回事。

所以 ContactPose 可以用来训练：

\[
\text{rigid transport}
\]

但我不认为它应该和 OakInk2 / GRAB / ARCTIC 等价地进入 **action-conditioned dynamics** 主监督。

---

## 这可能也是为什么 ContactPose 看起来特别容易改善

当前 step0：

| 数据 | h24 EPE |
|---|---:|
| OakInk2 | 9.79 mm |
| GRAB | 84.33 mm |
| ARCTIC | 76.01 mm |
| ContactPose | 40.11 mm |

step500：

| 数据 | h24 EPE |
|---|---:|
| OakInk2 | 10.49 |
| GRAB | 68.38 |
| ARCTIC | 69.34 |
| ContactPose | 34.76 |

ContactPose 改善并不能说明“模型学到了更好的 hand→object dynamics”。

它可能只是越来越会学：

\[
\text{hand rigid transport}\leftrightarrow
\text{object rigid transport}.
\]

这个需要单独解释。

---

# 三、四个数据源现在共用 OakInk2 normalization，我也不满意

这个现在代码里是明确写死的：

```text
normalization='reuse unchanged train-only OakInk parent statistics'
```

也就是说：

\[
\mu,\sigma
\]

全部来自 OakInk2：

- scene feature；
- action；
- translation；
- flow；
- rotation scale。

然后 GRAB / ARCTIC / ContactPose 全部拿 OakInk2 的统计量 normalize。

这对于 warm-start 有一个好处：

> 原模型参数不用重解释。

但是作为新的**四源联合训练**，问题很明显。

例如：

\[
EPE_{Oak}=9.8mm
\]

而：

\[
EPE_{GRAB}=84mm
\]

差快一个数量级。

这说明不同 source 的：

- 运动幅度；
- 手动作；
- object geometry；
- interaction statistics

明显不在同一分布。

而现在 loss 又使用：

\[
\frac{\Delta x-\mu_{Oak}}{\sigma_{Oak}}
\]

进行 normalized Huber。

于是不同 source 实际贡献给 optimizer 的梯度尺度不一定一致。

所以虽然 sampler 是：

\[
0.5/0.2/0.2/0.1
\]

但真实的梯度贡献并不一定是：

\[
50/20/20/10.
\]

这很可能也是为什么：

\[
OakInk:
9.79\rightarrow10.62
\]

一开始明显退化。

这当然可能只是 domain adaptation，但 normalization mismatch 会放大这个问题。

---

# 四、category 虽然代码没错，但语义没有真正统一

OakInk2 的三类是：

\[
0=\text{moving}
\]

\[
1=\text{static but program/near-hand}
\]

\[
2=\text{background static}.
\]

而 GRAB / ARCTIC / ContactPose 没有 program/background 信息，所以当前映射：

\[
1_{\rm native}\rightarrow0_{\rm mixed}
\]

\[
2_{\rm native}\rightarrow1_{\rm mixed}
\]

即：

\[
moving\rightarrow moving
\]

\[
near-static\rightarrow selected-static
\]

而 category2 永远不存在。

然后：

```python
[0.6, 0.2, 0.2]
```

会在 native source 里重新归一化成：

\[
\boxed{75\%\ moving +25\%\ static}
\]

不是原来的：

\[
60/20/20.
\]

代码里是明确这么设计的，因此不是 bug。

但结果就是：

> “balanced source panel”在四个数据集上的语义并不真正相同。

所以不能用一个统一的：

\[
balanced\ validation
\]

说四个 source 是完全 matched 的。

主指标最好明确限定为：

\[
\boxed{\text{moving-anchor}}
\]

这一点你们当前实验已经有做，但数据 sampler 的命名容易让人误解。

---

# 五、最新 EPIC-Contact 还有一个根本性的数据质量问题

EPIC-Contact 官方自己明确说：

它有两层标签：

**中央帧：**

高质量人工验证 GT。

**其他帧：**

传播得到的 pseudo-GT，而且：

> relative geometry 质量较好，但 **absolute camera placement 是 approximate**。[Hugging Face](https://huggingface.co/datasets/Sid2697/epic-contact/blob/main/README.md?utm_source=chatgpt.com)

而我们真正要训练的是：

\[
\Delta T_{object}^{world}.
\]

所以如果直接：

\[
T^w_c(\text{SpaTracker})
T^c_o(\text{EPIC-Contact})
\]

两个 independently estimated 系统的误差都会进入：

\[
\Delta T^w_o.
\]

尤其相机在动的时候：

\[
\text{camera error}
\]

很容易被误认为：

\[
\text{object effect}.
\]

这其实和我们之前 EgoDex 出现的问题有一点类似，只是轻很多。

所以在放进训练前必须做一个检查：

\[
\boxed{
\text{hand-object relative geometry 是否稳定}
}
\]

而不是只检查：

\[
R^\top R=I.
\]

我会重点看：

\[
d_{\rm hand-object}(t)
\]

是否平滑；

\[
T_h^{-1}T_o
\]

在稳定 grasp 段是否合理；

以及：

\[
\Delta T_o
\]

和 ObjectForesight 独立 pose 的一致程度。

---

# 六、所以我现在怎么看这批数据？

我会重新分级，而不是全都塞进一个 loader。

### A级：真正适合作为主 world-model supervision

\[
\boxed{
OakInk2 + GRAB + ARCTIC
}
\]

它们都有真正的动态：

\[
hand(t)
+
object(t)
\]

监督。

---

### B级：只做特定辅助监督

ContactPose：

\[
\boxed{\text{rigid grasp transport auxiliary}}
\]

不要把它当普通动态 manipulation data。

甚至可以不给它进入 action-conditioned main loss，而只做：

\[
SE(3)\ consistency / object-motion pretraining.
\]

---

### C级：pseudo-GT，先 audit 再训练

EPIC-Contact：

现在还处于：

\[
\boxed{CANDIDATE\_ONLY}
\]

这个状态其实是对的。

先把：

- c2w/w2c；
- valid mask；
- left/right hand；
- gaps；
- world alignment；
- ObjectForesight overlap；

这几个问题解决。

---

### D级：真正的大规模 weak data

ObjectForesight-EPIC / EgoDex。

这部分以后应该采用我们前面说的：

\[
\text{weak point trajectory supervision}
\]

而不是硬要求全部变成 OakInk2 那种 high-quality rigid SE3 supervision。

---

# 我现在最建议马上改的两件事

第一件非常明确：

**暂时把 ContactPose 从四源主 action-conditioned 训练里拿掉。**

先跑：

\[
OakInk2 + GRAB + ARCTIC
\]

三源。

因为这三个 source 的监督语义最接近。

ContactPose 单独做 auxiliary ablation。

第二件：

修 EPIC-Contact audit，至少必须做到：

\[
\boxed{
\text{读取 OF extrinsics convention}
}
\]

而不是默认 c2w；

并且：

\[
\boxed{
valid_{30Hz}
=
valid_{source}
\land high\ confidence
\land no\ large\ gap
}
\]

不能再：

```python
candidate_hand_valid=True
pose_valid=True
```

全部放开。

---

**所以你说“数据集部分似乎还是有问题”是对的。**

当前最严重的不是文件格式，而是：

\[
\boxed{
\text{不同 source 的 supervision semantics 没有真正对齐}
}
\]

尤其 **ContactPose 的 future hand 是 object-conditioned rigid transport**，这与我们要学习的“hand action → object consequence”不是同一种数据。

我现在会优先修这个，而不是继续扩第五、第六个数据集。