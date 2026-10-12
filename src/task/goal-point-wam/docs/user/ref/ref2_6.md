看了。现在 `main` 最新机器人主线到 **`054aba2`**。我们刚定的三项已经推进到一个很清楚的状态：

\[
\boxed{\text{A：R 已经成功；B：PTv3 hand readout 当前实现失败；C：还没开始。}}
\]

### 1. Chunk-\(R\) 已经救回来了

`9fca37b` / `6b841e3` 按我们刚才说的，把 loss 改成了：

\[
L=
4L_{a_0}
+
L_{a_{1:7}}
+
L_{\rm PD0}
\]

并且 checkpoint 按 **first-step PD XYZ p95 优先**选。

结果非常明显：

\[
\text{report first-PD p95}:
31.09\rightarrow19.51\text{ mm}
\]

recovery TRAIN 更明显：

\[
94.80\rightarrow12.78\text{ mm}
\]

最重要的真实 GT-\(\tau\) execution：

| 控制 | held |
|---|---:|
| teacher | 73 |
| 旧 \(R\) | 75 |
| **新 Chunk-\(R\)** | **74** |

全部 clipping=0。

所以这件事现在可以基本定下来：

\[
\boxed{\text{Chunk Transformer 的方向没问题，之前主要是 objective 不对。}}
\]

这个新 \(R\) 我建议直接保留，不要再回旧 MLP。

---

# 2. PTv3 explicit hand readout 已经实现，但当前接法不行

`d1a34e8` 到 `054aba2` 已经做了我们说的：

\[
\text{PTv3 full packed feature}
\rightarrow
\text{按 hand/keypoint ID 显式 gather }2\times11
\]

而且不经过 256 scene-token sampling。

实现现在是：

```python
hand, _ = gather_hand_tokens(scene, batch)
raw = self.observed_hand_condition(batch)

if raw is not None:
    hand = hand + raw
```

然后直接拿这个：

\[
f^{PTv3}_{hand}+f^{raw}_{hand}
\]

加到 trajectory query。

所以确实已经测试了我们刚讨论的结构。

但结果：

| Panel | 旧 h1 / first8 | 新 hand-anchor h1 / first8 |
|---|---:|---:|
| nominal | **6.51 / 16.06** | 8.59 / 25.00 |
| startup | **6.99 / 12.23** | 9.67 / 21.18 |
| t8 | **6.22 / 14.83** | 7.64 / 21.04 |
| t16 | **6.45 / 16.31** | 8.34 / 22.62 |
| t32 | **11.91 / 23.12** | 13.77 / 29.29 |

只有 startup first-PD：

\[
6.41\rightarrow4.44\text{ mm}
\]

有改善。

总体只有：

\[
10/21
\]

个 non-regression gate 通过，所以当前 B 是 `UNPROMISING`。

---

## 但我现在不认为“PTv3 hand token 没用”

原因非常具体。

新结构刚接进去时：

\[
\boxed{\text{nominal h1}=54.51\text{ mm}}
\]

而旧接口初始：

\[
6.94\text{ mm}.
\]

训练 200 steps 后，新接口从：

\[
54.51\rightarrow8.59\text{ mm}
\]

这其实说明一个很强的问题：

> **我们现在是把一个完全没有和 decoder 对齐过尺度/分布的 PTv3 feature，直接硬加进一个已经训练好的 query。**

也就是：

\[
q_{\rm new}
=
q_{\rm old}
+
f^{PTv3}_{hand}
\]

一上来就把 warm-start policy 整个扰乱了。

当前 `hand_anchored.py` 甚至**没有额外 projection、LayerNorm 或 gate**。

所以这个实验实际上混了两个问题：

1. PTv3 hand feature 有没有用；
2. 把一个新 feature 直接以系数 1 加进 pretrained decoder 会不会破坏原功能。

而结果显然第二个影响非常大。

---

# 3. 我认为 B 不应该就这样结束，应该做一个极小的 B2

不要再改 PTv3。

只把：

\[
q=q_{\rm old}+f_{hand}^{PTv3}
\]

改成：

\[
\boxed{
q=
q_{\rm old}
+
\alpha\,P(\operatorname{LN}(f_{hand}^{PTv3}))
}
\]

其中：

\[
\alpha=0
\]

初始化。

或者 projection \(P\) 最后一层直接 zero-init。

这样 frame0 时：

\[
\boxed{q_{\rm new}=q_{\rm old}}
\]

模型初始行为**严格等于现有 WAM**，然后训练过程中自己决定需要多少 PTv3 hand information。

这个设计能非常干净地区分：

> PTv3 hand feature 到底有用没用，

而不会因为直接 feature addition 把原模型炸掉。

甚至第一阶段只训练：

\[
P,\alpha,\text{trajectory decoder}
\]

PTv3 先冻住都可以。

我认为这个实验比现在直接宣布 B 失败合理得多。

---

# 4. C：predicted-\(\tau\) → E 目前确实还没做

仓库按照 gate 暂停了：

\[
\hat\tau
\rightarrow
W_{\rm frozen}
\rightarrow
\hat E.
\]

这个我觉得目前暂停是对的。

因为现在如果 B 本身让：

\[
\tau
\]

变差，再加入：

\[
L_E
\]

就没办法知道 effect consistency 是在帮 actor，还是在补偿一个坏 hand readout。

所以顺序现在应该是：

\[
\boxed{
\text{A 已通过}
\rightarrow
\text{B2 gated hand readout}
\rightarrow
\text{C predicted-}\tau\text{ world consistency}
}
\]

---

## 当前我会怎么定

**A 不再动。**

Chunk-\(R\) 已经完成它的目标：

\[
GT\ \tau
\rightarrow
74\ held
\]

基本和旧 \(R\) 的 75 一样。

**B 再给一次机会，但只改注入方式，不改 backbone。**

核心就是：

\[
\boxed{\text{direct add} \rightarrow \text{zero-init gated residual}}
\]

如果这个版本在**初始完全不破坏 old WAM**的情况下，经过适配仍然不能改善 trajectory/control，那时候才比较有资格说：

> explicit PTv3 hand readout 对我们目前 WAM 帮助有限。

然后再决定是保留旧 hand adapter 直接进入 C，还是继续改 hand 表征。

**C 现在还不该开。**

因为目前最大的新增信息就是：Chunk-\(R\) 已经证明可行，而 PTv3 hand readout 的第一次失败更像**接口适配失败**，还不像表征本身失败。