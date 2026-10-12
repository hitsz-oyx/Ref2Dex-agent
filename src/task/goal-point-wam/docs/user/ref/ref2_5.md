看了。现在机器人主线最新有效提交是 **`d840ec3`**，整体 `main` HEAD 后面还有一个 `c64a42d`，但那个主要是 EgoDex/video 线。

目前我们刚定的三项里，**只做完了第 1 项 Chunk-\(R\)**；第 2 项 PTv3 hand anchor 和第 3 项 predicted-\(\tau\) world consistency 都按 gate 暂停了。

### Chunk-\(R\) 的结果

现在确实实现了我们说的结构：

\[
24\times\tau\text{ token} + state\ token
\rightarrow
2\text{-layer Transformer}
\rightarrow
8\times12D\ action
\]

参数是：

- hidden 128；
- 4 heads；
- FFN 512；
- 2 decoder layers；
- 8 learned action queries；
- 每 tick 重规划；
- **只执行 action0**。

并且不是实现 bug。标签、GT \(\tau\)、当前 state、native action conversion、首步 dispatch 都独立审计过，误差 max=0。

但是物理 gate 很明确：

| 控制方式 | held |
|---|---:|
| Teacher | 73 |
| **GT \(\tau\) + 旧 \(R\)** | **75** |
| **GT \(\tau\) + Chunk-\(R\)** | **0** |

全部 clipping=0。

所以现在：

\[
\boxed{\text{问题已经和 WAM 无关，Chunk-R 自己的 GT execution 就没过。}}
\]

因此仓库暂停 B/C 是合理的。

---

## 但我不认为这说明 Chunk Transformer 方向错了

我看了现在的训练代码，反而发现一个很具体的问题。

### 当前 loss 和真正部署目标不一致

现在训练的是：

\[
L=
\frac1{8\times12}
\sum_{k=0}^{7}
\left|
\frac{\hat a_k-a_k}{\sigma_k}
\right|
\]

也就是：

> **8 个 action × 12 个维度全部等权 standardized L1。**

checkpoint 选择也是：

\[
\text{development all8 L1}
\]

最低的那个。

但真实部署：

\[
\boxed{\text{每次只执行 }a_0}
\]

所以实际我们真正关心的是：

\[
a_0
\]

是否极准。

这两个优化目标并不一致。

现在 report 上：

\[
\text{first native MAE}\approx0.0229
\]

看起来数值不大，但转换成腕部 PD：

\[
\boxed{\text{first PD XYZ p95}=31.09\text{ mm}}
\]

而 recovery TRAIN 更是：

\[
94.80\text{ mm}.
\]

对于抓取来说 3 cm 的首步 wrist tail error 已经非常大了。

旧 \(R\) 能 75 held，恰恰说明：

\[
\boxed{\tau^{GT}\text{本身没问题，执行任务也不是不可解。}}
\]

新 \(R\) 单纯没学到足够准确的**当前控制**。

---

# 所以我现在不会回退到旧 \(R\)，而是先修 Chunk-\(R\) 的训练目标

我建议保留整个 Transformer 结构，只改 objective。

例如：

\[
L=
\lambda_0L_{a_0}
+
\lambda_cL_{\rm chunk}
\]

其中：

\[
L_{a_0}
=
D(\hat a_0,a_0)
\]

\[
L_{\rm chunk}
=
\frac1{7}
\sum_{k=1}^{7}D(\hat a_k,a_k)
\]

第一版我甚至会直接：

\[
\boxed{\lambda_0=4,\quad\lambda_c=1}
\]

或者时间衰减：

\[
w=[4,2,1,1,0.5,0.5,0.5,0.5].
\]

这样 chunk 仍然提供 temporal context / regularization，但是模型明确知道：

> **第一步才是实际会被执行的动作。**

checkpoint selection 也不能再按 all8 L1，应该优先：

\[
\boxed{\text{first-step PD XYZ p95}}
\]

再用 chunk L1 当 secondary metric。

---

## 还有一个值得改的点：loss 最好加入 PD-space

现在直接优化 native action：

\[
\hat a_k\approx a_k.
\]

但不同 native action 维度对应的物理尺度不一样。

我们真正关心：

\[
\text{wrist XYZ / rotation / finger target}
\]

所以至少对 \(a_0\) 可以加入：

\[
L_{\rm PD0}
=
D(
PD(\hat a_0,q_t),
PD(a_0,q_t)
).
\]

于是：

\[
\boxed{
L=
L_{\rm chunk}
+
\lambda_0L_{a_0}
+
\lambda_{PD}L_{\rm PD0}
}
\]

这个比继续单纯加训练步数更合理。

因为现在 3000 steps 后 development loss 还在下降，说明模型没有完全拟合；但即使再训，它优化的仍然不是我们的真实 execution metric。

---

# 所以当前三项计划需要微调一下顺序

原来：

\[
R
\rightarrow
PTv3
\rightarrow
\hat\tau\rightarrow E
\]

现在仍然不变。

但是 \(R\) 这一项还不能算完成。

应该变成：

\[
\boxed{
\text{A1 Chunk-R结构}
\rightarrow
\text{A2 first-action/control-aware loss}
\rightarrow
\text{GT execution gate}
}
\]

只有达到至少接近：

\[
\text{GT }\tau+\text{Chunk-R held}\approx\text{旧R}
\]

比如预设的 ≥60 held，才进入：

\[
\boxed{\text{B：PTv3 explicit hand-token readout}}
\]

然后才：

\[
\boxed{\text{C：predicted-}\tau\rightarrow W_{\rm frozen}\rightarrow E}
\]

---

### 当前我的判断

这次失败其实信息量很高。

现在至少已经把问题切得非常干净：

\[
\boxed{
\tau^{GT}+\text{old R}\rightarrow75\ held
}
\]

但：

\[
\boxed{
\tau^{GT}+\text{Chunk R}\rightarrow0\ held
}
\]

所以暂时完全不用怀疑：

- PTv3；
- goal；
- WAM trajectory；
- predicted-\(\tau\) world coupling。

**先把 Chunk-\(R\) 本身修到 GT trajectory 可执行。**

而我认为当前最值得改的不是 Transformer 层数，而是：

\[
\boxed{\text{all8平均模仿} \rightarrow \text{first-action/control-aware chunk learning}}
\]

这一刀比继续盲目扩大 Chunk-\(R\) 更有针对性。