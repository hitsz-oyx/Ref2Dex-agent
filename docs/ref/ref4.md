可以参与后续，但我不会让它成为主干。当前证据更适合把它定位成：

\[
\boxed{\text{辅助 interaction target / diagnostic}}
\]

而不是：

\[
\boxed{\text{已经成熟到可以决定策略的核心 representation}}
\]

原因很直接：当前 hand-agnostic 13D \(I\) 的定义是合理的，但预测性能还没有过关。Cmv2-only 在 K1/K4 都落后于 direct state/action baseline；即便加 state/action，K1 也只是接近而不是稳定超过。因此现在如果直接做

\[
(E,\hat I)\rightarrow G\rightarrow policy
\]

一旦后面失败，我们又无法分辨是 \(I\) 本身没用，还是 \(I\) 没预测准。

### 我认为下一步应该收敛成一条线

**先冻结已经有正向证据的 E 路线。**

现在我们已经知道：

\[
\text{geometry + hand flow}\rightarrow E_1
\]

有正向信号，而且

\[
\text{OI spatial encoder + GRU}\rightarrow E_{1:4}
\]

也有正向信号。

这部分不要再改。

接下来只解决一个问题：

\[
\boxed{\text{怎样定义一个既 hand-agnostic、又保留 interaction topology 的 }I}
\]

我不建议继续调现在的 13D global \(I\)。它的问题很可能不是 MLP 不够强，而是压缩太早。

当前：

\[
I_{\text{global}}
=
[\text{mass},\text{centroid},\text{covariance},v_n,v_t,\dots]
\]

把整个物体上的 interaction 压成一个向量，空间结构损失太多。

我更推荐直接利用 OI-Cmv2 已经存在的 object-surface token 机制，定义：

\[
\boxed{
I=\{I_1,\dots,I_K\},\quad K=8\text{ or }16
}
\]

每个 object surface token 只保存很小的一组量，例如：

\[
I_k=
[
\text{contact mass},
\text{hand distance},
v_n,
v_t,
\text{contact change}
]
\]

必要时再加一个 support/slip indicator。

这仍然完全不依赖：

- finger ID；
- Inspire link ID；
- MANO joint；
- hand topology。

所以：

\[
\text{MANO}\rightarrow I_{\text{surface}}
\]

和

\[
\text{Inspire}\rightarrow I_{\text{surface}}
\]

仍然统一。

但它比 13D global vector 保留了：

> **哪里在接触、接触如何分布、哪些区域正在滑动或脱离。**

这才比较接近我们一开始对 OI-Cm 的设想。

---

### 然后只做一个 K=1 实验

不要马上再做 chunk。

固定现有 point-flow spatial encoder：

\[
\text{geometry}+F^{hand}_1
\]

同时输出：

\[
\hat E_1
\]

和新的：

\[
\hat I^{surface}_1
\]

然后只看三件事：

\[
E^{GT}_1\rightarrow G
\]

\[
(E^{GT}_1,I^{GT}_1)\rightarrow G
\]

以及

\[
(\hat E_1,\hat I_1)\rightarrow G
\]

如果 GT \(I\) 在已有 \(E\) 的基础上根本不增加价值：

\[
V_{HEI}\approx V_{HE}
\]

那就没必要继续花时间预测 interaction。

可以直接走 E-only。

如果：

\[
V_{HEI}\gg V_{HE}
\]

而预测 \(\hat I\) 也能保留这个增益，那么 I 才正式进入主链。

这一步很重要，因为之前我们证明的是：

> E+I 整体和 \(G\) 有关系。

现在需要更直接问：

\[
\boxed{\text{在 E 已经存在时，I 到底有没有独立价值？}}
\]

---

## 所以“现在的 I 能不能参与后续”

我的答案分两层。

**可以立即参与：**

作为 auxiliary target：

\[
L=L_E+\lambda_I L_I
\]

以及作为 offline bridge 的 ablation：

\[
E\quad vs\quad E+I
\]

这样可以继续利用它，而不会让它决定整个路线。

**暂时不应该：**

直接把当前 13D \(\hat I\) 当成 policy critic / action score 的核心输入，或者基于它开始 PPO/蒸馏。

因为它目前还没有证明：

\[
\hat I
\]

比直接 state/action 信息更可靠。

---

我现在建议的路线可以压缩成一句话：

\[
\boxed{
\text{保留已经有效的 point-flow E 主干；
把 I 从 global 13D 改成 object-surface token field；
先在 K=1 验证 I 对 E 的独立增益。}
}
\]

如果这个 K1 surface-I 成立，再把 **同一个 spatial encoder + GRU** 扩到 K4 chunk。

如果不成立，就果断让 I 退回辅助监督，主路线走：

\[
\text{point-flow}\rightarrow E\rightarrow G
\]

不要让 interaction 继续拖慢整个项目。