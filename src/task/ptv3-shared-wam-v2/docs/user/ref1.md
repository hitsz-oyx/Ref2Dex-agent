我觉得**非常值得做，而且现在是一个很合理的分支**。但我不会把 \(\tau\) 路线直接删掉，而是把它变成一个对照：

\[
\boxed{\tau\to R}
\qquad\text{vs}\qquad
\boxed{z_{\text{WAM}}\to Actor}
\]

因为我们现在已经有一个很明确的疑点：**WAM 在把内部表示 \(z\) 解码成显式 \(\tau\) 的过程中，可能丢掉了对动作有用的信息。**

目前 WAM-v2 的内部 `shared` token 是 768D，里面已经融合了：

- scene；
- hand；
- goal；
- 当前几何；
- shared interaction。

而现在我们做的是：

\[
z
\rightarrow
\hat\tau
\rightarrow
R
\rightarrow
q_{PD}
\]

这相当于强行要求：

\[
\boxed{\text{所有 action-relevant 信息必须先压缩成 24×11×3 的手轨迹}}
\]

这未必必要。

---

### 我更推荐的直接结构

不要简单把整个 \(z\) flatten 后拼到 MLP。

直接复用我们现在的 token 结构：

\[
Z_t=\text{WAM shared tokens}
\]

然后：

\[
s_t\rightarrow \text{state token}
\]

再用 8 个 action query cross-attend：

\[
\boxed{
A_{1:8}
=
Actor(Q_{\rm action},[Z_t,s_t])
}
\]

输出仍然保持我们现在的：

\[
8\times15
\]

`geometry-lift q_PD target`。

也就是说：

```text
H + goal + scene
       ↓
      WAM
       ↓
shared latent tokens z
       ↓
 + current robot state
       ↓
 action decoder / actor
       ↓
 q_PD chunk
```

这其实非常自然，因为现在 `SharedWAMV2.encode()` 已经直接给我们：

\[
\texttt{shared}\in\mathbb R^{N\times768}
\]

而现有 hand trajectory head只是从其中的 hand tokens 再解码 \(\tau\)。

---

## 这个方案最大的好处

现在：

\[
z\rightarrow\tau
\]

只受到 trajectory loss 约束。

所以即使 \(z\) 里面包含：

- 接触趋势；
- object motion intent；
- goal relation；
- hand-object interaction；

到了：

\[
\tau=(x,y,z)_{11\times24}
\]

以后，这些信息可能就没了。

而 direct actor：

\[
\boxed{z+s\rightarrow q_{PD}}
\]

允许 action loss 直接塑造 WAM latent：

\[
L_{\rm action}
\rightarrow Actor
\rightarrow z
\rightarrow WAM.
\]

这其实和我们之前想的：

> “让世界模型信息真正参与动作”

更一致。

而不是先逼它显式预测一条轨迹，再让另一个网络逆推出控制。

---

# 但我不建议一上来直接 PPO

第一步还是监督学习。

用现有 RL660：

\[
\boxed{
(z_t,s_t)\rightarrow q^{GT}_{PD}
}
\]

先和当前：

\[
(\hat\tau_t,s_t)\rightarrow q^{GT}_{PD}
\]

做 matched comparison。

其他全部固定：

- 同一 80k parent；
- 同一 RL660 split；
- 同一 q_PD labels；
- 同一 action representation；
- 同一 12k updates；
- 同一 batch；
- 同一个 action-head 参数量。

这样结果特别干净。

如果：

\[
z\text{-Actor}
\]

明显优于：

\[
\tau\text{-R},
\]

例如 first-control wrist error：

\[
50\text{ mm}\rightarrow20\text{ mm}
\]

那就基本证明：

\[
\boxed{\tau\text{ 是一个过强的信息瓶颈}}
\]

这会是一个非常有价值的结论。

---

## 我甚至建议保留三个 arm

最干净：

\[
\textbf{A: } s+\tau\rightarrow A
\]

当前 baseline。

\[
\textbf{B: } s+z\rightarrow A
\]

你现在说的方案。

\[
\textbf{C: } s+z+\tau\rightarrow A
\]

看 \(\tau\) 在 latent 已经存在的情况下还有没有增量价值。

这个实验的信息量非常高。

如果：

\[
B\approx C\gg A
\]

那么说明：

\[
\boxed{z\text{ 已经包含了动作所需信息，显式 }\tau\text{ 反而限制了它}}
\]

如果：

\[
C>B>A
\]

说明 latent + explicit trajectory 互补。

如果：

\[
A\approx B\approx C
\]

那瓶颈就不在 \(\tau\)，更可能是 q_PD 数据/RL660 本身。

---

# 还有一点：不要把 forecast loss 去掉

我会保留：

\[
L=
L_{\rm hand}
+
L_{\rm scene}
+
\lambda_A L_{\rm action}.
\]

因为如果只做：

\[
z\rightarrow action
\]

然后 action loss 随便改 WAM latent，

WAM 很容易退化成一个普通 policy encoder，最后：

\[
z
\]

不再有 world-model semantics。

这对你的毕设主线反而不好。

所以更合理的是：

\[
\boxed{
\text{WAM auxiliary prediction}
+
\text{latent action head}
}
\]

即：

\[
H,G\rightarrow z
\]

同时：

\[
z\rightarrow(\hat\tau,\hat E)
\]

和：

\[
z+s\rightarrow A.
\]

这样你依旧可以说：

> 世界模型通过显式未来预测约束学习未来表征，同时 latent representation 直接服务控制。

这比：

\[
\tau\rightarrow R
\]

还更像一个完整的 **world-model policy**。

---

## RL 应该什么时候上？

如果 BC 的 z-Actor 能先做到：

\[
GT/Pred\ z\rightarrow q_{PD}
\]

明显好于 \(\tau\)-R，而且闭环至少开始出现抓取，

再接 PPO：

\[
A_t=A^{BC}_t+\delta A^{RL}_t
\]

或者直接 fine-tune actor。

我不建议现在直接：

\[
z+s\rightarrow PPO
\]

从零开始。

因为我们之前 residual PPO 已经证明：**nominal 没过门时，PPO 很容易把问题变成“既学轨迹又学接触”，最后什么都学不好。**

---

# 具体到当前代码，改动其实不大

现在是：

\[
\texttt{forecaster(batch)}
\rightarrow
\texttt{prediction["trajectory"]}
\rightarrow
\texttt{native\_trajectory()}
\rightarrow
R.
\]

可以新增：

\[
\texttt{encode(batch)}
\rightarrow
\texttt{shared}
\]

然后：

```text
shared tokens 768D
      +
native_state 87D
      ↓
action decoder
      ↓
8 × 15 q_PD
```

甚至 action decoder 可以直接复用我们现在的 Transformer decoder 思路，只把 memory 从：

\[
24\times33\ \tau
\]

换成：

\[
N\times768\ z.
\]

所以工程成本并不高。

---

### 我现在的倾向

我会把这条路线的优先级放得很高：

\[
\boxed{\text{值得马上做}}
\]

但实验定义应该是：

> **不放弃 \(\tau\)，先验证 \(\tau\) 是否真的是控制信息瓶颈。**

如果 direct \(z\)-Actor 能明显超过当前 \(\tau\)-R，这其实比“继续把 R 从 0.6M 扩到 4.6M”更有研究价值。

因为它说明的不是工程容量问题，而是一个方法问题：

\[
\boxed{
\text{显式未来轨迹并不是 world model 到 control 的最佳接口；
latent future representation 可能更适合直接驱动 policy。}
}
\]

这条路线和你最开始想的“WAM 最终应该真正参与 actor，而不是只预测一个中间量”其实是完全一致的。