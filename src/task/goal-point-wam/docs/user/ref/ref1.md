**我建议继续用现在 PointWorld 的 PTv3 backbone，但不是把当前 `PointWorldWM` 整个原封不动搬过来。**

准确说：

\[
\boxed{\text{保留 PointWorld 的 scene backbone，重做 goal/action/world 三个头}}
\]

因为我们现在的 PointWorld 已经有一个很有价值的东西：它在 OakInk2/GRAB/ARCTIC 上学过

\[
\text{4-frame point history}
+
\text{hand motion}
\rightarrow
\text{object future}.
\]

这部分物理先验没有理由扔掉。

---

# 我建议的具体架构

整体：

\[
\boxed{
\begin{aligned}
H_{\rm point}
&\xrightarrow{\text{PTv3}}
Z_H\\
G=(P_s^G,P_g^G)
&\xrightarrow{\text{Goal Encoder}}
Z_G\\
H_{\rm robot}
&\xrightarrow{\text{MLP}}
z_R\\
(Z_H,Z_G,z_R)
&\xrightarrow{\text{Trajectory Decoder}}
\hat\tau\\
(Z_H,\tau_{\rm executed})
&\xrightarrow{\text{World Decoder}}
\hat E
\\
(s_t,\hat\tau)
&\xrightarrow{R}
q^{des}_{PD}
\end{aligned}}
\]

其中最关键的是：

\[
\boxed{\text{Trajectory branch 可以看 }G,\quad
\text{World branch 绝对不看 }G.}
\]

---

# 1. Scene Backbone：直接复用现在的 PointWorld

当前 PointWorld-small：

- PTv3-small；
- feature dim = 128；
- 固定 1 cm voxel；
- fixed workspace；
- 约 50.5M 参数。

现在输入每个 scene point 是 18D：

\[
f_i=
[
normal,\,
velocity,\,
acceleration,\,
history\ displacement,\,
metadata
].
\]

也就是现在已有的：

```text
scene_proj: 18 -> 128
PTv3-small
```

这一块我基本不改：

\[
\boxed{
H_{t-3:t}^{point}
\rightarrow
Z_H^{point}
}
\]

并直接从现有 PW checkpoint 初始化。

所以你之前花时间训练 PointWorld 的 backbone **不是废掉，而是成为 WAM 的预训练 scene encoder。**

---

## 但不再把 future hand action token 塞进这个 backbone

现在 PointWorld 是：

\[
[scene\ points + future\ hand\ points]
\rightarrow PTv3.
\]

也就是说未来动作直接被拼进 sparse point cloud。

新的 WAM 不这么干。

Scene encoder 永远只编码：

\[
\boxed{\text{已经观察到的世界}}
\]

即：

\[
H_{t-3:t}.
\]

因为未来 hand trajectory：

\[
\tau
\]

现在是我们**要预测的输出**。

---

# 2. Scene token 输出不能只剩一个 pooled vector

当前 PointWorld 最后按 object pooling 成：

\[
z_{obj}\in\mathbb R^{128}.
\]

如果只是预测 rigid SE(3)，够用。

但我们现在要生成 hand trajectory，而且后面希望支持：

- 不同 object shape；
- human video；
- deformable/articulated；
- interaction region。

所以不能只保留一个：

\[
z_{obj}.
\]

我建议保留 PTv3 decoder 输出的一组 spatial tokens。

例如从有效 scene points 中得到：

\[
Z_H\in\mathbb R^{256\times128}.
\]

可以用：

- FPS 选 256 个 token；
- 或当前 PTv3 pooling 层的中间 token；
- hand/object interaction region 优先保留。

第一版固定：

\[
\boxed{256\text{ scene tokens}\times128D}
\]

就够了。

---

# 3. Goal Encoder 不需要再来一个 50M PTv3

这里我反而不建议复制一个完整 PointWorld。

\(G\) 只是：

\[
G=(P_s,P_g)
\]

两组短程 goal clouds。

例如各采：

\[
N_G=128
\]

个点。

所以 goal encoder 可以很小：

### 每个点

\[
p_i\in\mathbb R^3
\]

先：

\[
3
\rightarrow64
\rightarrow128.
\]

分别得到：

\[
Z_s,\ Z_g\in\mathbb R^{128\times128}.
\]

加：

- `start/goal type embedding`
- 可选 visual descriptor；
- valid mask。

然后 2 层 Set/Cross Transformer：

\[
(Z_s,Z_g)
\rightarrow
Z_G.
\]

输出例如：

\[
\boxed{
Z_G\in\mathbb R^{64\times128}
}
\]

这里模型自己理解：

> start 这团点变成 goal 那团点意味着什么。

不要求：

\[
p_i^s\leftrightarrow p_i^g.
\]

---

# 4. Robot State 还保留

我不建议因为用了 point backbone 就把 H328 全删了。

第一版继续保留现在已经验证过的：

\[
H_{robot}=H328
\]

包括：

- q/dq；
- hand history；
- object motion；
- gravity/height 等。

小 MLP：

\[
328\rightarrow256\rightarrow128.
\]

得到：

\[
z_R\in\mathbb R^{128}.
\]

这样我们不会因为切到 point backbone，同时又把 robot proprioception 搞坏。

以后再做 ablation：

\[
\text{是否 H328 有冗余}.
\]

---

# 5. Trajectory Decoder：这里才是真正的 Action Model

我们需要输出：

\[
\hat\tau
\in
\mathbb R^{24\times11\times3}.
\]

我不再用现在那个简单 MLP：

\[
736\rightarrow256\rightarrow256\rightarrow792.
\]

而改成 Transformer trajectory queries。

建立：

\[
24\times11=264
\]

个 query token。

每个 query：

\[
q_{k,j}
=
e^{time}_k+e^{handpoint}_j.
\]

然后 3～4 层 Transformer Decoder：

\[
Q_{\tau}
\overset{cross-attn}{\longleftarrow}
[Z_H,Z_G,z_R].
\]

最后：

\[
128\rightarrow3.
\]

得到：

\[
\boxed{
\hat\tau_{k,j}
}
\]

建议仍然预测：

\[
\Delta p_{k,j}
=
p_{k,j}-p_{t,j}
\]

而不是绝对坐标。

---

## 所以 Action Model 实际是

\[
\boxed{
\hat\tau
=
A_\theta
(
Z_H,\,
Z_G,\,
z_R
)
}
\]

这是我们新的 policy 主体。

---

# 6. World branch 怎么做

这里我建议也不要完全照搬当前 PointWorld head。

当前 PW：

\[
(H,\tau)
\rightarrow
\text{rigid }SE(3)_{1:24}.
\]

最终我们希望吃大规模视频，所以更通用的是：

\[
\boxed{
(H,\tau)
\rightarrow
\text{future point motion}
}
\]

不过可以分两步实施。

---

## V1：先保留当前 rigid PointWorld head

当前数据上先验证 WAM 思路：

\[
Z_H+\tau_{\rm executed}
\rightarrow
\hat T^{obj}_{1:24}.
\]

这样现有：

- object projector；
- rotation6D；
- translation head；
- PointWorld checkpoint；

都能最大程度复用。

这是最省工程量的版本。

---

## V2：再换成 point-flow head

为了真正吃视频：

\[
(Z_H,\tau_{\rm executed})
\rightarrow
\Delta \hat P_{1:24}.
\]

例如选：

\[
N_E=128
\]

个 world query points。

每个：

\[
(point_i,time_k)
\]

作为 query，对：

\[
[Z_H,Z_\tau]
\]

cross-attention。

输出：

\[
\Delta p_{i,k}\in\mathbb R^3.
\]

最终：

\[
\boxed{
\hat E\in\mathbb R^{24\times128\times3}
}
\]

如果有 point track：

\[
L_E=L_{\rm flow}.
\]

没有对应关系：

\[
L_E=CD+\lambda HD_{\rm soft}.
\]

所以最终架构不依赖 rigid assumption。

---

# 7. \(\tau\) 怎么进入 World Decoder？

这部分可以直接复用现在 PointWorld 很多设计。

当前 PointWorld 每个 future hand point 是 9D：

\[
a_{k,j}
=
[
p_{k,j},\,
p_{k,j}-p_{t,j},\,
p_{k,j}-p_{k-1,j}
].
\]

继续保留。

然后：

\[
9\rightarrow128
\]

用当前已有：

```text
action_proj
```

并继续加：

- time embedding；
- hand-point embedding。

于是：

\[
Z_\tau
\in\mathbb R^{264\times128}.
\]

World decoder：

\[
\boxed{
W(Z_H,Z_\tau)\rightarrow E.
}
\]

所以 PointWorld 的：

- action representation；
- time embedding；
- keypoint embedding；

其实也能继续继承。

---

# 8. 一个非常重要的结构隔离

网络图应该是：

```text
                   Goal start/goal point sets
                            │
                       Goal Encoder
                            │
                            ▼
                         Z_G
                          │
                          │
Point history ──> PTv3 ── Z_H ───────┐
                          │           │
Robot H328 ───────> MLP ─ z_R        │
                          │           │
                          ▼           │
                  Trajectory Decoder  │
                          │           │
                         τ_hat        │
                          │           │
                    ┌─────┴─────┐     │
                    ▼           ▼     │
                 frozen R   World Decoder
                    │       ▲
                    ▼       │
                  q_des   τ_executed
                            │
                            ▼
                         E_hat
```

重点：

\[
\boxed{
Z_G\not\rightarrow WorldDecoder
}
\]

否则又会出现抄 goal 的 shortcut。

---

# 9. 大规模视频预训练时怎么反传

这里也很关键。

视频样本有：

\[
H,\quad
G,\quad
\tau^{human},\quad
E^{actual}.
\]

Action branch：

\[
(H,G)\rightarrow\hat\tau
\]

用：

\[
L_\tau.
\]

World branch：

\[
(H,\tau^{human})
\rightarrow\hat E
\]

用：

\[
L_E.
\]

这两个 loss **同时更新同一个 PTv3 scene backbone**：

\[
\boxed{
\nabla F_{\rm scene}
=
\nabla L_\tau
+
\lambda_E\nabla L_E.
}
\]

所以 PointWorld 的动力学 supervision 真正进入 action representation。

这才是我们想要的 WAM。

---

# 10. Robot fine-tune 时

在 DExplore 数据上再加入：

\[
L_{PD}
=
\|
R(s,\hat\tau)-q^{des}_{teacher}
\|.
\]

于是：

\[
\boxed{
L
=
L_\tau
+
\lambda_E L_E
+
\lambda_{PD}L_{PD}.
}
\]

再接现在已经有效的 PD DAgger。

---

# 11. 参数规模大概是什么样

第一版大概：

| 模块 | 规模 |
|---|---:|
| PointWorld PTv3 scene backbone | ~50M |
| Goal encoder | ~1–3M |
| Robot MLP | <0.2M |
| Trajectory decoder | ~3–6M |
| World head | ~2–5M |
| R | 现有冻结模型 |

所以总量大概：

\[
\boxed{55\sim65M}
\]

级别。

这其实不大。

而且最大的 50M 是我们已经有 pretrained weights 的 PointWorld backbone。

---

# 12. 要不要直接用两个 PTv3？

我不建议。

也就是不要：

\[
PTv3_{\rm scene}
+
PTv3_{\rm goal}.
\]

原因：

1. goal 只有 128 个点，太浪费；
2. 会让模型轻易增长到 100M+；
3. goal 的任务只是描述 state transition，不需要完整 spatial world encoder；
4. 第一版难以判断收益到底来自 world pretraining 还是参数量。

所以：

\[
\boxed{
\text{一个大 PTv3 做 scene/world prior}
+
\text{一个小 Goal Encoder}
}
\]

是我认为最合理的。

---

## 如果让我现在定 V1，我会直接定成这套

**Scene：**

\[
4096\times18
\rightarrow
\text{PW-PTv3-small}
\rightarrow
256\times128
\]

**Goal：**

\[
2\times128\times3
\rightarrow
\text{shared MLP + 2-layer cross-attention}
\rightarrow
64\times128
\]

**Robot：**

\[
328\rightarrow128
\]

**Action：**

\[
264\text{ trajectory queries}
\xrightarrow[\text{4 layers}]{cross-attn}
[Z_H,Z_G,z_R]
\rightarrow
24\times11\times3
\]

**World：**

\[
Z_H+
Z_{\tau}^{executed}
\rightarrow
\text{2–4 layer dynamics decoder}
\rightarrow
24\times128\times3
\]

**Execution：**

\[
\tau\rightarrow R_{\rm frozen}\rightarrow q^{des}.
\]

---

所以直接回答你的问题：

\[
\boxed{\text{是，继续用现在 PointWorld 的 PTv3 backbone。}}
\]

但**只复用它作为 goal-blind scene/dynamics encoder**，而不是继续使用现在完整的：

\[
(H,\tau)\rightarrow SE(3)
\]

网络作为 policy。

真正新的东西应该是：

\[
\boxed{
\text{PW-PTv3 scene backbone}
+
\text{small goal-set encoder}
+
\text{trajectory-query decoder}
+
\text{goal-blind dynamics head}
}
\]

这套才是我认为最适合我们当前研究目标的具体 WAM 架构。