可以，而且我反而更推荐 **“当前状态 + 终态 delta”**，比直接给 \(+8/+16/+24\) 三个 waypoint 更适合作为第一版 Shared-WAM 的 goal。

但要区分两种“终态”。

如果你说的是 **当前 WAM horizon 的终态**，例如：

\[
G_t=P^{obj}_{t+24}
\]

那我推荐。

如果你说的是整条 543 帧任务最后一帧，例如最终放置位置，那么我不推荐直接作为所有时刻唯一 goal。因为抓取、抬升、搬运、放置过程中可能出现：

\[
P^{obj}_{final}\approx P^{obj}_{initial}
\]

但中间其实要先拿起来再放回去。只给整任务终点会把中间意图抹掉。

---

### 对我们现在这版，我最推荐的 goal 是

当前目标物体的 32 个点：

\[
P_t\in\mathbb R^{32\times3}
\]

以及 horizon 终点相对于当前的位移：

\[
\boxed{
\Delta P_G=P_{t+24}^{goal}-P_t
}
\]

输入形式：

\[
\boxed{
G_t=(P_t,\Delta P_G)
}
\]

而不是直接给：

\[
(P_t,P_{t+24})
\]

两个绝对点集。

这其实就是我们仓库里已经探索过的 `start_delta` 思路，只不过在新 Shared-WAM 里把它作为正式 goal contract。

---

## 为什么 delta 比 absolute endpoint 更合适

第一，**它直接表达任务意图**。

例如：

\[
\Delta P_G\approx[0,0,+0.1]
\]

很明确就是：

> 希望这个物体相对当前位置向上移动。

而绝对目标：

\[
P_G=(0.42,-0.17,0.83)
\]

网络还需要自己先减去当前位置，才能理解“要往哪走”。

---

第二，**更容易跨场景泛化**。

假设两个飞机初始位置不同：

\[
P_t^A\neq P_t^B
\]

但任务都是向上抬 10 cm：

\[
\Delta P_G^A
\approx
\Delta P_G^B.
\]

用绝对坐标时模型必须同时学位置和动作关系。

用 delta：

\[
\boxed{\text{goal 更接近 interaction intent，而不是 workspace position}}
\]

这和我们希望学到的东西更一致。

---

第三，**归一化更容易**。

我们现在场景坐标可能跨度接近米级，但目标运动通常只有：

\[
\mathcal O(1\sim20\text{ cm}).
\]

直接预测/编码：

\[
\Delta P
\]

数值尺度更集中。

---

## 对 rigid object 还有一个更简洁的表达

因为 airplane 本身是刚体，其实 32 个：

\[
\Delta p_i
\]

不是完全独立的。

本质上由一个：

\[
\Delta T\in SE(3)
\]

决定。

所以还可以编码：

\[
\boxed{
G=
[\Delta x,\Delta y,\Delta z,R_{6D}]
}
\]

即：

\[
3+6=9D.
\]

然后再让网络把这个 goal embedding 注入 hand branch。

不过我第一版反而**不建议立刻压成 9D**。

因为我们整个 WAM 是 point-based，保留：

\[
(P_t,\Delta P_t)
\]

有几个好处：

- 和 OakInk2 / GRAB / ARCTIC 统一；
- 不需要依赖每个数据集都有高质量 SE(3)；
- 后面 EgoDex 只要有目标点轨迹也能接；
- 对非刚体/关节物体以后更容易扩展。

所以第一版：

\[
\boxed{
32\times [p_i,\Delta p_i]
}
\]

每个目标点就是一个 6D token，非常自然。

例如：

\[
g_i=
MLP([p_i,\Delta p_i])
+
E_{\rm target}.
\]

---

# 这样新 Shared-WAM 可以非常简单

共享 interaction 表征：

\[
Z=F(S_{\le t},H_{\le t})
\]

世界预测：

\[
\boxed{
\hat U=D_U(Z)
}
\]

手规划：

\[
\boxed{
\hat V=D_V(Z,G_t)
}
\]

其中：

\[
G_t=\{[p_i,\Delta p_i]\}_{i=1}^{32}.
\]

然后：

\[
R(s_t,\hat V)\rightarrow A.
\]

也就是：

```text id="5f58vw"
Scene ─→ PTv3 ───────┐
                     │
Hand H4 ─→ HandToken ├→ Shared Interaction Trunk → Scene Future U
                     │
                     └→ Hand representation
                               │
               [current point, goal delta] × 32
                               │
                               ▼
                      Goal-conditioned decoder
                               │
                               ▼
                        Hand future V / τ
                               │
                               ▼
                            Chunk-R
```

---

## 终态-only 最大的代价是什么？

就是它不再告诉模型：

\[
\text{中间什么时候应该到哪里}.
\]

比如：

\[
\Delta z=+10cm
\]

它不知道应该：

```text id="fjz67f"
前8帧接近
8-16帧抓稳
16-24帧抬升
```

还是：

```text id="u7orxi"
一开始就快速向上
```

但我认为**第一版这反而是好事**。

因为我们正想检验：

\[
\boxed{
\text{Shared scene-hand representation 能否自己从 demonstration 学出 interaction timing}
}
\]

而不是把 \(+8/+16/+24\) 的 reference timing 全都喂进去。

这会让它更接近 PointWAM：

PointWAM 的 language 也不会告诉它：

> 第8帧移动2cm，第16帧移动5cm，第24帧移动10cm。

它是从 demonstration 学这些中间运动模式的。

---

# 所以我会做这样的层级对照

第一版主模型：

\[
\boxed{
G_{\rm endpoint}
=
(P_t,\Delta P_{t\rightarrow t+24})
}
\]

如果这个能工作，就最干净。

然后第二个 ablation：

\[
\boxed{
G_{\rm waypoint}
=
\{\Delta P_8,\Delta P_{16},\Delta P_{24}\}
}
\]

比较：

- hand first8 / all24 EPE；
- action chunk error；
- airplane held；
- 对 goal shuffle 的敏感性。

如果三 waypoint 明显更好，我们再说明：

> dexterous manipulation 需要更强的 temporal goal specification。

而不是一开始就默认必须给。

---

## 我的选择

如果现在只能选一个，我会选：

\[
\boxed{
\textbf{当前目标物体点 + t+24终态相对位移 delta}
}
\]

也就是：

\[
\boxed{
[p_i,\;p^{goal}_{i,t+24}-p_{i,t}]
}
\]

而不是 absolute endpoint，也暂时不放 +8/+16。

它信息足够表达：

> 哪个物体，要往哪里、移动多少、怎么旋转。

但还没有把中间 hand trajectory 和 interaction timing 直接喂给模型。**这正好处在“任务条件足够明确”和“仍需要 WAM 学 interaction”之间。**