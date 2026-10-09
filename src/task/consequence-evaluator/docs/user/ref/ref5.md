对。**现在应该停止继续扩大 fork panel，也不应该再停留在 C2a。**你说的核心问题是对的：

\[
\boxed{\text{当前 PointWorld C2a 用的是 GT future hand，所以它还不能在决策时评价一个没执行过的候选。}}
\]

真正要往“抓取成功率”推进，下一步就是把这条链补齐：

\[
\boxed{
(H_t,A_i)
\rightarrow
\hat H^{hand}_{i,t+1:t+24}
\rightarrow
\hat Z^{PW}_{i,t+1:t+24}
\rightarrow
Q(H_t,A_i,\hat Z_i)
\rightarrow
\text{选动作}
}
\]

然后真实执行选中的动作，看最终成功率。**这一步不需要再为每个状态物理 fork 7 次。**

---

## 1. 现在 PointWorld 确实用了 GT future hand

当前 C2a 的实际合同是：

\[
H_t + H^{hand,GT}_{t+1:t+24}
\xrightarrow{PW}
\hat O_{t+1:t+24}.
\]

再组成：

\[
\hat Z=
(\hat O,\ H^{hand,GT})
\]

交给 evaluator。

所以目前的 73% 左右 ranking retention 只能说明：

> 如果我事先知道这只手未来实际怎么运动，PointWorld 对物体后果的预测大致保留了 evaluator 所需的信息。

这是一个有用的 oracle，但部署时显然不知道：

\[
H^{hand,GT}_{future}.
\]

所以现在真正 blocker 已经不是 \(Y\)，也不是 evaluator，而是：

\[
\boxed{
(H,A)\rightarrow H^{hand}_{future}
}
\]

这个 **动作执行桥**。

---

# 2. 但这个桥不需要 fork 数据

这是我现在最想纠正的地方。

以前我们一说 candidate，就想到：

```text
同一个 H
├─ candidate 0 → 物理跑24步
├─ candidate 1 → 物理跑24步
├─ ...
└─ candidate 6 → 物理跑24步
```

这个确实贵。

但训练 execution bridge 根本不需要这样。

我们只需要普通 rollout：

\[
(H_t,A_t,H^{hand,GT}_{t+1:t+24})
\]

就够了。

一个环境只执行**一个随机 action plan**，然后记录实际手轨迹。

例如并行 96 env：

```text
env0   随机 baseline
env1   随机 thumb-
env2   随机 middle+
env3   随机 grip+
...
```

每个环境只真正走自己的那一个动作。

这就是普通 intervention data collection，不是 counterfactual fork。

因此成本从近似：

\[
N_{\text{state}}\times K
\]

变成：

\[
N_{\text{rollout}}.
\]

而且 64/96 env 可以一次并行采。

---

# 3. 我建议现在直接训练一个 Hand Execution Model

不要先把 ACT、proposal、复杂 world model 又混进来。

定义：

\[
\boxed{
G_\phi(H_t,A_{t:t+23})
\rightarrow
\hat H^{hand}_{t+1:t+24}
}
\]

输出我们 PointWorld 已经接受的：

\[
24\times11\times3
\]

未来 Inspire 手语义点。

这里的 \(A\) 就继续使用我们现在已经定义好的：

\[
24\times18
\]

residual plan。

当前 old candidate 可以很简单：

- baseline；
- thumb-yaw ±；
- middle ±；
- grip+；
- wrist-z+。

前 8 步加 residual，后面 residual 为 0。

注意这里 \(G_\phi\) 学的不是纯 FK。

它学的是：

\[
\boxed{
\text{当前状态}
+
\text{未来 residual request}
\rightarrow
\text{冻结反馈控制器实际会执行出的手轨迹}
}
\]

所以 baseline policy 后面会怎么响应物体、接触怎么影响手，都允许模型自己学。

这其实就是我们以前说的 execution predictor，但**现在到了它真正应该出现的阶段**。

以前为了证明 consequence necessity，我们故意不用它；现在要部署 planner，就绕不过去了。

---

# 4. 仓库里其实已经说明这个桥“不是完全不可学”

旧 execution-geometry 的结果值得重新利用。

以前：

\[
H+A\rightarrow q_{t+8}
\]

对 finger 12 DoF 的预测改善很明显：

\[
\text{finger MSE gain}\approx80\%.
\]

真正出问题的是 wrist：

实际 wrist 换进去以后，hand endpoint RMSE：

\[
34.1\text{ mm}\rightarrow2.1\text{ mm}.
\]

这说明：

\[
\boxed{\text{finger execution 本身其实相当可预测，主要难点是公共 wrist/base 运动。}}
\]

所以现在不要再做“一个模型硬预测所有关节 endpoint”。

更合理是直接预测**手点轨迹**，并把 wrist/common motion 单独显式建模：

\[
G_\phi=
G_{\text{wrist}}
+
G_{\text{finger-relative}}.
\]

例如：

\[
(H,A)\rightarrow
\begin{cases}
\hat T^{wrist}_{1:24}\\
\hat p^{finger/wrist}_{1:24}
\end{cases}
\]

最后组合成：

\[
\hat H^{hand}_{1:24}.
\]

这个比以前的“预测 q8，再 FK 成 hand endpoint”更贴 PointWorld 接口。

---

# 5. 然后就可以真正做在线候选判断了

整个 planner 可以直接写成：

### 每 8 步决策一次

当前状态：

\[
H_t.
\]

生成固定 7 个候选：

\[
A_t^{(1)},\dots,A_t^{(7)}.
\]

每个候选先经过 execution bridge：

\[
\hat H_i^{hand}=G(H_t,A_i).
\]

再送 PointWorld：

\[
\hat O_i^{future}
=
PW(
H_t^{geom},
\hat H_i^{hand}
).
\]

构造 evaluator future：

\[
\hat Z_i=
(\hat O_i^{future},
\hat H_i^{hand}\text{ relative to }\hat O_i).
\]

然后用**同一个已经训练好的 C1 evaluator**：

\[
s_i=
Q_{C1}(H_t,A_i,\hat Z_i).
\]

选：

\[
i^*=\arg\max_i s_i.
\]

执行这个 candidate 的前 8 步：

\[
A^{(i^*)}_{1:8}.
\]

然后：

\[
t\leftarrow t+8
\]

重新观察、重新规划。

---

# 6. 这次真正评价的就不是 old-U ranking 了，而是抓取成功率

这是最重要的。

到这一步以后不要再把：

\[
U_{\rm old}
\]

当最终实验指标。

它只是 evaluator teacher。

真正对照应该是：

\[
\boxed{\text{Baseline policy}}
\]

vs

\[
\boxed{\text{Model-guided planner}}
\]

在**独立 episode / seed** 上比较最终：

- 抓取/任务成功率；
- rescue；
- harm；
- 中途失抓与 recovery；
- 平均干预次数。

这根本不要求 same-H fork。

比如直接跑：

\[
N=100
\]

个 baseline episodes，

再跑匹配 seed 的：

\[
N=100
\]

个 planner episodes。

即使 GPU PhysX 没法 bitwise counterfactual fork，也可以做**策略级统计比较**。

而这才是最终我们真正关心的：

\[
\boxed{
P(\text{success}\mid \pi_{\rm planner})
>
P(\text{success}\mid \pi_0)?
}
\]

---

# 7. 候选现在也不要搞复杂

我不建议现在又把 ACT proposal 拉回来。

第一版就继续用**已经验证有 headroom 的老 7 候选**。

这有三个好处：

1. old-U rolling 已经证明过这组候选能：
   \[
   20/25\rightarrow23/25
   \]
   所以 candidate set 不是完全没动作权威；

2. \(A_i\) 是明确已知的 18D residual schedule；

3. 便于检查 execution bridge 到底有没有学会 candidate sensitivity。

先证明：

\[
\boxed{
\text{固定7候选 + learned execution + PW + evaluator}
}
\]

能提高成功率。

如果这个能成立，之后再上：

\[
ACT/diffusion proposal
\]

扩大动作空间。

否则现在引入 proposal 只会多一个误差源。

---

# 8. 数据怎么来：不要 fork，做随机 intervention rollout

我建议现在专门采一批 **execution-bridge 数据**，但成本远低于 fork。

例如在当前 e260 actor 上：

每个 env 在预定几个阶段随机选择：

\[
A\sim\mathcal D_{\rm candidate}
\]

其中包含：

- zero；
- 6 个老 residual；
- 可以少量 amplitude jitter。

然后连续执行、记录：

\[
H_t,\quad
A_{t:t+23},\quad
H^{hand,GT}_{t+1:t+24}.
\]

不需要记录“同一个 H 的其他 6 个未来”。

一条轨迹可以产生很多 window。

而且这批数据还有一个额外用途：

同一个 rollout 里我们也自然得到：

\[
O^{GT}_{future}
\]

和：

\[
U_{\rm old}.
\]

所以同一套普通 rollout 可以同时用于：

\[
G_\phi:\quad(H,A)\rightarrow H^{hand}
\]

以及 evaluator / PointWorld robot adaptation。

这比继续 fork 数据划算得多。

---

# 9. 接下来我会明确分成四个误差层

现在终于可以真的 waterfall：

\[
\text{C1-oracle}:
GT\ hand+GT\ object
\]

↓

\[
\text{C2a}:
GT\ hand+PW\ object
\]

↓

\[
\boxed{
\text{C2b}:
Pred\ hand+PW\ object
}
\]

↓

\[
\boxed{
\text{Online rolling planner}
}
\]

我们现在：

- C1：75.64%
- C2a same-weight swap：73.08%

所以 C1→C2a 暂时看起来不是大问题。

真正未知的是：

\[
\boxed{\text{C2a}\rightarrow\text{C2b}}
\]

即：

\[
GT\ hand
\rightarrow
Predicted\ hand.
\]

所以**下一分钱算力应该花在 execution bridge，不是再采 fork candidate panel。**

---

## 我现在建议的最短路线

只有这一条：

\[
\boxed{
\text{普通随机 rollout}
\rightarrow
G(H,A)\to \hat H_{future}
\rightarrow
PW
\rightarrow
Q_{C1}
\rightarrow
7候选 rolling planner
\rightarrow
\text{实际 success rate}
}
\]

第一阶段先要求 execution bridge 在 held-out rollout 上：

\[
\hat H_{future}
\]

明显优于：

- zero hand motion；
- nominal FK；
- current-hand persistence。

然后马上把它接到已有 C1+PointWorld 上。

**不再为每个候选跑物理未来，也不再要求新的 same-H fork 数据。**

如果这样最终 planner 能把 baseline 成功率提高，我们整条链就真正通了：

\[
\boxed{
\text{动作}
\rightarrow
\text{实际手运动预测}
\rightarrow
\text{物体后果预测}
\rightarrow
\text{价值判断}
\rightarrow
\text{成功率提升}
}
\]

这才是现在应该推进的主线。