有，而且从“成熟度”和“适合我们”两个维度看，我认为路线其实已经比较清楚了。**最成熟的不是继续追求一个精确的 \(\tau\leftrightarrow A\) 映射，而是分层：规划层只给目标，执行层用闭环控制吸收 gap。**

我会把现有方法分成下面几类：

| 方法 | 核心结构 | 对 plan-execution gap 的处理 | 对我们适合度 |
|---|---|---|---|
| Position/impedance target + feedback controller | plan → \(q^{des}/x^{des}\) → PD/impedance | 用实时误差闭环吸收 | **很高** |
| Reference tracking + residual RL | reference → nominal control + learned residual | RL 专门修 dynamics/contact gap | **最高** |
| Action chunk + receding horizon | \(o\to A_{1:K}\)，执行几步再规划 | 不要求长期计划精确执行 | 高 |
| Goal-conditioned low-level policy | local goal/trajectory → closed-loop action | 把 tracking 本身学成 skill | **很高** |
| Full model-based MPC | model rollout → optimize → short execution | 每次观测后重新优化 | 中，接触手上成本高 |

### 1. 最经典也最成熟：目标轨迹 + PD / impedance

不是：

\[
\tau\rightarrow A_{\rm raw}
\]

而是：

\[
\boxed{
\tau
\rightarrow
q^{des},\dot q^{des}
\rightarrow
C(q,\dot q,q^{des},\dot q^{des})
\rightarrow
A
}
\]

接触时也不要求 actual pose 等于 desired pose。

实际上：

\[
q^{des}-q
\]

本身就可以产生接触力。

这正是 impedance / position control 的基本思想：**planner 给“想去哪”，controller 负责根据真实接触状态决定“现在施加什么控制”。** 接触控制领域长期都是这样分层的，近年来也仍大量使用自适应/可学习 impedance 来处理未知接触。[IEEE Xplore](https://ieeexplore.ieee.org/document/10458138?utm_source=chatgpt.com)

我们现在一直尝试的：

\[
\tau^{actual}\rightarrow A^{native}
\]

反而绕过了这个天然的控制层。

---

## 2. 对我们最有参考价值：reference + residual RL

这个现在在 dexterous manipulation 里非常常见，而且跟我们的情况高度一致。

例如 REGRIND 的路线就是：

\[
\text{human motion}
\rightarrow
\text{retargeted robot reference}
\rightarrow
\boxed{\text{residual RL}}
\rightarrow
\text{physical execution}.
\]

它明确承认：**retarget 出来的运动学 reference 并不等于动态可执行策略。**

所以 RL 不重新学整个动作，而是专门修：

- contact mismatch；
- tracking error；
- dynamics；
- grasp stability。

REGRIND 用 residual RL 去跟踪 object-centric reference，并在不同灵巧手和接触丰富任务上完成实际执行。[arXiv](https://arxiv.org/abs/2607.11874?utm_source=chatgpt.com)

更直接的是 2026 年的 Contact-Anchored Retargeting 工作：

\[
\text{human trajectory}
\rightarrow
\text{physics refinement}
\rightarrow
\text{contact retargeting}
\rightarrow
\boxed{\text{residual policy learning}}
\]

他们明确把 residual policy 用来恢复 retarget 后的 **dynamic feasibility**。论文报告加入这些物理修正后成功率有很大提升。[arXiv](https://arxiv.org/abs/2609.24093?utm_source=chatgpt.com)

这和我们现在的问题几乎是一回事：

\[
\text{几何轨迹正确}
\not\Rightarrow
\text{物理抓取稳定}.
\]

---

# 3. 但这里的 residual 不一定意味着“永久依赖 basepolicy”

这是我们之前一直担心的。

其实成熟做法可以写成：

\[
\boxed{
A_t=
C(s_t,g_t)
+
\Delta A_\theta(s_t,g_t)
}
\]

其中：

\[
g_t
\]

是当前局部目标，比如：

\[
g_t=
(q^{des},\dot q^{des},\text{desired contact geometry}).
\]

\(C\) 是一个通用的 nominal controller，不是某个固定 task-specific basepolicy。

所以 residual 是围绕：

\[
\boxed{\text{control law}}
\]

而不是：

\[
\boxed{\text{basepolicy action}}
\]

这点非常重要。

最终低层 skill 可以变成：

\[
\boxed{
\pi_{\rm low}(s_t,g_t)\rightarrow A_t
}
\]

它已经是独立的 goal-conditioned controller。

旧 basepolicy 只负责给它提供初始化数据。

---

# 4. Action chunk + receding horizon 也是非常成熟的处理办法

Diffusion Policy 就是典型：

\[
o_t\rightarrow
A_{t:t+T}
\]

但不把整段全执行。

而是：

\[
\boxed{
\text{predict long}
\quad+\quad
\text{execute short}
\quad+\quad
\text{observe again}
}
\]

也就是 receding horizon。[arXiv](https://arxiv.org/abs/2303.04137?utm_source=chatgpt.com)

这其实告诉我们：

> 不要要求计划本身覆盖所有未来扰动。

比如预测 24 步：

\[
T_{\rm plan}=24
\]

实际只执行：

\[
K=2\sim8.
\]

然后重新观测。

我们现在 24/8 的思路本身没问题，问题主要在于**这 8 步的低层执行接口太脆**。

---

# 5. 对我们的现实方案，我现在会选“1 + 2 + 3”

也就是：

\[
\boxed{
\text{trajectory planner}
+
\text{goal-conditioned residual controller}
+
\text{receding horizon}
}
\]

具体到我们的 Inspire：

### 高层

预测未来 hand plan：

\[
F(P_t,H_t,G)\rightarrow\tau_{t:t+24}.
\]

### 中间层

把它转成局部机器人目标：

\[
\tau
\rightarrow
g_t=
\left[
x^{des}_{wrist},
R^{des}_{wrist},
q^{des}_{finger}
\right].
\]

### 低层

每个 control step：

\[
\boxed{
A_t
=
C(s_t,g_t)
+
\Delta A_\theta(s_t,g_t)
}
\]

或者干脆训练：

\[
\pi_{\rm low}(s_t,g_t)\rightarrow A_t.
\]

这里可以读实时：

\[
q,dq,\text{object},\text{hand}
\]

以后需要时还可以加 tactile/contact。

### replanning

执行 2～8 步：

\[
H_{t+K}\rightarrow\tau_{\rm new}.
\]

---

# 6. 这条路线最大的好处是：我们已经有低层 skill 的 teacher

我们不是从零训练。

现有 competent basepolicy：

\[
\pi_{\rm base}\rightarrow A
\]

可以产生数据：

\[
(s_t,g_t,A_t).
\]

先 BC 得到：

\[
\pi_{\rm low}(s,g)\rightarrow A.
\]

然后再用 RL 做 contact adaptation。

这和我们之前：

\[
H\rightarrow288D\tau
\]

然后希望 PPO 从 0/4 学会一切，难度完全不同。

---

# 7. 我甚至认为可以先不学 wrist

因为仓库已经显示：

\[
\tau\rightarrow \text{wrist pose}
\]

几乎是 exact。

那么第一版：

\[
A_{wrist}=C_{\rm analytic}(q,\dot q,x^{des},\dot x^{des})
\]

只让 neural controller 负责：

\[
\boxed{
A_{finger}
=
C_{\rm finger}(q,q^{des})
+
\Delta A_{\rm finger}
}
\]

这会把问题进一步缩小到真正最困难的：

> contact/preload。

而不是又让网络重新学 wrist kinematics。

---

## 最终可以长成这样

\[
\boxed{
\begin{aligned}
(P,H,G)&\rightarrow\tau^{plan}\\
\tau^{plan}&\rightarrow g^{robot}\\
(s,g^{robot})&\rightarrow \pi_{\rm low}\rightarrow A\\
\text{PhysX}&\rightarrow s'\\
s'&\rightarrow \text{replan}
\end{aligned}
}
\]

PointWorld则仍然可以在高层工作：

\[
(P,H,\tau^{plan})
\rightarrow
E/z^{phys}.
\]

所以：

- world model 不负责低层 control；
- low-level controller 不负责 task planning；
- planner 不负责 preload。

这就是比较成熟的职责分离。

---

### 如果让我现在只选一条实施路线

我会选：

\[
\boxed{
\tau
\rightarrow
\text{robot local target}
\rightarrow
\text{goal-conditioned low-level residual policy}
}
\]

而不是继续：

\[
\tau\rightarrow raw\ native\ action.
\]

然后让低层每一步闭环，外层每 4～8 步重规划。

**这其实就是把 REGRIND/残差控制的成熟思想，与 PointWAM 的 hand-trajectory representation 结合起来。** PointWAM 给我们“怎么表示计划”，REGRIND/传统控制告诉我们“不要指望几何计划直接等于底层执行”。[arXiv](https://arxiv.org/abs/2607.11874?utm_source=chatgpt.com)

对我们当前仓库而言，我认为这是比继续调 retargeter Transformer 更成熟、更现实的方向。