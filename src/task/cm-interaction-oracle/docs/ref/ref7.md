不是“对所有关节加同方差高斯噪声”。我刚看了当前代码，实际是**结构化、离散的 residual intervention**。

当前 18D action 里：

\[
a'=\mathrm{clip}(a_{\text{base}}+\alpha\delta,-1,1)
\]

随机选的是 7 个 arm：

\[
\{\text{zero},x^+,x^-,z^+,z^-,\text{finger}^+,\text{finger}^-\}.
\]

其中 wrist 只改 action 0/2：

\[
\delta_x=\pm0.01,\qquad \delta_z=\pm0.01.
\]

而 finger arm 是同时改：

\[
\boxed{[6,8,10,12,15]}
\]

这 5 个 native action 维度，每个都加同一个：

\[
\pm0.1\alpha.
\]

在最近 amplitude 实验里：

\[
\alpha\in\{1,2,4\}.
\]

所以 alpha4 时这些 finger action 维度统一加：

\[
\pm0.4.
\]

### 这 5 个维度分别是什么

当前 Inspire 的 6 个独立 finger DOF 顺序是：

\[
[\text{index},\text{middle},\text{pinky},\text{ring},
\text{thumb-yaw},\text{thumb-pitch}]
\]

native indices：

\[
[6,8,10,12,14,15].
\]

而我们扰动的是：

\[
[6,8,10,12,15]
\]

也就是：

\[
\boxed{
\text{index, middle, pinky, ring, thumb-pitch}
}
\]

**没有扰动 thumb-yaw。**

而且它不是只改 5 个物理关节。后面还有 mimic coupling：

\[
q_7=1.05q_6
\]

\[
q_9=1.05q_8
\]

\[
q_{11}=1.05q_{10}
\]

\[
q_{13}=1.05q_{12}
\]

thumb-pitch 还会带：

\[
q_{16}=0.6q_{15},\qquad
q_{17}=0.8q_{15}.
\]

所以一个 `finger+` 实际是在**同时收紧四根手指 + thumb pitch，并联动多个从属关节**。

---

## 你的担心是成立的

虽然在 normalized action space 里我们给这些维度的是相同的：

\[
\Delta a_i=0.1\alpha,
\]

但绝对不能理解成：

\[
\Delta q_i=\text{相同弧度}.
\]

因为之后还有：

1. 每个 DOF 自己的 joint range；
2. normalized action → PD target 的映射；
3. mimic ratio；
4. 每根手指不同的 link length / Jacobian；
5. 当前姿态不同；
6. 是否已经接触物体。

因此实际 fingertip displacement / force change 可能差很多：

\[
\Delta x_{\text{tip}}
=
J(q)\Delta q.
\]

即使两个关节：

\[
|\Delta q_1|=|\Delta q_2|,
\]

如果：

\[
\|J_1(q)\|\gg\|J_2(q)\|,
\]

末端运动也会差很多。

---

# 而且最新结果已经提示这种不对称确实存在

最近 amplitude probe 中，`finger±` 最明显的响应是：

\[
\boxed{\text{thumb\_distal}}
\]

alpha4 时：

`finger+`：

\[
d_{\rm thumb}-d_0=-8.44\text{ mm}
\]

force norm 增加。

`finger-`：

\[
d_{\rm thumb}-d_0=+22.95\text{ mm}
\]

force norm 大幅下降。

这非常值得警惕。

因为我们的 `finger+` 本意有点像：

> “整体增加手指抓紧程度”。

但真实执行出来可能更像：

> “一个高度耦合的 composite direction，其中 thumb 的几何/受力响应特别显著”。

所以你说：

> “这样会不会导致有一些指受到的影响特别大？”

答案是：

\[
\boxed{\text{会，而且目前数据很可能已经表现出了这个问题。}}
\]

---

## 这也可能解释为什么我们一直看不到 task-aligned \(I\)

假设当前 grasp 主要靠：

- index + middle + ring 三点支撑，

但我们的 synergy direction 同时改变：

- index
- middle
- pinky
- ring
- thumb pitch

而最终最大的变化恰好发生在一个**冗余 thumb contact** 上。

那么就会出现：

\[
a\rightarrow I_{\rm thumb}
\]

非常强，

但：

\[
a\not\rightarrow
I_{\rm load-bearing}
\]

所以 object retention 几乎没变化。

这正是我们现在看到的模式。

---

# 我认为这里确实有实验设计上的问题

目前 `finger±` 不是一个很干净的 action intervention。

它实际是在测试一个固定 synergy：

\[
\delta_{\rm finger}
=
[+,+,+,+,0,+]
\]

而不是在测试：

\[
\text{“某一个 contact finger 的控制变化”}.
\]

因此我们无法回答：

> 哪一根手指的动作真正影响 load-bearing interaction？

多个 finger effect 会：

- 互相补偿；
- 一个变差另一个变好；
- 或让整个 hand/object 重新调整。

global contact 仍然保持。

---

# 下一步如果专门解决这个问题，我会改成 per-finger / normalized-physical intervention

而不是继续：

\[
\text{finger synergy}\times\alpha.
\]

例如在 6 个 independent finger DOF 中分别 perturb：

\[
\delta^{index},\
\delta^{middle},\
\delta^{pinky},\
\delta^{ring},\
\delta^{thumb-yaw},\
\delta^{thumb-pitch}.
\]

而且幅度不要按统一 normalized action 定义，最好按**物理 DOF range 比例**或更进一步按**预期 fingertip displacement**归一。

最低成本版本可以规定每个独立 DOF 都产生例如：

\[
5\%\text{ of its own physical joint range}
\]

这样至少：

\[
\frac{\Delta q_i}{q_i^{max}-q_i^{min}}
\]

是一致的。

更干净的是用 Jacobian/FK，把幅度调到各 finger tip 约产生同一级别，例如：

\[
\|\Delta x_{\rm tip}\|\approx 5\text{ mm}.
\]

然后比较：

\[
\text{finger intervention}
\rightarrow
\text{its own distance/force}
\rightarrow
\text{retention}.
\]

---

## 还有一个细节很重要

当前 finger synergy **完全没有扰动 thumb-yaw（index 14）**。

如果 thumb 的 load-bearing 能力很依赖 opposition / abduction 方向，而不是单纯 pitch/flexion，那么现在这个 action set 本身就可能没有覆盖真正重要的 thumb control dimension。

这也是为什么我现在不愿意根据当前：

\[
\text{finger±}\rightarrow\text{thumb response}
\]

就判断“thumb 是最可控/最重要”。

我们只是恰好沿着一种特定的 coupled direction 去推它。

---

所以对你问题的最准确回答是：

> **当前不是等方差随机改所有关节角。它是在 normalized native action space 中，对一个预定义的 5-DOF finger synergy 同时施加等数值 residual；经过各关节 range、mimic coupling 和手指几何后，各指实际受到的物理影响可以非常不均匀。最新 thumb-dominant response 已经说明这个风险是真实的。**

我认为这可能是目前“局部 \(I\) 明显变化、任务结果却不变化”的一个重要混杂因素。下一次如果继续做 control attribution，应该拆开 finger synergy，而不是继续扩大同一个 synergy。