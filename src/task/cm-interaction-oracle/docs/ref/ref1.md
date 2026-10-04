可以，而且结合你们**今天最新的仓库状态**，我认为这比继续直接回归现在的 RTG/G 更值得优先试。但我会把方案稍微改成：

> **RECAP-style 的 advantage 打标签 → 训练 action critic \(C(H,a)\) 判断“这个动作相对当前状态是不是更好的动作”**  
> 而不是  
> **success/failure 打标签 → 训练 critic。**

这两者差别很重要。

### 为什么现在适合转这条路线

我看了最新 `agent/cm-interaction-oracle`。目前最关键的几个事实是：

- 最新 commit 是 `2b1c158cd`，已经专门开始审计 **hold / drop target coverage**。
- 当前 112 个 e260 episode 里只有 **2 个达到 45-step held success**，而且这两个之后**都发生了 drop**。
- 现有 K1 point-flow → G：
  - \(H\): MAE 23.3652
  - \(H+\text{GT E}\): 23.2837，只提升 **0.35%**
  - predicted-E fit: 22.8961，表面提升 **2.01%**
  - 但去掉那个高 RTG episode 后反而**差 5.82%**。
- 因此当前问题已经不是简单“G 网络容量够不够”，而是 **G/RTG 本身未必是合适的动作监督信号**。

所以你现在想到 RECAP 是很自然的转向。

RECAP 本身的核心也不是“成功轨迹动作=1、失败轨迹动作=0”。它先学习 state value，再用一段未来 reward + bootstrap value 估计 action advantage，然后将 advantage 离散成 improvement indicator。也就是说，它问的是：

\[
\text{这个动作之后的结果，相比在这个状态下通常应有的结果，是更好还是更差？}
\]

而不是：

\[
\text{这一整条 episode 最后成功了吗？}
\]

这正好解决你们现在“前面很多动作其实是好的，但 episode 后面又掉了”的 credit assignment 问题。[arxiv.org](https://arxiv.org/abs/2511.14759?utm_source=chatgpt.com)

---

## 我建议你们做的版本

先不要把 Cm 塞进去。第一版可以非常简单。

已有 trajectory：

\[
(H_t,a_t,r_t,H_{t+1},\ldots)
\]

先训练一个 **state-only baseline**

\[
V(H_t)
\]

然后对于当前动作 \(a_t\)，计算一个 N-step outcome：

\[
Y_t^{(N)}
=
\sum_{i=0}^{N-1}\gamma^i r_{t+i}
+
\gamma^N V(H_{t+N})
\]

再计算：

\[
A_t=Y_t^{(N)}-V(H_t)
\]

这个 \(A_t\) 才是真正适合给动作打标签的量。

然后不要强行精确回归它，可以先离散成：

\[
y_t=
\begin{cases}
+1 & A_t > \tau_+\\
0 & \tau_- \le A_t \le \tau_+\\
-1 & A_t < \tau_-
\end{cases}
\]

比如同一个 motion / phase 内：

- top 30% → positive
- bottom 30% → negative
- 中间 40% → neutral / 不参与第一版训练

RECAP 也是把连续 advantage 转成离散 improvement indicator，再用这个信号区分好坏行为。[GitHub](https://github.com/asimfish/embodied-sb-research/blob/main/tracks/pi0.6%2B/paper/detailed_reports/03_pi_star_0.6_RECAP.md?utm_source=chatgpt.com)

然后训练：

\[
C_\theta(H_t,a_t)\rightarrow
P(\text{positive})
\]

这时这个网络才真正是你说的：

> **“给当前动作评分的 critic”。**

---

## 但你们这里我会改掉 RECAP 的一个地方

**不要直接把现有 total RTG 当 \(Y_t\)。**

因为仓库现在已经很明确地告诉我们：

> 高 RTG ≠ 最终稳定 grasp。

那个影响最大的高 RTG episode 最长 hold 8.87 秒，但后来仍然 drop。

你们现在 collector 已经有：

- base reward
- approach
- held
- lift progress
- stable
- contact
- `max_hold_seconds`
- `drop_after_success`

所以我更倾向于让 label 反映 **未来 N 步任务后果**。

第一版甚至可以仍然使用当前 reward，只是加入一个非常明确的持续性约束，例如将近期 drop / contact loss 纳入 outcome。重点不是再设计一个巨复杂 reward，而是：

> **标签评价的是“这个动作之后的一段局部物理结果”，而不是整局最后的成功标签。**

---

# 为什么这可能比你们现在的 G 好

你们目前 G 实际在学：

\[
H,E,I \rightarrow \text{absolute RTG}
\]

这是一个很难的问题。

同一个不错的当前动作，可能因为 3 秒以后策略做错而得到很差的 G；反过来，一个当前一般的动作也可能因为后面 actor 救回来了得到不错的 G。

而我们现在改成：

\[
(H,a)\rightarrow
\underbrace{Y(H,a)-V(H)}_{\text{relative action quality}}
\]

state difficulty 被 \(V(H)\) 消掉了一大部分。

举个很贴合你们数据的情况：

同样一个已经接触物体的状态：

- 动作 A：保持接触并开始 lift；
- 动作 B：让某几个手指松掉；
- 两条 episode 后面都可能最终失败。

如果用终局 success：

\[
A=0,\quad B=0
\]

没有监督。

但 advantage label 可能是：

\[
A=+,\qquad B=-
\]

这才是我们真正想让 critic 学到的东西。

---

## 而且你们已经有一个很好的数据条件

现在 `run_cm_physical_value_environment.py` 本来就在执行：

```text
a = π(H) + noise
```

noise level 已经包括：

\[
0,\ 0.05,\ 0.10,\ 0.20
\]

这实际上已经提供了**动作干预数据**。

这点很重要。

因为如果数据永远只有

\[
a_t=\pi(H_t)
\]

那么训练 \(C(H,a)\) 非常危险：critic 完全可以忽略 \(a\)，只靠 \(H\) 判断这是不是容易成功的状态。

而你们现在有随机 action perturbation，相当于已经有一个初步的 randomized treatment。

所以一定同时训练两个 control：

\[
C_H(H)
\]

和

\[
C_{Ha}(H,a)
\]

真正的第一道 Gate 应该是：

\[
C_{Ha} \gg C_H
\]

如果 \(H+a\) 根本不能比 \(H\) 更好地预测 advantage label，那么：

> **当前数据里并没有足够的“动作选择信息”。**

这时候继续搞 Cm 也没意义。

---

# Cm 在这个方案中应该放在哪里

这一点尤其重要，因为否则最后很容易得到：

> critic 工作了，但 Cm 完全没用。

你之前已经抓到这个问题了：

\[
(H,a)\rightarrow E,I
\]

从信息论上基本是确定的。

那么只要网络能力够：

\[
C(H,a)
\]

理论上完全可以自己隐式学习

\[
(H,a)\rightarrow E/I\rightarrow A.
\]

所以我们的实验必须是：

\[
\boxed{C_H}
\]

↓

\[
\boxed{C_{Ha}}
\]

↓

\[
\boxed{C_{H,a,\hat E,\hat I}}
\]

或者更干净一点：

\[
C(H,a)
\quad\text{vs}\quad
C(H,\hat E,\hat I)
\quad\text{vs}\quad
C(H,a,\hat E,\hat I)
\]

这样最终 Cm 的 claim 才会变成：

> **显式物理 consequence representation 能否让 action-quality critic 更容易泛化 / 排序？**

而不是：

> Cm 给 critic 增加了它本来不知道的信息。

这是两个完全不同的 claim。

我认为前者更合理。

---

## 我甚至不建议第一步回归 continuous Q

第一版就做分类 / ranking。

原因很简单：你们过去已经反复遇到 absolute value regression 被 outlier 和 episode distribution 控制。

所以第一版：

\[
C(H,a)\rightarrow\{-1,0,+1\}
\]

会比

\[
Q(H,a)\rightarrow 23.718
\]

稳定得多。

更进一步，可以直接做 pairwise ranking：

\[
C(H,a_i)>C(H,a_j)
\]

当

\[
A_i>A_j.
\]

最终我们实际使用 critic 的时候，本来也不是特别关心：

> “这个动作 Q=17.42。”

而是关心：

> “8 个 candidate 里面哪个最好？”

所以 **ranking accuracy 比 MAE 更贴近最终用途。**

---

# action chunk 怎么处理

你前面也正好问过 chunk。

这里我认为这套方案和 chunk 非常自然。

单步可以是：

\[
C(H_t,a_t)
\]

chunk 就是：

\[
C(H_t,a_{t:t+k-1})
\]

但**label horizon 和 action horizon 不需要相等**。

例如：

\[
\underbrace{a_{t:t+7}}_{\text{8-step chunk}}
\rightarrow
\underbrace{A_t^{32}}_{\text{未来 32 steps outcome}}
\]

这是完全合理的。

而且比让 critic 只判断一个 30 Hz 的瞬时 action 更有可能形成清晰 contrast。

第一版我会试：

**action chunk = 4 或 8，label horizon = 32。**

`h32` 不是凭空来的：你们 Gate-1 最近的 consequence probe 里，h32 是目前较稳定的探索性候选。不过这个只应作为起始 probe，不应因为之前结果就把它永久固定。

网络也不用复杂。

\[
H\xrightarrow{\text{GRU}}z_H
\]

短 action chunk：

\[
a_{t:t+k-1}\xrightarrow{\text{MLP/小GRU}}z_a
\]

然后：

\[
[z_H,z_a]\rightarrow C
\]

足够了。

---

# 我建议现在把整个路线变成这 4 个 Gate

| Gate | 要回答的问题 |
|---|---|
| **G0 Label** | advantage label 是否稳定、不会被少数 success/drop episode 控制 |
| **G1 Action information** | \(C(H,a)\) 是否明显优于 \(C(H)\) |
| **G2 Cm contribution** | 加 predicted E/I 后，action ranking 是否进一步提高 |
| **G3 Policy utility** | 用 critic 选 candidate 后，真实 closed-loop actor 是否优于 baseline |

这样链条非常干净：

\[
\boxed{\text{future outcome}}
\]

↓

\[
\boxed{\text{advantage label}}
\]

↓

\[
\boxed{\text{action critic}}
\]

↓

\[
\boxed{\text{Cm-enhanced action critic}}
\]

↓

\[
\boxed{\text{candidate selection}}
\]

↓

\[
\boxed{\text{policy improvement}}
\]

---

## 和你之前方案最大的区别

之前我们一直试图走：

\[
E/I\rightarrow G\rightarrow policy
\]

现在应该改成：

\[
\text{真实 outcome}
\rightarrow
\text{action advantage label}
\rightarrow
C(H,a)
\]

然后再问：

\[
\hat E,\hat I
\]

能不能帮助这个 critic。

**我认为这是目前更合理的路线。**

而且它不会把上限锁在 baseline 上：baseline 的 \(V(H)\) 只是用来定义“比当前策略预期更好还是更差”，并不是 critic 最终输出的动作。通过带噪 rollout / 后续 candidate selection，数据分布可以不断扩展。RECAP 本质上也是利用当前策略的数据定义 relative advantage，再用它进行策略改进，而不是要求一开始就知道全局最优动作。[物理智能](https://www.physicalintelligence.company/download/pistar06.pdf?utm_source=chatgpt.com)

**所以我的结论是：可以做，而且我建议把它作为现在 `cm-interaction-oracle` 后面的下一条主线。第一步先不要训练 Cm，也不要重新设计 G；直接用现有带 action noise 的数据做 `cross-fitted V(H) → advantage label → C(H,a) vs C(H)`。如果这一步都过不了，路线可以很便宜地停掉；如果过了，再把 Cm 接进来。**