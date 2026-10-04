可以预测 \(G\)，而且我认为**用 \(G\) 比只预测“成功概率”更合理**。成功率应该保留，但作为终局指标，不应该承担全部监督。

把整条链固定下来，我建议以后不要再改定义：

\[
\boxed{
(s_t,a_t)
\rightarrow
C_t=(E_t,I_t)
\rightarrow
G_t
\rightarrow
\text{action improvement}
}
\]

其中：

- \(E_t\)：物体 effect；
- \(I_t\)：interaction consequence；
- \(G_t\)：从 \(t\) 到 episode 结束的真实 return-to-go。

我们现在首先验证的是中间这一段：

\[
\boxed{C_t^{GT}\rightarrow G_t}
\]

而**蒸馏是这条链验证通过之后，如何把这些知识装进最终策略的工程路线**。两者不是一回事。

---

## 一、为什么我不建议只预测成功率

因为抓取成功：

\[
Y\in\{0,1\}
\]

太粗。

例如下面三个轨迹可能都是失败：

```text
A：刚接触就掉了
B：已经稳定抬起，最后一秒掉了
C：一直抓稳，但最终高度差 2 mm 没过阈值
```

如果只监督：

\[
P(success)
\]

三者标签全部是 0。

这会浪费大量物理信息。

所以主要监督应该是：

\[
\boxed{
G_t=\sum_{k=t}^{T}\gamma^{k-t}r_k
}
\]

这是完整真实 simulator trajectory 给出的 Monte-Carlo return，不 bootstrap，不用 TD。

同时保留：

\[
Y_{\text{success}}
\]

作为第二个 head / 最终评价指标。

也就是说最好：

\[
F(\cdot)
\rightarrow
(\hat G_t,\hat p_{\text{success}})
\]

而不是只有一个 success classifier。

前提当然是你们现在 PPO reward 本身基本反映任务目标。如果 reward 与最终抓取严重错位，就需要同时报告 success、held-lift duration、drop 等，而不能只相信 \(G\)。

---

# 二、第一阶段：我们现在真正要验证什么

数据来自正常的真实 PhysX rollout。

对每个 \(t\) 保存：

\[
H_t=\text{过去一段状态/history}
\]

\[
a_t=\text{实际动作}
\]

然后事后从未来轨迹提取：

\[
C_t^{GT}
=
(E_t^{GT},I_t^{GT})
\]

再计算：

\[
G_t
\]

所以一条数据就是：

\[
\boxed{
(H_t,a_t,E_t^{GT},I_t^{GT},G_t,Y_t)
}
\]

注意：这里完全没有 Cm。

然后做严格消融：

\[
V_H(H)\rightarrow G
\]

\[
V_{HA}(H,a)\rightarrow G
\]

\[
V_{HE}(H,E^{GT})\rightarrow G
\]

\[
V_{HI}(H,I^{GT})\rightarrow G
\]

\[
V_{HEI}(H,E^{GT},I^{GT})\rightarrow G
\]

以及最关键的：

\[
V_{HAEI}(H,a,E^{GT},I^{GT})\rightarrow G
\]

这几个结果回答的是不同问题。

### 关键判断 1

如果：

\[
V_{HEI}\gg V_H
\]

说明：

> 知道真实未来 effect+interaction 后，长期价值确实更容易判断。

这证明 \(E+I\) 有 task-relevant information。

### 关键判断 2

如果：

\[
V_{HAEI}\approx V_{HEI}
\]

更重要。

说明：

> 已经知道 effect+interaction 后，原始 action 本身几乎不再提供额外信息。

这才支持：

\[
a\rightarrow(E,I)\rightarrow G
\]

即 **\(E+I\) 是比较充分的 action consequence representation**。

如果：

\[
V_{HAEI}\gg V_{HEI}
\]

那说明 \(E+I\) 仍然漏了东西。

这时候继续训练 Cm 没意义，应该先改 Cm 的预测目标。

---

# 三、但预测 \(G\) 和“动作怎么变好”之间还差一步

是的。

单纯知道：

\[
C\rightarrow G
\]

还不能自动产生一个动作。

最终需要的是：

\[
Q(H,a)\approx E[G\mid H,a]
\]

或者更适合 residual policy 的：

\[
\boxed{
A(H,a)=Q(H,a)-V(H)
}
\]

即：

> 相比当前 baseline，在当前状态执行这个动作有多值得。

这里 effect+interaction 起到的是**中间解释变量**：

\[
(H,a)
\rightarrow
(E,I)
\rightarrow
G
\]

而不是拿：

\[
-\|E-E_{ref}\|
\]

这种手工式子直接当 action score。

---

# 四、Cm 在这条链里的位置非常明确

第一阶段证明 GT consequence 有价值以后，才训练真正的 Cm：

\[
\boxed{
Cm(H,a)
\rightarrow
(\hat E,\hat I)
}
\]

然后把第一阶段已经训练并冻结的 value bridge 接上：

\[
g(H,\hat E,\hat I)
\rightarrow
\hat G
\]

所以最终得到：

\[
\boxed{
Q_{Cm}(H,a)
=
g(H,Cm(H,a))
}
\]

这时候就可以非常干净地比较：

\[
g(H,E^{GT},I^{GT})
\]

和：

\[
g(H,\hat E,\hat I)
\]

如果前者很好、后者差：

> representation 是对的，Cm 预测是瓶颈。

如果两个都好，但实际策略不涨：

> action learning / policy integration 是瓶颈。

如果 GT 的都不好：

> idea/representation 本身有问题。

这才是完整归因链。

---

# 五、蒸馏到底是什么位置？

**蒸馏在最后。**

它不是现在的验证方法。

假设最终我们已经有：

\[
Q_{Cm}(H,a)
\]

或者更理想：

\[
A_{Cm}(H,a)
\]

那么我们可以让一个 teacher 根据这个价值判断产生动作偏好。

比如 actor 当前动作：

\[
a_\pi
\]

teacher 找到一个更好的 correction：

\[
a_T
\]

然后训练最终 actor：

\[
\pi_\theta(H)\rightarrow a_T
\]

这叫：

\[
\boxed{\text{teacher-to-actor distillation}}
\]

部署时可以不跑 teacher，甚至不跑 Cm：

\[
H
\rightarrow
\pi_{\theta}
\rightarrow
a
\]

所以蒸馏解决的是：

> **已经有一套有用知识以后，怎么把它压进一个快速策略。**

它不回答：

> effect+interaction 本身是不是有用。

---

## 这和 V1.18 有什么区别？

你们 V1.18 已经做过 planner-to-actor distillation 的框架。

但那里的 teacher 大致是：

\[
Cm
\rightarrow
\hat E
\rightarrow
\text{reference-effect cost}
\rightarrow
a_T
\]

然后：

\[
\pi\leftarrow a_T
\]

也就是说 teacher 的“价值判断”很大程度来自**人工定义的 effect distance**。

而现在如果前面的链成立，新的逻辑应该是：

\[
Cm
\rightarrow
(\hat E,\hat I)
\rightarrow
\underbrace{g(\hat E,\hat I)}_{\text{由真实 }G\text{ 监督学到}}
\rightarrow
A/Q
\rightarrow
a_T
\rightarrow
\pi
\]

最大的区别就在中间：

以前：

\[
\boxed{\text{effect}\rightarrow\text{人工 score}}
\]

现在：

\[
\boxed{\text{effect+interaction}\rightarrow\text{真实长期 return 学到的 value}}
\]

所以这不是简单重复 V1.18。

---

# 六、还有一条“直接蒸馏未来”的路线，但要和 Cm 路线分开

我们也可以训练一个拥有 GT future 的 teacher：

\[
T(H,a,E^{GT},I^{GT})\rightarrow G
\]

然后直接蒸馏成：

\[
S(H,a)\rightarrow G
\]

甚至直接蒸馏 actor。

这当然可能有效。

但必须注意：

\[
\boxed{\text{它绕开了 Cm}}
\]

如果这个 student 最后表现很好，我们只能说：

> future privileged information 可以帮助训练策略。

不能说：

> OI-Cm 有用。

所以我会把它定义成**独立 upper-bound/control experiment**，而不是主路线。

它很有价值，因为它还能回答：

> 如果我不强制 student 显式预测 \(E,I\)，只要求它模仿 future-aware teacher，它最多能学到多少？

---

# 七、因此整个研究链现在可以固定成四个 Gate

### Gate 1：Representation sufficiency

完全没有 Cm：

\[
(E^{GT},I^{GT})\rightarrow G
\]

问：

> GT effect+interaction 是否足以解释长期价值？

**现在就做这个。**

---

### Gate 2：Cm predictability

\[
(H,a)\rightarrow(\hat E,\hat I)
\]

问：

> 部署时可用的当前信息，能否预测 Gate 1 所证明有价值的 consequence？

---

### Gate 3：Value preservation

同一个冻结 bridge：

\[
g(H,E^{GT},I^{GT})
\]

对比：

\[
g(H,\hat E,\hat I)
\]

问：

> Cm 的预测误差到底损失了多少真正的长期价值信息？

---

### Gate 4：Policy transfer

最后才比较：

- 在线 `Cm + bridge`；
- planner/selector；
- actor distillation；
- 或 advantage-weighted PPO auxiliary。

问：

> 已经证明有用的信息能否转换成最终 policy gain？

---

所以我现在明确的建议不是“马上做蒸馏”。

**现在做 Gate 1。**

而且 Gate 1 的主要 target 用：

\[
\boxed{G_t=\text{真实 Monte-Carlo return-to-go}}
\]

成功率作为辅助/最终 sanity check。

只有当：

\[
(E^{GT},I^{GT})
\]

能够明显提高 \(G_t\) 的 held-out 预测，并且原始 action 在给定 \(E,I\) 后不再增加多少信息，我们才有理由说：

> **OI-Cm 预测 effect+interaction 这个科学目标本身值得继续。**

然后 Cm、bridge、蒸馏自然就顺着这一条链往下走，而不是再选一个接法碰运气。