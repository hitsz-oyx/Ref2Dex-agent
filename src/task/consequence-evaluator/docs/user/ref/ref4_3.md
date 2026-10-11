对，你这个质疑是成立的。**如果我们一发现 \(\Delta P\) 不好，就退回到“高度、接触、掉落、held fraction”这些手工项重新拼 reward，那确实又绕回手工奖励了。**这不应该是主路线。

但这里有个关键点：**Motus2 也确实用了时序监督，只是它用时序的方式和我们现在不一样。**我查了它的正式公式后，这个差别其实很大。

Motus2 对成功轨迹的一个长度为 \(\Delta t\) 的 segment，直接定义：

\[
r_t=+\frac{\Delta t}{T-t}
\]

而失败或任务无关轨迹的 segment 定义：

\[
r_t=-\frac{\Delta t}{T-t}.
\]

然后不是直接拿这个公式在线算 candidate value，而是训练一个：

\[
\boxed{
V_\theta(c_t,A_t,Z_t)
}
\]

让它根据**当前历史 + 动作 + 真实/预测未来**学习这个时序 value。测试时真正排序 candidate 的，是这个 learned evaluator，不是“candidate 离成功参考走近了多少”。[Anonymous View](https://rt.http3.lol/index.php?q=aHR0cHM6Ly9hcnhpdi5vcmcvaHRtbC8yNjA4LjMwMjM3djE)

这和我们的：

\[
Y=P(H_{t+24})-P(H_t)
\]

不是一回事。

---

## 我们现在最大的偏差其实在这里

我们为了避免直接用 \(t/T\)，做了一个更复杂的：

\[
H_t
\rightarrow
\text{TCC/reference matching}
\rightarrow
P_t
\]

然后：

\[
Y=P_{t+24}-P_t.
\]

它的逻辑是：

> “未来更像成功 reference 的后面部分，就更好。”

这会天然惩罚**偏离 reference 但其实是有用的 corrective action**。

Motus2 没有要求未来：

\[
Z_t
\]

去贴某一条成功参考。

它只告诉 evaluator：

> 这段数据来自一个成功执行，时间在往任务终点推进，所以给正监督。

以及：

> 这段数据来自失败/无关执行，所以给负监督。

然后让大模型自己从：

\[
(H,A,Z)
\]

学“为什么这个未来好或坏”。[Anonymous View](https://rt.http3.lol/index.php?q=aHR0cHM6Ly9hcnhpdi5vcmcvaHRtbC8yNjA4LjMwMjM3djE)

所以本质区别是：

\[
\boxed{
\text{我们：先手工构造 phase，再把 phase difference 当 value}
}
\]

而 Motus2 是：

\[
\boxed{
\text{时序只负责提供弱标签，真正的 value 由数据学习}
}
\]

这其实更符合我们最开始想做 evaluator 的思路。

---

# 为什么这种很简单的时序标签反而可能比我们的 TCC 更好？

因为成功轨迹里的“时间”并不是在告诉模型：

> 第 60 帧一定比第 40 帧几何上好。

它只是一个**弱排序约束**：

\[
\text{successful trajectory: later segment generally means more task progress}.
\]

真正决定 value 的输入仍然有：

\[
H,\quad A,\quad Z.
\]

比如两个 candidate 都在同一个 \(t\)：

\[
A_1\rightarrow Z_1
\]

\[
A_2\rightarrow Z_2.
\]

它们的训练标签不是根据“谁更靠近某条 reference”在线生成的。

Evaluator 过去已经从很多：

- 成功轨迹；
- 失败轨迹；
- 次优轨迹；
- task-irrelevant interaction

里学习到：

\[
Z_1\text{ 这种未来通常是什么性质。}
\]

Motus2 特别强调，**failed 和 suboptimal interactions 不用于 action imitation，而用于 dynamics/value learning**。这其实就是它避免“时间=价值”过于简单的关键。[arXiv](https://arxiv.org/abs/2608.30237?utm_source=chatgpt.com)

---

## 举一个和我们特别相关的例子

假设当前抓着飞机。

候选 A：

> 手暂时往 reference 外偏一点，但把快滑掉的飞机重新夹紧。

候选 B：

> 沿着 reference 的标准轨迹继续抬，但飞机开始滑。

我们的 TCC：

\[
\Delta P_A < \Delta P_B
\]

很可能选 B。

因为 A 偏离成功 reference。

但如果 evaluator 在训练里看过大量：

\[
\text{slip}\rightarrow\text{failure}
\]

和：

\[
\text{regrasp}\rightarrow\text{success},
\]

它完全可能学出：

\[
V(H,A,Z_A)>V(H,A,Z_B).
\]

即便：

\[
Z_A
\]

几何上没有沿着某条 reference phase 向前。

这就是 learned evaluator 比 reference-progress 更有表达能力的地方。

---

# 那 Motus2 不也是“手工定义正负”吗？

严格来说，是。

它至少人为规定了：

\[
\text{success trajectory}\rightarrow+
\]

\[
\text{failed/irrelevant trajectory}\rightarrow-
\]

以及一个相对时间公式：

\[
\frac{\Delta t}{T-t}.
\]

所以它并不是“完全无人工先验”。

但它需要的人工信息非常弱：

\[
\boxed{
\text{episode成功/失败}
+
\text{时间戳}
}
\]

而不是：

- 高度多少厘米；
- 接触力阈值；
- held 多少帧；
- slip 扣多少；
- wrist error 权重多少；
- 各项 reward coefficient 怎么配。

这两类“手工”的性质完全不同。

前者更接近：

\[
\boxed{\text{weak supervision}}
\]

后者才是真正的：

\[
\boxed{\text{reward engineering}}.
\]

所以我们完全可以保持“非手工 reward”的路线。

---

# 很多其他时序方法为什么也能工作？

原因也类似。

例如 TimeRewarder 直接从被动视频里学习 frame-wise temporal distance，把“朝任务未来推进”作为 dense reward，然后用这个 reward 帮 RL；它在多个 manipulation task 上确实有效。[arXiv](https://arxiv.org/abs/2509.26627?utm_source=chatgpt.com)

更早的 HOLD 也是从人类视频中学习时间/目标距离表示，再拿这种 learned distance 给机器人提供 reward prior。[arXiv](https://arxiv.org/abs/2211.09019?utm_source=chatgpt.com)

这些方法共同的思想不是：

\[
\boxed{\text{时间本身就是真实价值}}
\]

而是：

\[
\boxed{
\text{时间顺序是非常廉价、规模很大的弱监督}
}
\]

用它把 representation/value model 学出来。

成功的前提通常是：

1. demonstration 大体是朝任务完成方向推进的；
2. 有很多不同 demonstration，模型不能单纯记某条轨迹；
3. 最好还有 failure / negative / goal conditioning；
4. 最终用真正任务 success 验证，而不是默认 temporal reward 就正确。

---

# 这反而说明我们可能把问题做复杂了

我现在会重新审视我们从：

\[
t/T
\]

一路发展到：

\[
TCC
\rightarrow physical\ reference\ bank
\rightarrow P_t
\rightarrow \Delta P
\]

这条链。

最开始我们担心：

> 同一个绝对时间，不同 episode 的 task progress 不一样。

这个担心没错。

但是我们的解决方法相当于试图先构造一个**非常准确的 progress coordinate**：

\[
P(H)
\]

再让：

\[
\Delta P
\]

直接承担 value。

可能没必要。

Motus2 给的答案其实更直接：

\[
\boxed{
\text{不要先把 progress 算准；直接让 evaluator 从时序弱标签学。}
}
\]

---

# 对我们来说可以非常自然地改

我们已经有大量 Gym trajectory，而且天然知道 episode 最后：

\[
S\in\{0,1\}.
\]

对于一个 \(K=24\) 的 window：

\[
(H_t,A_{t:t+23},Z_{t:t+23})
\]

可以给最简单的 supervision：

\[
y_t=
\begin{cases}
+\dfrac{24}{T-t}, & \text{success episode}\\[6pt]
-\dfrac{24}{T-t}, & \text{failed episode}
\end{cases}
\]

然后训练：

\[
\boxed{
Q_\theta(H,A,Z)\rightarrow y
}
\]

这和 Motus2 非常接近。

不需要：

- TCC；
- reference matching；
- physical reference bank；
- 设计 S/P/M；
- 手工 contact reward。

甚至 \(Z\) 可以先直接用 **GT future**。

于是我们的链条重新变得特别干净：

\[
(H,A,Z^{GT})
\rightarrow Q
\]

先验证：

\[
\arg\max_i Q(H,A_i,Z_i^{GT})
\]

是否提高 \(Z_{\rm task}\)。

如果有效，再换：

\[
Z^{GT}\rightarrow \hat Z_{\rm PointWorld}.
\]

这其实就是我们原本“从后往前”的逻辑。

---

## 但我不建议完全照抄 Motus2 的正负标签

因为我们有一个比它更好的条件：

\[
\boxed{\text{我们在 simulator 里有大量真实 outcome，而且可以知道 recovery。}}
\]

一个失败 episode 的前半段可能完全正确，只是最后掉了。

如果把失败 episode 的所有 segment 都标成负：

\[
-\frac{\Delta t}{T-t},
\]

会产生 label noise。

Motus2靠大数据、失败/次优样本和大模型容量可能扛得住这种弱监督；它甚至在附录里展示了失败轨迹的预测 value 可以先升后降，说明 learned model 并不是简单输出 episode 的固定正负。[Anonymous View](https://rt.http3.lol/index.php?q=aHR0cHM6Ly9hcnhpdi5vcmcvaHRtbC8yNjA4LjMwMjM3djE\&utm_source=chatgpt.com)

我们数据没那么大，所以可以稍微做得更合理，但仍然**不需要手工 reward**。

例如只使用：

\[
\boxed{\text{success / failure + temporal position}}
\]

作为最基础标签，然后把 recoverable failure 作为独立类别或不确定样本，而不是设计几十项 reward。

---

# 我现在最推荐的是把 TCC 降级，而不是放弃时序

结构变成：

\[
\boxed{
\text{时序}=\text{监督来源}
}
\]

而不是：

\[
\boxed{
\text{时序 progress}=\text{最终 value 公式}
}
\]

即：

\[
(H,A,Z)
\xrightarrow{\text{learned evaluator}}
V.
\]

监督仍然来源于：

\[
\text{success/failure}
+
\text{trajectory time}.
\]

TCC 可以留作辅助 feature：

\[
P_t
\]

但不再规定：

\[
V=\Delta P.
\]

这个差别非常重要。

---

所以你问“为什么 Motus2 可以，我们不行”，答案不是：

> 我们这个任务特殊，时序方法不适合。

而更可能是：

\[
\boxed{\text{我们把时序监督用错了层级。}}
\]

Motus2把时间当**weak label**，让 evaluator 学价值；我们却先把时间/参考轨迹压成一个 \(P\)，然后直接规定：

\[
Y=\Delta P.
\]

后者约束强得多，也更容易把“偏离 reference 但有益”的动作判错。

因此我现在不赞成退回手工 reward。**更合理的主线恰恰是回到 learned evaluator：用成功/失败轨迹的时序作为弱监督，GT future 作为输入，让网络自己学习什么样的 future consequence 值得执行。**这和 Motus2 的方法论是一致的，而且比我们当前的 physical-reference \(\Delta P\) 简洁很多。