可以。**如果我们现在正式决定“旧 \(Y\) 就是 teacher label”，那主线可以直接进入 evaluator，不需要再为 \(Y\) 做一轮定义研究。**

而且你提出的三组对照基本就是现在最应该做的实验。不过我会稍微修正一下输入，保证三组严格可比：

| 组 | 输入 | 目标 | 回答的问题 |
|---|---|---|---|
| C0 | \(H+A\) | \(U_{\text{old}}\) | 不看未来，仅靠当前状态和动作能预测多少？ |
| C1 | \(H+A+Z^{GT}\) | \(U_{\text{old}}\) | **真实未来 consequence 到底增加多少信息？** |
| C2 | \(H+A+\hat Z^{PW}\) | \(U_{\text{old}}\) | PointWorld 保留了多少 GT-future 的价值信息？ |

这里我不建议 C1/C2 去掉 \(A\)。三组都应该有相同的 \(H+A\) 基础输入，只让“有没有 future、future 是 GT 还是预测”成为唯一变量。

而且这里的 \(A\) 不能只是“当前一步 action”，而应该是**当前决策时刻已知的 candidate action chunk / hand trajectory**。

---

### 标签就用冻结的老 \(Y\)

严格来说，我们以前的 `short_y` 是 8 维，然后实际决策用：

\[
\boxed{
U_{\rm old}
=
Y_7+0.25Y_3-Y_6
}
\]

所以 evaluator 最终 target 我建议直接冻结成这个：

\[
Q_\theta\rightarrow U_{\rm old}.
\]

不要再改权重，不再调阈值。

如果想保留可解释性，可以额外加一个 auxiliary head 去预测原来的 8 个 \(Y\) 分量，但 planner 用的主输出仍然是冻结的：

\[
U_{\rm old}.
\]

原因很简单：真正经过 rolling 验证的是这个 utility，而不是我们重新发明的其它组合。

---

## 候选还要不要？

这里要把“训练”和“验证”分开。

**训练 evaluator 不要求每个样本都有 7 个 candidate。**

任何一条轨迹，只要我们有：

\[
(H,A,Z^{GT})
\]

以及后面的 32 步：

- height；
- pair/contact；

就可以算：

\[
U_{\rm old}.
\]

所以完全可以把大量 intervention trajectory 拆成普通监督样本：

\[
(H,A,Z^{GT})\rightarrow U_{\rm old}.
\]

因此：

\[
\boxed{\text{不用再等一个新的 candidate oracle 实验，才能开始训练 evaluator。}}
\]

旧 \(Y\) 是否值得作为 teacher，我们其实已经有真实 rolling evidence：

\[
23/32\rightarrow27/32
\]

以及最近恢复出来的：

\[
20/25\rightarrow23/25.
\]

这一步已经足够让我们往下走了。

---

但是：

\[
\boxed{\text{candidate 对 evaluator 的最终验证仍然必须有。}}
\]

因为我们最终根本不是要做：

> “把 \(U\) 回归得 MSE 很低。”

我们要做的是：

\[
A_1,\ldots,A_K
\rightarrow
Q_1,\ldots,Q_K
\rightarrow
\arg\max_iQ_i.
\]

所以 candidate 不再是**定义 Y 的前置实验**，而是 evaluator 的**考试题**。

这个定位非常关键。

---

# 我会把实验拆成两层

第一层现在立刻可以训练。

用已有 intervention 数据训练 C0/C1/C2，比较：

\[
\text{MAE/RMSE}
\]

只是基础指标，更重要的是：

\[
\text{Spearman / pairwise ranking}.
\]

另外给 C1 做一个零成本诊断：

\[
Z^{GT}\rightarrow \text{shuffle}(Z^{GT})
\]

如果 C1 真利用未来，那么 shuffle 以后性能应该明显掉。

第二层才拿完全 held-out 的 same-\(H\) candidate panel：

\[
H\rightarrow
\{A_0,\ldots,A_6\}
\]

对应：

\[
\{Z_0^{GT},\ldots,Z_6^{GT}\}.
\]

直接比较：

\[
\operatorname{rank}U_{\rm old}
\]

和：

\[
\operatorname{rank}Q_{C0},
\quad
\operatorname{rank}Q_{C1},
\quad
\operatorname{rank}Q_{C2}.
\]

最终指标：

\[
\boxed{\text{pairwise accuracy}}
\]

\[
\boxed{\text{top-1 agreement}}
\]

\[
\boxed{
\text{regret}
=
U^*_{\rm old}-U_{\rm old}(\hat A)
}
\]

以及最重要的：

\[
\boxed{\text{真实 Z90 / task success 的 rescue / harm}}
\]

这就够了。

---

## 现在已有 candidate 数据其实可以直接发挥作用

我们刚刚恢复的 ref13 已经有：

\[
25\times7
\]

same-\(H\) candidate panel，而且 prefix 合同通过。

所以它特别适合当一个**冻结的 evaluator screening test**。

不要拿这 25 个 anchor 去训练，然后再在它们上面汇报结果。

把它们整个留作 test：

\[
\boxed{\text{train on other intervention data}}
\]

然后：

\[
\boxed{\text{test on ref13 same-H candidates}}
\]

这样非常干净。

如果 C1 在这里能够明显：

\[
C1>C0,
\]

就证明：

\[
\boxed{\text{GT future consequence 确实帮助恢复 old-Y 的动作排序。}}
\]

这一步一过，我们的核心假设其实就成立了一大半。

---

# 但有两个合同问题现在必须提前固定

第一个是 **PointWorld 只有 24 步，旧 \(Y\) 用 32 步**。

这个不要通过修改旧 \(Y\) 来解决。

保持：

\[
\boxed{U_{\rm old}^{32}}
\]

完全不变。

三个 evaluator 都只允许使用：

\[
Z_{1:24}.
\]

于是 C1 实际问：

\[
(H,A,Z^{GT}_{1:24})
\rightarrow
U_{\rm old}^{32}.
\]

这反而是一个非常好的 gate。

如果连 GT 的前 24 步都预测不好 32-step old utility：

\[
C1\approx C0,
\]

那说明：

\[
\boxed{\text{24步 horizon 本身信息不足}}
\]

这时候我们才考虑把 PointWorld 扩到 32。

如果：

\[
C1\gg C0,
\]

那说明 24 步 consequence 足够，PointWorld 当前 horizon 可以继续用。

这样不用现在贸然重训 PointWorld。

---

第二个问题是旧 \(Y\) 用了 `force-pair/contact`，而 PointWorld 当前主要预测几何。

这个也**不要偷偷把 contact 喂给 C1，却不给 C2**。

C1 和 C2 的 \(Z\) 必须完全同构。

例如都只给：

\[
\boxed{
\text{object SE(3)/velocity}
+
\text{future hand geometry}
+
\text{hand-object relative geometry}
}
\]

C1 用 GT object future。

C2 把它换成：

\[
\hat Z^{PW}.
\]

旧 \(Y\) 中的 contact 只用于生成监督标签：

\[
U_{\rm old}.
\]

这样实际上是在做：

> 能不能从纯几何 consequence 中恢复一个原本包含 contact 信息、且已经验证有控制价值的 teacher？

这是一个非常好的研究问题。

如果：

\[
C1
\]

都恢复不了 old \(Y\)，那说明我们的 consequence representation 缺 interaction/contact 信息，应该给 PointWorld/interaction head 补信息。

如果：

\[
C1\text{ 很好，C2差}
\]

才说明主要是 PointWorld 精度问题。

如果：

\[
C1\approx C2\gg C0,
\]

那就是我们最想看到的结果：

\[
\boxed{\text{PointWorld 的预测已经足以支撑任务价值判断。}}
\]

---

# 所以主线现在终于可以真正固定下来

不再研究第四个 Y。

直接：

\[
\boxed{
U_{\rm old}\text{ frozen}
}
\]

↓

\[
\boxed{
C0:\ H+A
}
\]

\[
\boxed{
C1:\ H+A+Z^{GT}
}
\]

\[
\boxed{
C2:\ H+A+\hat Z^{PW}
}
\]

↓

在 held-out same-\(H\) candidates 上看排序

↓

如果：

\[
C1>C0
\]

说明 consequence 有必要；

如果：

\[
C2\approx C1
\]

说明 PointWorld 精度够；

如果：

\[
C2<C1
\]

就量化 world-model gap。

然后才进入：

\[
\text{candidate planning}
\rightarrow
\text{execution bridge}
\rightarrow
\text{RL}.
\]

所以回答你最后两个问题：

**可以直接开始训练 evaluator。**

而“候选还有没有必要”的准确答案是：

\[
\boxed{\text{不再需要候选来继续证明/定义老Y；但必须保留候选作为 evaluator 和 planner 的最终考试。}}
\]

我认为这一步之后，主线就不应该再停在 Y 上了。