我看了。现在 `agent/cm-interaction-oracle` 的最新 HEAD 已经到 **`fe96efccff`**，比我们刚才看的 `2b1c158cd` 又前进了 4 个 commit。最重要的是：**仓库已经把我们刚讨论的 RECAP-style relative advantage → action critic 方案实现并跑了一轮，但在 G0 标签可靠性检查就主动停了。**

### 当前实际做到了哪里

新增的主实验是：

`P-20261004-recap-relative-action.md`

流程基本就是我们刚才讨论的：

\[
H_t \rightarrow V(H_t)
\]

然后构造

\[
A_t^{32}
=
\sum_{i=0}^{31}\gamma^i r_{t+i}
+
\gamma^{32}V(H_{t+32})
-
V(H_t)
\]

再根据 advantage 给动作打 positive / neutral / negative 标签，之后原计划训练：

\[
C(H,a)
\]

并比较：

\[
C(H)\quad vs\quad C(H,a_t)\quad vs\quad C(H,a_{t:t+3})
\]

这里做得比较严谨的一点是，它没有直接把 episode success/failure 贴到所有动作上，而是用了 **cross-fitted V**：给某个 episode 生成标签的 V 没看过这个 episode。这一点我认可。

而且它显式控制了现有数据里的 `noise_std`，因为你们的 action noise 是 episode-fixed 的，否则 critic 很容易只识别“这是 0.2 noise 的轨迹”。

---

## 结果为什么停在 G0

数据量本身不是问题：

| 指标 | 结果 |
|---|---:|
| train/test episode | 90 / 22 |
| train query | 11,520 |
| test query | 2,816 |
| positive train/test | 3417 / 840 |
| negative train/test | 3453 / 710 |
| held-out V MSE | 229.58 |
| zero-V MSE | 324.80 |

所以：

\[
V(H)
\]

确实比直接预测 0 好，而且正负标签数量也够。

真正失败的是**标签稳定性**。

两套独立 V fit 得到的离散 advantage label 一致率只有：

\[
34.72\%\quad\text{train}
\]

\[
39.15\%\quad\text{test}
\]

预注册 gate 是：

\[
\ge75\%
\]

所以代码按合同直接停止，**没有训练 G1 action critic**。

这一点我认为仓库处理是正确的：没有因为结果不好就偷偷放低 gate。

---

# 但这个结果其实比“39% 一致率”看起来乐观

这里有一个很重要的细节。

两套 V 得到的**连续 advantage**：

\[
A^{(1)},A^{(2)}
\]

相关性其实非常高：

\[
\rho_{\text{test}}=0.978
\]

去掉最大的 MC-return episode 后仍然：

\[
0.952
\]

而如果只看“两套 V 都认为这是 extreme”的样本，符号一致率有：

\[
82.39\%.
\]

真正把 G0 拉垮的主要是：

> 一套 V 觉得这个样本是 positive/negative，另一套觉得它只是 neutral。

也就是 **neutral/extreme boundary instability**，占 union disagreement 的约 **52.5%**。

真正完全反号：

\[
+\leftrightarrow-
\]

只有约：

\[
8.37\%.
\]

所以现在不能简单理解成：

> “advantage 根本算不出来。”

更准确的是：

> **advantage 的排序/方向相当稳定，但用固定阈值把它硬切成三类不稳定。**

这和我们刚才讨论“第一版可以做分类，但 ranking 可能更符合最终用途”正好对应。

---

## 更关键的问题：当前 32-step label 绝大部分还是靠 V bootstrap

仓库自己也已经查出来了。

test query 中，未来 32 step 里真的出现非零 task reward 的只有：

\[
16.3\%.
\]

在这些真实发生 hold/lift/stable 事件的区域，两套 label 的一致率：

\[
78.0\%
\]

已经达到预期附近。

但在未来 32 步 task reward 完全为 0 的区域：

\[
29.6\%
\]

一致率非常差。

于是绝大多数这类标签实际上是：

\[
A_t
\approx
\gamma^{32}V(H_{t+32})-V(H_t)
\]

而不是由实际发生的物理事件决定。

所以现在的问题已经很清楚：

\[
\boxed{\text{不是 action critic 失败，而是当前 V 差分主导了 label}}
\]

两个高度相关的 V，只要 absolute error 有一点差异，做：

\[
V(s')-V(s)
\]

以后误差会被放大，再经过 ±0.05 threshold，就会出现大量 neutral/extreme 翻转。

这就是为什么：

\[
\rho(A_1,A_2)=0.978
\]

但 discrete agreement 只有 39%。

---

# 我对这轮实现的评价

我认为**方向基本正确，而且比之前直接 E/I → RTG 的路线更干净**。

仓库这次也没有犯以前那种“一个负 Probe 就关闭整个科学假设”的错误。最新 `STATE.md` 写得比较准确：

> 当前暂停原因是 label/value contract reliability，不是 RECAP 被否定，也不是 action information 被否定。

这个判断我同意。

不过我觉得现在的 stop 条件**对“离散分类 label”是正确的，对整个 relative-action 路线则稍微过于保守**。

因为已经有一个相当强的信号：

\[
A_1\leftrightarrow A_2
\]

连续排序高度一致。

这正意味着下一步**不应该继续修这个三分类 threshold**，而应该问：

> 我们真的需要 \(-1/0/+1\) 吗？

---

# 我建议下一步不要继续调 V

不要：

- 换 V seed；
- 改 30/70% quantile；
- 把 threshold 从 0.05 改 0.03；
- 只选 future-reward-active 样本；
- 继续调 GRU width。

这些都会变成对这个 G0 的后验修补。

我会新开一个明确不同的 Probe：

\[
\boxed{\text{continuous / ranking action critic}}
\]

直接保留：

\[
A_t
\]

的连续信息。

最简单可以训练：

\[
C(H,a)\rightarrow A_t
\]

但评价**不要主要看 MSE**，而看：

\[
\text{Spearman}(C,A)
\]

以及最重要的：

\[
P(C(H,a_i)>C(H,a_j)\mid A_i>A_j).
\]

或者直接做 pairwise loss：

\[
\mathcal L
=
-\log \sigma
\left(
(C_i-C_j)\operatorname{sign}(A_i-A_j)
\right).
\]

这样 neutral boundary 根本不存在。

---

## 但这里还有一个更本质的问题

现在的数据依然是：

\[
a_t=\pi(H_t)+\epsilon_t
\]

但每个状态基本只观察过**一个 action**。

所以即使：

\[
C(H,a)>C(H)
\]

最后通过，也只能说明：

> action 与未来 relative outcome 有额外预测关系。

还不能证明：

> critic 能在**同一个状态下比较 candidate actions**。

这是我们最终真正需要的。

所以我现在会把路线稍微调整成：

\[
\text{现有数据}
\rightarrow
\text{先证明 action-sensitive ranking}
\]

如果过了，再做一个非常小的真正 counterfactual 数据集：

同一类/极近邻状态附近采：

\[
a_0,\quad
a_0+\delta_1,\quad
a_0+\delta_2,\ldots
\]

然后观察真实短期物理 consequence / outcome。

这才是真正测试：

\[
\boxed{C(H,a_i)>C(H,a_j)}
\]

是否能指导 candidate selection。

---

# 所以当前仓库状态，我会这样概括

之前是：

\[
E/I \rightarrow G
\]

这条路的 G target 很弱，而且被少数 episode 控制。

现在已经真正转成：

\[
\boxed{
\text{trajectory}
\rightarrow
V(H)
\rightarrow
A(H,a)
\rightarrow
critic
}
\]

这是正确的结构性变化。

但当前停在：

\[
\boxed{G0:\ \text{离散 advantage label 不稳定}}
\]

并且**critic 根本还没被测试**。

所以现在绝对不能得出：

> `C(H,a)` 不行。

也不能得出：

> Cm 不适合做 critic。

目前我们只知道：

> **用两个 point-estimate V 的 n-step difference，再用固定 threshold 离散成三类，作为监督标签，不够稳定。**

我认为下一步最值得做的是 **保留现有 cross-fit advantage，不再离散，直接做 matched `H` vs `H+a` 的 continuous/ranking G1 probe**。如果 `H+a` 连 action ranking 都没有增益，再停这条路线；如果有，再接 predicted \(E/I\)。这会比继续修当前 G0 更直接。