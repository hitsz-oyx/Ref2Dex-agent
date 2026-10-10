对，**做过，而且已经做到了比“\(E_{\mathrm{GT}}\) 对 \(Y\) 有没有帮助”更后面。** 我刚才把这个 Gate 重复提出来了，这是不对的。

把仓库历史按链条整理，实际上已经有下面几层证据：

| 链路 | 已有结果 | 结论 |
|---|---:|---|
| \(E_{\mathrm{GT}},I_{\mathrm{GT}}\rightarrow Y\) | Ref8：H-only `0.73135` → HEI `0.39311`，误差 **↓46.25%** | **明确有用** |
| 同一合同下 GT E/I→Y | Ref12：H `0.63951` → GT E/I `0.45440`，**↓28.95%** | **明确有用** |
| actual future flow→Y | `0.63951 → 0.45788`，**↓28.40%** | **明确有用** |
| predicted E/I→Y | `0.54712`，改善 **14.45%**，CI 跨 0 | 有趋势但不稳 |
| GT E/I same-state candidate ranking | **67.35%** strict pair accuracy | 有排序信息，但没过当时 70% gate |
| learned actual-flow→Y ranking | **57.38%** | 不够 |
| learned predicted-E/I→Y ranking | **57.82%** | 不够 |
| hybrid | **57.95%** | 不够 |
| GT-Y one-shot → Z | `23/32 → 24/32` | 很弱 |
| **rolling GT-Y → Z** | **23/32 → 27/32，救回4，伤害0** | **PROMISING** |

所以我们其实已经回答了两个非常关键的问题：

\[
\boxed{E_{\rm GT}/I_{\rm GT}\text{ 中确实有 }Y\text{ 所需的信息}}
\]

以及

\[
\boxed{\text{如果 }Y\text{ 足够好并滚动使用，确实能提高最终 }Z}
\]

尤其第二个最重要。`P-20261005-rolling-oracle-control` 已经不是纯离线 correlation，而是真正执行了 same-current-state rolling GT-Y，得到 **23/32→27/32，+12.5pp，4 rescue / 0 harm**。

---

因此，**现在绝对不应该再做一次 \(E_{\rm GT}\rightarrow Y\) 必要性实验。**

真正断掉的是中间这一段：

\[
\boxed{
\text{可部署的未来表示}
\rightarrow
\hat Y
}
\]

因为 oracle 已经给出了上下界：

\[
E/I_{\rm GT}\rightarrow Y
\]

有信息，

\[
Y_{\rm GT}\rightarrow Z
\]

有收益。

但实际学习出来的东西到不了这个水平。

最典型的就是 `P-20261006-actual-flow-y-ranking`：

\[
\begin{aligned}
\text{Direct actual flow}&=57.38\%\\
\text{Pred-EI}&=57.82\%\\
\text{Hybrid}&=57.95\%\\
\text{GT-EI}&=67.35\%
\end{aligned}
\]

而专门做的 noise-tolerance 实验已经给出了经验门槛：**大约要到 70% same-state pair ranking accuracy** 才值得继续往 rolling learned-Y 推。

甚至加噪 GT-Y：

- \(\sigma=.02\)：93.1%
- \(\sigma=.05\)：84.0%
- \(\sigma=.10\)：75.8%
- \(\sigma=.20\)：69.6%

所以那个 `.70` 不是随便定的。

---

### 这也解释了为什么我们后来一直绕

过去我们不是不知道 \(E\) 有没有用。

而是不断卡在：

\[
E_{\rm GT}
\quad\longrightarrow\quad
\hat E
\quad\longrightarrow\quad
\hat Y
\]

这里。

Ref12 当时其实已经给出了非常重要的判断：

\[
\text{actual Flow}\rightarrow Y:\quad 28.40\%
\]

\[
GT(E,I)\rightarrow Y:\quad28.95\%
\]

两者几乎一样。

但：

\[
\text{Flow}\rightarrow \widehat{E,I}\rightarrow Y
\]

只保留了约 14.45% 的收益。

所以我们当时已经得出过一个结论：

> **没有证据证明 E/I 必须作为 Flow→Y 的强制信息瓶颈。**

这就是为什么继续不断修改 `E/I decoder → evaluator` 接法一直效果不好。

---

## 那么现在应该把主线改成什么？

我现在会把优先级改成：

\[
\boxed{
\text{未来手轨迹 }\tau
\rightarrow
Y
\rightarrow
Z
}
\]

先把这个桥做通，而不是重新证明 \(E_{\rm GT}\rightarrow Y\)。

原因是最新仓库实际上出现了一个很值得重视的新信号：

最近 consequence-evaluator 的 **T-only observed-\(\tau\)** 已经做到：

\[
70.07\%
\]

而 shuffle \(\tau\) 后：

\[
45.99\%
\]

下降 **24.09pp**。

而且在 7 个 informative anchors 上，T-only argmax **全部命中 teacher maximum**；加 baseline abstention 后 informative regret 仍为 0。

这和以前 actual-flow learned-Y 的 57.38% 已经不是一个水平了。

所以我认为现在真正值得追的是：

> **为什么现在 observed-\(\tau\rightarrow Y\) 已经摸到 70% gate，而一旦变成生成的 \(\hat\tau\)、或者加入 H / E 分支，就又坏掉？**

这才是当前新的核心问题。

---

于是整条路线应该变成：

\[
\boxed{
H
\rightarrow
\hat\tau
\rightarrow
\hat Y
\rightarrow
\text{selector}
\rightarrow
\tau^*
\rightarrow
A
\rightarrow
Z
}
\]

其中目前各段状态非常清楚：

**\(\tau_{\rm GT}\rightarrow Y\)**：已经出现约 70% 正信号。

**\(Y_{\rm GT}\rightarrow Z\)**：已经证明 rolling 有 +12.5pp。

**\(\tau_{\rm GT}\rightarrow A\)**：今天已经做到 12/16 long hold，PROMISING。

现在最差的一环反而是：

\[
\boxed{H\rightarrow\hat\tau}
\]

因为最近那个 held split：

\[
RMSE_{\rm model}=302.8\text{ mm}
\]

而 persistence：

\[
262.1\text{ mm}
\]

模型甚至比 persistence 差。

所以如果重新审视整个仓库，**我认为目前真正的第一 blocker 已经不是 \(E_{\rm GT}\rightarrow Y\)，而是“如何得到可用的候选 \(\tau\)”**。

PointWorld \(E\) 可以暂时冻结。它后面可以作为额外 consequence feature 再回来，但现在没必要围绕它重复做必要性证明。

因此我会修正上一条建议为：

> **下一步先集中解决 candidate \(\tau\) proposal，而不是再做 GT-E→Y。**
>
> 具体要研究的是：为什么 observed-\(\tau\) 能排序，H→\(\tau\) 却跨 episode 崩掉；是 proposal 数据不足、ACT/chunk 表达问题、还是我们根本不该直接回归唯一未来，而应该生成多模态 candidate \(\tau\)。

这个问题其实正好接上我们之前讨论 ACT、PointWAM、DexWM、DExplore 那一串。下一步我建议就沿着这里继续，把**“H→候选 τ 为什么失败，以及应该用什么 proposal 机制替代现在的单预测器”**彻底梳理清楚。